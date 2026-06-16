import os

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
import json
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
import torch
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.utils.helpers import get_backward_observation
import joblib
import mediapy as media
import numpy as np
from torch.utils._pytree import tree_map

import humanoidverse
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

# ---------- inference metric helpers ----------
_BODY_DOF_SLICE = slice(0, 22)
_HAND_DOF_SLICE = slice(22, 29)


def _wrist_local_pos_slice(use_root_height_obs: bool) -> slice:
    """Right wrist (right_wrist_yaw_link) local pos in privileged_state.

    G1 29-DOF: body index 29 → local_body_pos[(29-1)*3] = 84.
    +1 offset when root_height_obs=True (root_height occupies privstate[0]).
    """
    start = 85 if use_root_height_obs else 84
    return slice(start, start + 3)
# ---------- end metric helpers ----------


def main(model_folder: Path, data_path: Path | None = None, headless: bool = True, device="cuda", simulator: str = "isaacsim", save_mp4: bool=False, disable_dr: bool = False, disable_obs_noise: bool = False, motion_list: list[int] = [25], episode_len: int | None = 100, video_name: str | None = None):
    # motion_list: motion ids to evaluate (default [25])
    
    model_folder = Path(model_folder)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = "model"
    model_name = model.__class__.__name__
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)

    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"
    # import ipdb; ipdb.set_trace()
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    # Outputs under model_folder/tracking_inference (sibling of exported/)
    output_dir = model_folder / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    z_export_dim = model.cfg.archi.total_z_dim
    export_meta_policy_as_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + z_export_dim)},
        z_dim=z_export_dim,
        history=('history_actor' in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"Exported model to {output_dir}/{model_name}.onnx")

    def tracking_inference(obs) -> torch.Tensor:
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    # rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("="*80)
    print(env.config.simulator)
    print("-"*80)
    
    output_dir = model_folder / "tracking_inference"

    for MOTION_ID in motion_list:
        env.set_is_evaluating(MOTION_ID)
        # we visulize the first env
        obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)

        expert_qpos = np.concatenate([
            obs_dict["ref_body_pos"][:,0].cpu().numpy(),
            np.roll(obs_dict["ref_body_rots"][:,0].cpu().numpy(),1,axis=-1),
            obs_dict["dof_pos"].cpu().numpy()
        ], axis=-1)

        # import ipdb; ipdb.set_trace()

        z = tracking_inference(tree_map(lambda x: x[1:], obs))
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(z.cpu().numpy(), output_dir / f"zs_{MOTION_ID}.pkl")
        print(f"Saved zs_{MOTION_ID}.pkl")
        
    observation, info = wrapped_env.reset(to_numpy=False)

    # Root state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3). Isaac expects quat as wxyz; motion lib uses xyzw.
    ref_body_rots = obs_dict["ref_body_rots"][0, 0].clone()
    if simulator == "isaacsim":
        ref_body_rots = ref_body_rots[[3, 0, 1, 2]]  # xyzw -> wxyz for correct humanoid facing in Isaac
    ref_root_init_state = torch.cat(
            [
                obs_dict["ref_body_pos"][0, 0],
                ref_body_rots,
                obs_dict["ref_body_vels"][0, 0],
                obs_dict["ref_body_angular_vels"][0, 0],
            ]
        )
    dof_init_state = torch.zeros_like(wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0])
    dof_init_state[..., 0] = obs_dict["dof_pos"][0]
    dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]
    target_states = {
        "dof_states": dof_init_state,
        "root_states": torch.stack([ref_root_init_state.clone() for i in range(num_envs)])
    }
    env_ids = torch.arange(num_envs, dtype=torch.long)
    observation, info = wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
    # refresh_env_ids = wrapped_env._env.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
    # wrapped_env._env.simulator.set_actor_root_state_tensor(refresh_env_ids, wrapped_env._env.target_robot_root_states)
    # wrapped_env._env.simulator.set_dof_state_tensor(refresh_env_ids, wrapped_env._env.target_robot_dof_state)
    # wrapped_env._env.need_to_refresh_envs[refresh_env_ids] = False
    observation_new, reward, terminated, truncated, info = wrapped_env.step(torch.zeros((num_envs, wrapped_env.action_space.shape[-1]), dtype=torch.float32), to_numpy=False)
    observation = wrapped_env._get_g1env_observation(to_numpy=False)
    qpos, qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
    assert np.allclose(wrapped_env._env.simulator.dof_pos.clone().cpu(), expert_qpos[0,7:])
    joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]
    _wrist_slice = _wrist_local_pos_slice(use_root_height_obs)
    try:
        _z_body_dim: int = model.cfg.archi.z_body_dim
    except AttributeError:
        _z_body_dim = model.cfg.archi.total_z_dim
    _action_dim = wrapped_env.action_space.shape[-1]
    sim_dev = wrapped_env._env.device
    _last_act_buf = torch.zeros((num_envs, _action_dim), dtype=torch.float32, device=sim_dev)
    _z_actual_list:    list[np.ndarray] = []
    _ee_local_pred_list: list[np.ndarray] = []

    # Rollout / video length (None = full motion from z)
    if episode_len is None:
        episode_len = z.shape[0]
    else:
        episode_len = min(episode_len, z.shape[0])
    print(f"Saving video for tracking ({episode_len} steps)")
    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
        # Only render 1 + episode_len frames (same as frames list), not the full motion
        expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + episode_len])
        frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

    print(f"Running tracking inference for {episode_len} steps")
    for i in range(episode_len):
        print(f"Step {i} of {episode_len}")
        # --- collect metrics at current robot state (before acting) ---
        _bmap_obs = {
            "state": observation["state"],
            "privileged_state": observation["privileged_state"],
            "last_action": observation.get("last_action", _last_act_buf),
        }
        with torch.no_grad():
            _z_raw = model.backward_map(_bmap_obs)
            _z_now = model.project_z(_z_raw)
        _z_actual_list.append(_z_now[0].detach().cpu().numpy())
        _ee_local_pred_list.append(
            observation["privileged_state"][0, _wrist_slice].detach().cpu().numpy()
        )
        # ---
        action = model.act(observation, z[i % len(z)].repeat(num_envs, 1), mean=True)
        _last_act_buf = action.detach()
        observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
        joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])

    # --- compute and save all inference metrics ---
    _joint_pos_arr = np.stack(joint_pos, axis=0).squeeze(1)     # [episode_len+1, 29]
    _z_actual_arr  = np.stack(_z_actual_list)                    # [episode_len, z_dim]
    _ee_pred_arr   = np.stack(_ee_local_pred_list)               # [episode_len, 3]

    _ref_dof_arr = obs_dict["dof_pos"].cpu().numpy()             # [T, 29]  absolute DOF from motion lib
    _ref_ee_arr  = obs["privileged_state"][:, _wrist_slice].cpu().numpy() if isinstance(
        obs["privileged_state"], torch.Tensor
    ) else obs["privileged_state"][:, _wrist_slice].astype(np.float32)  # [T, 3]

    _n_cmp = min(episode_len, _ref_dof_arr.shape[0] - 1, len(_z_actual_arr))
    _jp_pred = _joint_pos_arr[1 : _n_cmp + 1]
    _jp_ref   = _ref_dof_arr[1 : _n_cmp + 1]
    _z_act   = _z_actual_arr[:_n_cmp]
    _z_exp   = z.detach().cpu().numpy()[:_n_cmp]
    _ee_pred = _ee_pred_arr[:_n_cmp]
    _ee_ref  = _ref_ee_arr[:_n_cmp]

    _all_dof_err  = np.linalg.norm(_jp_pred - _jp_ref,                                   axis=-1)
    _body_dof_err = np.linalg.norm(_jp_pred[:, _BODY_DOF_SLICE] - _jp_ref[:, _BODY_DOF_SLICE], axis=-1)
    _hand_dof_err = np.linalg.norm(_jp_pred[:, _HAND_DOF_SLICE] - _jp_ref[:, _HAND_DOF_SLICE], axis=-1)
    _hand_ee_err  = np.linalg.norm(_ee_pred - _ee_ref, axis=-1)

    _z_hand_act = _z_act[:, _z_body_dim:]
    if _z_hand_act.shape[1] > 0 and len(_z_hand_act) > 1:
        _dz_h       = np.diff(_z_hand_act, axis=0)
        _hand_scale = np.sqrt(max(1, _z_hand_act.shape[1]))
        _spike_rate = float(np.mean(np.linalg.norm(_dz_h, axis=1) / _hand_scale > 0.5))
    else:
        _spike_rate = 0.0

    _metrics = {
        "motion_id":              int(MOTION_ID),
        "n_frames":               int(_n_cmp),
        "all_dof_error_norm":     float(np.mean(_all_dof_err)),
        "body_dof_error_norm":    float(np.mean(_body_dof_err)),
        "hand_dof_error_norm":    float(np.mean(_hand_dof_err)),
        "hand_ee_local_err_m":    float(np.mean(_hand_ee_err)),
        "hand_ee_local_err_mm":   float(np.mean(_hand_ee_err) * 1000),
        "z_hand_spike_rate":      _spike_rate,
        "z_body_dim":             int(_z_body_dim),
        "z_hand_dim":             int(_z_act.shape[1] - _z_body_dim),
    }
    print(f"\n=== Inference Metrics (motion {MOTION_ID}) ===")
    for _k, _v in _metrics.items():
        print(f"  {_k}: {_v:.4f}" if isinstance(_v, float) else f"  {_k}: {_v}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _analysis = {
        "metrics":        _metrics,
        "z_expert":       _z_exp,
        "z_actual":       _z_act,
        "joint_pos_pred": _jp_pred,
        "joint_pos_ref":  _jp_ref,
        "ee_local_pred":  _ee_pred,
        "ee_local_ref":   _ee_ref,
    }
    joblib.dump(_analysis, output_dir / f"analysis_{MOTION_ID}.pkl")
    print(f"Saved analysis → {output_dir}/analysis_{MOTION_ID}.pkl")
    with open(output_dir / f"metrics_{MOTION_ID}.json", "w", encoding="utf-8") as _f:
        json.dump(_metrics, _f, indent=2)
    print(f"Saved metrics  → {output_dir}/metrics_{MOTION_ID}.json")
    # ---

    # breakpoint()  # use PYTHONBREAKPOINT=0 to disable, or install ipdb for a nicer debugger

    if save_mp4:
        new_frames = []
        for a, b in zip(expert_video, frames):
            new_frames.append(np.concatenate([a, b], axis=1))
        _video_stem = video_name or "tracking.mp4"
        if not _video_stem.endswith(".mp4"):
            _video_stem = f"{_video_stem}.mp4"
        video_path = output_dir / _video_stem
        media.write_video(str(video_path), new_frames, fps=50)
        print(f"Saved video for tracking: {video_path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
