import os

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import json
import torch
import joblib
import mediapy as media
import numpy as np
import math
from torch.utils._pytree import tree_map
from tqdm import tqdm

import humanoidverse
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.utils.helpers import get_backward_observation

# Resolve humanoidverse root directory
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).parent.parent.parent


def _get_right_hand_pose_from_wrapper(wrapped_env) -> torch.Tensor:
    """Return right hand pose (pos[3] + quat[4]) for env 0 from env wrapper."""
    if hasattr(wrapped_env, "_get_right_hand_pose"):
        pose = wrapped_env._get_right_hand_pose(to_numpy=False)
        return pose[0].detach().clone()
    raise AttributeError("Environment wrapper does not expose _get_right_hand_pose.")


def _inject_right_hand_noise(model, z: torch.Tensor, noise_scale: float, seed: int, step: int) -> torch.Tensor:
    if noise_scale <= 0:
        return z
    z_parts = model.split_z(z)
    z_right = z_parts["right_hand"]
    if z_right.shape[-1] == 0:
        return z
    gen = torch.Generator(device=z.device)
    gen.manual_seed(seed + step)
    noise = torch.randn(z_right.shape, device=z_right.device, dtype=z_right.dtype, generator=gen) * noise_scale
    z_right_perturbed = z_right + noise
    norm = torch.norm(z_right_perturbed, dim=-1, keepdim=True).clamp_min(1e-8)
    z_right_perturbed = math.sqrt(z_right_perturbed.shape[-1]) * z_right_perturbed / norm
    z_perturbed = model.merge_z({"right_hand": z_right_perturbed, "rest": z_parts["rest"]})
    return model.project_z(z_perturbed)


def main(
    model_folder: Path,
    data_path: Path | None = None,
    headless: bool = True,
    device="cuda",
    simulator: str = "isaacsim",
    save_mp4: bool = False,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    episode_len: int = 500,
    video_folder: str | None = None,
    run_right_hand_perturb_eval: bool = True,
    right_hand_noise_scale: float = 0.4,
    perturb_seed: int = 0,
    perturb_goal_switch_every: int = 100,
):

    model_folder = Path(model_folder)
    video_folder = Path(video_folder) if video_folder is not None else model_folder / "goal_inference" / "videos"
    video_folder.mkdir(parents=True, exist_ok=True)

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

    output_dir = model_folder / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_meta_policy_as_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim)},
        z_dim=model.cfg.archi.z_dim,
        history=('history_actor' in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"Exported model to {output_dir}/{model_name}.onnx")
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("="*80)
    print(env.config.simulator)
    print("-"*80)
    
    # Try to find goal_frames JSON file
    goal_json_paths = [
        HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "goal_frames_lafan29dof.json",
        HUMANOIDVERSE_DIR / "data" / "goal_frames_lafan29dof.json",
    ]
    goal_json = None
    for path in goal_json_paths:
        if path.exists():
            goal_json = str(path)
            break
    
    if goal_json is None:
        raise FileNotFoundError(
            f"Could not find goal_frames_lafan29dof.json. Searched in: {goal_json_paths}"
        )
    
    with open(goal_json, "r") as f:
        goals_to_evaluate = json.load(f)
    pbar = tqdm(goals_to_evaluate, leave=False, disable=False)
    z_dict = {}
    with torch.no_grad():
        for goal in pbar:
            env.set_is_evaluating(goal["motion_id"])
                # we visulize the first env
            gobs, gobs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=0)
            num_frames = next(iter(gobs.values())).shape[0]
            frame_pbar = tqdm(goal["frames"], leave=False, disable=False, desc="frames")
            for frame_idx in frame_pbar:
                if frame_idx >= num_frames:
                    pbar.write(f"  Skipping frame_idx {frame_idx} (motion has {num_frames} frames)")
                    continue
                goal_name = f"{goal['motion_name']}_{frame_idx}"
                goal_observation = {k: v[frame_idx][None,...] for k,v in gobs.items()}
                goal_observation = tree_map(lambda x: torch.tensor(x, device=model.device, dtype=torch.float32), goal_observation)

                z_dict[goal_name] = model.goal_inference(goal_observation).cpu().numpy()
    path = model_folder / "goal_inference"
    path.mkdir(exist_ok=True)
    with open(os.path.join(path, "goal_reaching.pkl"), "wb") as f:
        joblib.dump(z_dict, f)

    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)

    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)

    frames = []
    counter = 0
    # episode_len: number of sim steps for the optional goal-reaching video (~12 it/s → 5000 ≈ 7 min)
    _pbar = tqdm(desc="steps", disable=False, leave=False, total=episode_len)
    goal_idx = -1
    goal_names = list(z_dict.keys())
    if len(goal_names) == 0:
        raise RuntimeError("No inferred goals found in z_dict.")

    while counter < episode_len:
        if counter % perturb_goal_switch_every == 0:
            goal_idx = (goal_idx + 1) % len(goal_names)
            print(f"Switching to goal {goal_names[goal_idx]} at step {counter}")
            z = z_dict[goal_names[goal_idx]].copy()
            z = torch.tensor(z, device=model.device, dtype=torch.float32)

        action = model.act(observation, z.repeat(num_envs, 1), mean=True)
        observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
        counter += 1
        _pbar.update(1)
    _pbar.close()
    if save_mp4:
        media.write_video(video_folder / "goal.mp4", frames, fps=50)
        print("Saved video for goal")

    if run_right_hand_perturb_eval:
        print("Running right-hand perturbation ablation (sequential baseline/perturbed rollouts).")
        baseline_right_hand_poses = []
        perturbed_right_hand_poses = []
        baseline_states = []
        perturbed_states = []
        baseline_frames = []
        perturbed_frames = []

        # Baseline rollout
        baseline_obs, _ = wrapped_env.reset(to_numpy=False)
        goal_idx = -1
        for step in tqdm(range(episode_len), desc="ablation-baseline", leave=False, disable=False):
            if step % perturb_goal_switch_every == 0:
                goal_idx = (goal_idx + 1) % len(goal_names)
                z = torch.tensor(z_dict[goal_names[goal_idx]].copy(), device=model.device, dtype=torch.float32)
            action = model.act(baseline_obs, z.repeat(num_envs, 1), mean=True)
            baseline_obs, _, _, _, _ = wrapped_env.step(action, to_numpy=False)
            try:
                baseline_right_hand_poses.append(_get_right_hand_pose_from_wrapper(wrapped_env).cpu().numpy())
            except Exception as exc:
                print(f"Warning: right hand pose metric unavailable: {exc}")
                run_right_hand_perturb_eval = False
                break
            if "state" in baseline_obs:
                baseline_states.append(baseline_obs["state"][0].detach().cpu().numpy())
            if save_mp4:
                baseline_frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])

        # Perturbed rollout
        if run_right_hand_perturb_eval:
            perturbed_obs, _ = wrapped_env.reset(to_numpy=False)
            goal_idx = -1
            for step in tqdm(range(episode_len), desc="ablation-perturbed", leave=False, disable=False):
                if step % perturb_goal_switch_every == 0:
                    goal_idx = (goal_idx + 1) % len(goal_names)
                    z_base = torch.tensor(z_dict[goal_names[goal_idx]].copy(), device=model.device, dtype=torch.float32)
                z = _inject_right_hand_noise(model, z_base, noise_scale=right_hand_noise_scale, seed=perturb_seed, step=step)
                action = model.act(perturbed_obs, z.repeat(num_envs, 1), mean=True)
                perturbed_obs, _, _, _, _ = wrapped_env.step(action, to_numpy=False)
                perturbed_right_hand_poses.append(_get_right_hand_pose_from_wrapper(wrapped_env).cpu().numpy())
                if "state" in perturbed_obs:
                    perturbed_states.append(perturbed_obs["state"][0].detach().cpu().numpy())
                if save_mp4:
                    perturbed_frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])

        if run_right_hand_perturb_eval and len(baseline_right_hand_poses) > 0 and len(perturbed_right_hand_poses) > 0:
            n = min(len(baseline_right_hand_poses), len(perturbed_right_hand_poses))
            baseline_right_hand_poses_np = np.stack(baseline_right_hand_poses[:n], axis=0)
            perturbed_right_hand_poses_np = np.stack(perturbed_right_hand_poses[:n], axis=0)
            delta_pose = perturbed_right_hand_poses_np - baseline_right_hand_poses_np
            right_hand_pos_delta_l2 = np.linalg.norm(delta_pose[:, :3], axis=-1)
            right_hand_rot_delta_l2 = np.linalg.norm(delta_pose[:, 3:], axis=-1)

            metrics = {
                "episode_len": int(episode_len),
                "compared_steps": int(n),
                "right_hand_noise_scale": float(right_hand_noise_scale),
                "right_hand_pos_delta_l2_mean": float(np.mean(right_hand_pos_delta_l2)),
                "right_hand_pos_delta_l2_std": float(np.std(right_hand_pos_delta_l2)),
                "right_hand_rot_delta_l2_mean": float(np.mean(right_hand_rot_delta_l2)),
                "right_hand_rot_delta_l2_std": float(np.std(right_hand_rot_delta_l2)),
            }

            if len(baseline_states) > 0 and len(perturbed_states) > 0:
                n_state = min(len(baseline_states), len(perturbed_states))
                baseline_states_np = np.stack(baseline_states[:n_state], axis=0)
                perturbed_states_np = np.stack(perturbed_states[:n_state], axis=0)
                state_delta = np.linalg.norm(perturbed_states_np - baseline_states_np, axis=-1)
                metrics["state_delta_l2_mean"] = float(np.mean(state_delta))
                metrics["state_delta_l2_std"] = float(np.std(state_delta))

            ablation_path = model_folder / "goal_inference" / "ablation"
            ablation_path.mkdir(parents=True, exist_ok=True)
            with open(ablation_path / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved right-hand perturbation metrics to {ablation_path / 'metrics.json'}")
            print(metrics)

            if save_mp4:
                media.write_video(ablation_path / "baseline.mp4", baseline_frames, fps=50)
                media.write_video(ablation_path / "right_hand_perturbed.mp4", perturbed_frames, fps=50)
                print(f"Saved ablation videos to {ablation_path}")

if __name__ == "__main__":
    import tyro

    tyro.cli(main)
