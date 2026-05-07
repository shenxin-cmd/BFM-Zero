"""
Isaac 上 single-frame goal inference（Split-z）。与 ``goal_inference.py`` 同源，只取一帧作目标。

**目标姿态（可行性与默认）**
    ``backward_map`` / ``goal_inference`` 只需要与训练一致的 ``state``、``last_action``、``privileged_state``。
    这与 ``inference_tutorial.ipynb`` / ``split_z_hand_validation.py`` 思路一致：在 MuJoCo 里设关节角 → ``mj_forward`` →
    组装 privileged + state（速度置零）。**不必**再遍历 pkl。Isaac 仍负责 rollout；参考动作由
    ``set_is_evaluating(rollout_motion_id)`` 单独指定，可与目标 z 解耦。

**Rollout z 轨迹**：默认 ``--save-z-trace`` 将每步 ``backward_map → project_z`` 得到的 **推断 z** 与 **policy z** 写入 ``.npz``（``--z-trace-out`` 可指定路径）；用 ``python -m humanoidverse.plot_split_z_rollout_pca`` 画 PCA 图。

噪声与录像说明见 ``main`` 的 docstring；``python -m humanoidverse.split_z_goal_inference_isaac --help``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mediapy as media
import numpy as np
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, get_backward_observation

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# MuJoCo 手写 goal：build_goal_arrays_from_mujoco_stand（关节名见 split_z_hand_validation._DOF_NAMES）。
# - GOAL_USE_ABSOLUTE_JOINT_RAD=True：GOAL_POSE_JOINT_ABS_RAD 中为 **铰链角绝对值** [rad]，未写的关节用
#   _REF_DEFAULT_DOF_POS（与 g1_29dof.yaml 默认站姿一致）。
# - False：则用 GOAL_POSE_JOINT_DELTA_RAD 在默认站姿上加增量。
# 关闭手写 goal：`--no-use-mujoco-custom-goal` 或 USE_MUJOCO_CUSTOM_GOAL = False → 下方 motion 库逻辑。
# -----------------------------------------------------------------------------
USE_MUJOCO_CUSTOM_GOAL = True

GOAL_USE_ABSOLUTE_JOINT_RAD = True

# 绝对角 [rad]（与往日「增量版」观感接近的一组示例；可自行改）
GOAL_POSE_JOINT_ABS_RAD: dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": -0.09,
    "left_shoulder_roll_joint": 0.41,
    "left_shoulder_yaw_joint": 0.38,
    "left_elbow_joint": -0.05,
    "left_wrist_roll_joint": 0.03,
    "left_wrist_pitch_joint": 0.38,
    "left_wrist_yaw_joint": 0.38,
    "right_shoulder_pitch_joint": -0.09,
    "right_shoulder_roll_joint": 0.01,
    "right_shoulder_yaw_joint": 0.38,
    "right_elbow_joint": -0.05,
    "right_wrist_roll_joint": 0.03,
    "right_wrist_pitch_joint": 0.38,
    "right_wrist_yaw_joint": 0.38,
}

# 仅当 GOAL_USE_ABSOLUTE_JOINT_RAD=False 时生效。
GOAL_POSE_JOINT_DELTA_RAD: dict[str, float] = {}

# -----------------------------------------------------------------------------
# Motion 库目标（仅当 ``USE_MUJOCO_CUSTOM_GOAL`` 为 False，且未显式关掉自动选帧等）
# -----------------------------------------------------------------------------
USE_AUTO_ONE_LEG_GOAL = True
GOAL_MOTION_ID = 0
GOAL_FRAME_IDX = 0
# rollout 参考轨迹起始 id（eval 装入 num_envs 条 clip）；与 MuJoCo 目标 z 独立
ROLLOUT_REF_MOTION_ID = 0


def _pick_one_leg_goal_frame(env, motion_id: int | None) -> tuple[int, int]:
    """在 motion 库中选一帧：双脚高度差大且脚/根线速度小，用作单脚站立目标近似。

    注意：``MotionLib`` 在 eval 时往往只把 **num_envs 条** motion 载入显存，``len(_motion_lengths)==已载入条数``，
    不能与 ``_num_unique_motions``（数据集里总条数）混用。此处通过 ``load_motions(start_idx=..., num_motions_to_load=1)``
    逐条载入并在 **局部槽位 0** 上打分；全库扫描时对每条数据集 motion 各 load 一次（时间 O(N)，内存友好）。
    """
    ml = env._motion_lib
    feet = env.feet_indices.long()
    if feet.numel() < 2:
        raise RuntimeError("feet_indices 至少需要两只脚以自动选单脚站立帧。")

    n_unique = int(ml._num_unique_motions)

    def best_in_loaded_slot0() -> tuple[int, float]:
        """当前库内只载入了批量 motion，本环境 num_envs=1 时长度恒为 1，对应局部索引 0。"""
        motion_len = ml._motion_lengths[0]
        n = int(torch.ceil(motion_len / env.dt).item())
        if n < 1:
            return 0, -1.0
        t = torch.arange(n, device=env.device, dtype=torch.float32) * env.dt
        with torch.no_grad():
            res = ml.get_motion_state(0, t)
        rg = res["rg_pos_t"]
        bv = res["body_vel_t"]
        zl = rg[:, feet[0], 2]
        zr = rg[:, feet[1], 2]
        diff = (zl - zr).abs()
        v_foot = bv[:, feet].norm(dim=-1).sum(dim=1)
        v_root = bv[:, 0].norm(dim=-1)
        score = diff / (0.08 + v_foot + 0.05 * v_root)
        j = int(score.argmax().item())
        return j, float(score[j].item())

    def load_one_dataset_motion(dataset_idx: int) -> None:
        dataset_idx = int(np.clip(dataset_idx, 0, max(n_unique - 1, 0)))
        ml.load_motions(
            random_sample=False,
            num_motions_to_load=1,
            start_idx=dataset_idx,
        )

    if motion_id is not None:
        ds = int(np.clip(motion_id, 0, max(n_unique - 1, 0)))
        load_one_dataset_motion(ds)
        fid, _ = best_in_loaded_slot0()
        return ds, fid

    best_mid, best_fid, best_s = 0, 0, -1.0
    for ds_idx in range(n_unique):
        load_one_dataset_motion(ds_idx)
        fid, s = best_in_loaded_slot0()
        if s > best_s:
            best_mid, best_fid, best_s = ds_idx, fid, s
    return best_mid, best_fid


def main(
    model_folder: Path,
    data_path: Path | None = None,
    motion_id: int | None = None,
    goal_frame: int | None = None,
    use_mujoco_custom_goal: bool = USE_MUJOCO_CUSTOM_GOAL,
    mujoco_goal_xml: Path | None = None,
    headless: bool = True,
    device: str = "cuda",
    simulator: str = "isaacsim",
    hand_noise_std: float = 0.0,
    seed: int = 0,
    episode_len: int = 500,
    save_mp4: bool = True,
    video_folder: Path | None = None,
    video_suffix: str = "",
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    export_onnx: bool = False,
    video_camera: str = "face_torso",
    face_torso_camera_distance: float = 2.75,
    face_torso_camera_elevation_deg: float = 14.0,
    policy_sample: bool = False,
    save_z_trace: bool = True,
    z_trace_out: Path | None = None,
) -> None:
    """
    **z_hand 噪声**：先做 ``zh <- zh_ref + hand_noise_std * ε``（ε∼标准正态），再 ``project_z(cat(z_body, zh))``.
    Split 模式下 ``project_z`` 对 ``z_hand`` 做 normalize，故 **极大 std 不会改变投影后的范数量级**，只会改变与其它随机方向的混合，
    ``std`` 从 2 调到 20 往往看起来差不多属正常。

    **为何不“乱挥”**：策略 ``act(..., mean=True)`` 取 **均值**，无采样噪声；映射到 **有界关节动作**，
    ``z_hand`` 是训练中 **语义空间** 的子向量，并不等于在动作末尾直接加大幅度白噪。
    可试 ``--policy-sample`` 略增手部随机性。

    **目标姿态**：默认 ``use_mujoco_custom_goal`` 取模块 ``USE_MUJOCO_CUSTOM_GOAL``（默认 True），
    用 ``GOAL_POSE_JOINT_ABS_RAD``（绝对 rad）或 ``GOAL_POSE_JOINT_DELTA_RAD``（由 ``GOAL_USE_ABSOLUTE_JOINT_RAD`` 切换）。
    传 ``--no-use-mujoco-custom-goal`` 时从 lafan motion 取帧；``USE_AUTO_ONE_LEG_GOAL`` / ``--goal-frame`` 等规则同旧版。

    **Rollout 参考轨迹**：MuJoCo 目标下 ``--motion-id`` 仅影响 ``set_is_evaluating`` 装入的参考 clip；未传时用 ``ROLLOUT_REF_MOTION_ID``。
    """

    model_folder = Path(model_folder)
    vid_dir = Path(video_folder) if video_folder is not None else model_folder / "split_z_goal_isaac" / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()

    if not model.cfg.archi.is_split_mode:
        raise RuntimeError(
            "此脚本面向 split-z checkpoint（archi.z_body_dim>0 且 z_hand_dim>0）。"
            "非 split 模型请用 humanoidverse.goal_inference。"
        )

    with open(model_folder / "config.json", "r", encoding="utf-8") as f:
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

    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    if export_onnx:
        output_dir = model_folder / "exported"
        output_dir.mkdir(parents=True, exist_ok=True)
        z_export_dim = model.cfg.archi.total_z_dim
        model_name = model.__class__.__name__
        export_meta_policy_as_onnx(
            model,
            output_dir,
            f"{model_name}.onnx",
            {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + z_export_dim)},
            z_dim=z_export_dim,
            history=("history_actor" in model.cfg.archi.actor.input_filter.key),
            use_29dof=True,
        )
        print(f"Exported ONNX to {output_dir}/{model_name}.onnx")

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env

    umc = use_mujoco_custom_goal
    if umc and goal_frame is not None:
        print("提示: 已启用 MuJoCo 关节目标，忽略 --goal-frame。")

    if umc:
        from humanoidverse.split_z_hand_validation import DEFAULT_MUJOCO_SCENE, build_goal_arrays_from_mujoco_stand
        from humanoidverse.utils.g1_env_config import get_g1_robot_xml_root

        rollout_mid = int(motion_id) if motion_id is not None else ROLLOUT_REF_MOTION_ID
        xml_path = Path(mujoco_goal_xml).resolve() if mujoco_goal_xml is not None else (get_g1_robot_xml_root() / DEFAULT_MUJOCO_SCENE)
        arrs = build_goal_arrays_from_mujoco_stand(
            mujoco_scene_xml=xml_path,
            root_height_obs=use_root_height_obs,
            joint_abs_rad=GOAL_POSE_JOINT_ABS_RAD if GOAL_USE_ABSOLUTE_JOINT_RAD else None,
            joint_delta_rad=None if GOAL_USE_ABSOLUTE_JOINT_RAD else GOAL_POSE_JOINT_DELTA_RAD,
        )
        backward_keys = ["state", "last_action", "privileged_state"]
        goal_observation = {
            k: torch.as_tensor(arrs[k], device=model.device, dtype=torch.float32).unsqueeze(0) for k in backward_keys
        }
        goal_source = "mujoco_joint_delta"
        trace_file_tag = "mujoco_goal"
        mid_meta = -1
        fid_meta = 0
        _mkey_s = f"mujoco:{xml_path.name}"
    else:
        if goal_frame is not None:
            mid = GOAL_MOTION_ID if motion_id is None else motion_id
            fid = goal_frame
        elif USE_AUTO_ONE_LEG_GOAL:
            focus = motion_id
            mid, fid = _pick_one_leg_goal_frame(env, focus)
        else:
            mid = GOAL_MOTION_ID if motion_id is None else motion_id
            fid = GOAL_FRAME_IDX

        rollout_mid = mid
        goal_source = "motion_lib"
        trace_file_tag = f"M{mid}_f{fid}".replace(".", "p")
        mid_meta = mid
        fid_meta = fid

        env.set_is_evaluating(rollout_mid)
        gobs, _gobs_dict = get_backward_observation(
            env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=0
        )
        n_frames = int(gobs["state"].shape[0])
        if fid < 0 or fid >= n_frames:
            raise IndexError(f"goal_frame {fid} out of range [0, {n_frames}) for motion {mid}")

        goal_observation = {k: v[fid : fid + 1] for k, v in gobs.items()}
        goal_observation = tree_map(
            lambda x: x.detach().clone().to(device=model.device, dtype=torch.float32)
            if isinstance(x, torch.Tensor)
            else torch.as_tensor(x, device=model.device, dtype=torch.float32),
            goal_observation,
        )
        backward_keys = list(goal_observation.keys())

        try:
            _mkey = env._motion_lib._motion_data_keys[mid]
            _mkey_s = _mkey.item() if hasattr(_mkey, "item") else str(_mkey)
        except Exception:
            _mkey_s = "?"

    vc = video_camera.lower().strip()
    if vc not in ("face_torso", "track"):
        raise ValueError("video_camera must be 'face_torso' or 'track'.")

    print(
        f"goal_source={goal_source} | rollout_ref_motion_id={rollout_mid} | "
        f"motion_id_meta={mid_meta} goal_frame_meta={fid_meta} | motion_key={_mkey_s} | "
        f"use_root_height_obs={use_root_height_obs}"
    )
    print(f"hand_noise_std={hand_noise_std} | video_camera={vc} | policy_sample={policy_sample}")

    if umc:
        env.set_is_evaluating(rollout_mid)

    with torch.no_grad():
        z_ref = model.goal_inference(goal_observation)

    d_body = int(model.cfg.archi.z_body_dim)
    zb = z_ref[:, :d_body]
    zh = z_ref[:, d_body:]
    policy_goal_z_body_np = zb.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    policy_goal_z_hand_np = zh.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    policy_goal_z_full_np = z_ref.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    print(
        f"z_ref: total_dim={z_ref.shape[-1]}, ||z||={z_ref.norm().item():.4f}, "
        f"||z_body||={zb.norm().item():.4f}, ||z_hand||={zh.norm().item():.4f}"
    )

    rng = torch.Generator(device=z_ref.device)
    rng.manual_seed(seed)
    if hand_noise_std > 0:
        noise = torch.randn(zh.shape, device=zh.device, dtype=zh.dtype, generator=rng)
        zh = zh + float(hand_noise_std) * noise
    z = model.project_z(torch.cat([zb, zh], dim=-1))

    zb2 = z[:, :d_body]
    zh2 = z[:, d_body:]
    print(
        f"z after noise+project_z: ||z||={z.norm().item():.4f}, "
        f"||z_body||={zb2.norm().item():.4f}, ||z_hand||={zh2.norm().item():.4f}"
    )

    render_inner = wrapped_env._env

    def _backward_obs_from_wrapped(obs_dict: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k in backward_keys:
            if k not in obs_dict:
                raise KeyError(
                    f"Observation missing key {k!r} for backward_map; have {list(obs_dict.keys())}",
                )
            t = obs_dict[k]
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t, device=model.device, dtype=torch.float32)
            else:
                t = t.to(device=model.device, dtype=torch.float32)
            out[k] = t
        return out

    def _record_inferred_z(obs_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ob = _backward_obs_from_wrapped(obs_dict)
        zt = model.project_z(model.backward_map(ob))
        zb_np = zt[:, :d_body].detach().float().cpu().numpy().squeeze(0).astype(np.float32)
        zh_np = zt[:, d_body:].detach().float().cpu().numpy().squeeze(0).astype(np.float32)
        zf_np = zt.detach().float().cpu().numpy().squeeze(0).astype(np.float32)
        return zb_np, zh_np, zf_np

    inferred_b: list[np.ndarray] = []
    inferred_h: list[np.ndarray] = []
    inferred_f: list[np.ndarray] = []

    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(
            render_size=256,
            video_camera=vc,
            face_torso_distance=face_torso_camera_distance,
            face_torso_elevation_deg=face_torso_camera_elevation_deg,
        )

    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)

    frames: list = []
    _pbar = tqdm(desc="steps", disable=False, leave=False, total=episode_len)
    zn = z.repeat(render_inner.num_envs, 1)
    act_mean = not policy_sample

    with torch.no_grad():
        b0, h0, f0 = _record_inferred_z(observation)
        inferred_b.append(b0)
        inferred_h.append(h0)
        inferred_f.append(f0)

    for _counter in range(episode_len):
        action = model.act(observation, zn, mean=act_mean)
        observation, _reward, _terminated, _truncated, _info = wrapped_env.step(action, to_numpy=False)
        with torch.no_grad():
            b1, h1, f1 = _record_inferred_z(observation)
            inferred_b.append(b1)
            inferred_h.append(h1)
            inferred_f.append(f1)
        if save_mp4:
            frames.append(rgb_renderer.render(render_inner, 0)[0])
        _pbar.update(1)
    _pbar.close()

    if save_z_trace:
        zb_arr = np.stack(inferred_b, axis=0)
        zh_arr = np.stack(inferred_h, axis=0)
        zf_arr = np.stack(inferred_f, axis=0)
        trace_path = Path(z_trace_out) if z_trace_out is not None else vid_dir / (
            f"ztrace_{trace_file_tag}_hnoise{hand_noise_std:.4f}".replace(".", "p") + ".npz"
        )
        meta = {
            "goal_source": goal_source,
            "rollout_ref_motion_id": rollout_mid,
            "motion_id": mid_meta,
            "goal_frame": fid_meta,
            "hand_noise_std": hand_noise_std,
            "episode_len": episode_len,
            "num_inferred": int(zb_arr.shape[0]),
            "z_body_dim": int(model.cfg.archi.z_body_dim),
            "z_hand_dim": int(model.cfg.archi.z_hand_dim),
            "seed": int(seed),
            "policy_sample": bool(policy_sample),
            "backward_keys": backward_keys,
        }
        np.savez(
            trace_path,
            inferred_z_body=zb_arr,
            inferred_z_hand=zh_arr,
            inferred_z_full=zf_arr,
            policy_goal_z_body=policy_goal_z_body_np,
            policy_goal_z_hand=policy_goal_z_hand_np,
            policy_goal_z_full=policy_goal_z_full_np,
            policy_actor_z_body=zb2.detach().cpu().numpy().reshape(-1).astype(np.float32),
            policy_actor_z_hand=zh2.detach().cpu().numpy().reshape(-1).astype(np.float32),
            policy_actor_z_full=z.detach().cpu().numpy().reshape(-1).astype(np.float32),
            meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
        )
        print(f"Saved z trace: {trace_path}  (samples={zb_arr.shape[0]}, backward_keys={backward_keys})")

    if save_mp4 and frames:
        sfx = f"_{video_suffix}" if video_suffix else ""
        noise_tag = f"hnoise{hand_noise_std:.4f}".replace(".", "p")
        vc_tag = vc
        samp = "_psample" if policy_sample else ""
        out_mp4 = vid_dir / f"goal_{trace_file_tag}_{noise_tag}_{vc_tag}{samp}{sfx}.mp4"
        media.write_video(str(out_mp4), frames, fps=50)
        print(f"Saved video: {out_mp4}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
