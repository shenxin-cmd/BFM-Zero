"""
tracking_inference_npz.py
=========================
从单个 .npz 轨迹文件（而非 lafan 运动库）做 tracking inference。

输入 NPZ 格式（与 g1_circle_7_obs.npz 一致）
------------------------------------------------
  state             (N, 64)   float32
      = [dof_pos_rel(29), dof_vel(29), proj_grav(3), ang_vel(3)]
      dof_pos_rel = qpos[7:36] - DEFAULT_JOINT_POS
      dof_vel     = qvel[6:35]     （MuJoCo qvel，物理单位 rad/s）
      proj_grav   = R⁻¹ · [0,0,-1]
      ang_vel     = R · ω_body     （根角速度）

  last_action       (N, 29)   float32

  privileged_state  (N, D)    float32
      = compute_humanoid_observations_max 的输出，
        [root_h(1)?, local_body_pos, local_body_rot_6d, local_body_vel, local_body_ang_vel]
        D 典型值：448（30刚体含root_h）、447（不含）、463（31刚体）、462

速度尺度差异（重要）
--------------------
  训练时（motion library）：
      body_vel = np.gradient(wbody_pos) / motion_dt
               = Δpos / motion_dt   （物理 m/s）
      其中 motion_dt = 1/motion_fps（LAFAN 通常 30~120 Hz）

  NPZ 录制时（g1_player.py）：
      body_vel = (body_pos_t - body_pos_{t-1}) / PRIV_STATE_DT
               = Δpos / 0.001       （PRIV_STATE_DT 固定 1000 Hz）
      其中 Δpos 是相邻录制帧的差，帧间隔 = 1/npz_fps

  换算关系：
      NPZ 存储值 = 物理速度 × (帧间隔 / PRIV_STATE_DT)
                 = 物理速度 × (1/npz_fps) / 0.001
                 = 物理速度 × (1000 / npz_fps)

  因此，要把 NPZ 速度转回物理 m/s（模型期望值），需乘以：
      correction = 0.001 × npz_fps   （= 1/(1000/npz_fps)）

  例：npz_fps=100 → correction=0.1（除以10）
      npz_fps=50  → correction=0.05（除以20）
      npz_fps=30  → correction=0.03（除以33）

  若不指定 --npz-fps，脚本会尝试从 NPZ 内的时间戳自动推断，
  否则使用默认值 100 Hz 并打印警告。

关于第 0 帧跳过
----------------
NPZ 第 0 帧的 privileged_state body 速度 = 0（录制时无前帧），
第 1 帧起有限差分才有意义。与原版一致 obs[1:] 跳过第 0 帧。

用法
----
    python -m humanoidverse.tracking_inference_npz \\
        --model-folder  workdir/bfmzero-split-z/<run-id> \\
        --npz-path      g1_circle_7_obs.npz \\
        [--npz-fps      50]      # 录制帧率，默认自动推断
        [--episode-len  500] \\
        [--save-mp4] \\
        [--no-vel-rescale]       # 禁用速度缩放（调试用）
"""
from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import joblib
import json
import numpy as np
import torch
from torch.utils._pytree import tree_map

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    IsaacRendererWithMuJoco,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import export_meta_policy_as_onnx

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

# NPZ 录制时 g1_player.py 使用的固定 dt（分母），与 PRIV_STATE_DT 保持一致
_PRIV_STATE_DT = 0.001  # seconds


# ---------------------------------------------------------------------------
# NPZ 加载工具
# ---------------------------------------------------------------------------

def _normalize_key(k: str) -> str:
    k = k.strip().lower().replace(" ", "_")
    if k == "priviledged_state":
        k = "privileged_state"
    return k


def load_npz_obs(path: Path) -> dict[str, np.ndarray]:
    """加载 NPZ，返回 {key: (T, D) float32 ndarray}。"""
    raw = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in raw.files:
        out[_normalize_key(k)] = np.asarray(raw[k], dtype=np.float32)
    raw.close()
    for req in ("state", "last_action", "privileged_state"):
        if req not in out:
            raise KeyError(
                f"NPZ 文件 {path.name!r} 缺少必须的键 '{req}'；"
                f"现有键: {list(out.keys())}"
            )
    return out


def _infer_npz_fps(traj: dict[str, np.ndarray]) -> float | None:
    """
    尝试从 NPZ 内的时间戳数组推断录制帧率。
    常见键名：simtime、sim_time、timestamps、time、times。
    """
    for key in ("simtime", "sim_time", "timestamps", "timestamp", "time", "times"):
        if key in traj:
            t = np.asarray(traj[key], dtype=np.float64).ravel()
            if len(t) >= 2:
                diffs = np.diff(t)
                dt_med = float(np.median(diffs[diffs > 0]))
                if dt_med > 0:
                    return round(1.0 / dt_med, 1)
    return None


# ---------------------------------------------------------------------------
# privileged_state 维度对齐
# ---------------------------------------------------------------------------

def _get_model_priv_dim(model) -> int:
    return int(model.obs_space.spaces["privileged_state"].shape[0])


def _align_privileged_state(
    priv: np.ndarray,
    target_dim: int,
    root_height_obs: bool,
) -> np.ndarray:
    """
    若 privileged_state 维度比模型期望多（含 head_link 扩展体），自动裁剪对齐。
    支持：463→448（含 root_h）、462→447（不含）。
    """
    src_dim = int(priv.shape[-1])
    if src_dim == target_dim:
        return priv

    trim_map: dict[tuple[int, int], bool] = {
        (463, 448): True,
        (462, 447): False,
    }
    key = (src_dim, target_dim)
    if key not in trim_map:
        raise ValueError(
            f"privileged_state 维数 {src_dim} 与模型期望 {target_dim} 不符，"
            "且不在已知映射（463→448 / 462→447）中。"
        )
    has_root = trim_map[key]
    if has_root != root_height_obs:
        print(
            f"[警告] 由维数推断 has_root_height={has_root}，"
            f"但 config.root_height_obs={root_height_obs}，以维数推断为准。"
        )

    seg_full_and_chop = [(90, 3), (186, 6), (93, 3), (93, 3)]
    parts: list[np.ndarray] = []
    if has_root:
        parts.append(priv[:, :1])
    cur = 1 if has_root else 0
    for length, chop in seg_full_and_chop:
        parts.append(priv[:, cur: cur + length - chop])
        cur += length

    result = np.concatenate(parts, axis=-1).astype(np.float32)
    assert result.shape[-1] == target_dim
    print(f"  privileged_state 维度对齐: {src_dim} → {target_dim}")
    return result


# ---------------------------------------------------------------------------
# privileged_state 速度分量缩放
# ---------------------------------------------------------------------------

def _vel_slice(target_dim: int, root_height_obs: bool) -> tuple[int, int]:
    """
    返回 privileged_state 中 (body_vel ++ body_ang_vel) 的切片 [start, end)。

    layout（已对齐后）：
      [root_h(1)?] [local_pos((N-1)*3)] [local_rot(N*6)] [local_vel(N*3)] [local_ang_vel(N*3)]

    从 target_dim 和 root_height_obs 反解 N（刚体数）：
      有 root_h:  1 + (N-1)*3 + N*6 + N*3 + N*3 = target_dim → 15N = target_dim + 2
      无 root_h:  (N-1)*3 + N*6 + N*3 + N*3 = target_dim     → 15N = target_dim + 3
    """
    if root_height_obs:
        N = (target_dim + 2) // 15
        offset = 1
    else:
        N = (target_dim + 3) // 15
        offset = 0

    pos_len = (N - 1) * 3
    rot_len = N * 6
    vel_start = offset + pos_len + rot_len
    vel_end   = vel_start + N * 3 + N * 3   # linear_vel + ang_vel
    return vel_start, vel_end


def _rescale_privileged_vel(
    priv: np.ndarray,
    npz_fps: float,
    root_height_obs: bool,
) -> np.ndarray:
    """
    将 privileged_state 中 body 线速度和角速度从 NPZ 录制尺度转换到物理 m/s。

    NPZ 存储值 = 物理速度 × (1/npz_fps) / PRIV_STATE_DT
    物理速度   = NPZ 存储值 × PRIV_STATE_DT × npz_fps
    correction = _PRIV_STATE_DT × npz_fps

    训练时 motion_lib 使用的是物理速度（np.gradient / motion_dt），
    所以需要把 NPZ 速度缩回物理尺度才能和模型期望对齐。
    """
    correction = _PRIV_STATE_DT * npz_fps  # e.g., 0.001 * 100 = 0.1
    if abs(correction - 1.0) < 1e-4:
        return priv  # 刚好 1000 Hz 录制时无需缩放（不常见）

    target_dim = priv.shape[-1]
    vel_start, vel_end = _vel_slice(target_dim, root_height_obs)

    priv = priv.copy()
    priv[:, vel_start:vel_end] *= correction
    print(
        f"  privileged_state 速度缩放: dims[{vel_start}:{vel_end}] × {correction:.4f}"
        f"  （NPZ {npz_fps:.0f} Hz → 物理 m/s，PRIV_STATE_DT={_PRIV_STATE_DT}s）"
    )
    return priv


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(
    model_folder: Path,
    npz_path: Path,
    npz_fps: float | None = None,
    data_path: Path | None = None,
    headless: bool = True,
    device: str = "cuda",
    simulator: str = "isaacsim",
    save_mp4: bool = False,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    episode_len: int | None = None,
    no_vel_rescale: bool = False,
) -> None:
    """
    参数说明
    --------
    model_folder   : checkpoint 上一级目录（内含 checkpoint/ 和 config.json）
    npz_path       : 轨迹 NPZ 文件路径（如 g1_circle_7_obs.npz）
    npz_fps        : NPZ 录制帧率（Hz）。若不指定，自动从 NPZ 时间戳推断；
                     推断失败则默认 100 Hz 并打印警告。常见值：30 / 50 / 100。
    data_path      : 覆盖 config.json 里的 lafan_tail_path（仅用于 Isaac 环境初始化）
    episode_len    : rollout 最大步数（None = 使用 z 的完整长度）
    no_vel_rescale : 禁用 privileged_state 速度缩放（仅调试用）
    """
    model_folder = Path(model_folder)
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ 文件不存在: {npz_path}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    dev = next(model.parameters()).device

    # ── 读取训练配置 ──────────────────────────────────────────────────────────
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs: bool = bool(config["env"].get("root_height_obs", False))
    priv_expected: int = _get_model_priv_dim(model)
    print(f"模型期望 privileged_state 维度: {priv_expected}")
    print(f"use_root_height_obs: {use_root_height_obs}")

    # Isaac 环境初始化需要合法的 lafan_tail_path
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

    # ── 导出 ONNX ────────────────────────────────────────────────────────────
    export_dir = model_folder / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    z_export_dim = model.cfg.archi.total_z_dim
    export_meta_policy_as_onnx(
        model,
        export_dir,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(
            1, model._actor.input_filter.output_space.shape[0] + z_export_dim
        )},
        z_dim=z_export_dim,
        history=("history_actor" in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"Exported model to {export_dir}/{model_name}.onnx")

    # ── backward_map → cumulative mean → project_z ───────────────────────────
    def tracking_inference(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """obs 已跳过第 0 帧（privileged_state 第 0 帧 body 速度为 0 不准）。"""
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    # ── 构建 Isaac 环境 ───────────────────────────────────────────────────────
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("=" * 80)
    print(env.config.simulator)
    print("-" * 80)

    ddp = env.default_dof_pos.detach().float()
    default_dof_np: np.ndarray = (ddp[0] if ddp.ndim >= 2 else ddp).cpu().numpy()

    # ── 加载 NPZ ─────────────────────────────────────────────────────────────
    print(f"Loading trajectory: {npz_path}")
    traj = load_npz_obs(npz_path)
    N = traj["state"].shape[0]
    print(f"  Trajectory length: {N} frames")
    print(f"  state shape:            {traj['state'].shape}")
    print(f"  last_action shape:      {traj['last_action'].shape}")
    print(f"  privileged_state shape: {traj['privileged_state'].shape}")

    # ── 确定 NPZ 录制帧率 ─────────────────────────────────────────────────────
    if npz_fps is None:
        inferred = _infer_npz_fps(traj)
        if inferred is not None:
            npz_fps = inferred
            print(f"  从 NPZ 时间戳自动推断 npz_fps = {npz_fps:.1f} Hz")
        else:
            npz_fps = 100.0
            print(
                f"  [警告] 无法从 NPZ 推断帧率，使用默认 npz_fps = {npz_fps:.0f} Hz。\n"
                f"  若实际帧率不同（如 30/50 Hz），请用 --npz-fps 显式指定，\n"
                f"  否则 privileged_state 速度缩放会不准确！"
            )
    else:
        print(f"  使用用户指定 npz_fps = {npz_fps:.1f} Hz")

    # ── privileged_state 维度对齐 ─────────────────────────────────────────────
    traj["privileged_state"] = _align_privileged_state(
        traj["privileged_state"], priv_expected, use_root_height_obs
    )

    # ── privileged_state 速度缩放 ─────────────────────────────────────────────
    # 原理：
    #   NPZ 存储值 = Δpos / PRIV_STATE_DT = 物理速度 × (帧间隔/PRIV_STATE_DT)
    #   模型期望   = 物理速度（motion_lib 用 np.gradient/motion_dt 计算）
    #   correction = PRIV_STATE_DT × npz_fps
    #              = 0.001 × npz_fps   （把 NPZ 尺度还原回物理 m/s）
    if not no_vel_rescale:
        traj["privileged_state"] = _rescale_privileged_vel(
            traj["privileged_state"], npz_fps, use_root_height_obs
        )
    else:
        print("  [调试] 速度缩放已禁用（--no-vel-rescale）")

    # ── 构建 backward_map 输入（跳过第 0 帧）────────────────────────────────
    obs_full: dict[str, torch.Tensor] = {
        "state":            torch.from_numpy(traj["state"]).to(dev),
        "last_action":      torch.from_numpy(traj["last_action"]).to(dev),
        "privileged_state": torch.from_numpy(traj["privileged_state"]).to(dev),
    }
    # x[1:] 跳过第 0 帧：privileged_state 第 0 帧 body 速度 = 0（无前帧可差分）
    obs_for_bmap = tree_map(lambda x: x[1:], obs_full)

    # ── backward_map → z ──────────────────────────────────────────────────────
    print("Running backward_map to get z...")
    with torch.no_grad():
        z = tracking_inference(obs_for_bmap)   # (N-1, z_dim)
    print(f"  z shape: {z.shape}")

    # ── 保存 z ───────────────────────────────────────────────────────────────
    out_dir = model_folder / "tracking_inference_npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    z_save_path = out_dir / f"zs_{stem}.pkl"
    joblib.dump(z.cpu().numpy(), z_save_path)
    print(f"Saved z → {z_save_path}")

    # ── 初始化 Isaac 环境到轨迹第 0 帧 ──────────────────────────────────────
    # state[:, :29]  = dof_pos_rel → 加 default_dof_pos 还原绝对关节角
    # state[:, 29:58] = dof_vel   （MuJoCo qvel，物理 rad/s，无需缩放）
    dof_abs_np = traj["state"][:, :29] + default_dof_np   # (N, 29)
    dof_vel_np = traj["state"][:, 29:58]                   # (N, 29)

    # NPZ 不含根轨迹，使用 env.base_init_state
    bis = env.base_init_state.detach().float().cpu()
    root7 = (bis[:7] if bis.ndim == 1 else bis[0, :7]).numpy().astype(np.float32)
    root_pos       = root7[:3]
    root_quat_xyzw = root7[3:7]
    root_quat = (
        root_quat_xyzw[[3, 0, 1, 2]] if simulator == "isaacsim" else root_quat_xyzw
    )

    T_vis = min(N, (episode_len or N) + 1)
    expert_qpos = np.zeros((T_vis, 36), dtype=np.float32)
    expert_qpos[:, :3]  = root_pos
    expert_qpos[:, 3:7] = root_quat
    expert_qpos[:, 7:]  = dof_abs_np[:T_vis]

    sim_dev = env.device
    ref_root = torch.tensor(
        np.concatenate([root_pos, root_quat, np.zeros(6)]),
        dtype=torch.float32,
    )
    dof_init = torch.zeros_like(
        wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0]
    )
    dof_init[..., 0] = torch.from_numpy(dof_abs_np[0]).float().to(dof_init.device)
    dof_init[..., 1] = torch.from_numpy(dof_vel_np[0]).float().to(dof_init.device)

    target_states = {
        "dof_states":  dof_init,
        "root_states": torch.stack([ref_root.clone().to(sim_dev)
                                    for _ in range(num_envs)]),
    }
    env_ids = torch.arange(num_envs, dtype=torch.long, device=sim_dev)
    wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
    wrapped_env.step(
        torch.zeros((num_envs, wrapped_env.action_space.shape[-1]),
                    dtype=torch.float32, device=sim_dev),
        to_numpy=False,
    )
    observation = wrapped_env._get_g1env_observation(to_numpy=False)

    # ── Rollout ───────────────────────────────────────────────────────────────
    n_steps = z.shape[0]
    if episode_len is not None:
        n_steps = min(n_steps, episode_len)
    print(f"Running rollout for {n_steps} steps "
          f"(z length={z.shape[0]}, episode_len={episode_len})")

    joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]

    if save_mp4:
        try:
            import mediapy as media
        except ImportError:
            raise ImportError("save_mp4 需要 mediapy：pip install mediapy")
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
        expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + n_steps])
        frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

    for i in range(n_steps):
        action = model.act(observation, z[i % len(z)].repeat(num_envs, 1), mean=True)
        observation, reward, terminated, truncated, info = wrapped_env.step(
            action, to_numpy=False
        )
        joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1}/{n_steps}")

    joint_pos_arr = np.stack(joint_pos, axis=0).squeeze(1)
    print(f"Rollout complete. joint_pos shape: {joint_pos_arr.shape}")

    if save_mp4:
        new_frames = [
            np.concatenate([a, b], axis=1)
            for a, b in zip(expert_video, frames)
        ]
        video_path = out_dir / f"tracking_{stem}.mp4"
        media.write_video(str(video_path), new_frames, fps=50)
        print(f"Saved video: {video_path}")

    print(f"\n=== Done ===")
    print(f"  z pkl  : {z_save_path}")
    if save_mp4:
        print(f"  video  : {out_dir / f'tracking_{stem}.mp4'}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
