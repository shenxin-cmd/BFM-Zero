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

import mujoco
import joblib
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
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

# 右手末端执行器（与 g1_traj_gen.py 保持一致）
_EE_BODY       = "right_wrist_yaw_link"
_SPHERE_RADIUS = 0.028                    # 视频标注球半径 (m)
_SPHERE_RGBA   = (1.0, 0.40, 0.05, 1.0)  # 橙色


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
# 右手末端轨迹工具
# ---------------------------------------------------------------------------

def _quat_wxyz_inv_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """用四元数逆旋转将世界系向量变到基座系。quat 格式 MuJoCo wxyz。"""
    q_w, q_vec = float(quat_wxyz[0]), quat_wxyz[1:4]
    a = vec * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, vec) * (2.0 * q_w)
    c = q_vec * (2.0 * np.dot(q_vec, vec))
    return a - b + c


def _world_to_base(
    ee_world: np.ndarray,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """世界系 EE 位置 → 基座（pelvis）局部坐标系。"""
    return _quat_wxyz_inv_rotate(root_quat_wxyz, ee_world - root_pos)


def _fk_ee_positions_base(
    fk_model: "mujoco.MjModel",
    fk_data: "mujoco.MjData",
    ee_id: int,
    qpos7_root: np.ndarray,    # (7,) or (N, 7) MuJoCo [x,y,z,qw,qx,qy,qz]
    dof_abs_arr: np.ndarray,   # (N, 29)
) -> np.ndarray:
    """
    Batch FK: return (N, 3) EE positions in pelvis-local frame.
    ``qpos7_root`` may be a single root reused for all frames, or per-frame (N, 7).
    """
    N = dof_abs_arr.shape[0]
    qpos7_root = np.asarray(qpos7_root, dtype=np.float64)
    per_frame_root = qpos7_root.ndim == 2
    if per_frame_root:
        if qpos7_root.shape != (N, 7):
            raise ValueError(f"qpos7_root expected ({N}, 7), got {qpos7_root.shape}")
    elif qpos7_root.shape != (7,):
        raise ValueError(f"qpos7_root expected (7,), got {qpos7_root.shape}")

    out = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        fk_data.qpos[:7] = qpos7_root[i] if per_frame_root else qpos7_root
        fk_data.qpos[7:] = dof_abs_arr[i]
        fk_data.qvel[:] = 0.0
        mujoco.mj_forward(fk_model, fk_data)
        ee_world = fk_data.xpos[ee_id].copy()
        root_pos = fk_data.qpos[:3].copy()
        root_quat = fk_data.qpos[3:7].copy()
        out[i] = _world_to_base(ee_world, root_pos, root_quat)
    return out


def _fk_one_step_base(
    fk_model: "mujoco.MjModel",
    fk_data: "mujoco.MjData",
    ee_id: int,
    root_states_row: np.ndarray,  # (>=13,) Isaac 格式 [x,y,z, qx,qy,qz,qw, ...]
    dof_pos: np.ndarray,          # (29,)
) -> np.ndarray:
    """
    单步 FK，返回 EE 在基座局部坐标系下的位置 (3,)。
    Isaac root_states 索引 [0,1,2,6,3,4,5] → MuJoCo [x,y,z,qw,qx,qy,qz]。
    """
    fk_data.qpos[:7] = root_states_row[[0, 1, 2, 6, 3, 4, 5]]
    fk_data.qpos[7:] = dof_pos
    fk_data.qvel[:] = 0.0
    mujoco.mj_forward(fk_model, fk_data)
    ee_world = fk_data.xpos[ee_id].copy()
    root_pos = fk_data.qpos[:3].copy()
    root_quat = fk_data.qpos[3:7].copy()
    return _world_to_base(ee_world, root_pos, root_quat)


def _fk_ee_positions(
    fk_model: "mujoco.MjModel",
    fk_data: "mujoco.MjData",
    ee_id: int,
    qpos7_root: np.ndarray,    # (7,) [x,y,z,qw,qx,qy,qz]  MuJoCo 格式
    dof_abs_arr: np.ndarray,   # (N, 29)
) -> np.ndarray:
    """批量 FK：返回 (N, 3) EE 世界坐标（保留供调试）。"""
    N = dof_abs_arr.shape[0]
    out = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        fk_data.qpos[:7] = qpos7_root
        fk_data.qpos[7:] = dof_abs_arr[i]
        fk_data.qvel[:] = 0.0
        mujoco.mj_forward(fk_model, fk_data)
        out[i] = fk_data.xpos[ee_id].copy()
    return out


def _fk_one_step(
    fk_model: "mujoco.MjModel",
    fk_data: "mujoco.MjData",
    ee_id: int,
    root_states_row: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    """单步 FK，返回 EE 世界坐标 (3,)（保留供调试）。"""
    fk_data.qpos[:7] = root_states_row[[0, 1, 2, 6, 3, 4, 5]]
    fk_data.qpos[7:] = dof_pos
    fk_data.qvel[:] = 0.0
    mujoco.mj_forward(fk_model, fk_data)
    return fk_data.xpos[ee_id].copy()


def _render_with_ee_sphere(
    rgb_renderer: "IsaacRendererWithMuJoco",
    hv_env,
    ee_id: int,
) -> np.ndarray:
    """
    在 MuJoCo 渲染帧里注入右手末端的橙色球体标记，返回 (H, W, 3) uint8。

    实现：直接调用 g1.renderer.update_scene + 注入 mjvGeom sphere + render，
    绕过 g1.render() 以便在 update_scene 和 render 之间插入球体。
    """
    g1 = IsaacRendererWithMuJoco._inner_g1_env(rgb_renderer.mujoco_env)

    # 1. 从 Isaac 读取当前状态，写入 MuJoCo
    base_pos = hv_env.simulator.robot_root_states[:, [0, 1, 2, 6, 3, 4, 5]].float().cpu().numpy()
    joint_pos = hv_env.simulator.dof_pos.float().cpu().numpy()
    mujoco_qpos = np.concatenate([base_pos, joint_pos], axis=1)[0]  # (36,)

    qvel = g1._mj_data.qvel.copy()
    rgb_renderer.mujoco_env.reset(options={"qpos": mujoco_qpos, "qvel": qvel})
    mujoco.mj_forward(g1.model, g1.data)

    # 2. 末端世界坐标
    ee_pos = g1.data.xpos[ee_id].copy().astype(np.float64)

    # 3. 确保 renderer 已初始化
    if g1.renderer is None:
        g1.renderer = mujoco.Renderer(
            g1._mj_model,
            width=g1._config.render_width,
            height=g1._config.render_height,
        )
        mujoco.mj_forward(g1._mj_model, g1._mj_data)

    # 4. update_scene（与 g1.render() 内部一致）
    cam_arg = getattr(g1._config, "camera", "track")
    g1.renderer.update_scene(g1._mj_data, camera=cam_arg)

    # 5. 注入球体 geom
    scene = getattr(g1.renderer, "scene", None) or getattr(g1.renderer, "_scene", None)
    if scene is not None and scene.ngeom < scene.maxgeom:
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        try:
            mujoco.mjv_initGeom(
                geom,
                int(mujoco.mjtGeom.mjGEOM_SPHERE),
                np.full(3, _SPHERE_RADIUS, dtype=np.float64),
                ee_pos,
                np.eye(3, dtype=np.float64).flatten(),
                np.array(_SPHERE_RGBA, dtype=np.float32),
            )
        except Exception:
            geom.type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
            geom.size[:] = _SPHERE_RADIUS
            geom.pos[:] = ee_pos
            geom.mat[:] = np.eye(3, dtype=np.float64).flatten()
            geom.rgba[:] = np.array(_SPHERE_RGBA, dtype=np.float32)
            geom.dataid = -1

    # 6. render
    return g1.renderer.render()


def _save_ee_traj_plot(
    expert_ee_base: np.ndarray,   # (N, 3) 基座系
    policy_ee_base: np.ndarray,   # (M, 3) 基座系
    out_path: Path,
    title: str = "",
) -> None:
    """
    将专家轨迹（蓝色虚线）和 policy 轨迹（红色实线）投影到
    **机器人基座（pelvis）局部坐标系的 Y-Z 平面** 绘制并保存。

    每帧 EE 位置先变换到该帧 pelvis 局部系，再取 Y、Z 分量作图，
    避免 inference 时 base 漂移导致世界系投影失真。
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        expert_ee_base[:, 1], expert_ee_base[:, 2],
        color="#1D7FD4", lw=1.8, linestyle="--", alpha=0.9,
        label=f"Expert / NPZ FK  (T={len(expert_ee_base)})",
    )
    ax.scatter(expert_ee_base[0, 1], expert_ee_base[0, 2],
               marker="*", s=220, color="#1D7FD4", zorder=8, label="expert start")

    ax.plot(
        policy_ee_base[:, 1], policy_ee_base[:, 2],
        color="#E63946", lw=2.0, linestyle="-", alpha=0.9,
        label=f"Policy rollout FK  (T={len(policy_ee_base)})",
    )
    ax.scatter(policy_ee_base[0, 1], policy_ee_base[0, 2],
               marker="*", s=220, color="#E63946", zorder=8, label="policy start")

    min_len = min(len(expert_ee_base), len(policy_ee_base))
    yz_dev = np.linalg.norm(
        expert_ee_base[:min_len, 1:3] - policy_ee_base[:min_len, 1:3], axis=1
    )
    mean_dev = float(yz_dev.mean())
    max_dev  = float(yz_dev.max())

    ax.set_xlabel("Y_base  (m)", fontsize=12)
    ax.set_ylabel("Z_base  (m)", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35)
    ax.set_title(
        "Right-hand EE Trajectory (pelvis frame, Y-Z plane)"
        + (f"  |  {title}" if title else "")
        + f"\nmean_dev={mean_dev*100:.2f} cm   max_dev={max_dev*100:.2f} cm",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"EE 轨迹对比图已保存（基座系 Y-Z）: {out_path}")


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

    # ── FK 模型（轻量 MuJoCo 实例，专用于末端轨迹计算） ─────────────────────
    from humanoidverse.utils.g1_env_config import G1EnvConfig
    _fk_env, _ = G1EnvConfig(render_height=16, render_width=16).build(num_envs=1)
    _fk_g1     = IsaacRendererWithMuJoco._inner_g1_env(_fk_env)
    fk_model   = _fk_g1._mj_model
    fk_data    = mujoco.MjData(fk_model)
    ee_id      = int(mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_BODY, _EE_BODY))
    print(f"  EE body '{_EE_BODY}' id = {ee_id}")

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

    # ── 专家末端轨迹（FK on NPZ 关节角，根固定在初始位置） ─────────────────
    # root_quat 此处已是 MuJoCo wxyz 格式（isaacsim 路径已做 [3,0,1,2] 重排）
    qpos7_root = np.concatenate([root_pos, root_quat]).astype(np.float64)
    print("Computing expert EE trajectory via FK (base frame)...")
    expert_ee_base = _fk_ee_positions_base(
        fk_model, fk_data, ee_id, qpos7_root, dof_abs_np
    )
    print(f"  expert_ee_base shape: {expert_ee_base.shape}  "
          f"Y_base=[{expert_ee_base[:,1].min():.3f},{expert_ee_base[:,1].max():.3f}]  "
          f"Z_base=[{expert_ee_base[:,2].min():.3f},{expert_ee_base[:,2].max():.3f}]")

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

    # policy EE 轨迹（基座系）：从第 0 帧开始
    _rs0 = wrapped_env._env.simulator.robot_root_states[0].float().detach().cpu().numpy()
    _d0  = wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0, :, 0].float().detach().cpu().numpy()
    policy_ee_base_list = [_fk_one_step_base(fk_model, fk_data, ee_id, _rs0, _d0)]

    if save_mp4:
        try:
            import mediapy as media
        except ImportError:
            raise ImportError("save_mp4 需要 mediapy：pip install mediapy")
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
        expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + n_steps])
        # 初始帧：policy 视频带末端球体标记
        frames = [_render_with_ee_sphere(rgb_renderer, wrapped_env._env, ee_id)]

    for i in range(n_steps):
        action = model.act(observation, z[i % len(z)].repeat(num_envs, 1), mean=True)
        observation, reward, terminated, truncated, info = wrapped_env.step(
            action, to_numpy=False
        )
        joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())

        # 收集 policy EE 位置（基座系，每帧用当前 root 变换）
        _rs = wrapped_env._env.simulator.robot_root_states[0].float().detach().cpu().numpy()
        _d  = wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0, :, 0].float().detach().cpu().numpy()
        policy_ee_base_list.append(_fk_one_step_base(fk_model, fk_data, ee_id, _rs, _d))

        if save_mp4:
            frames.append(_render_with_ee_sphere(rgb_renderer, wrapped_env._env, ee_id))
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1}/{n_steps}")

    joint_pos_arr = np.stack(joint_pos, axis=0).squeeze(1)
    print(f"Rollout complete. joint_pos shape: {joint_pos_arr.shape}")

    # ── EE 轨迹对比图（始终保存） ─────────────────────────────────────────────
    policy_ee_base = np.array(policy_ee_base_list)   # (n_steps+1, 3)
    ee_plot_path = out_dir / f"ee_traj_{stem}.png"
    _save_ee_traj_plot(
        expert_ee_base[: len(policy_ee_base)],
        policy_ee_base,
        ee_plot_path,
        title=stem,
    )

    if save_mp4:
        new_frames = [
            np.concatenate([a, b], axis=1)
            for a, b in zip(expert_video, frames)
        ]
        video_path = out_dir / f"tracking_{stem}.mp4"
        media.write_video(str(video_path), new_frames, fps=50)
        print(f"Saved video: {video_path}")

    print(f"\n=== Done ===")
    print(f"  z pkl      : {z_save_path}")
    print(f"  EE traj    : {ee_plot_path}")
    if save_mp4:
        print(f"  video      : {out_dir / f'tracking_{stem}.mp4'}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
