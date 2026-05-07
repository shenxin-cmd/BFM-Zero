"""
验证「拆解 z」后策略对 z_hand 的敏感性（不修改 goal_inference.py）。

Goal 来源（二选一）
------------------
1. **内置 MuJoCo 站立（推荐）**：在下方「用户可编辑区」设置
   ``STAND_POSE_JOINT_DELTA_RAD``（在训练用默认关节角上加增量，单位 rad）。
   脚本用 ``compute_humanoid_observations_max`` + 与 ``get_backward_observation``
   一致的 ``state`` / ``history_actor`` 拼装，无需外部 npz。

2. **外部文件**：传入 ``--goal-file`` 仍为 .npz / .pkl。

**推理方式**
------------
- **打印表格（L2 敏感度）**：对固定 goal 观测做 **开环** ``model.act(obs_actor, z)``，
  只扰动 ``z_hand``，用于对比动作变化量。
- **录视频**：必须用 **闭环**：每一步用 MuJoCo 环境返回的
  ``state`` / ``last_action`` 更新策略输入（与 ``goal_inference`` 类似），否则
  观测不变则每步动作相同，运动学引擎会重复同一姿态，视频像定格幻灯片。

开环说明、split-z 要求等见脚本后半 docstring 与打印。

用法
----
.. code-block:: bash

   # 内置站立（不传 goal-file）
   python -m humanoidverse.split_z_hand_validation \\
     --checkpoint-dir workdir/bfmzero-split-z/<run>/checkpoint

   # 只录「无噪 / 大噪」两档短视频（闭环渲染，默认同表用 hand-noise-stds）
   python -m humanoidverse.split_z_hand_validation \\
     --checkpoint-dir ... \\
     --hand-noise-stds 0.0 2.0 \\
     --record-video \\
     --video-hand-noise-stds 0.0 2.0

注意：``env.root_height_obs`` 与训练 ``config.json`` 一致时，
``privileged_state`` 维数才对（通常带 root 高度为 463）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium.spaces as gym_spaces
import joblib
import mediapy as media
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyR

# -----------------------------------------------------------------------------
# 用户可编辑区：站立姿势 = 训练 YAML 默认角 + 增量（弧度）
# （默认角与 config/robot/g1/g1_29dof.yaml robot.init_state.default_joint_angles 一致）
# -----------------------------------------------------------------------------

# 在此 dict 中填写要调整的关节名 -> 相对默认值的增量 [rad]；未列出的关节保持默认。
STAND_POSE_JOINT_DELTA_RAD: dict[str, float] = {
    # 例：略抬右肩
    # "right_shoulder_pitch_joint": 0.3,
}

# 使用的 MuJoCo 场景（可通过 CLI --mujoco-xml 覆盖）
DEFAULT_MUJOCO_SCENE = "scene_29dof_freebase_mujoco.xml"

# 与训练一致的 **MuJoCo / 仿真体** 顺序（30 个 link，不含虚拟 head）
_MJ_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

# motion.extend_config：与 g1_29dof.yaml 一致，用于 max_local_self 第 31 个 body（nums_extend_bodies=1）
_HEAD_LINK_PARENT = "torso_link"
_HEAD_LINK_OFFSET_PARENT = (0.0, 0.0, 0.35)  # 与训练 extend_config.pos 一致
# rot 为 w,x,y,z = [1,0,0,0] -> 子相对父四元数（xyzw, w_last）为单位元
_HEAD_LINK_REL_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

# 与训练一致的自由度顺序（robot.dof_names）
_DOF_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_REF_DEFAULT_DOF_POS: dict[str, float] = {
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
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# -----------------------------------------------------------------------------
# Indexing: body joints 0..21, right-hand joints 22..28 (match training action order).
# -----------------------------------------------------------------------------
BODY_ACTION_DIM = 22
HAND_ACTION_DIM = 7
_EXPECTED_PRIV_WITH_ROOT_H = 463
_EXPECTED_PRIV_NO_ROOT_H = 462


def _append_head_extend_torch(
    bp: torch.Tensor,
    br: torch.Tensor,
    bv: torch.Tensor,
    bw: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 LeggedRobotMotions 中 extend body 逻辑一致：在 30 链节后追加 head_link（静止，线速度 0）。"""
    from humanoidverse.utils.torch_utils import my_quat_rotate, quat_mul

    torso_i = _MJ_BODY_NAMES.index(_HEAD_LINK_PARENT)
    pr = br[:, torso_i, :].reshape(1, 4)
    ppos = bp[:, torso_i, :].reshape(1, 3)
    off = torch.tensor([_HEAD_LINK_OFFSET_PARENT], dtype=torch.float32, device=device)
    rel = torch.tensor([_HEAD_LINK_REL_QUAT_XYZW], dtype=torch.float32, device=device)

    rotated = my_quat_rotate(pr, off)
    head_pos = my_quat_rotate(rel, rotated) + ppos
    head_pos = head_pos.unsqueeze(1)
    head_rot = quat_mul(pr, rel, w_last=True).unsqueeze(1)

    z3 = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
    bp2 = torch.cat([bp, head_pos], dim=1)
    br2 = torch.cat([br, head_rot], dim=1)
    bv2 = torch.cat([bv, z3], dim=1)
    bw2 = torch.cat([bw, z3], dim=1)
    return bp2, br2, bv2, bw2
def _training_default_dof_vec() -> np.ndarray:
    return np.array([_REF_DEFAULT_DOF_POS[n] for n in _DOF_NAMES], dtype=np.float64)


def build_goal_arrays_from_mujoco_stand(
    *,
    mujoco_scene_xml: Path,
    root_height_obs: bool,
    joint_delta_rad: dict[str, float],
    gravity_z: float = -9.81,
) -> dict[str, np.ndarray]:
    """
    组装与 ``helpers.get_backward_observation(..., use_obs_filter 逻辑)`` 对齐的一帧观测
    （速度置零，等价于velocity_multiplier=0）。
    """
    from humanoidverse.envs.legged_robot_motions.legged_robot_motions import compute_humanoid_observations_max
    from humanoidverse.utils.g1_env_config import get_g1_robot_xml_root
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    xml_path = Path(mujoco_scene_xml).resolve()
    if not xml_path.is_file():
        alt = get_g1_robot_xml_root() / DEFAULT_MUJOCO_SCENE
        xml_path = alt.resolve()

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)

    kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid < 0:
        raise RuntimeError(f"No keyframe 'stand' in model {xml_path}")
    mj_data.qpos[:] = mj_model.key_qpos[kid]
    mj_data.qvel[:] = 0.0

    mj_type_hinge = int(mujoco.mjtJoint.mjJNT_HINGE)

    defaults = np.array([_REF_DEFAULT_DOF_POS[n] for n in _DOF_NAMES], dtype=np.float64)

    def _effective_angle(name: str) -> float:
        return float(defaults[_DOF_NAMES.index(name)]) + float(joint_delta_rad.get(name, 0.0))

    # 写入与训练 dof 列表一致的铰链关节角（MJCF 必须与这些 joint 名称一致）。
    for jn in _DOF_NAMES:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise KeyError(f"Joint {jn!r} not found in {xml_path}")
        if mj_model.jnt_type[jid] != mj_type_hinge:
            raise TypeError(f"Joint {jn!r} must be mjJNT_HINGE for this script (got type {mj_model.jnt_type[jid]})")
        qadr = int(mj_model.jnt_qposadr[jid])
        mj_data.qpos[qadr] = _effective_angle(jn)

    mujoco.mj_forward(mj_model, mj_data)

    # ---- Bodies: position + orientation (qx,qy,qz,qw / w_last) ----
    pos_list: list[np.ndarray] = []
    rot_list: list[np.ndarray] = []
    zeros_vel = []

    device = torch.device("cpu")

    for bn in _MJ_BODY_NAMES:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bn)
        if bid < 0:
            raise KeyError(f"Body {bn!r} not found")
        xp = mj_data.xpos[bid].astype(np.float64).copy()
        xmat = mj_data.xmat[bid].reshape(3, 3).astype(np.float64).copy()
        qx, qy, qz, qw = ScipyR.from_matrix(xmat).as_quat()
        pos_list.append(xp)
        rot_list.append(np.array([qx, qy, qz, qw], dtype=np.float64))

    bp = torch.tensor(np.stack(pos_list), dtype=torch.float32, device=device).unsqueeze(0)
    br = torch.tensor(np.stack(rot_list), dtype=torch.float32, device=device).unsqueeze(0)
    bv = torch.zeros((1, br.shape[1], 3), dtype=torch.float32, device=device)
    bw = torch.zeros_like(bv)

    bp, br, bv, bw = _append_head_extend_torch(bp, br, bv, bw, device)

    obs_parts = compute_humanoid_observations_max(bp, br, bv, bw, True, root_height_obs)
    max_local_np = torch.cat([v for v in obs_parts.values()], dim=-1).detach().numpy().squeeze(0)

    expected = _EXPECTED_PRIV_WITH_ROOT_H if root_height_obs else _EXPECTED_PRIV_NO_ROOT_H
    if max_local_np.shape[-1] != expected:
        raise RuntimeError(
            f"privileged_state dim {max_local_np.shape[-1]} != expected {expected} "
            f"(root_height_obs={root_height_obs}). Check train config.json env.root_height_obs."
        )

    # Root orientation (MJ free joint quaternion in qpos: w,x,y,z)
    qw_r, qx_r, qy_r, qz_r = (float(x) for x in mj_data.qpos[3:7])
    base_quat = torch.tensor([[qx_r, qy_r, qz_r, qw_r]], dtype=torch.float32, device=device)

    dof_abs = []
    dof_vel_zeros = []
    for jn in _DOF_NAMES:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr = int(mj_model.jnt_qposadr[jid])
        vadr = int(mj_model.jnt_dofadr[jid])
        dof_abs.append(float(mj_data.qpos[qadr]))
        dof_vel_zeros.append(float(mj_data.qvel[vadr]))

    dof_abs = np.asarray(dof_abs, dtype=np.float32)
    ref_dof_pos_rel = dof_abs - defaults.astype(np.float32)

    dof_vel_np = np.zeros(29, dtype=np.float32)
    g_world = torch.tensor([[0.0, 0.0, gravity_z]], dtype=torch.float32, device=device)
    projected_gravity = (
        quat_rotate_inverse(base_quat, g_world, w_last=True).detach().numpy().squeeze(0).astype(np.float32)
    )

    root_ang_w = torch.tensor([mj_data.qvel[:3].astype(np.float32)], device=device)
    ref_ang_vel = (
        quat_rotate_inverse(base_quat, root_ang_w, w_last=True).detach().numpy().squeeze(0).astype(np.float32)
    )

    state = np.concatenate([ref_dof_pos_rel, dof_vel_np, projected_gravity, ref_ang_vel]).astype(np.float32)
    if state.shape[-1] != 64:
        raise RuntimeError(f"state dim {state.shape[-1]} != 64")

    bogus_actions = ref_dof_pos_rel.copy()
    last_action = bogus_actions.copy()
    bogus_history_piece = np.concatenate([bogus_actions, ref_ang_vel, ref_dof_pos_rel, dof_vel_np, projected_gravity])
    bogus_history_piece = bogus_history_piece.astype(np.float32)
    if bogus_history_piece.shape[-1] * 4 != 372:
        raise RuntimeError(
            f"single history slice dim {bogus_history_piece.shape[-1]} -> stacked != 372 "
            "(check history layout vs HumanoidVerse)."
        )

    history_actor = np.tile(bogus_history_piece, 4)

    return {
        "state": state,
        "privileged_state": max_local_np.astype(np.float32),
        "last_action": last_action,
        "history_actor": history_actor,
    }


def _load_goal_dict(path: Path) -> dict[str, np.ndarray]:
    suf = path.suffix.lower()
    if suf == ".npz":
        raw = np.load(path, allow_pickle=False)
        return {k: np.asarray(raw[k], dtype=np.float32) for k in raw.files}
    if suf in (".pkl", ".joblib"):
        obj = joblib.load(path)
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict in {path}, got {type(obj)}")
        return {k: np.asarray(v, dtype=np.float32) for k, v in obj.items()}
    raise ValueError(f"Unsupported goal file: {path} (use .npz or .pkl/.joblib)")


def _require_keys(d: dict[str, np.ndarray], keys: list[str]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Goal dict missing keys: {missing}. Present: {list(d.keys())}")


def _to_model_obs(d: dict[str, np.ndarray], keys: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        t = torch.as_tensor(d[k], dtype=torch.float32, device=device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        out[k] = t
    return out


def _dict_box_dim(space: gym_spaces.Dict, key: str) -> int:
    if key not in space.spaces:
        raise KeyError(f"model.obs_space has no key {key!r}; keys={list(space.spaces.keys())}")
    s = space.spaces[key]
    if not isinstance(s, gym_spaces.Box):
        raise TypeError(f"Expected Box for {key!r}, got {type(s)}")
    return int(np.prod(s.shape))


def _align_vec(x: np.ndarray, expected: int) -> np.ndarray:
    """Pad with zeros or truncate so vector length matches BatchNorm / obs_space (e.g. Isaac 35 vs MuJoCo 29)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == expected:
        return x
    if x.size > expected:
        return x[:expected].copy()
    out = np.zeros(expected, dtype=np.float32)
    out[: x.size] = x
    return out


def _actor_obs_from_g1_step(
    obs: dict[str, np.ndarray],
    history_actor_fixed: torch.Tensor,
    device: torch.device,
    state_dim: int,
    last_action_dim: int,
) -> dict[str, torch.Tensor]:
    """用 G1 单步观测更新 state / last_action；history 沿用 goal 初始化（与 humenv 不提供 4 帧栈一致）。"""
    s = _align_vec(obs["state"], state_dim)
    la = _align_vec(obs["last_action"], last_action_dim)
    return {
        "state": torch.from_numpy(s).to(device=device, dtype=torch.float32).unsqueeze(0),
        "last_action": torch.from_numpy(la).to(device=device, dtype=torch.float32).unsqueeze(0),
        "history_actor": history_actor_fixed,
    }


def _open_loop_compare(
    model,
    obs_actor: dict[str, torch.Tensor],
    z_ref: torch.Tensor,
    hand_noise_stds: list[float],
    seed: int,
) -> tuple[list[tuple[float, float, float]], torch.Tensor]:
    rng = torch.Generator(device=z_ref.device)
    rng.manual_seed(seed)
    z_body_dim = model.cfg.archi.z_body_dim

    with torch.no_grad():
        a0 = model.act(obs_actor, z_ref, mean=True)
    rows: list[tuple[float, float, float]] = []
    zb = z_ref[:, :z_body_dim]
    zh0 = z_ref[:, z_body_dim:]

    for std in hand_noise_stds:
        noise = torch.randn(zh0.shape, device=zh0.device, dtype=zh0.dtype, generator=rng)
        zh = zh0 + float(std) * noise
        z = model.project_z(torch.cat([zb, zh], dim=-1))
        with torch.no_grad():
            a = model.act(obs_actor, z, mean=True)
        db = (a[:, :BODY_ACTION_DIM] - a0[:, :BODY_ACTION_DIM]).norm(dim=-1).mean().item()
        dh = (a[:, BODY_ACTION_DIM : BODY_ACTION_DIM + HAND_ACTION_DIM] - a0[:, BODY_ACTION_DIM : BODY_ACTION_DIM + HAND_ACTION_DIM]).norm(
            dim=-1
        ).mean().item()
        rows.append((float(std), db, dh))
    return rows, a0


def main(
    checkpoint_dir: Path,
    goal_file: Path | None = None,
    mujoco_xml: Path | None = None,
    device: str = "cuda",
    hand_noise_stds: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    seed: int = 0,
    record_video: bool = False,
    video_path: Path | None = None,
    video_steps_per_noise: int = 200,
    video_hand_noise_stds: tuple[float, ...] | None = None,
    fps: int = 50,
) -> None:
    """CLI：不传 ``goal-file`` 时用内置 MuJoCo 站立 + 本文件顶部 ``STAND_POSE_JOINT_DELTA_RAD``。"""
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from humanoidverse.utils.g1_env_config import G1EnvConfig, NoiseConfig, get_g1_robot_xml_root

    checkpoint_dir = Path(checkpoint_dir).resolve()

    train_cfg_path = checkpoint_dir.parent / "config.json" if checkpoint_dir.name == "checkpoint" else checkpoint_dir / "config.json"
    if not train_cfg_path.is_file():
        train_cfg_path = checkpoint_dir / "config.json"
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    model = load_model_from_checkpoint_dir(str(checkpoint_dir), device=device)
    model.eval()
    if not model.cfg.archi.is_split_mode:
        raise RuntimeError(
            "Checkpoint is not split-z mode. Use ``train_bfm_zero_split_z`` checkpoint.",
        )

    sp = model.obs_space
    if not isinstance(sp, gym_spaces.Dict):
        raise RuntimeError(
            "This script expects a Dict obs_space (Isaac/HumEnv-style keys). Got "
            f"{type(sp).__name__}.",
        )

    root_height_obs = bool(train_cfg.get("env", {}).get("root_height_obs", False))
    mz = mujoco_xml if mujoco_xml is not None else get_g1_robot_xml_root() / DEFAULT_MUJOCO_SCENE

    if goal_file is None:
        print(f"Builtin stand goal from MuJoCo: {mz}")
        print(f"Joint deltas (rad): {STAND_POSE_JOINT_DELTA_RAD or '(none)'}")
        g_numpy = build_goal_arrays_from_mujoco_stand(
            mujoco_scene_xml=mz,
            root_height_obs=root_height_obs,
            joint_delta_rad=STAND_POSE_JOINT_DELTA_RAD,
        )
    else:
        gf = Path(goal_file).resolve()
        print(f"Loading goal file: {gf}")
        g_numpy = _load_goal_dict(gf)

    _require_keys(g_numpy, ["state", "privileged_state", "last_action", "history_actor"])

    # Isaac 训练里 last_action / actions 常与 MuJoCo nu（29）不等（多 padding），否则 BatchNorm 维度报错。
    g_numpy["state"] = _align_vec(g_numpy["state"], _dict_box_dim(sp, "state"))
    g_numpy["last_action"] = _align_vec(g_numpy["last_action"], _dict_box_dim(sp, "last_action"))
    g_numpy["history_actor"] = _align_vec(g_numpy["history_actor"], _dict_box_dim(sp, "history_actor"))

    z_body_dim = model.cfg.archi.z_body_dim
    z_hand_dim = model.cfg.archi.z_hand_dim
    print(f"Split-z: z_body_dim={z_body_dim}, z_hand_dim={z_hand_dim}, total={z_body_dim + z_hand_dim}")

    dev = torch.device(device)
    obs_b = _to_model_obs(g_numpy, ["state", "privileged_state"], dev)
    obs_actor = _to_model_obs(g_numpy, ["state", "last_action", "history_actor"], dev)

    if root_height_obs:
        print("(train config env.root_height_obs=True; privileged expects 463)")

    with torch.no_grad():
        z_ref = model.goal_inference(obs_b)
    print(f"z_ref shape: {z_ref.shape}, ||z_ref||: {z_ref.norm().item():.4f}")

    stds = list(hand_noise_stds)
    rows, a0 = _open_loop_compare(model, obs_actor, z_ref, stds, seed)
    print("\nOpen-loop (fixed goal obs): L2 change vs z_ref action")
    print(f"  baseline ||action|| per batch sample: {a0.norm(dim=-1).mean().item():.4f}")
    print("  noise_std |  ||Delta_a_body||   ||Delta_a_hand||")
    for std, db, dh in rows:
        print(f"  {std:8.4f} | {db:12.6f}  {dh:12.6f}")

    if not record_video:
        return

    env_wrapped, _ = G1EnvConfig(
        max_episode_steps=None,
        add_time=False,
        noise_config=NoiseConfig(level=0.0),
        render_height=480,
        render_width=640,
        camera="track",
    ).build(num_envs=1)

    out = Path(video_path) if video_path is not None else Path.cwd() / "split_z_hand_sweep.mp4"
    frames: list[np.ndarray] = []

    rng = torch.Generator(device=z_ref.device)
    rng.manual_seed(seed)
    zb = z_ref[:, :z_body_dim]
    zh0 = z_ref[:, z_body_dim:]

    video_stds = list(video_hand_noise_stds) if video_hand_noise_stds is not None else stds
    history_fixed = obs_actor["history_actor"]

    state_dim = _dict_box_dim(sp, "state")
    last_action_dim = _dict_box_dim(sp, "last_action")
    env_act_dim = int(np.prod(env_wrapped.action_space.shape))

    for std in video_stds:
        noise = torch.randn(zh0.shape, device=zh0.device, dtype=zh0.dtype, generator=rng)
        zh = zh0 + float(std) * noise
        z = model.project_z(torch.cat([zb, zh], dim=-1))
        obs, _ = env_wrapped.reset(seed=seed + int(100 * std))
        # 闭环：每步用环境真值 state/last_action，否则动作重复 → 运动学姿态不变（幻灯片）
        r0 = env_wrapped.render()
        if r0 is not None:
            frames.append(np.asarray(r0))
        for _ in range(video_steps_per_noise):
            obs_t = _actor_obs_from_g1_step(obs, history_fixed, dev, state_dim, last_action_dim)
            with torch.no_grad():
                action = model.act(obs_t, z, mean=True)
            a_flat = action.squeeze(0).detach().cpu().numpy().reshape(-1)
            if a_flat.size >= env_act_dim:
                a_np = a_flat[:env_act_dim]
            else:
                a_np = _align_vec(a_flat, env_act_dim)
            obs, _r, _te, _tu, _i = env_wrapped.step(a_np)
            fr = env_wrapped.render()
            if fr is not None:
                frames.append(np.asarray(fr))

    if frames:
        media.write_video(str(out), frames, fps=fps)
        print(f"\nSaved video to {out} ({len(frames)} frames, {video_stds=})")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
