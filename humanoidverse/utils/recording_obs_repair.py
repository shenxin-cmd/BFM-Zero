"""Repair recording NPZ obs for tracking inference.

Recording format (``*_obs.npz``):
  - ``state``            (T, 64)  = [dof_pos(29), dof_vel(29), proj_grav(3), ang_vel(3)]
  - ``last_action``      (T, 29)
  - ``privileged_state`` (T, 463) — often has wrong body velocities (~30× too large)
    because the recorder used finite-difference velocities instead of MuJoCo ``cvel``.

This module reconstructs ``qpos``/``qvel`` from ``state`` (matching the user's
``compute_state`` convention), runs MuJoCo FK + ``compute_humanoid_observations_max``-style
privileged encoding, and returns a repaired trajectory dict ready for
``tracking_inference_split.py``.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

G1_DEFAULT_JOINT_POS = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
G1_DEFAULT_ROOT_HEIGHT = 0.793
HEAD_LINK_OFFSET_TORSO_FRAME = np.array([0.0, 0.0, 0.35], dtype=np.float64)

G1_BODY_NAMES_30 = [
    'pelvis',
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
    'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
    'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
    'waist_yaw_link', 'waist_roll_link', 'torso_link',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    'left_elbow_link',
    'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link',
]
TORSO_BODY_IDX = G1_BODY_NAMES_30.index('torso_link')

_MJCF_CANDIDATES = (
    "humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml",
    "humanoidverse/data/robot/g1/scene_29dof_freebase_mujoco.xml",
    "humanoidverse/data/robots/g1/g1_29dof.xml",
    "humanoidverse/data/robot/g1/g1_29dof.xml",
)


def find_g1_mjcf(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    for rel in _MJCF_CANDIDATES:
        p = root / rel
        if p.is_file():
            return p
    raise FileNotFoundError(
        "G1 MJCF not found. Tried:\n  " + "\n  ".join(str(root / r) for r in _MJCF_CANDIDATES)
    )


def state_to_qpos_qvel(
    state: np.ndarray,
    *,
    root_heights: np.ndarray | None = None,
    dt: float = 1.0 / 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct MuJoCo ``qpos`` (T,36) and ``qvel`` (T,35) from ``state`` (T,64).

    Matches the user's ``compute_state``:
      dof_pos  = qpos[7:36] - DEFAULT_JOINT_POS
      dof_vel  = qvel[6:35]
      proj_grav = R_root.inv().apply([0,0,-1])
      ang_vel  = R_root.apply(qvel[3:6])   # body-frame root angular velocity
    """
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] < 64:
        raise ValueError(f"state must be (T, 64+), got {state.shape}")

    T = state.shape[0]
    dof_rel = state[:, :29]
    dof_vel = state[:, 29:58]
    proj_grav = state[:, 58:61]
    ang_vel_body = state[:, 61:64]

    dof_abs = dof_rel + G1_DEFAULT_JOINT_POS[None, :]

    root_quats_wxyz = np.empty((T, 4), dtype=np.float64)
    g_down = np.array([0.0, 0.0, -1.0])
    for t in range(T):
        pg = proj_grav[t]
        pg_norm = pg / (np.linalg.norm(pg) + 1e-8)
        # R_root @ pg_norm = g_down  (inverse of proj_grav = R_root^T @ g_down)
        r = sRot.align_vectors(g_down[None], pg_norm[None])[0]
        root_quats_wxyz[t] = r.as_quat()[[3, 0, 1, 2]]  # xyzw → wxyz

    root_pos = np.zeros((T, 3), dtype=np.float64)
    if root_heights is not None:
        root_pos[:, 2] = np.asarray(root_heights, dtype=np.float64).reshape(-1)[:T]
    else:
        root_pos[:, 2] = G1_DEFAULT_ROOT_HEIGHT

    qpos = np.concatenate([root_pos, root_quats_wxyz, dof_abs], axis=-1)

    qvel = np.zeros((T, 35), dtype=np.float64)
    qvel[:, 6:35] = dof_vel
    if T > 1:
        qvel[1:, :3] = (root_pos[1:] - root_pos[:-1]) / dt
        qvel[0, :3] = qvel[1, :3]
    for t in range(T):
        rot = sRot.from_quat(root_quats_wxyz[t, [1, 2, 3, 0]])  # wxyz → xyzw
        qvel[t, 3:6] = rot.inv().apply(ang_vel_body[t])

    return qpos, qvel


def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)


def _quat_rotate_batch_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    for i in range(len(q_xyzw)):
        out[i] = sRot.from_quat(q_xyzw[i]).apply(v[i])
    return out


def _calc_heading_quat_inv_np(q_xyzw: np.ndarray) -> np.ndarray:
    ref = np.zeros((len(q_xyzw), 3))
    ref[:, 0] = 1.0
    rot_dir = _quat_rotate_batch_np(q_xyzw, ref)
    heading = np.arctan2(rot_dir[:, 1], rot_dir[:, 0])
    half = -heading / 2.0
    return np.stack([np.zeros_like(half), np.zeros_like(half), np.sin(half), np.cos(half)], axis=-1)


def _quat_to_tan_norm_np(q_xyzw: np.ndarray) -> np.ndarray:
    ref_x = np.zeros((len(q_xyzw), 3)); ref_x[:, 0] = 1.0
    ref_z = np.zeros((len(q_xyzw), 3)); ref_z[:, 2] = 1.0
    tan = _quat_rotate_batch_np(q_xyzw, ref_x)
    norm = _quat_rotate_batch_np(q_xyzw, ref_z)
    return np.concatenate([tan, norm], axis=-1)


def compute_privileged_state_from_bodies(
    body_pos: np.ndarray,
    body_rot: np.ndarray,
    body_vel: np.ndarray,
    body_ang_vel: np.ndarray,
) -> np.ndarray:
    """``compute_humanoid_observations_max`` numpy port (root_height_obs=True, 31 bodies)."""
    T, n_bodies = body_pos.shape[0], body_pos.shape[1]
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    root_h = root_pos[:, 2:3]

    h_inv = _calc_heading_quat_inv_np(root_rot)
    h_inv_exp = np.repeat(h_inv[:, None, :], n_bodies, axis=1)

    local_pos = body_pos - root_pos[:, None, :]
    flat_lp = local_pos.reshape(T * n_bodies, 3)
    flat_hi = h_inv_exp.reshape(T * n_bodies, 4)
    local_body_pos = _quat_rotate_batch_np(flat_hi, flat_lp).reshape(T, n_bodies * 3)[:, 3:]

    flat_br = body_rot.reshape(T * n_bodies, 4)
    flat_local_rot = _quat_mul_np(flat_hi, flat_br)
    local_body_rot = _quat_to_tan_norm_np(flat_local_rot).reshape(T, n_bodies * 6)

    flat_bv = body_vel.reshape(T * n_bodies, 3)
    local_body_vel = _quat_rotate_batch_np(flat_hi, flat_bv).reshape(T, n_bodies * 3)

    flat_bav = body_ang_vel.reshape(T * n_bodies, 3)
    local_body_ang_vel = _quat_rotate_batch_np(flat_hi, flat_bav).reshape(T, n_bodies * 3)

    return np.concatenate(
        [root_h, local_body_pos, local_body_rot, local_body_vel, local_body_ang_vel],
        axis=-1,
    ).astype(np.float32)


def _get_mj_body_ids(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    ids = []
    for n in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid < 0:
            raise ValueError(f"Body '{n}' not found in MuJoCo model")
        ids.append(bid)
    return np.array(ids, dtype=np.int32)


def mujoco_body_states_from_qpos_qvel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_seq: np.ndarray,
    qvel_seq: np.ndarray,
    body_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return body states (T, 31, *) including virtual head_link extend body."""
    T = qpos_seq.shape[0]
    body_pos = np.empty((T, 31, 3), dtype=np.float64)
    body_rot = np.empty((T, 31, 4), dtype=np.float64)
    body_vel = np.empty((T, 31, 3), dtype=np.float64)
    body_ang_vel = np.empty((T, 31, 3), dtype=np.float64)

    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        data.qvel[:] = qvel_seq[t]
        mujoco.mj_forward(model, data)

        bp = data.xpos[body_ids].copy()
        br = data.xquat[body_ids][:, [1, 2, 3, 0]].copy()
        bv = data.cvel[body_ids, 3:6].copy()
        bav = data.cvel[body_ids, 0:3].copy()

        body_pos[t, :30] = bp
        body_rot[t, :30] = br
        body_vel[t, :30] = bv
        body_ang_vel[t, :30] = bav

        torso_pos = bp[TORSO_BODY_IDX]
        torso_rot = br[TORSO_BODY_IDX]
        head_pos = torso_pos + sRot.from_quat(torso_rot).apply(HEAD_LINK_OFFSET_TORSO_FRAME)
        body_pos[t, 30] = head_pos
        body_rot[t, 30] = torso_rot
        body_vel[t, 30] = bv[TORSO_BODY_IDX]
        body_ang_vel[t, 30] = bav[TORSO_BODY_IDX]

    return (
        body_pos.astype(np.float32),
        body_rot.astype(np.float32),
        body_vel.astype(np.float32),
        body_ang_vel.astype(np.float32),
    )


def repair_recording_traj(
    traj: dict[str, np.ndarray],
    *,
    mjcf_path: Path | None = None,
    dt: float = 1.0 / 30.0,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Repair ``privileged_state`` in a recording trajectory dict.

    Keeps ``state`` and ``last_action`` unchanged (user-confirmed joint data).
    Adds ``mujoco_qpos`` / ``mujoco_qvel`` for accurate Isaac reset & rendering.
    """
    for k in ("state", "last_action", "privileged_state"):
        if k not in traj:
            raise KeyError(f"recording traj missing {k!r}; keys={sorted(traj.keys())}")

    state = np.asarray(traj["state"], dtype=np.float64)
    T = state.shape[0]
    root_heights = None
    priv_old = np.asarray(traj["privileged_state"], dtype=np.float64)
    if priv_old.ndim == 2 and priv_old.shape[0] == T and priv_old.shape[1] >= 1:
        root_heights = priv_old[:, 0]

    if "qpos" in traj and traj["qpos"].shape == (T, 36):
        qpos_seq = np.asarray(traj["qpos"], dtype=np.float64)
        if "qvel" in traj and traj["qvel"].shape == (T, 35):
            qvel_seq = np.asarray(traj["qvel"], dtype=np.float64)
            source = "qpos+qvel"
        else:
            qvel_seq = state_to_qpos_qvel(state, root_heights=root_heights, dt=dt)[1]
            source = "qpos + state-derived qvel"
    else:
        qpos_seq, qvel_seq = state_to_qpos_qvel(state, root_heights=root_heights, dt=dt)
        source = "state → qpos/qvel"

    mjcf = mjcf_path or find_g1_mjcf()
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    body_ids = _get_mj_body_ids(model, G1_BODY_NAMES_30)

    bp, br, bv, bav = mujoco_body_states_from_qpos_qvel(model, data, qpos_seq, qvel_seq, body_ids)
    priv_new = compute_privileged_state_from_bodies(bp, br, bv, bav)

    if verbose:
        print(f"  repair source: {source}")
        print(f"  privileged_state: {priv_new.shape}  range [{priv_new.min():.3f}, {priv_new.max():.3f}]")
        if priv_old.shape == priv_new.shape:
            pos_diff = float(np.abs(priv_new[:, 1:91] - priv_old[:, 1:91]).mean())
            vel_diff = float(np.abs(priv_new[:, 277:370] - priv_old[:, 277:370]).mean())
            print(f"  Δlocal_body_pos: {pos_diff:.4f} m  Δlocal_body_vel: {vel_diff:.4f} m/s")

    out = dict(traj)
    out["privileged_state"] = priv_new
    out["mujoco_qpos"] = qpos_seq.astype(np.float32)
    out["mujoco_qvel"] = qvel_seq.astype(np.float32)
    return out
