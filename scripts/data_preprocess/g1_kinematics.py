"""Minimal numpy-only kinematics helpers for the G1 29-DOF robot.

Mirrors the conventions of ``humanoidverse/utils/motion_lib/torch_humanoid_batch.py``
(``Humanoid_Batch``) so that the pkl files produced by the conversion scripts are
bit-compatible with what ``MotionLibRobot`` expects:

  * ``pose_aa``           : (T, 30, 3) axis-angle.
                            - index 0      : GLOBAL root (pelvis) rotation as rotvec
                            - index 1..29  : per-joint LOCAL rotation = angle * joint_axis
                              (MJCF body order == dof_names order, one hinge per body)
  * ``root_trans_offset`` : (T, 3) world-frame pelvis position [m]
  * ``fps``               : int

Only numpy + scipy are required (no torch), so the scripts can run on any machine.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot

# ---------------------------------------------------------------------------
# Constants (G1 29-DOF, must match humanoidverse/config/robot/g1/g1_29dof*.yaml)
# ---------------------------------------------------------------------------

G1_DOF_NAMES: list[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# default_joint_angles from g1_29dof_hard_waist.yaml (== DEFAULT_JOINT_POS used by the
# data1 NPZ generation pipeline, see BATCH_DATA_README.md §4.1)
G1_DEFAULT_JOINT_POS = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # left leg
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # right leg
     0.0, 0.0, 0.0,                          # waist
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # left arm
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # right arm
    dtype=np.float64,
)

RIGHT_ARM_DOF_IDX = list(range(22, 29))
LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"


@dataclass
class G1Skeleton:
    body_names: list[str]
    parents: np.ndarray            # (30,) int, parents[0] == -1
    local_translation: np.ndarray  # (30, 3) body offset in parent frame
    local_rotation: np.ndarray     # (30, 4) wxyz quat of the fixed body frame
    dof_axes: np.ndarray           # (29, 3) hinge axis of body i+1 in its own frame
    joint_names: list[str]         # (29,) joint name of body i+1
    joints_range: np.ndarray       # (29, 2) joint limits [rad]

    def body_index(self, name: str) -> int:
        return self.body_names.index(name)


def load_g1_skeleton(mjcf_path: str | Path) -> G1Skeleton:
    """Parse the G1 MJCF the same way ``Humanoid_Batch.from_mjcf`` does (depth-first)."""
    tree = ET.parse(str(mjcf_path))
    worldbody = tree.getroot().find("worldbody")
    root_body = worldbody.find("body")

    body_names, parents = [], []
    local_translation, local_rotation = [], []
    dof_axes, joint_names, joints_range = [], [], []

    def _add(body, parent_idx):
        idx = len(body_names)
        body_names.append(body.attrib["name"])
        parents.append(parent_idx)
        local_translation.append(np.fromstring(body.attrib.get("pos", "0 0 0"), dtype=float, sep=" "))
        local_rotation.append(np.fromstring(body.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" "))
        for joint in body.findall("joint"):
            if joint.attrib.get("type") == "free":
                continue  # pelvis floating base
            joint_names.append(joint.attrib["name"])
            dof_axes.append(np.fromstring(joint.attrib["axis"], dtype=float, sep=" "))
            rng = joint.attrib.get("range")
            joints_range.append(
                np.fromstring(rng, dtype=float, sep=" ") if rng else np.array([-np.pi, np.pi])
            )
        for child in body.findall("body"):
            _add(child, idx)

    _add(root_body, -1)

    skel = G1Skeleton(
        body_names=body_names,
        parents=np.asarray(parents, dtype=np.int64),
        local_translation=np.asarray(local_translation, dtype=np.float64),
        local_rotation=np.asarray(local_rotation, dtype=np.float64),
        dof_axes=np.asarray(dof_axes, dtype=np.float64),
        joint_names=joint_names,
        joints_range=np.asarray(joints_range, dtype=np.float64),
    )
    assert len(skel.body_names) == 30, f"expected 30 bodies, got {len(skel.body_names)}"
    assert len(skel.joint_names) == 29, f"expected 29 joints, got {len(skel.joint_names)}"
    assert skel.joint_names == G1_DOF_NAMES, (
        "MJCF joint order differs from G1_DOF_NAMES - conversion would scramble dofs:\n"
        f"{skel.joint_names}"
    )
    return skel


# ---------------------------------------------------------------------------
# pose_aa construction
# ---------------------------------------------------------------------------

def build_pose_aa(skel: G1Skeleton, root_rotvec: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
    """(T,3) root rotvec + (T,29) joint angles [rad] -> (T,30,3) pose_aa."""
    T = dof_pos.shape[0]
    assert root_rotvec.shape == (T, 3) and dof_pos.shape == (T, 29)
    pose_aa = np.zeros((T, 30, 3), dtype=np.float32)
    pose_aa[:, 0] = root_rotvec.astype(np.float32)
    # each hinge joint: axis-angle = angle * axis  (axes are unit vectors in MJCF)
    pose_aa[:, 1:] = (dof_pos[:, :, None] * skel.dof_axes[None, :, :]).astype(np.float32)
    return pose_aa


def quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    """(T,4) wxyz -> (T,3) rotvec (scipy uses xyzw internally)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    xyzw = np.concatenate([q[:, 1:4], q[:, 0:1]], axis=-1)
    return sRot.from_quat(xyzw).as_rotvec()


def euler_xyz_extrinsic_deg_to_rotvec(euler_deg: np.ndarray) -> np.ndarray:
    """(T,3) extrinsic XYZ euler [deg] -> (T,3) rotvec.

    scipy's lowercase 'xyz' == extrinsic, applied in x->y->z order, matching the
    BONES-SEED CSV convention (see selected_one_per_type/README.md §2).
    """
    return sRot.from_euler("xyz", np.asarray(euler_deg, dtype=np.float64), degrees=True).as_rotvec()


# ---------------------------------------------------------------------------
# Forward kinematics (numpy mirror of Humanoid_Batch.forward_kinematics_batch)
# ---------------------------------------------------------------------------

def fk_body_positions(
    skel: G1Skeleton,
    pose_aa: np.ndarray,
    trans: np.ndarray,
    body_indices: list[int] | None = None,
) -> np.ndarray:
    """Compute world positions of bodies.

    pose_aa: (T, 30, 3), trans: (T, 3).
    Returns (T, len(body_indices or 30), 3).

    Convention (identical to torch_humanoid_batch.fk_batch):
        R_world[0] = R(pose_aa[:,0]);                P_world[0] = trans
        R_world[i] = R_world[p] @ R_local[i] @ R(pose_aa[:,i])
        P_world[i] = R_world[p] @ offset[i] + P_world[p]
    """
    T = pose_aa.shape[0]
    rot_mats = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_matrix().reshape(T, 30, 3, 3)
    local_rot_mats = sRot.from_quat(
        np.concatenate([skel.local_rotation[:, 1:4], skel.local_rotation[:, 0:1]], axis=-1)
    ).as_matrix()  # (30, 3, 3)

    R_world = np.zeros((T, 30, 3, 3))
    P_world = np.zeros((T, 30, 3))
    R_world[:, 0] = rot_mats[:, 0]
    P_world[:, 0] = trans
    for i in range(1, 30):
        p = skel.parents[i]
        R_world[:, i] = R_world[:, p] @ local_rot_mats[i] @ rot_mats[:, i]
        P_world[:, i] = np.einsum("tij,j->ti", R_world[:, p], skel.local_translation[i]) + P_world[:, p]

    if body_indices is not None:
        return P_world[:, body_indices]
    return P_world


_ANKLE_Z0_CACHE: dict[int, float] = {}


def reference_ankle_height(skel: G1Skeleton) -> float:
    """Ankle-roll link height in the MJCF home configuration.

    The MJCF home pose (all joint angles zero, pelvis at z=0.793) has the feet flat
    on the ground, so this value is the calibrated 'foot on ground' ankle height.
    """
    key = id(skel)
    if key not in _ANKLE_Z0_CACHE:
        pose0 = np.zeros((1, 30, 3))
        trans0 = np.array([[0.0, 0.0, 0.793]])
        feet = [skel.body_index(LEFT_FOOT_BODY), skel.body_index(RIGHT_FOOT_BODY)]
        z0 = fk_body_positions(skel, pose0, trans0, feet)[0, :, 2].min()
        _ANKLE_Z0_CACHE[key] = float(z0)
    return _ANKLE_Z0_CACHE[key]


def ground_align_translation(
    skel: G1Skeleton,
    pose_aa: np.ndarray,
    trans: np.ndarray,
    percentile: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Shift root z so the lowest-foot height matches the calibrated standing height.

    Uses a low percentile (instead of min) of the per-frame lowest ankle height to be
    robust to brief retargeting penetrations.  Returns (aligned_trans, applied_shift).
    """
    feet = [skel.body_index(LEFT_FOOT_BODY), skel.body_index(RIGHT_FOOT_BODY)]
    foot_z = fk_body_positions(skel, pose_aa, trans, feet)[:, :, 2].min(axis=1)  # (T,)
    shift = float(np.percentile(foot_z, percentile)) - reference_ankle_height(skel)
    aligned = trans.copy()
    aligned[:, 2] -= shift
    return aligned, shift


# ---------------------------------------------------------------------------
# Clip slicing
# ---------------------------------------------------------------------------

def slice_clips(
    n_frames: int,
    clip_len: int = 300,
    min_len: int = 160,
) -> list[tuple[int, int]]:
    """Cut [0, n_frames) into non-overlapping (start, end) segments.

    * full ``clip_len`` segments first (head-to-tail, no overlap)
    * a tail segment is kept if >= ``min_len``
    * a sequence shorter than ``clip_len`` is kept whole if >= ``min_len``

    ``min_len=160`` frames @30fps = 5.33 s; at the 50 Hz control rate this gives
    ~267 expert-buffer steps, comfortably above the 251-step minimum required by
    ``rollout_expert_trajectories_length=250`` so every produced clip is usable by
    all training components (discriminator, z_expert AND expert z-tracking rollouts).
    """
    segments = []
    start = 0
    while start + clip_len <= n_frames:
        segments.append((start, start + clip_len))
        start += clip_len
    if n_frames - start >= min_len:
        segments.append((start, n_frames))
    return segments
