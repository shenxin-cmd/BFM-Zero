import pytest

from humanoidverse.agents.stage4 import resolve_right_arm_joint_indices


G1_29DOF_NAMES = (
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


def test_resolve_right_arm_joint_indices_from_names():
    indices = resolve_right_arm_joint_indices(
        dof_names=G1_29DOF_NAMES,
        active_joint_names=(
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ),
        wrist_joint_names=(
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ),
    )

    assert indices.active_action_indices == (22, 23, 24, 25)
    assert indices.wrist_action_indices == (26, 27, 28)
    assert indices.active_dof_indices == (22, 23, 24, 25)
    assert indices.wrist_dof_indices == (26, 27, 28)
    assert set(indices.active_action_indices).isdisjoint(indices.wrist_action_indices)


def test_resolve_right_arm_joint_indices_fails_on_missing_names():
    with pytest.raises(ValueError, match="Missing active DOF"):
        resolve_right_arm_joint_indices(
            dof_names=G1_29DOF_NAMES,
            active_joint_names=(
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "not_a_joint",
            ),
            wrist_joint_names=(
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ),
        )
