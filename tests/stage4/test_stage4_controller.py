import torch

from humanoidverse.agents.stage4 import (
    DLSHandController,
    HandTaskCommand,
    active_target_to_action,
    body_action_indices_for_stage4,
    resolve_right_arm_joint_indices,
)

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


def test_dls_hand_controller_outputs_full_action_with_locked_wrist():
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
    body_indices = tuple(idx for idx in range(29) if idx not in indices.controlled_action_indices)
    controller = DLSHandController(indices=indices, action_dim=29, body_indices=body_indices, max_joint_delta=0.05)
    command = HandTaskCommand(
        target_pos_root=torch.tensor([[0.1, 0.0, 0.0]]),
        target_lin_vel_root=torch.zeros(1, 3),
        position_mask=torch.ones(1, 1),
        velocity_mask=torch.zeros(1, 1),
        command_id=torch.zeros(1, dtype=torch.long),
        command_done=torch.zeros(1, 1, dtype=torch.bool),
    )
    jac = torch.zeros(1, 3, 4)
    jac[:, :3, :3] = torch.eye(3)

    out = controller.step(
        body_action=torch.zeros(1, len(body_indices)),
        active_q=torch.zeros(1, 4),
        wrist_pos_root=torch.zeros(1, 3),
        command=command,
        active_position_jacobian=jac,
        active_lower=torch.full((1, 4), -1.0),
        active_upper=torch.full((1, 4), 1.0),
        active_pd_reference_pos=torch.zeros(1, 4),
        action_scale=1.0,
    )

    assert out.full_action.shape == (1, 29)
    assert torch.all(out.full_action[:, list(indices.wrist_action_indices)] == 0.0)
    assert out.active_joint_delta.shape == (1, 4)
    assert out.active_joint_delta[0, 0] > 0.0


def test_body_action_indices_for_stage4_excludes_active_hand_and_locked_wrist():
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

    body_indices = body_action_indices_for_stage4(action_dim=29, indices=indices)

    assert len(body_indices) == 22
    assert not set(body_indices) & set(indices.controlled_action_indices)
    assert set(body_indices) | set(indices.controlled_action_indices) == set(range(29))


def test_active_target_to_action_round_trips_env_pd_semantics_with_offset():
    q_current = torch.tensor([[0.2, -0.1, 0.05, 0.4], [0.0, 0.3, -0.2, 0.1]])
    dq = torch.tensor([[0.03, -0.02, 0.01, 0.0], [-0.01, 0.02, 0.0, 0.03]])
    q_cmd = q_current + dq
    default_dof_pos = torch.tensor([[0.1, -0.2, 0.0, 0.25]])
    default_dof_pos_offset = torch.tensor([[0.02, 0.03, -0.01, 0.04], [-0.03, 0.01, 0.02, -0.02]])
    action_scale = 0.25
    active_pd_reference_pos = default_dof_pos + default_dof_pos_offset

    action = active_target_to_action(
        q_cmd,
        active_pd_reference_pos=active_pd_reference_pos,
        action_scale=action_scale,
    )
    reconstructed_pd_target = action * action_scale + default_dof_pos + default_dof_pos_offset

    assert torch.allclose(reconstructed_pd_target, q_cmd)
