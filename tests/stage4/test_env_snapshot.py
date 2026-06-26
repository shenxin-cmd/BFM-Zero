from types import SimpleNamespace

import torch

from humanoidverse.agents.stage4 import HandTaskCommand, build_stage4_env_snapshot, dls_hand_action_from_snapshot


class _DummyRootPhysxView:
    def __init__(self, jacobians):
        self._jacobians = jacobians

    def get_jacobians(self):
        return self._jacobians


def _make_dummy_env():
    dof_names = [
        *(f"joint_{i}" for i in range(22)),
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    body_names = [*(f"body_{i}" for i in range(29)), "right_wrist_yaw_link"]
    num_envs = 2
    dof_pos = torch.arange(num_envs * 29, dtype=torch.float32).reshape(num_envs, 29) * 0.01
    default_dof_pos = torch.zeros(1, 29)
    default_dof_pos_offset = torch.zeros(num_envs, 29)
    default_dof_pos[:, [22, 23, 24, 25]] = torch.tensor([[0.10, -0.20, 0.05, 0.30]])
    default_dof_pos_offset[:, [22, 23, 24, 25]] = torch.tensor([[0.01, 0.02, -0.03, 0.04], [-0.02, 0.03, 0.01, -0.01]])
    root_states = torch.zeros(num_envs, 13)
    root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    rigid_body_pos = torch.zeros(num_envs, 30, 3)
    rigid_body_pos[:, 29] = torch.tensor([[0.3, -0.2, 1.1], [0.4, -0.1, 1.2]])
    jacobians = torch.zeros(num_envs, 30, 6, 35)
    body_ids = [*range(29), 7]
    dof_ids = [*range(22), 12, 13, 14, 15, 26, 27, 28]
    for local_col, raw_dof_id in enumerate((12, 13, 14, 15)):
        jacobians[:, 7, :3, raw_dof_id + 6] = torch.eye(3, 4)[:, local_col]
    simulator = SimpleNamespace(
        dof_pos=dof_pos,
        dof_ids=dof_ids,
        body_ids=body_ids,
        robot_root_states=root_states,
        _rigid_body_pos=rigid_body_pos,
        _robot=SimpleNamespace(root_physx_view=_DummyRootPhysxView(jacobians)),
    )
    config = SimpleNamespace(
        robot=SimpleNamespace(control=SimpleNamespace(action_scale=0.25)),
        get=lambda key, default=None: {
            "stage4": {
                "hand_control_mode": "task_space_4dof",
                "end_effector_body_name": "right_wrist_yaw_link",
            }
        }.get(key, default),
    )
    return SimpleNamespace(
        config=config,
        simulator=simulator,
        dof_names=dof_names,
        body_names=body_names,
        dof_pos_limits=torch.stack([torch.full((29,), -1.0), torch.full((29,), 1.0)], dim=-1),
        default_dof_pos=default_dof_pos,
        default_dof_pos_offset=default_dof_pos_offset,
    )


def test_build_stage4_env_snapshot_reads_env_tensors_and_heading_jacobian():
    env = _make_dummy_env()

    snapshot = build_stage4_env_snapshot(env)

    assert snapshot.end_effector_body_index == 29
    assert snapshot.indices.active_dof_indices == (22, 23, 24, 25)
    assert snapshot.indices.wrist_dof_indices == (26, 27, 28)
    assert snapshot.active_q.shape == (2, 4)
    assert snapshot.active_lower.shape == (2, 4)
    assert snapshot.active_upper.shape == (2, 4)
    assert snapshot.end_effector_pos_heading.shape == (2, 3)
    assert snapshot.active_position_jacobian_world.shape == (2, 3, 4)
    assert snapshot.active_position_jacobian_heading.shape == (2, 3, 4)
    assert torch.allclose(snapshot.active_position_jacobian_heading, snapshot.active_position_jacobian_world)
    assert snapshot.active_pd_reference_pos.shape == (2, 4)
    assert torch.allclose(
        snapshot.active_pd_reference_pos,
        torch.tensor([[0.11, -0.18, 0.02, 0.34], [0.08, -0.17, 0.06, 0.29]]),
    )
    assert snapshot.action_scale == 0.25


def test_dls_hand_action_from_snapshot_preserves_body_action_and_locks_wrist():
    env = _make_dummy_env()
    snapshot = build_stage4_env_snapshot(env)
    body_action = torch.randn(2, 22, requires_grad=True)
    command = HandTaskCommand(
        target_pos_heading=snapshot.end_effector_pos_heading + torch.tensor([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        target_lin_vel_heading=torch.zeros(2, 3),
        position_mask=torch.ones(2, 1),
        velocity_mask=torch.zeros(2, 1),
        command_id=torch.zeros(2, dtype=torch.long),
        command_done=torch.zeros(2, 1, dtype=torch.bool),
    )

    out = dls_hand_action_from_snapshot(
        body_action=body_action,
        snapshot=snapshot,
        command=command,
        action_dim=29,
        max_joint_delta=0.03,
    )

    body_indices = [idx for idx in range(29) if idx not in snapshot.indices.controlled_action_indices]
    assert out.full_action.shape == (2, 29)
    assert torch.allclose(out.full_action[:, body_indices], body_action)
    assert torch.all(out.full_action[:, list(snapshot.indices.wrist_action_indices)] == 0.0)
    assert out.active_hand_action.shape == (2, 4)
    reconstructed_active_pd_target = out.active_hand_action * snapshot.action_scale + snapshot.active_pd_reference_pos
    assert torch.allclose(reconstructed_active_pd_target, out.active_joint_target)
    out.full_action[:, body_indices].sum().backward()
    assert body_action.grad is not None
    assert torch.all(body_action.grad == 1.0)
