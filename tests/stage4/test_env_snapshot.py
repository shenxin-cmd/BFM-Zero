from types import SimpleNamespace

import torch

from humanoidverse.agents.stage4 import build_stage4_env_snapshot


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
    root_states = torch.zeros(num_envs, 13)
    root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    rigid_body_pos = torch.zeros(num_envs, 30, 3)
    rigid_body_pos[:, 29] = torch.tensor([[0.3, -0.2, 1.1], [0.4, -0.1, 1.2]])
    jacobians = torch.zeros(num_envs, 30, 6, 35)
    for local_col, jac_col in enumerate((28, 29, 30, 31)):
        jacobians[:, 29, :3, jac_col] = torch.eye(3, 4)[:, local_col]
    simulator = SimpleNamespace(
        dof_pos=dof_pos,
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
        default_dof_pos=torch.zeros(1, 29),
    )


def test_build_stage4_env_snapshot_reads_env_tensors_and_heading_jacobian():
    env = _make_dummy_env()

    snapshot = build_stage4_env_snapshot(env)

    assert snapshot.end_effector_body_index == 29
    assert snapshot.indices.active_dof_indices == (22, 23, 24, 25)
    assert snapshot.indices.wrist_dof_indices == (26, 27, 28)
    assert snapshot.active_q.shape == (2, 4)
    assert snapshot.end_effector_pos_heading.shape == (2, 3)
    assert snapshot.active_position_jacobian_world.shape == (2, 3, 4)
    assert snapshot.active_position_jacobian_heading.shape == (2, 3, 4)
    assert torch.allclose(snapshot.active_position_jacobian_heading, snapshot.active_position_jacobian_world)
    assert snapshot.default_active_joint_pos.shape == (1, 4)
    assert snapshot.action_scale == 0.25
