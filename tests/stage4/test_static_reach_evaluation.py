from types import SimpleNamespace

import torch

from humanoidverse.agents.stage4 import evaluate_static_reach_dls


class _DummyRootPhysxView:
    def __init__(self, jacobians):
        self._jacobians = jacobians

    def get_jacobians(self):
        return self._jacobians


class _DummyEnv:
    single_action_space = SimpleNamespace(shape=(29,))

    def __init__(self):
        self.base = SimpleNamespace()
        self.base.num_envs = 2
        self.base.num_dof = 29
        self.base.device = "cpu"
        self.base.dt = 0.02
        self.base.dof_names = [
            *(f"joint_{i}" for i in range(22)),
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self.base.body_names = [*(f"body_{i}" for i in range(29)), "right_wrist_yaw_link"]
        self.base.default_dof_pos = torch.zeros(1, 29)
        self.base.default_dof_pos_offset = torch.zeros(2, 29)
        self.base.dof_pos_limits = torch.stack([torch.full((29,), -1.0), torch.full((29,), 1.0)], dim=-1)
        jac = torch.zeros(2, 30, 6, 35)
        for local_col, jac_col in enumerate((28, 29, 30, 31)):
            jac[:, 29, :3, jac_col] = torch.eye(3, 4)[:, local_col]
        self.base.simulator = SimpleNamespace(
            dof_pos=torch.zeros(2, 29),
            dof_vel=torch.zeros(2, 29),
            robot_root_states=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 6] * 2),
            _rigid_body_pos=torch.zeros(2, 30, 3),
            _robot=SimpleNamespace(root_physx_view=_DummyRootPhysxView(jac)),
        )
        self.base.config = SimpleNamespace(
            robot=SimpleNamespace(control=SimpleNamespace(action_scale=1.0)),
            get=lambda key, default=None: {
                "stage4": {
                    "hand_control_mode": "task_space_4dof",
                    "end_effector_body_name": "right_wrist_yaw_link",
                }
            }.get(key, default),
        )

    @property
    def unwrapped(self):
        return self.base

    def step(self, action):
        self.base.simulator.dof_pos[:, 22:26] += action[:, 22:26].clamp(-0.01, 0.01)
        self.base.simulator._rigid_body_pos[:, 29, :3] = self.base.simulator.dof_pos[:, 22:25]
        reward = torch.ones(self.base.num_envs)
        done = torch.zeros(self.base.num_envs, dtype=torch.bool)
        return {}, reward, done, done, {}


def test_evaluate_static_reach_dls_returns_per_target_metrics():
    env = _DummyEnv()
    offsets = torch.tensor([[0.02, 0.0, 0.0], [0.0, 0.01, 0.0]])

    result = evaluate_static_reach_dls(env, target_offsets_heading=offsets, steps_per_target=3, max_joint_velocity=1.0)

    assert result.steady_state_error.shape == (2,)
    assert result.max_action_jump.shape == (2,)
    assert result.min_sigma_min.shape == (2,)
    assert result.min_joint_margin.shape == (2,)
    assert torch.all(torch.isfinite(result.steady_state_error))
    assert torch.all(result.target_offsets_heading == offsets)
