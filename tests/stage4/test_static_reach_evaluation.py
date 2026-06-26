from types import SimpleNamespace

import torch

from humanoidverse.agents.stage4 import (
    DLSParameterSet,
    Stage4CommandSamplerConfig,
    evaluate_static_reach_dls,
    evaluate_static_reach_dls_parameter_scan,
)


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
    assert result.final_ik_residual.shape == (2,)
    assert len(result.target_categories) == 2
    assert len(result.target_diagnostics) == 2
    assert torch.all(torch.isfinite(result.steady_state_error))
    assert torch.all(result.target_offsets_heading == offsets)
    assert "reachable/steady_error_mean" in result.grouped_metrics
    assert "boundary/success_2cm" in result.grouped_metrics
    assert "coordination_required/final_ik_residual" in result.grouped_metrics
    assert "max_action_jump" in result.global_metrics
    assert "active_joint_velocity_max" in result.global_metrics
    assert "wrist_q_abs_max" in result.global_metrics
    assert result.global_metrics["active_joint_velocity_max"] <= 1.0 + 1e-5
    assert result.global_metrics["wrist_q_abs_max"] == 0.0
    assert result.global_metrics["wrist_dq_abs_max"] == 0.0
    diagnostic = result.target_diagnostics[0]
    assert diagnostic["target_category"] in ("reachable", "boundary", "coordination_required")
    assert diagnostic["position_error_curve"].shape == (3, 2)
    assert diagnostic["active_joint_q_curve"].shape == (3, 2, 4)
    assert diagnostic["active_joint_target_curve"].shape == (3, 2, 4)
    assert diagnostic["action_curve"].shape == (3, 2, 29)
    assert diagnostic["sigma_min_curve"].shape == (3, 2)
    assert diagnostic["joint_margin_curve"].shape == (3, 2)
    assert diagnostic["damping_curve"].shape == (3, 2)
    assert diagnostic["nullspace_action_curve"].shape == (3, 2, 4)
    assert diagnostic["failure_mode"] in (
        "success",
        "horizon_too_short",
        "plateau_or_model_mismatch",
        "joint_limit_pressure",
        "low_manipulability",
        "limiter_velocity_saturated",
        "limiter_acceleration_saturated",
        "target_unreachable_by_arm_only_presolve",
    )


def test_evaluate_static_reach_dls_uses_configurable_classification_thresholds():
    env = _DummyEnv()
    offsets = torch.tensor([[0.20, 0.0, 0.0]])
    sampler_config = Stage4CommandSamplerConfig(
        reachable_threshold=0.001,
        coordination_required_threshold=0.01,
        presolve_steps=1,
        presolve_max_joint_delta=0.01,
    )

    result = evaluate_static_reach_dls(
        env,
        target_offsets_heading=offsets,
        steps_per_target=2,
        max_joint_velocity=1.0,
        sampler_config=sampler_config,
    )

    assert result.target_categories == ("coordination_required",)
    assert result.sampler_config.coordination_required_threshold == 0.01


def test_static_reach_parameter_scan_runs_fresh_env_per_parameter_set():
    make_count = 0

    def make_env():
        nonlocal make_count
        make_count += 1
        return _DummyEnv()

    results = evaluate_static_reach_dls_parameter_scan(
        make_env,
        target_offsets_heading=torch.tensor([[0.02, 0.0, 0.0]]),
        parameter_sets=(
            DLSParameterSet(label="slow", max_joint_velocity=0.5, steps_per_target=2),
            DLSParameterSet(label="fast", max_joint_velocity=1.0, steps_per_target=2),
        ),
    )

    assert make_count == 2
    assert set(results) == {"slow", "fast"}
    assert results["slow"].target_diagnostics[0]["position_error_curve"].shape[0] == 2
