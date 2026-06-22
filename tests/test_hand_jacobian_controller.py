from __future__ import annotations

import unittest

import torch

from humanoidverse.controllers.hand_jacobian_controller import (
    HandJacobianController,
    HandJacobianControllerConfig,
    compute_linear_gate,
    damped_least_squares,
)


RIGHT_ARM = torch.arange(22, 29)


def make_controller(batch: int, **overrides) -> HandJacobianController:
    values = dict(
        enabled=True,
        kp_position=1.0,
        kd_position=0.0,
        damping=0.1,
        gate_mode="none",
        max_task_velocity=10.0,
        max_delta_q=10.0,
        max_delta_action=10.0,
        lowpass_beta=0.0,
        joint_limit_margin=0.0,
    )
    values.update(overrides)
    return HandJacobianController(
        HandJacobianControllerConfig(**values), batch, "cpu", RIGHT_ARM
    )


def inputs(batch: int) -> dict[str, torch.Tensor]:
    jacobian = torch.zeros(batch, 3, 7)
    jacobian[:, :, :3] = torch.eye(3)
    return {
        "action_bfm": torch.zeros(batch, 29),
        "current_wrist_pos_root": torch.zeros(batch, 3),
        "target_wrist_pos_root": torch.zeros(batch, 3),
        "wrist_linear_vel_root": torch.zeros(batch, 3),
        "jacobian_pos_root": jacobian,
        "current_right_arm_q": torch.zeros(batch, 7),
        "default_right_arm_q": torch.zeros(7),
        "lower_joint_limits": -10.0 * torch.ones(7),
        "upper_joint_limits": 10.0 * torch.ones(7),
        "action_scale": torch.ones(7),
        "action_lower": -10.0 * torch.ones(7),
        "action_upper": 10.0 * torch.ones(7),
    }


class HandJacobianControllerTest(unittest.TestCase):
    def test_zero_error_is_zero_correction(self) -> None:
        data = inputs(1)
        action, metrics = make_controller(1).compute(**data)
        torch.testing.assert_close(action, data["action_bfm"])
        self.assertEqual(float(metrics["delta_q_raw_norm"][0]), 0.0)

    def test_dls_identity_direction(self) -> None:
        jacobian = torch.zeros(1, 3, 7)
        jacobian[:, :, :3] = torch.eye(3)
        velocity = torch.tensor([[1.0, -2.0, 3.0]])
        result = damped_least_squares(jacobian, velocity, 0.1)
        expected = torch.tensor([[1.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0]]) / 1.01
        torch.testing.assert_close(result, expected)

    def test_singular_jacobian_is_finite_and_bounded(self) -> None:
        data = inputs(4)
        data["jacobian_pos_root"].zero_()
        data["target_wrist_pos_root"][:, 0] = 1.0
        action, _ = make_controller(4, max_delta_q=0.02).compute(**data)
        self.assertTrue(bool(torch.isfinite(action).all()))
        self.assertLessEqual(float(action[:, 22:29].abs().max()), 0.02)

    def test_batched_shapes(self) -> None:
        for batch in (1, 32, 1024):
            action, metrics = make_controller(batch).compute(**inputs(batch))
            self.assertEqual(action.shape, (batch, 29))
            self.assertEqual(metrics["wrist_error"].shape, (batch,))

    def test_action_unit_conversion(self) -> None:
        data = inputs(1)
        data["target_wrist_pos_root"][0, 0] = 0.1
        data["action_scale"] = 0.5 * torch.ones(7)
        action, _ = make_controller(1).compute(**data)
        expected_delta_q = 0.1 / 1.01
        self.assertAlmostEqual(float(action[0, 22] * 0.5), expected_delta_q, places=6)

    def test_linear_gate_boundaries(self) -> None:
        errors = torch.tensor([[0.0], [0.02], [0.06], [0.10], [0.20]])
        gate = compute_linear_gate(errors, near=0.02, far=0.10)
        torch.testing.assert_close(gate.squeeze(-1), torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]))

    def test_partial_reset_only_clears_selected_filter_state(self) -> None:
        controller = make_controller(3)
        controller.prev_delta_q[:] = torch.tensor([[1.0] * 7, [2.0] * 7, [3.0] * 7])
        controller.reset(torch.tensor([1]))
        torch.testing.assert_close(controller.prev_delta_q[0], torch.ones(7))
        torch.testing.assert_close(controller.prev_delta_q[1], torch.zeros(7))
        torch.testing.assert_close(controller.prev_delta_q[2], 3.0 * torch.ones(7))

    def test_nan_jacobian_disables_only_invalid_environment(self) -> None:
        data = inputs(2)
        data["target_wrist_pos_root"][:, 0] = 0.1
        data["jacobian_pos_root"][0, 0, 0] = float("nan")
        action, metrics = make_controller(2).compute(**data)
        torch.testing.assert_close(action[0], data["action_bfm"][0])
        self.assertGreater(float(action[1, 22]), 0.0)
        self.assertTrue(bool(metrics["invalid_input"][0]))
        self.assertFalse(bool(metrics["invalid_input"][1]))

    def test_only_right_arm_changes(self) -> None:
        data = inputs(2)
        data["action_bfm"] = torch.randn(2, 29) * 0.1
        data["target_wrist_pos_root"][:, 1] = 0.05
        action, _ = make_controller(2).compute(**data)
        torch.testing.assert_close(action[:, :22], data["action_bfm"][:, :22])

    def test_joint_limit_projection(self) -> None:
        data = inputs(1)
        data["action_bfm"][0, 22] = 0.09
        data["target_wrist_pos_root"][0, 0] = 1.0
        data["lower_joint_limits"] = -0.1 * torch.ones(7)
        data["upper_joint_limits"] = 0.1 * torch.ones(7)
        action, metrics = make_controller(1, joint_limit_margin=0.01).compute(**data)
        self.assertLessEqual(float(action[0, 22]), 0.09 + 1e-7)
        self.assertTrue(bool(metrics["joint_limit_projected"][0]))


if __name__ == "__main__":
    unittest.main()
