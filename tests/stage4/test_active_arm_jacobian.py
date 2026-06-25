import pytest
import torch

from humanoidverse.agents.stage4 import (
    finite_difference_position_jacobian,
    resolve_body_index,
    select_active_position_jacobian,
    select_isaacsim_active_position_jacobian,
)


def _toy_fk(q: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.sin(q[..., 0]) + q[..., 1].square(),
            q[..., 2] * q[..., 3],
            q[..., 0] - q[..., 3],
        ],
        dim=-1,
    )


def test_select_active_position_jacobian_shape_is_3_by_4():
    full = torch.randn(5, 3, 29)
    active = select_active_position_jacobian(full, (22, 23, 24, 25))

    assert active.shape == (5, 3, 4)
    assert torch.allclose(active, full[..., [22, 23, 24, 25]])


def test_finite_difference_position_jacobian_matches_toy_analytic_jacobian():
    q = torch.tensor([[0.2, -0.3, 0.4, 0.5], [-0.6, 0.7, -0.8, 0.9]], dtype=torch.float64)
    jac = finite_difference_position_jacobian(_toy_fk, q, eps=1e-6)
    expected = torch.zeros(q.shape[0], 3, 4, dtype=torch.float64)
    expected[:, 0, 0] = torch.cos(q[:, 0])
    expected[:, 0, 1] = 2.0 * q[:, 1]
    expected[:, 1, 2] = q[:, 3]
    expected[:, 1, 3] = q[:, 2]
    expected[:, 2, 0] = 1.0
    expected[:, 2, 3] = -1.0

    assert torch.allclose(jac, expected, atol=1e-6)


def test_resolve_body_index_uses_name_not_position_guess():
    body_names = ("pelvis", "right_elbow_link", "right_wrist_yaw_link")

    assert resolve_body_index(body_names, "right_wrist_yaw_link") == 2

    with pytest.raises(ValueError, match="right_hand_link"):
        resolve_body_index(body_names, "right_hand_link")


def test_select_isaacsim_active_position_jacobian_handles_actuated_columns():
    jac = torch.randn(2, 30, 6, 29)
    selected = select_isaacsim_active_position_jacobian(
        jac,
        body_index=29,
        active_dof_indices=(22, 23, 24, 25),
        num_dofs=29,
    )

    assert selected.shape == (2, 3, 4)
    assert torch.allclose(selected, jac[:, 29, :3, [22, 23, 24, 25]])


def test_select_isaacsim_active_position_jacobian_handles_floating_base_columns():
    jac = torch.randn(2, 30, 6, 35)
    selected = select_isaacsim_active_position_jacobian(
        jac,
        body_index=29,
        active_dof_indices=(22, 23, 24, 25),
        num_dofs=29,
    )

    assert selected.shape == (2, 3, 4)
    assert torch.allclose(selected, jac[:, 29, :3, [28, 29, 30, 31]])


def test_select_isaacsim_active_position_jacobian_rejects_unknown_column_count():
    jac = torch.randn(2, 30, 6, 34)

    with pytest.raises(ValueError, match="Unexpected IsaacSim Jacobian column count"):
        select_isaacsim_active_position_jacobian(
            jac,
            body_index=29,
            active_dof_indices=(22, 23, 24, 25),
            num_dofs=29,
        )
