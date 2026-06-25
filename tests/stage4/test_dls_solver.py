import torch

from humanoidverse.agents.stage4 import damped_least_squares


def test_dls_solver_returns_batched_4dof_delta_without_nan():
    jac = torch.zeros(2, 3, 4)
    jac[:, :3, :3] = torch.eye(3)
    err = torch.tensor([[1.0, -2.0, 0.5], [0.2, 0.3, -0.4]])

    dq = damped_least_squares(jac, err, damping=0.05, gain=0.5, max_joint_delta=0.25)

    assert dq.shape == (2, 4)
    assert torch.isfinite(dq).all()
    assert dq.abs().max() <= 0.25


def test_dls_solver_handles_rank_deficient_and_zero_jacobian():
    jac = torch.zeros(2, 3, 4)
    jac[0, 0, 0] = 1.0
    err = torch.ones(2, 3)

    dq = damped_least_squares(jac, err, damping=torch.tensor([0.1, 0.2]), gain=1.0, max_joint_delta=0.05)

    assert torch.isfinite(dq).all()
    assert dq.shape == (2, 4)
    assert torch.all(dq[1] == 0.0)
