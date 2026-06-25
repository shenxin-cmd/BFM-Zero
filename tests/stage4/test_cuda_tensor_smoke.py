import pytest
import torch

from humanoidverse.agents.stage4 import adaptive_damping, damped_least_squares, zero_wrist_actions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_stage4_core_tensor_ops_run_on_cuda():
    device = torch.device("cuda")
    jac = torch.zeros(4, 3, 4, device=device)
    jac[:, :3, :3] = torch.eye(3, device=device)
    err = torch.randn(4, 3, device=device)

    damping, sigma_min = adaptive_damping(jac)
    dq = damped_least_squares(jac, err, damping, max_joint_delta=0.05)

    action = torch.randn(4, 29, device=device)
    locked = zero_wrist_actions(action, (26, 27, 28))

    assert damping.device.type == "cuda"
    assert sigma_min.device.type == "cuda"
    assert dq.device.type == "cuda"
    assert torch.isfinite(dq).all()
    assert torch.all(locked[:, [26, 27, 28]] == 0.0)
