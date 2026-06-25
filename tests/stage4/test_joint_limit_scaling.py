import torch

from humanoidverse.agents.stage4 import apply_joint_limit_scaling, joint_margin_scale


def test_joint_margin_scale_is_small_near_limits():
    q = torch.tensor([[0.98, 0.0, -0.98]])
    lower = torch.full_like(q, -1.0)
    upper = torch.full_like(q, 1.0)
    scale = joint_margin_scale(q, lower, upper, margin=0.2)

    assert scale[0, 0] < 0.2
    assert scale[0, 1] == 1.0
    assert scale[0, 2] < 0.2


def test_apply_joint_limit_scaling_blocks_motion_deeper_into_limit():
    q = torch.tensor([[0.98, -0.98, 0.0]])
    dq = torch.tensor([[0.1, -0.1, 0.1]])
    lower = torch.full_like(q, -1.0)
    upper = torch.full_like(q, 1.0)
    scaled = apply_joint_limit_scaling(q, dq, lower, upper, margin=0.2)

    assert scaled[0, 0] < 0.02
    assert scaled[0, 1] > -0.02
    assert scaled[0, 2] == dq[0, 2]
