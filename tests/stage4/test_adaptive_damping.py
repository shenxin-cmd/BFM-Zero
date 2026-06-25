import torch

from humanoidverse.agents.stage4 import adaptive_damping


def test_adaptive_damping_increases_as_sigma_min_decreases():
    strong = torch.zeros(3, 4)
    strong[:3, :3] = torch.eye(3)
    weak = strong.clone()
    weak[2, 2] = 0.01
    jac = torch.stack([strong, weak])

    damping, sigma_min = adaptive_damping(
        jac,
        damping_min=0.02,
        damping_max=0.20,
        singular_value_threshold=0.08,
    )

    assert sigma_min[0] > sigma_min[1]
    assert damping[1] >= damping[0]
    assert torch.all(damping >= 0.02)
    assert torch.all(damping <= 0.20)
