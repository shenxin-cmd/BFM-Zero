import torch

from humanoidverse.agents.stage4 import JointCommandLimiter


def test_joint_command_limiter_enforces_velocity_limit():
    limiter = JointCommandLimiter(
        action_dim=4,
        num_envs=2,
        device="cpu",
        max_joint_velocity=1.0,
    )
    limiter.step(torch.zeros(2, 4), dt=0.1)
    out = limiter.step(torch.full((2, 4), 10.0), dt=0.1)

    assert torch.allclose(out, torch.full((2, 4), 0.1))


def test_joint_command_limiter_enforces_acceleration_limit():
    limiter = JointCommandLimiter(
        action_dim=1,
        num_envs=1,
        device="cpu",
        max_joint_acceleration=2.0,
    )
    limiter.step(torch.zeros(1, 1), dt=0.5)
    out = limiter.step(torch.ones(1, 1) * 10.0, dt=0.5)

    assert torch.allclose(out, torch.tensor([[0.5]]))


def test_joint_command_limiter_reset_is_per_environment():
    limiter = JointCommandLimiter(action_dim=2, num_envs=2, device="cpu", max_joint_velocity=1.0)
    limiter.step(torch.zeros(2, 2), dt=0.1)
    limiter.step(torch.ones(2, 2), dt=0.1)
    limiter.reset([1])
    out = limiter.step(torch.tensor([[1.0, 1.0], [5.0, 5.0]]), dt=0.1)

    assert torch.allclose(out[0], torch.tensor([0.2, 0.2]))
    assert torch.allclose(out[1], torch.tensor([5.0, 5.0]))
