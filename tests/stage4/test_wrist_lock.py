import torch

from humanoidverse.agents.stage4 import clamp_wrist_pd_target, zero_wrist_actions


def test_zero_wrist_actions_hard_locks_only_wrist_indices():
    wrist_indices = (26, 27, 28)
    action = torch.randn(3, 29)
    locked = zero_wrist_actions(action, wrist_indices)

    assert torch.all(locked[:, list(wrist_indices)] == 0.0)
    non_wrist = [idx for idx in range(29) if idx not in wrist_indices]
    assert torch.allclose(locked[:, non_wrist], action[:, non_wrist])


def test_clamp_wrist_pd_target_sets_absolute_zero_not_default_offset():
    wrist_indices = (26, 27, 28)
    default_offset_target = torch.full((2, 29), 0.25)
    clamped = clamp_wrist_pd_target(default_offset_target, wrist_indices, wrist_absolute_target=0.0)

    assert torch.all(clamped[:, list(wrist_indices)] == 0.0)
    non_wrist = [idx for idx in range(29) if idx not in wrist_indices]
    assert torch.all(clamped[:, non_wrist] == 0.25)
