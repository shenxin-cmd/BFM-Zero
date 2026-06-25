import torch
from torch import nn


def _has_nonzero_grad(module: nn.Module) -> bool:
    return any(param.grad is not None and torch.any(param.grad != 0) for param in module.parameters())


def test_hand_residual_loss_does_not_update_frozen_body_base():
    body_base = nn.Linear(4, 4)
    hand_residual = nn.Linear(4, 4)
    for param in body_base.parameters():
        param.requires_grad_(False)

    obs = torch.randn(8, 4)
    base_action = body_base(obs)
    hand_delta = hand_residual(obs)
    loss = (base_action.detach() + hand_delta).square().mean()
    loss.backward()

    assert not _has_nonzero_grad(body_base)
    assert _has_nonzero_grad(hand_residual)


def test_coordination_loss_does_not_update_frozen_body_base():
    body_base = nn.Linear(4, 4)
    coordination_head = nn.Linear(4, 4)
    for param in body_base.parameters():
        param.requires_grad_(False)

    obs = torch.randn(8, 4)
    base_action = body_base(obs)
    coord_delta = coordination_head(torch.cat([obs[:, :2], base_action.detach()[:, :2]], dim=-1))
    loss = coord_delta.square().mean()
    loss.backward()

    assert not _has_nonzero_grad(body_base)
    assert _has_nonzero_grad(coordination_head)
