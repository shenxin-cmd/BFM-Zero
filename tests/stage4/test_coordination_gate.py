import torch

from humanoidverse.agents.stage4 import CoordinationGate, coordination_gate_raw


def test_coordination_gate_low_for_easy_target():
    gate = coordination_gate_raw(
        ik_residual=torch.tensor([0.0]),
        sigma_min=torch.tensor([0.2]),
        joint_margin=torch.tensor([0.5]),
        position_error_norm=torch.tensor([0.01]),
    )

    assert gate.item() < 0.1


def test_coordination_gate_high_for_bad_reachability_signals():
    gate = coordination_gate_raw(
        ik_residual=torch.tensor([0.2]),
        sigma_min=torch.tensor([0.0]),
        joint_margin=torch.tensor([0.0]),
        position_error_norm=torch.tensor([0.5]),
    )

    assert gate.item() > 0.9


def test_coordination_gate_ema_limits_jump():
    gate = CoordinationGate(num_envs=1, device="cpu", ema_alpha=0.8)
    first = gate.step(torch.tensor([0.0]))
    second = gate.step(torch.tensor([1.0]))

    assert first.item() == 0.0
    assert 0.0 < second.item() < 1.0
