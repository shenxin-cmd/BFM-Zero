import pytest
import torch

from humanoidverse.agents.stage4 import assemble_full_action


def test_assemble_full_action_places_body_active_hand_and_zero_wrist_values():
    body_indices = tuple(range(22))
    active_indices = (22, 23, 24, 25)
    wrist_indices = (26, 27, 28)
    body_action = torch.full((2, len(body_indices)), 1.5)
    active_hand_action = torch.tensor([[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])

    full = assemble_full_action(
        body_action,
        active_hand_action,
        action_dim=29,
        body_indices=body_indices,
        active_hand_indices=active_indices,
        wrist_indices=wrist_indices,
    )

    assert full.shape == (2, 29)
    assert torch.all(full[:, list(body_indices)] == 1.5)
    assert torch.all(full[:, list(active_indices)] == active_hand_action)
    assert torch.all(full[:, list(wrist_indices)] == 0.0)


def test_assemble_full_action_rejects_overlapping_indices():
    with pytest.raises(ValueError, match="disjoint"):
        assemble_full_action(
            torch.zeros(1, 2),
            torch.zeros(1, 1),
            action_dim=4,
            body_indices=(0, 1),
            active_hand_indices=(1,),
            wrist_indices=(3,),
        )
