from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class RightArmJointIndices:
    active_action_indices: tuple[int, int, int, int]
    wrist_action_indices: tuple[int, int, int]
    active_dof_indices: tuple[int, int, int, int]
    wrist_dof_indices: tuple[int, int, int]
    active_joint_names: tuple[str, str, str, str]
    wrist_joint_names: tuple[str, str, str]

    @property
    def controlled_action_indices(self) -> tuple[int, ...]:
        return self.active_action_indices + self.wrist_action_indices


def _index_names(names: Sequence[str], requested: Sequence[str], *, kind: str) -> tuple[int, ...]:
    index_by_name = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in requested if name not in index_by_name]
    if missing:
        raise ValueError(f"Missing {kind} names: {missing}. Available names: {list(names)}")
    return tuple(index_by_name[name] for name in requested)


def _assert_unique(indices: Sequence[int], *, label: str) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} contains duplicate indices: {tuple(indices)}")


def resolve_right_arm_joint_indices(
    *,
    dof_names: Sequence[str],
    active_joint_names: Sequence[str],
    wrist_joint_names: Sequence[str],
    action_names: Sequence[str] | None = None,
) -> RightArmJointIndices:
    """Resolve Stage 4 right-arm indices by joint name.

    The current environments use the same 29D ordering for actions and DOFs. The
    optional `action_names` argument keeps this function explicit for future envs
    where action order may differ from DOF order.
    """

    if len(active_joint_names) != 4:
        raise ValueError(f"Expected exactly 4 active right-arm joints, got {len(active_joint_names)}")
    if len(wrist_joint_names) != 3:
        raise ValueError(f"Expected exactly 3 locked wrist joints, got {len(wrist_joint_names)}")

    action_names = dof_names if action_names is None else action_names
    active_dof = _index_names(dof_names, active_joint_names, kind="active DOF")
    wrist_dof = _index_names(dof_names, wrist_joint_names, kind="wrist DOF")
    active_action = _index_names(action_names, active_joint_names, kind="active action")
    wrist_action = _index_names(action_names, wrist_joint_names, kind="wrist action")

    _assert_unique(active_dof, label="active_dof_indices")
    _assert_unique(wrist_dof, label="wrist_dof_indices")
    _assert_unique(active_action, label="active_action_indices")
    _assert_unique(wrist_action, label="wrist_action_indices")
    if set(active_dof) & set(wrist_dof):
        raise ValueError("Active DOF indices overlap locked wrist DOF indices")
    if set(active_action) & set(wrist_action):
        raise ValueError("Active action indices overlap locked wrist action indices")

    return RightArmJointIndices(
        active_action_indices=active_action,  # type: ignore[arg-type]
        wrist_action_indices=wrist_action,  # type: ignore[arg-type]
        active_dof_indices=active_dof,  # type: ignore[arg-type]
        wrist_dof_indices=wrist_dof,  # type: ignore[arg-type]
        active_joint_names=tuple(active_joint_names),  # type: ignore[arg-type]
        wrist_joint_names=tuple(wrist_joint_names),  # type: ignore[arg-type]
    )


def _as_index_tensor(indices: Sequence[int] | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        return indices.to(device=device, dtype=torch.long)
    return torch.tensor(tuple(indices), dtype=torch.long, device=device)


def zero_wrist_actions(action: torch.Tensor, wrist_indices: Sequence[int] | torch.Tensor) -> torch.Tensor:
    wrist_idx = _as_index_tensor(wrist_indices, device=action.device)
    locked = action.clone()
    locked[..., wrist_idx] = 0.0
    if not torch.all(locked[..., wrist_idx] == 0.0):
        raise RuntimeError("Failed to hard-lock wrist action values to zero")
    return locked


def clamp_wrist_pd_target(
    joint_target: torch.Tensor,
    wrist_indices: Sequence[int] | torch.Tensor,
    *,
    wrist_absolute_target: float = 0.0,
) -> torch.Tensor:
    wrist_idx = _as_index_tensor(wrist_indices, device=joint_target.device)
    clamped = joint_target.clone()
    clamped[..., wrist_idx] = wrist_absolute_target
    expected = torch.full_like(clamped[..., wrist_idx], wrist_absolute_target)
    if not torch.allclose(clamped[..., wrist_idx], expected):
        raise RuntimeError("Failed to hard-lock wrist PD targets")
    return clamped


def assemble_full_action(
    body_action: torch.Tensor,
    active_hand_action: torch.Tensor,
    *,
    action_dim: int,
    body_indices: Sequence[int] | torch.Tensor,
    active_hand_indices: Sequence[int] | torch.Tensor,
    wrist_indices: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    body_idx = _as_index_tensor(body_indices, device=body_action.device)
    active_idx = _as_index_tensor(active_hand_indices, device=body_action.device)
    wrist_idx = _as_index_tensor(wrist_indices, device=body_action.device)

    if body_action.shape[:-1] != active_hand_action.shape[:-1]:
        raise ValueError(f"Batch shapes differ: body={body_action.shape}, hand={active_hand_action.shape}")
    if body_action.shape[-1] != body_idx.numel():
        raise ValueError(f"body_action dim {body_action.shape[-1]} does not match body_indices {body_idx.numel()}")
    if active_hand_action.shape[-1] != active_idx.numel():
        raise ValueError(
            f"active_hand_action dim {active_hand_action.shape[-1]} does not match active_hand_indices {active_idx.numel()}"
        )

    all_indices = torch.cat([body_idx, active_idx, wrist_idx], dim=0)
    if all_indices.unique().numel() != all_indices.numel():
        raise ValueError("Action assembly indices must be disjoint")
    if int(all_indices.min().item()) < 0 or int(all_indices.max().item()) >= action_dim:
        raise ValueError(f"Action assembly indices out of range for action_dim={action_dim}")

    full_action = torch.zeros(
        (*body_action.shape[:-1], action_dim),
        device=body_action.device,
        dtype=body_action.dtype,
    )
    full_action[..., body_idx] = body_action
    full_action[..., active_idx] = active_hand_action
    full_action[..., wrist_idx] = 0.0
    if not torch.all(full_action[..., wrist_idx] == 0.0):
        raise RuntimeError("Failed to hard-lock wrist values during action assembly")
    return full_action
