from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from humanoidverse.utils.torch_utils import calc_heading_quat, calc_heading_quat_inv, my_quat_rotate


def _rotate_flat(quat_xyzw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if quat_xyzw.shape[-1] != 4 or vec.shape[-1] != 3:
        raise ValueError(f"Expected quat [...,4] and vec [...,3], got {quat_xyzw.shape} and {vec.shape}")
    if quat_xyzw.shape[:-1] != vec.shape[:-1]:
        raise ValueError(f"Quaternion and vector batch shapes differ: {quat_xyzw.shape} vs {vec.shape}")
    original_shape = vec.shape
    rotated = my_quat_rotate(quat_xyzw.reshape(-1, 4), vec.reshape(-1, 3))
    return rotated.reshape(original_shape)


def world_to_heading_frame(
    points_world: torch.Tensor,
    *,
    root_pos_world: torch.Tensor,
    root_quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Transform world points/vectors to the root heading frame.

    This removes root translation and yaw heading only, matching the local
    heading-coordinate convention used by current motion observations.
    """

    local = points_world - root_pos_world.unsqueeze(-2) if points_world.ndim == root_pos_world.ndim + 1 else points_world - root_pos_world
    heading_inv = calc_heading_quat_inv(root_quat_xyzw, w_last=True)
    if local.ndim == heading_inv.ndim + 1:
        heading_inv = heading_inv.unsqueeze(-2).expand(*local.shape[:-1], 4)
    return _rotate_flat(heading_inv, local)


def heading_to_world_frame(
    points_heading: torch.Tensor,
    *,
    root_pos_world: torch.Tensor,
    root_quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Transform root heading-frame points/vectors back to world coordinates."""

    heading = calc_heading_quat(root_quat_xyzw, w_last=True)
    if points_heading.ndim == heading.ndim + 1:
        heading = heading.unsqueeze(-2).expand(*points_heading.shape[:-1], 4)
    world_delta = _rotate_flat(heading, points_heading)
    return world_delta + root_pos_world.unsqueeze(-2) if points_heading.ndim == root_pos_world.ndim + 1 else world_delta + root_pos_world


def select_active_position_jacobian(
    position_jacobian: torch.Tensor,
    active_dof_indices: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Select the 3x4 active right-arm position Jacobian.

    Args:
        position_jacobian: tensor shaped `[..., 3, num_dof]`.
        active_dof_indices: the four active shoulder/elbow DOF indices.
    """

    if position_jacobian.shape[-2] != 3:
        raise ValueError(f"Expected position_jacobian shape [..., 3, num_dof], got {position_jacobian.shape}")
    if isinstance(active_dof_indices, torch.Tensor):
        active_idx = active_dof_indices.to(device=position_jacobian.device, dtype=torch.long)
    else:
        active_idx = torch.tensor(tuple(active_dof_indices), device=position_jacobian.device, dtype=torch.long)
    if active_idx.numel() != 4:
        raise ValueError(f"Expected exactly 4 active DOF indices, got {active_idx.numel()}")
    selected = position_jacobian.index_select(-1, active_idx)
    if selected.shape[-2:] != (3, 4):
        raise RuntimeError(f"Active position Jacobian must end with [3,4], got {selected.shape}")
    return selected


def finite_difference_position_jacobian(
    fk_fn: Callable[[torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Central finite-difference position Jacobian for `fk_fn(q) -> [..., 3]`."""

    if q.ndim < 1:
        raise ValueError("q must have at least one dimension")
    dof = q.shape[-1]
    base_pos = fk_fn(q)
    if base_pos.shape[-1] != 3:
        raise ValueError(f"fk_fn must return [..., 3], got {base_pos.shape}")

    columns = []
    for idx in range(dof):
        delta = torch.zeros_like(q)
        delta[..., idx] = eps
        pos_plus = fk_fn(q + delta)
        pos_minus = fk_fn(q - delta)
        columns.append(((pos_plus - pos_minus) / (2.0 * eps)).unsqueeze(-1))
    jac = torch.cat(columns, dim=-1)
    if jac.shape != (*base_pos.shape[:-1], 3, dof):
        raise RuntimeError(f"Finite-difference Jacobian has unexpected shape {jac.shape}")
    return jac
