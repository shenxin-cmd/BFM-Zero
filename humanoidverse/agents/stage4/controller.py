from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .actions import RightArmJointIndices, assemble_full_action
from .control import (
    JointCommandLimiter,
    adaptive_damping,
    apply_joint_limit_scaling,
    comfortable_posture_nullspace_delta,
    damped_least_squares,
    joint_margin_scale,
)
from .env_adapter import Stage4EnvSnapshot


@dataclass(frozen=True)
class HandTaskCommand:
    target_pos_heading: torch.Tensor
    target_lin_vel_heading: torch.Tensor
    position_mask: torch.Tensor
    velocity_mask: torch.Tensor
    command_id: torch.Tensor
    command_done: torch.Tensor


@dataclass(frozen=True)
class DLSHandControllerOutput:
    full_action: torch.Tensor
    active_hand_action: torch.Tensor
    active_joint_delta: torch.Tensor
    active_joint_target: torch.Tensor
    primary_joint_delta: torch.Tensor
    nullspace_joint_delta: torch.Tensor
    position_error_heading: torch.Tensor
    primary_task_error_before_nullspace: torch.Tensor
    primary_task_error_after_nullspace: torch.Tensor
    nullspace_action_norm: torch.Tensor
    j_times_nullspace_norm: torch.Tensor
    damping: torch.Tensor
    sigma_min: torch.Tensor
    joint_margin: torch.Tensor


def body_action_indices_for_stage4(*, action_dim: int, indices: RightArmJointIndices) -> tuple[int, ...]:
    controlled = set(indices.controlled_action_indices)
    body_indices = tuple(idx for idx in range(action_dim) if idx not in controlled)
    if len(body_indices) + len(controlled) != action_dim:
        raise ValueError("Stage 4 body/action indices do not cover action_dim exactly once")
    return body_indices


def active_target_to_action(
    active_joint_target: torch.Tensor,
    *,
    active_pd_reference_pos: torch.Tensor,
    action_scale: float,
) -> torch.Tensor:
    """Convert absolute active joint targets to env action offsets.

    Env PD semantics are:
    `joint_target = action * action_scale + default_dof_pos + default_dof_pos_offset`.
    Therefore Stage 4 must subtract the full active PD reference, not only the
    static default pose.
    """

    if action_scale == 0:
        raise ValueError("action_scale must be non-zero")
    return (active_joint_target - active_pd_reference_pos) / action_scale


class DLSHandController:
    """DLS-only Stage 4 hand controller.

    This wrapper is intentionally explicit about PD action semantics: DLS produces
    active joint targets, then `active_target_to_action` converts those absolute
    targets to current env action offsets via
    `(target - default_dof_pos - default_dof_pos_offset) / action_scale`.
    """

    def __init__(
        self,
        *,
        indices: RightArmJointIndices,
        action_dim: int,
        body_indices: Sequence[int],
        dls_gain: float = 1.0,
        max_joint_delta: float = 0.05,
        damping_min: float = 0.02,
        damping_max: float = 0.20,
        singular_value_threshold: float = 0.08,
        joint_limit_margin: float = 0.15,
        comfortable_q: torch.Tensor | Sequence[float] | None = None,
        nullspace_gain: float = 0.0,
        max_nullspace_delta: float = 0.02,
    ) -> None:
        self.indices = indices
        self.action_dim = action_dim
        self.body_indices = tuple(body_indices)
        self.dls_gain = dls_gain
        self.max_joint_delta = max_joint_delta
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.singular_value_threshold = singular_value_threshold
        self.joint_limit_margin = joint_limit_margin
        self.comfortable_q = comfortable_q
        self.nullspace_gain = nullspace_gain
        self.max_nullspace_delta = max_nullspace_delta

    def step(
        self,
        *,
        body_action: torch.Tensor,
        active_q: torch.Tensor,
        wrist_pos_heading: torch.Tensor,
        command: HandTaskCommand,
        active_position_jacobian: torch.Tensor,
        active_lower: torch.Tensor,
        active_upper: torch.Tensor,
        active_pd_reference_pos: torch.Tensor,
        action_scale: float,
        joint_command_limiter: JointCommandLimiter | None = None,
        dt: float | None = None,
    ) -> DLSHandControllerOutput:
        position_error = (command.target_pos_heading - wrist_pos_heading) * command.position_mask
        damping, sigma_min = adaptive_damping(
            active_position_jacobian,
            damping_min=self.damping_min,
            damping_max=self.damping_max,
            singular_value_threshold=self.singular_value_threshold,
        )
        dq = damped_least_squares(
            active_position_jacobian,
            position_error,
            damping,
            gain=self.dls_gain,
            max_joint_delta=self.max_joint_delta,
        )
        dq = apply_joint_limit_scaling(
            active_q,
            dq,
            active_lower,
            active_upper,
            margin=self.joint_limit_margin,
        )
        primary_dq = dq
        primary_task_error_before_nullspace = (
            position_error - (active_position_jacobian @ primary_dq.unsqueeze(-1)).squeeze(-1)
        ).norm(dim=-1)
        nullspace_dq = torch.zeros_like(dq)
        if self.comfortable_q is not None and self.nullspace_gain > 0.0:
            comfortable_q = torch.as_tensor(self.comfortable_q, dtype=active_q.dtype, device=active_q.device)
            if comfortable_q.ndim == 1:
                comfortable_q = comfortable_q.unsqueeze(0).expand_as(active_q)
            nullspace_dq = comfortable_posture_nullspace_delta(
                active_position_jacobian,
                active_q,
                comfortable_q,
                damping,
                gain=self.nullspace_gain,
                max_joint_delta=self.max_nullspace_delta,
            )
            dq = apply_joint_limit_scaling(
                active_q,
                dq + nullspace_dq,
                active_lower,
                active_upper,
                margin=self.joint_limit_margin,
            )
        primary_task_error_after_nullspace = (
            position_error - (active_position_jacobian @ (primary_dq + nullspace_dq).unsqueeze(-1)).squeeze(-1)
        ).norm(dim=-1)
        j_times_nullspace = (active_position_jacobian @ nullspace_dq.unsqueeze(-1)).squeeze(-1)
        active_target = active_q + dq
        if joint_command_limiter is not None:
            if dt is None:
                raise ValueError("dt must be provided when joint_command_limiter is used")
            active_target = joint_command_limiter.step(active_target, dt=dt)
            dq = active_target - active_q
        active_action = active_target_to_action(
            active_target,
            active_pd_reference_pos=active_pd_reference_pos,
            action_scale=action_scale,
        )
        full_action = assemble_full_action(
            body_action,
            active_action,
            action_dim=self.action_dim,
            body_indices=self.body_indices,
            active_hand_indices=self.indices.active_action_indices,
            wrist_indices=self.indices.wrist_action_indices,
        )
        margin = joint_margin_scale(
            active_q,
            active_lower,
            active_upper,
            margin=self.joint_limit_margin,
        ).amin(dim=-1)
        return DLSHandControllerOutput(
            full_action=full_action,
            active_hand_action=active_action,
            active_joint_delta=dq,
            active_joint_target=active_target,
            primary_joint_delta=primary_dq,
            nullspace_joint_delta=nullspace_dq,
            position_error_heading=position_error,
            primary_task_error_before_nullspace=primary_task_error_before_nullspace,
            primary_task_error_after_nullspace=primary_task_error_after_nullspace,
            nullspace_action_norm=nullspace_dq.norm(dim=-1),
            j_times_nullspace_norm=j_times_nullspace.norm(dim=-1),
            damping=damping,
            sigma_min=sigma_min,
            joint_margin=margin,
        )


def coordination_gate_raw(
    *,
    ik_residual: torch.Tensor,
    sigma_min: torch.Tensor,
    joint_margin: torch.Tensor,
    position_error_norm: torch.Tensor,
    ik_residual_threshold: float = 0.05,
    singular_value_threshold: float = 0.08,
    joint_margin_threshold: float = 0.10,
    position_error_threshold: float = 0.20,
    weights: tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15),
) -> torch.Tensor:
    ik_score = (ik_residual / ik_residual_threshold).clamp(0.0, 1.0)
    singular_score = ((singular_value_threshold - sigma_min) / singular_value_threshold).clamp(0.0, 1.0)
    limit_score = ((joint_margin_threshold - joint_margin) / joint_margin_threshold).clamp(0.0, 1.0)
    error_score = (position_error_norm / position_error_threshold).clamp(0.0, 1.0)
    w_ik, w_sing, w_limit, w_err = weights
    return (w_ik * ik_score + w_sing * singular_score + w_limit * limit_score + w_err * error_score).clamp(0.0, 1.0)


@dataclass
class CoordinationGate:
    num_envs: int
    device: torch.device | str
    ema_alpha: float = 0.8

    def __post_init__(self) -> None:
        if not (0.0 <= self.ema_alpha < 1.0):
            raise ValueError("ema_alpha must be in [0, 1)")
        self.device = torch.device(self.device)
        self.previous_gate = torch.zeros(self.num_envs, device=self.device)
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None) -> None:
        if env_ids is None:
            self.previous_gate.zero_()
            self.initialized[:] = False
            return
        idx = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.previous_gate[idx] = 0.0
        self.initialized[idx] = False

    def step(self, raw_gate: torch.Tensor) -> torch.Tensor:
        raw_gate = raw_gate.to(device=self.device).flatten()
        if raw_gate.shape != (self.num_envs,):
            raise ValueError(f"Expected raw_gate shape {(self.num_envs,)}, got {raw_gate.shape}")
        first = ~self.initialized
        self.previous_gate[first] = raw_gate[first]
        self.initialized[first] = True
        filtered = self.ema_alpha * self.previous_gate + (1.0 - self.ema_alpha) * raw_gate
        self.previous_gate = filtered.clamp(0.0, 1.0)
        return self.previous_gate


def dls_hand_action_from_snapshot(
    *,
    body_action: torch.Tensor,
    snapshot: Stage4EnvSnapshot,
    command: HandTaskCommand,
    action_dim: int,
    dls_gain: float = 1.0,
    max_joint_delta: float = 0.05,
    damping_min: float = 0.02,
    damping_max: float = 0.20,
    singular_value_threshold: float = 0.08,
    joint_limit_margin: float = 0.15,
    comfortable_q: torch.Tensor | Sequence[float] | None = None,
    nullspace_gain: float = 0.0,
    max_nullspace_delta: float = 0.02,
    joint_command_limiter: JointCommandLimiter | None = None,
    dt: float | None = None,
) -> DLSHandControllerOutput:
    """Assemble a full env action from body policy output plus Stage 4 DLS hand control."""

    body_indices = body_action_indices_for_stage4(action_dim=action_dim, indices=snapshot.indices)
    controller = DLSHandController(
        indices=snapshot.indices,
        action_dim=action_dim,
        body_indices=body_indices,
        dls_gain=dls_gain,
        max_joint_delta=max_joint_delta,
        damping_min=damping_min,
        damping_max=damping_max,
        singular_value_threshold=singular_value_threshold,
        joint_limit_margin=joint_limit_margin,
        comfortable_q=comfortable_q,
        nullspace_gain=nullspace_gain,
        max_nullspace_delta=max_nullspace_delta,
    )
    return controller.step(
        body_action=body_action,
        active_q=snapshot.active_q,
        wrist_pos_heading=snapshot.end_effector_pos_heading,
        command=command,
        active_position_jacobian=snapshot.active_position_jacobian_heading,
        active_lower=snapshot.active_lower,
        active_upper=snapshot.active_upper,
        active_pd_reference_pos=snapshot.active_pd_reference_pos,
        action_scale=snapshot.action_scale,
        joint_command_limiter=joint_command_limiter,
        dt=dt,
    )
