from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from .control import adaptive_damping, apply_joint_limit_scaling, damped_least_squares, joint_margin_scale
from .controller import HandTaskCommand
from .env_adapter import Stage4EnvSnapshot

CommandMode = Literal["static_reach", "motion_reference", "external_task"]
TargetCategory = Literal["reachable", "boundary", "coordination_required"]

TARGET_CATEGORIES: tuple[TargetCategory, ...] = (
    "reachable",
    "boundary",
    "coordination_required",
)


@dataclass(frozen=True)
class Stage4CommandSamplerConfig:
    """Configuration for Stage 4 commands in the root_heading_frame.

    The root_heading_frame uses the root/pelvis position as origin and only the
    root yaw/facing direction for orientation. It deliberately excludes root
    roll and pitch.
    """

    command_mode: CommandMode = "static_reach"
    reachable_threshold: float = 0.02
    coordination_required_threshold: float = 0.08
    boundary_sigma_threshold: float = 0.03
    boundary_joint_margin_threshold: float = 0.10
    presolve_steps: int = 20
    presolve_dls_gain: float = 1.0
    presolve_max_joint_delta: float = 0.05
    presolve_damping_min: float = 0.02
    presolve_damping_max: float = 0.20
    presolve_singular_value_threshold: float = 0.08
    presolve_joint_limit_margin: float = 0.15


@dataclass(frozen=True)
class TargetClassification:
    category: tuple[TargetCategory, ...]
    final_ik_residual: torch.Tensor
    min_sigma: torch.Tensor
    min_joint_margin: torch.Tensor


def make_hand_task_command_root_heading(
    *,
    target_pos_heading: torch.Tensor,
    target_lin_vel_heading: torch.Tensor | None = None,
    command_id: torch.Tensor | None = None,
) -> HandTaskCommand:
    """Build a 3D position command expressed in the root_heading_frame."""

    if target_pos_heading.ndim != 2 or target_pos_heading.shape[-1] != 3:
        raise ValueError(f"target_pos_heading must have shape [num_envs,3], got {target_pos_heading.shape}")
    num_envs = target_pos_heading.shape[0]
    if target_lin_vel_heading is None:
        target_lin_vel_heading = torch.zeros_like(target_pos_heading)
    if target_lin_vel_heading.shape != target_pos_heading.shape:
        raise ValueError(
            "target_lin_vel_heading must match target_pos_heading shape, "
            f"got {target_lin_vel_heading.shape} and {target_pos_heading.shape}"
        )
    if command_id is None:
        command_id = torch.zeros(num_envs, dtype=torch.long, device=target_pos_heading.device)
    return HandTaskCommand(
        target_pos_heading=target_pos_heading,
        target_lin_vel_heading=target_lin_vel_heading,
        position_mask=torch.ones(num_envs, 1, dtype=target_pos_heading.dtype, device=target_pos_heading.device),
        velocity_mask=torch.zeros(num_envs, 1, dtype=target_pos_heading.dtype, device=target_pos_heading.device),
        command_id=command_id.to(device=target_pos_heading.device),
        command_done=torch.zeros(num_envs, 1, dtype=torch.bool, device=target_pos_heading.device),
    )


class Stage4CommandSampler:
    """Command sampler and arm-only reachability classifier for Stage4B.

    Static reach commands are absolute targets in the root_heading_frame. The
    classifier uses a multi-step arm-only DLS presolve on the current snapshot's
    linearized Jacobian; this is a conservative routing signal for evaluation,
    not a replacement for real rollout metrics.
    """

    def __init__(self, cfg: Stage4CommandSamplerConfig | None = None) -> None:
        self.cfg = cfg or Stage4CommandSamplerConfig()

    def make_static_reach_command(
        self,
        snapshot: Stage4EnvSnapshot,
        *,
        target_offsets_heading: torch.Tensor | Sequence[Sequence[float]],
    ) -> HandTaskCommand:
        if self.cfg.command_mode != "static_reach":
            raise NotImplementedError(f"command_mode={self.cfg.command_mode!r} is not implemented for DLS-only Stage4B")
        offsets = torch.as_tensor(
            target_offsets_heading,
            dtype=snapshot.end_effector_pos_heading.dtype,
            device=snapshot.end_effector_pos_heading.device,
        )
        if offsets.ndim != 2 or offsets.shape[-1] != 3:
            raise ValueError(f"target_offsets_heading must have shape [num_envs,3], got {offsets.shape}")
        if offsets.shape[0] == 1:
            offsets = offsets.expand(snapshot.end_effector_pos_heading.shape[0], 3)
        if offsets.shape != snapshot.end_effector_pos_heading.shape:
            raise ValueError(
                "target_offsets_heading must have one row or num_envs rows, "
                f"got {offsets.shape} for snapshot {snapshot.end_effector_pos_heading.shape}"
            )
        return make_hand_task_command_root_heading(
            target_pos_heading=snapshot.end_effector_pos_heading + offsets,
        )

    def classify_static_targets(
        self,
        snapshot: Stage4EnvSnapshot,
        *,
        target_pos_heading: torch.Tensor,
    ) -> TargetClassification:
        cfg = self.cfg
        if cfg.presolve_steps <= 0:
            raise ValueError("presolve_steps must be positive")
        if target_pos_heading.shape != snapshot.end_effector_pos_heading.shape:
            raise ValueError(
                "target_pos_heading must match snapshot.end_effector_pos_heading shape, "
                f"got {target_pos_heading.shape} and {snapshot.end_effector_pos_heading.shape}"
            )

        jac = snapshot.active_position_jacobian_heading
        q = snapshot.active_q.clone()
        error = target_pos_heading - snapshot.end_effector_pos_heading
        min_margin = torch.full((q.shape[0],), float("inf"), dtype=q.dtype, device=q.device)
        min_sigma = torch.full_like(min_margin, float("inf"))

        for _ in range(cfg.presolve_steps):
            damping, sigma_min = adaptive_damping(
                jac,
                damping_min=cfg.presolve_damping_min,
                damping_max=cfg.presolve_damping_max,
                singular_value_threshold=cfg.presolve_singular_value_threshold,
            )
            dq = damped_least_squares(
                jac,
                error,
                damping,
                gain=cfg.presolve_dls_gain,
                max_joint_delta=cfg.presolve_max_joint_delta,
            )
            dq = apply_joint_limit_scaling(
                q,
                dq,
                snapshot.active_lower,
                snapshot.active_upper,
                margin=cfg.presolve_joint_limit_margin,
            )
            q = q + dq
            error = error - (jac @ dq.unsqueeze(-1)).squeeze(-1)
            margin = joint_margin_scale(
                q,
                snapshot.active_lower,
                snapshot.active_upper,
                margin=cfg.presolve_joint_limit_margin,
            ).amin(dim=-1)
            min_margin = torch.minimum(min_margin, margin)
            min_sigma = torch.minimum(min_sigma, sigma_min)

        final_residual = error.norm(dim=-1)
        low_sigma = min_sigma < cfg.boundary_sigma_threshold
        low_margin = min_margin < cfg.boundary_joint_margin_threshold
        reachable = (final_residual < cfg.reachable_threshold) & ~low_sigma & ~low_margin
        coordination_required = final_residual >= cfg.coordination_required_threshold
        categories: list[TargetCategory] = []
        for idx in range(final_residual.numel()):
            if bool(reachable[idx].item()):
                categories.append("reachable")
            elif bool(coordination_required[idx].item()):
                categories.append("coordination_required")
            else:
                categories.append("boundary")

        return TargetClassification(
            category=tuple(categories),
            final_ik_residual=final_residual,
            min_sigma=min_sigma,
            min_joint_margin=min_margin,
        )
