from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .control import JointCommandLimiter
from .controller import HandTaskCommand, dls_hand_action_from_snapshot
from .env_adapter import build_stage4_env_snapshot


@dataclass(frozen=True)
class StaticReachEvaluationResult:
    steady_state_error: torch.Tensor
    max_action_jump: torch.Tensor
    min_sigma_min: torch.Tensor
    min_joint_margin: torch.Tensor
    target_offsets_heading: torch.Tensor


def _make_static_reach_command(*, target_pos_heading: torch.Tensor) -> HandTaskCommand:
    num_envs = target_pos_heading.shape[0]
    return HandTaskCommand(
        target_pos_root=target_pos_heading,
        target_lin_vel_root=torch.zeros(num_envs, 3, dtype=target_pos_heading.dtype, device=target_pos_heading.device),
        position_mask=torch.ones(num_envs, 1, dtype=target_pos_heading.dtype, device=target_pos_heading.device),
        velocity_mask=torch.zeros(num_envs, 1, dtype=target_pos_heading.dtype, device=target_pos_heading.device),
        command_id=torch.zeros(num_envs, dtype=torch.long, device=target_pos_heading.device),
        command_done=torch.zeros(num_envs, 1, dtype=torch.bool, device=target_pos_heading.device),
    )


def evaluate_static_reach_dls(
    env,
    *,
    target_offsets_heading: torch.Tensor | Sequence[Sequence[float]],
    steps_per_target: int = 12,
    max_joint_velocity: float = 0.30,
    body_action_value: float = 0.0,
    comfortable_q: torch.Tensor | Sequence[float] | None = None,
    nullspace_gain: float = 0.2,
    max_nullspace_delta: float = 0.03,
) -> StaticReachEvaluationResult:
    """Run a short DLS-only static reach evaluation in the root_heading_frame.

    This is deliberately outside replay/training. It exercises the actual env
    rollout path while keeping commands static and inspectable.
    """

    if steps_per_target <= 0:
        raise ValueError("steps_per_target must be positive")
    base_env = env.unwrapped
    device = base_env.device
    offsets = torch.as_tensor(target_offsets_heading, dtype=torch.float32, device=device)
    if offsets.ndim != 2 or offsets.shape[-1] != 3:
        raise ValueError(f"target_offsets_heading must have shape [num_targets,3], got {offsets.shape}")

    initial_snapshot = build_stage4_env_snapshot(base_env)
    limiter = JointCommandLimiter(
        action_dim=4,
        num_envs=base_env.num_envs,
        device=device,
        ema_alpha=1.0,
        max_joint_velocity=max_joint_velocity,
    )
    limiter.step(initial_snapshot.active_q, dt=base_env.dt)
    previous_action = None

    steady_errors = []
    max_action_jumps = []
    min_sigmas = []
    min_margins = []

    for target_idx, offset in enumerate(offsets):
        snapshot = build_stage4_env_snapshot(base_env)
        target_pos = snapshot.end_effector_pos_heading + offset.unsqueeze(0).expand(base_env.num_envs, 3)
        command = _make_static_reach_command(target_pos_heading=target_pos)
        max_action_jump = torch.zeros((), dtype=torch.float32, device=device)
        min_sigma = torch.full((), float("inf"), dtype=torch.float32, device=device)
        min_margin = torch.full((), float("inf"), dtype=torch.float32, device=device)
        final_error = torch.full((), float("nan"), dtype=torch.float32, device=device)

        for _ in range(steps_per_target):
            snapshot = build_stage4_env_snapshot(base_env)
            body_action = torch.full(
                (base_env.num_envs, 22),
                float(body_action_value),
                dtype=snapshot.active_q.dtype,
                device=device,
            )
            out = dls_hand_action_from_snapshot(
                body_action=body_action,
                snapshot=snapshot,
                command=command,
                action_dim=env.single_action_space.shape[0],
                max_joint_delta=0.20,
                comfortable_q=comfortable_q,
                nullspace_gain=nullspace_gain,
                max_nullspace_delta=max_nullspace_delta,
                joint_command_limiter=limiter,
                dt=base_env.dt,
            )
            if previous_action is not None:
                max_action_jump = torch.maximum(max_action_jump, (out.active_hand_action - previous_action).abs().max())
            previous_action = out.active_hand_action.detach().clone()
            min_sigma = torch.minimum(min_sigma, out.sigma_min.amin())
            min_margin = torch.minimum(min_margin, out.joint_margin.amin())
            final_error = out.position_error_root.norm(dim=-1).mean()
            obs, reward, terminated, truncated, info = env.step(out.full_action)
            reward_tensor = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(reward, device=device)
            if not torch.isfinite(reward_tensor).all():
                raise RuntimeError(f"Non-finite reward during static reach target {target_idx}")
            if not torch.isfinite(base_env.simulator.dof_pos).all():
                raise RuntimeError(f"Non-finite dof_pos during static reach target {target_idx}")

        steady_errors.append(final_error.detach())
        max_action_jumps.append(max_action_jump.detach())
        min_sigmas.append(min_sigma.detach())
        min_margins.append(min_margin.detach())

    return StaticReachEvaluationResult(
        steady_state_error=torch.stack(steady_errors),
        max_action_jump=torch.stack(max_action_jumps),
        min_sigma_min=torch.stack(min_sigmas),
        min_joint_margin=torch.stack(min_margins),
        target_offsets_heading=offsets.detach(),
    )
