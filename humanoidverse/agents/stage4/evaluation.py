from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .commands import (
    Stage4CommandSampler,
    Stage4CommandSamplerConfig,
    TARGET_CATEGORIES,
    TargetCategory,
)
from .control import JointCommandLimiter
from .controller import dls_hand_action_from_snapshot
from .env_adapter import build_stage4_env_snapshot


@dataclass(frozen=True)
class StaticReachEvaluationResult:
    steady_state_error: torch.Tensor
    max_action_jump: torch.Tensor
    min_sigma_min: torch.Tensor
    min_joint_margin: torch.Tensor
    target_offsets_heading: torch.Tensor
    target_categories: tuple[TargetCategory, ...]
    final_ik_residual: torch.Tensor
    grouped_metrics: dict[str, float]
    global_metrics: dict[str, float]
    target_diagnostics: tuple[dict[str, torch.Tensor | str], ...]
    sampler_config: Stage4CommandSamplerConfig


@dataclass(frozen=True)
class DLSParameterSet:
    label: str
    dls_gain: float = 1.0
    max_joint_delta: float = 0.20
    ema_alpha: float = 1.0
    max_joint_velocity: float = 0.30
    max_joint_acceleration: float | None = None
    steps_per_target: int = 12
    damping_min: float = 0.02
    damping_max: float = 0.20
    singular_value_threshold: float = 0.08
    nullspace_gain: float = 0.2


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _nan(device: torch.device | str) -> torch.Tensor:
    return torch.tensor(float("nan"), dtype=torch.float32, device=device)


def _category_for_envs(categories: tuple[TargetCategory, ...]) -> TargetCategory:
    if "coordination_required" in categories:
        return "coordination_required"
    if "boundary" in categories:
        return "boundary"
    return "reachable"


def _first_time_to_threshold(error_curve: torch.Tensor, *, threshold: float, dt: float) -> torch.Tensor:
    below = error_curve <= threshold
    if not bool(below.any().item()):
        return torch.tensor(float("inf"), dtype=error_curve.dtype, device=error_curve.device)
    return torch.nonzero(below, as_tuple=False)[0, 0].to(dtype=error_curve.dtype) * dt


def _success_rate(values: torch.Tensor, threshold: float) -> torch.Tensor:
    if values.numel() == 0:
        return _nan(values.device)
    return (values <= threshold).to(torch.float32).mean()


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    return values.mean() if values.numel() else _nan(values.device)


def _safe_median(values: torch.Tensor) -> torch.Tensor:
    return values.median() if values.numel() else _nan(values.device)


def _safe_p90(values: torch.Tensor) -> torch.Tensor:
    return torch.quantile(values, 0.90) if values.numel() else _nan(values.device)


def _infer_failure_mode(
    *,
    error_curve: torch.Tensor,
    final_ik_residual: torch.Tensor,
    min_sigma: torch.Tensor,
    min_margin: torch.Tensor,
    max_active_velocity: torch.Tensor,
    max_active_acceleration: torch.Tensor,
    max_joint_velocity: float | None,
    max_joint_acceleration: float | None,
    boundary_sigma_threshold: float,
    boundary_joint_margin_threshold: float,
) -> str:
    initial_error = error_curve[0]
    final_error = error_curve[-1]
    improvement = initial_error - final_error
    if final_ik_residual >= 0.08:
        return "target_unreachable_by_arm_only_presolve"
    if min_margin < boundary_joint_margin_threshold:
        return "joint_limit_pressure"
    if min_sigma < boundary_sigma_threshold:
        return "low_manipulability"
    if max_joint_velocity is not None and max_active_velocity >= 0.95 * max_joint_velocity:
        return "limiter_velocity_saturated"
    if max_joint_acceleration is not None and max_active_acceleration >= 0.95 * max_joint_acceleration:
        return "limiter_acceleration_saturated"
    if improvement > 0 and final_error > 0.05 and improvement / initial_error.clamp_min(1e-6) > 0.25:
        return "horizon_too_short"
    if final_error > 0.05:
        return "plateau_or_model_mismatch"
    return "success"


def _compute_grouped_metrics(
    *,
    target_categories: tuple[TargetCategory, ...],
    steady_state_error: torch.Tensor,
    final_ik_residual: torch.Tensor,
    min_sigma: torch.Tensor,
    min_margin: torch.Tensor,
    time_to_5cm: torch.Tensor,
    sampler_config: Stage4CommandSamplerConfig,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for category in TARGET_CATEGORIES:
        mask = torch.tensor([item == category for item in target_categories], device=steady_state_error.device)
        values = steady_state_error[mask]
        residual = final_ik_residual[mask]
        sigma = min_sigma[mask]
        margin = min_margin[mask]
        t5 = time_to_5cm[mask]

        if category == "reachable":
            metrics[f"{category}/steady_error_mean"] = _as_float(_safe_mean(values))
            metrics[f"{category}/steady_error_median"] = _as_float(_safe_median(values))
            metrics[f"{category}/steady_error_p90"] = _as_float(_safe_p90(values))
            metrics[f"{category}/success_1cm"] = _as_float(_success_rate(values, 0.01))
            metrics[f"{category}/success_2cm"] = _as_float(_success_rate(values, 0.02))
            metrics[f"{category}/success_5cm"] = _as_float(_success_rate(values, 0.05))
            finite_t5 = t5[torch.isfinite(t5)]
            metrics[f"{category}/time_to_5cm"] = _as_float(_safe_mean(finite_t5))
        elif category == "boundary":
            metrics[f"{category}/steady_error_mean"] = _as_float(_safe_mean(values))
            metrics[f"{category}/success_2cm"] = _as_float(_success_rate(values, 0.02))
            metrics[f"{category}/joint_limit_hit_rate"] = _as_float(
                _safe_mean((margin < sampler_config.boundary_joint_margin_threshold).to(torch.float32))
            )
            metrics[f"{category}/singularity_rate"] = _as_float(
                _safe_mean((sigma < sampler_config.boundary_sigma_threshold).to(torch.float32))
            )
        else:
            metrics[f"{category}/steady_error_mean"] = _as_float(_safe_mean(values))
            metrics[f"{category}/final_ik_residual"] = _as_float(_safe_mean(residual))
    return metrics


def _compute_global_metrics(
    *,
    max_action_jump: torch.Tensor,
    max_active_velocity: torch.Tensor,
    max_active_acceleration: torch.Tensor,
    wrist_q_abs_max: torch.Tensor,
    wrist_dq_abs_max: torch.Tensor,
    min_sigma: torch.Tensor,
    min_margin: torch.Tensor,
) -> dict[str, float]:
    return {
        "max_action_jump": _as_float(max_action_jump.max()),
        "active_joint_velocity_max": _as_float(max_active_velocity.max()),
        "active_joint_acceleration_max": _as_float(max_active_acceleration.max()),
        "wrist_q_abs_max": _as_float(wrist_q_abs_max.max()),
        "wrist_dq_abs_max": _as_float(wrist_dq_abs_max.max()),
        "minimum_sigma": _as_float(min_sigma.min()),
        "minimum_joint_margin": _as_float(min_margin.min()),
    }


def evaluate_static_reach_dls(
    env,
    *,
    target_offsets_heading: torch.Tensor | Sequence[Sequence[float]],
    steps_per_target: int = 12,
    max_joint_velocity: float = 0.30,
    max_joint_acceleration: float | None = None,
    ema_alpha: float = 1.0,
    body_action_value: float = 0.0,
    comfortable_q: torch.Tensor | Sequence[float] | None = None,
    nullspace_gain: float = 0.2,
    max_nullspace_delta: float = 0.03,
    dls_gain: float = 1.0,
    max_joint_delta: float = 0.20,
    damping_min: float = 0.02,
    damping_max: float = 0.20,
    singular_value_threshold: float = 0.08,
    sampler_config: Stage4CommandSamplerConfig | None = None,
) -> StaticReachEvaluationResult:
    """Run stratified DLS-only static reach evaluation in the root_heading_frame.

    This path is deliberately outside replay/training. Static targets are
    classified by an arm-only DLS presolve, then executed through the real
    `snapshot -> DLS -> limiter -> env.step` path.
    """

    if steps_per_target <= 0:
        raise ValueError("steps_per_target must be positive")
    base_env = env.unwrapped
    device = base_env.device
    offsets = torch.as_tensor(target_offsets_heading, dtype=torch.float32, device=device)
    if offsets.ndim != 2 or offsets.shape[-1] != 3:
        raise ValueError(f"target_offsets_heading must have shape [num_targets,3], got {offsets.shape}")

    sampler_cfg = sampler_config or Stage4CommandSamplerConfig()
    sampler = Stage4CommandSampler(sampler_cfg)
    initial_snapshot = build_stage4_env_snapshot(base_env)
    limiter = JointCommandLimiter(
        action_dim=4,
        num_envs=base_env.num_envs,
        device=device,
        ema_alpha=ema_alpha,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
    )
    limiter.step(initial_snapshot.active_q, dt=base_env.dt)
    previous_action = None

    steady_errors = []
    max_action_jumps = []
    min_sigmas = []
    min_margins = []
    target_categories = []
    final_ik_residuals = []
    time_to_5cm_values = []
    max_active_velocities = []
    max_active_accelerations = []
    wrist_q_abs_values = []
    wrist_dq_abs_values = []
    diagnostics: list[dict[str, torch.Tensor | str]] = []

    wrist_idx = torch.tensor(initial_snapshot.indices.wrist_dof_indices, device=device, dtype=torch.long)
    previous_command_target = initial_snapshot.active_q.detach().clone()
    previous_command_velocity = torch.zeros_like(previous_command_target)

    for target_idx, offset in enumerate(offsets):
        snapshot = build_stage4_env_snapshot(base_env)
        expanded_offset = offset.unsqueeze(0).expand(base_env.num_envs, 3)
        target_pos = snapshot.end_effector_pos_heading + expanded_offset
        classification = sampler.classify_static_targets(snapshot, target_pos_heading=target_pos)
        target_category = _category_for_envs(classification.category)
        target_categories.append(target_category)
        final_ik_residuals.append(classification.final_ik_residual.mean().detach())
        command = sampler.make_static_reach_command(snapshot, target_offsets_heading=expanded_offset)

        max_action_jump = torch.zeros((), dtype=torch.float32, device=device)
        min_sigma = torch.full((), float("inf"), dtype=torch.float32, device=device)
        min_margin = torch.full((), float("inf"), dtype=torch.float32, device=device)
        max_active_velocity = torch.zeros((), dtype=torch.float32, device=device)
        max_active_acceleration = torch.zeros((), dtype=torch.float32, device=device)
        wrist_q_abs_max = torch.zeros((), dtype=torch.float32, device=device)
        wrist_dq_abs_max = torch.zeros((), dtype=torch.float32, device=device)
        final_error = torch.full((), float("nan"), dtype=torch.float32, device=device)

        initial_wrist_pos = snapshot.end_effector_pos_heading.detach().clone()
        position_error_curve = []
        active_joint_q_curve = []
        active_joint_target_curve = []
        action_curve = []
        sigma_min_curve = []
        joint_margin_curve = []
        damping_curve = []
        nullspace_action_curve = []
        primary_before_curve = []
        primary_after_curve = []
        nullspace_norm_curve = []
        j_nullspace_norm_curve = []

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
                dls_gain=dls_gain,
                max_joint_delta=max_joint_delta,
                damping_min=damping_min,
                damping_max=damping_max,
                singular_value_threshold=singular_value_threshold,
                comfortable_q=comfortable_q,
                nullspace_gain=nullspace_gain,
                max_nullspace_delta=max_nullspace_delta,
                joint_command_limiter=limiter,
                dt=base_env.dt,
            )
            if previous_action is not None:
                max_action_jump = torch.maximum(max_action_jump, (out.active_hand_action - previous_action).abs().max())
            previous_action = out.active_hand_action.detach().clone()

            command_velocity = (out.active_joint_target - previous_command_target) / base_env.dt
            command_acceleration = (command_velocity - previous_command_velocity) / base_env.dt
            max_active_velocity = torch.maximum(max_active_velocity, command_velocity.abs().max())
            max_active_acceleration = torch.maximum(max_active_acceleration, command_acceleration.abs().max())
            previous_command_target = out.active_joint_target.detach().clone()
            previous_command_velocity = command_velocity.detach().clone()

            min_sigma = torch.minimum(min_sigma, out.sigma_min.amin())
            min_margin = torch.minimum(min_margin, out.joint_margin.amin())
            final_error = out.position_error_heading.norm(dim=-1).mean()

            position_error_curve.append(out.position_error_heading.norm(dim=-1).detach().clone())
            active_joint_q_curve.append(snapshot.active_q.detach().clone())
            active_joint_target_curve.append(out.active_joint_target.detach().clone())
            action_curve.append(out.full_action.detach().clone())
            sigma_min_curve.append(out.sigma_min.detach().clone())
            joint_margin_curve.append(out.joint_margin.detach().clone())
            damping_curve.append(out.damping.detach().clone())
            nullspace_action_curve.append(out.nullspace_joint_delta.detach().clone())
            primary_before_curve.append(out.primary_task_error_before_nullspace.detach().clone())
            primary_after_curve.append(out.primary_task_error_after_nullspace.detach().clone())
            nullspace_norm_curve.append(out.nullspace_action_norm.detach().clone())
            j_nullspace_norm_curve.append(out.j_times_nullspace_norm.detach().clone())

            obs, reward, terminated, truncated, info = env.step(out.full_action)
            reward_tensor = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(reward, device=device)
            if not torch.isfinite(reward_tensor).all():
                raise RuntimeError(f"Non-finite reward during static reach target {target_idx}")
            if not torch.isfinite(base_env.simulator.dof_pos).all():
                raise RuntimeError(f"Non-finite dof_pos during static reach target {target_idx}")
            wrist_q_abs_max = torch.maximum(wrist_q_abs_max, base_env.simulator.dof_pos[:, wrist_idx].abs().max())
            if hasattr(base_env.simulator, "dof_vel"):
                wrist_dq_abs_max = torch.maximum(wrist_dq_abs_max, base_env.simulator.dof_vel[:, wrist_idx].abs().max())

        final_snapshot = build_stage4_env_snapshot(base_env)
        error_curve = torch.stack(position_error_curve).mean(dim=-1)
        time_to_5cm = _first_time_to_threshold(error_curve, threshold=0.05, dt=base_env.dt)
        failure_mode = _infer_failure_mode(
            error_curve=error_curve,
            final_ik_residual=classification.final_ik_residual.mean(),
            min_sigma=min_sigma,
            min_margin=min_margin,
            max_active_velocity=max_active_velocity,
            max_active_acceleration=max_active_acceleration,
            max_joint_velocity=max_joint_velocity,
            max_joint_acceleration=max_joint_acceleration,
            boundary_sigma_threshold=sampler_cfg.boundary_sigma_threshold,
            boundary_joint_margin_threshold=sampler_cfg.boundary_joint_margin_threshold,
        )

        steady_errors.append(final_error.detach())
        max_action_jumps.append(max_action_jump.detach())
        min_sigmas.append(min_sigma.detach())
        min_margins.append(min_margin.detach())
        time_to_5cm_values.append(time_to_5cm.detach())
        max_active_velocities.append(max_active_velocity.detach())
        max_active_accelerations.append(max_active_acceleration.detach())
        wrist_q_abs_values.append(wrist_q_abs_max.detach())
        wrist_dq_abs_values.append(wrist_dq_abs_max.detach())
        diagnostics.append(
            {
                "target_pos_heading": target_pos.detach().clone(),
                "target_category": target_category,
                "initial_wrist_pos_heading": initial_wrist_pos,
                "final_wrist_pos_heading": final_snapshot.end_effector_pos_heading.detach().clone(),
                "position_error_curve": torch.stack(position_error_curve),
                "active_joint_q_curve": torch.stack(active_joint_q_curve),
                "active_joint_target_curve": torch.stack(active_joint_target_curve),
                "action_curve": torch.stack(action_curve),
                "sigma_min_curve": torch.stack(sigma_min_curve),
                "joint_margin_curve": torch.stack(joint_margin_curve),
                "damping_curve": torch.stack(damping_curve),
                "nullspace_action_curve": torch.stack(nullspace_action_curve),
                "primary_task_error_before_nullspace": torch.stack(primary_before_curve),
                "primary_task_error_after_nullspace": torch.stack(primary_after_curve),
                "nullspace_action_norm": torch.stack(nullspace_norm_curve),
                "j_times_nullspace_norm": torch.stack(j_nullspace_norm_curve),
                "failure_mode": failure_mode,
            }
        )

    steady_state_error = torch.stack(steady_errors)
    max_action_jump = torch.stack(max_action_jumps)
    min_sigma_tensor = torch.stack(min_sigmas)
    min_margin_tensor = torch.stack(min_margins)
    final_ik_residual = torch.stack(final_ik_residuals)
    time_to_5cm = torch.stack(time_to_5cm_values)
    max_active_velocity = torch.stack(max_active_velocities)
    max_active_acceleration = torch.stack(max_active_accelerations)
    wrist_q_abs = torch.stack(wrist_q_abs_values)
    wrist_dq_abs = torch.stack(wrist_dq_abs_values)

    return StaticReachEvaluationResult(
        steady_state_error=steady_state_error,
        max_action_jump=max_action_jump,
        min_sigma_min=min_sigma_tensor,
        min_joint_margin=min_margin_tensor,
        target_offsets_heading=offsets.detach(),
        target_categories=tuple(target_categories),
        final_ik_residual=final_ik_residual,
        grouped_metrics=_compute_grouped_metrics(
            target_categories=tuple(target_categories),
            steady_state_error=steady_state_error,
            final_ik_residual=final_ik_residual,
            min_sigma=min_sigma_tensor,
            min_margin=min_margin_tensor,
            time_to_5cm=time_to_5cm,
            sampler_config=sampler_cfg,
        ),
        global_metrics=_compute_global_metrics(
            max_action_jump=max_action_jump,
            max_active_velocity=max_active_velocity,
            max_active_acceleration=max_active_acceleration,
            wrist_q_abs_max=wrist_q_abs,
            wrist_dq_abs_max=wrist_dq_abs,
            min_sigma=min_sigma_tensor,
            min_margin=min_margin_tensor,
        ),
        target_diagnostics=tuple(diagnostics),
        sampler_config=sampler_cfg,
    )


def evaluate_static_reach_dls_parameter_scan(
    make_env: Callable[[], object],
    *,
    target_offsets_heading: torch.Tensor | Sequence[Sequence[float]],
    parameter_sets: Sequence[DLSParameterSet],
    sampler_config: Stage4CommandSamplerConfig | None = None,
    comfortable_q: torch.Tensor | Sequence[float] | None = None,
    max_nullspace_delta: float = 0.03,
) -> dict[str, StaticReachEvaluationResult]:
    """Run a small Stage4B DLS parameter scan with one fresh env per setting."""

    results: dict[str, StaticReachEvaluationResult] = {}
    for params in parameter_sets:
        env = make_env()
        try:
            results[params.label] = evaluate_static_reach_dls(
                env,
                target_offsets_heading=target_offsets_heading,
                steps_per_target=params.steps_per_target,
                max_joint_velocity=params.max_joint_velocity,
                max_joint_acceleration=params.max_joint_acceleration,
                ema_alpha=params.ema_alpha,
                comfortable_q=comfortable_q,
                nullspace_gain=params.nullspace_gain,
                max_nullspace_delta=max_nullspace_delta,
                dls_gain=params.dls_gain,
                max_joint_delta=params.max_joint_delta,
                damping_min=params.damping_min,
                damping_max=params.damping_max,
                singular_value_threshold=params.singular_value_threshold,
                sampler_config=sampler_config,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    return results
