from __future__ import annotations

from dataclasses import dataclass

import torch


def adaptive_damping(
    jacobian: torch.Tensor,
    *,
    damping_min: float = 0.02,
    damping_max: float = 0.20,
    singular_value_threshold: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return damping and sigma_min for a batch of active-arm Jacobians."""

    if jacobian.shape[-2:] != (3, 4):
        raise ValueError(f"Expected jacobian shape [...,3,4], got {jacobian.shape}")
    if not (0.0 <= damping_min <= damping_max):
        raise ValueError("Expected 0 <= damping_min <= damping_max")
    if singular_value_threshold <= 0:
        raise ValueError("singular_value_threshold must be positive")

    sigma_min = torch.linalg.svdvals(jacobian)[..., -1]
    risk = ((singular_value_threshold - sigma_min) / singular_value_threshold).clamp(0.0, 1.0)
    damping = damping_min + (damping_max - damping_min) * risk
    return damping, sigma_min


def damped_least_squares(
    jacobian: torch.Tensor,
    task_error: torch.Tensor,
    damping: torch.Tensor | float,
    *,
    gain: float = 1.0,
    max_joint_delta: float = 0.05,
) -> torch.Tensor:
    """Damped least-squares solve for 3D position control with 4 active DOFs."""

    if jacobian.shape[-2:] != (3, 4):
        raise ValueError(f"Expected jacobian shape [...,3,4], got {jacobian.shape}")
    if task_error.shape[-1] != 3:
        raise ValueError(f"Expected task_error shape [...,3], got {task_error.shape}")
    if jacobian.shape[:-2] != task_error.shape[:-1]:
        raise ValueError(f"Batch shapes differ: jacobian={jacobian.shape}, task_error={task_error.shape}")
    if max_joint_delta <= 0:
        raise ValueError("max_joint_delta must be positive")

    jj_t = jacobian @ jacobian.transpose(-1, -2)
    eye = torch.eye(3, dtype=jacobian.dtype, device=jacobian.device).expand_as(jj_t)
    damping_tensor = torch.as_tensor(damping, dtype=jacobian.dtype, device=jacobian.device)
    while damping_tensor.ndim < jj_t.ndim - 1:
        damping_tensor = damping_tensor.unsqueeze(-1)
    damping_sq = damping_tensor.square().unsqueeze(-1)
    system = jj_t + damping_sq * eye
    task_step = torch.linalg.solve(system, (gain * task_error).unsqueeze(-1))
    dq = (jacobian.transpose(-1, -2) @ task_step).squeeze(-1)
    return dq.clamp(min=-max_joint_delta, max=max_joint_delta)


def joint_margin_scale(
    q: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    margin: float = 0.15,
) -> torch.Tensor:
    """Scale in [0,1] that becomes small near either joint limit."""

    if q.shape != lower.shape or q.shape != upper.shape:
        raise ValueError(f"q/lower/upper shapes must match, got {q.shape}, {lower.shape}, {upper.shape}")
    if margin <= 0:
        raise ValueError("margin must be positive")
    distance_to_lower = q - lower
    distance_to_upper = upper - q
    nearest = torch.minimum(distance_to_lower, distance_to_upper)
    return (nearest / margin).clamp(0.0, 1.0)


def apply_joint_limit_scaling(
    q: torch.Tensor,
    dq: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    margin: float = 0.15,
) -> torch.Tensor:
    """Direction-aware limit scaling for a proposed joint delta."""

    if q.shape != dq.shape:
        raise ValueError(f"q and dq shapes must match, got {q.shape} and {dq.shape}")
    distance_to_lower = q - lower
    distance_to_upper = upper - q
    toward_lower = dq < 0
    toward_upper = dq > 0
    lower_scale = (distance_to_lower / margin).clamp(0.0, 1.0)
    upper_scale = (distance_to_upper / margin).clamp(0.0, 1.0)
    scale = torch.ones_like(dq)
    scale = torch.where(toward_lower, lower_scale, scale)
    scale = torch.where(toward_upper, upper_scale, scale)
    return dq * scale


@dataclass
class JointCommandLimiter:
    action_dim: int
    num_envs: int
    device: torch.device | str
    ema_alpha: float = 1.0
    max_joint_velocity: float | None = None
    max_joint_acceleration: float | None = None

    def __post_init__(self) -> None:
        if self.action_dim <= 0 or self.num_envs <= 0:
            raise ValueError("action_dim and num_envs must be positive")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")
        self.device = torch.device(self.device)
        self.prev_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.prev_velocity = torch.zeros_like(self.prev_target)
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None) -> None:
        if env_ids is None:
            self.initialized[:] = False
            self.prev_target.zero_()
            self.prev_velocity.zero_()
            return
        idx = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.initialized[idx] = False
        self.prev_target[idx] = 0.0
        self.prev_velocity[idx] = 0.0

    def step(self, raw_target: torch.Tensor, dt: float) -> torch.Tensor:
        if raw_target.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected raw_target shape {(self.num_envs, self.action_dim)}, got {raw_target.shape}")
        if dt <= 0:
            raise ValueError("dt must be positive")
        raw_target = raw_target.to(device=self.device)

        first = ~self.initialized
        if first.any():
            self.prev_target[first] = raw_target[first]
            self.prev_velocity[first] = 0.0
            self.initialized[first] = True

        target = self.ema_alpha * raw_target + (1.0 - self.ema_alpha) * self.prev_target
        desired_velocity = (target - self.prev_target) / dt
        if self.max_joint_acceleration is not None:
            max_dv = self.max_joint_acceleration * dt
            desired_velocity = torch.minimum(desired_velocity, self.prev_velocity + max_dv)
            desired_velocity = torch.maximum(desired_velocity, self.prev_velocity - max_dv)
        if self.max_joint_velocity is not None:
            desired_velocity = desired_velocity.clamp(
                min=-self.max_joint_velocity,
                max=self.max_joint_velocity,
            )
        limited = self.prev_target + desired_velocity * dt
        self.prev_velocity = desired_velocity
        self.prev_target = limited
        return limited
