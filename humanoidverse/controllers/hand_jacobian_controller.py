"""Batched right-hand Jacobian precision controller.

The controller is an inference-only residual: it converts a root-frame wrist
position error into a seven-joint correction and composes that correction in
the same PD-target space as the actor action.  It never changes the actor or a
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HandJacobianControllerConfig:
    """Configuration for :class:`HandJacobianController` (SI units)."""

    enabled: bool = False
    frame: str = "root"
    kp_position: float = 3.0
    kd_position: float = 0.25
    damping: float = 0.08
    adaptive_damping: bool = False
    min_damping: float = 0.02
    max_damping: float = 0.20
    singularity_threshold: float = 0.05
    gate_mode: str = "linear"
    gate_near: float = 0.02
    gate_far: float = 0.10
    max_task_velocity: float = 0.30
    max_delta_q: float = 0.02
    max_delta_action: float = 0.30
    lowpass_beta: float = 0.85
    composition_mode: str = "task_priority"
    joint_limit_margin: float = 0.05
    max_valid_error: float = 1.0
    use_nullspace: bool = False
    nullspace_gain: float = 0.1

    def validate(self) -> None:
        if self.frame != "root":
            raise ValueError(f"Only frame='root' is supported, got {self.frame!r}")
        if self.gate_mode not in ("linear", "none"):
            raise ValueError("gate_mode must be 'linear' or 'none'")
        if self.gate_mode == "linear" and self.gate_far <= self.gate_near:
            raise ValueError("gate_far must be larger than gate_near")
        if self.damping <= 0 or self.min_damping <= 0 or self.max_damping <= 0:
            raise ValueError("DLS damping values must be positive")
        if self.max_damping < self.min_damping:
            raise ValueError("max_damping must be >= min_damping")
        if self.singularity_threshold <= 0:
            raise ValueError("singularity_threshold must be positive")
        if self.max_task_velocity <= 0 or self.max_delta_q <= 0 or self.max_delta_action <= 0:
            raise ValueError("velocity and correction limits must be positive")
        if not 0.0 <= self.lowpass_beta < 1.0:
            raise ValueError("lowpass_beta must be in [0, 1)")
        if self.composition_mode not in ("residual", "task_priority"):
            raise ValueError("composition_mode must be 'residual' or 'task_priority'")
        if self.joint_limit_margin < 0:
            raise ValueError("joint_limit_margin must be non-negative")
        if self.max_valid_error <= 0:
            raise ValueError("max_valid_error must be positive")


def compute_linear_gate(error_norm: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """Return a gate of one near the target and zero far from it."""

    if far <= near:
        raise ValueError("gate_far must be larger than gate_near")
    return ((far - error_norm) / (far - near)).clamp_(0.0, 1.0)


def _damped_least_squares_with_status(
    jacobian: torch.Tensor,
    task_velocity: torch.Tensor,
    damping: torch.Tensor | float,
    eye: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``J^T (J J^T + lambda^2 I)^-1 v`` without an explicit inverse.

    Args:
        jacobian: Batched task Jacobian ``[B, M, N]``.
        task_velocity: Batched task command ``[B, M]``.
        damping: Scalar or per-environment damping ``[B]``.
    Returns:
        Joint correction ``[B, N]`` and failure mask ``[B]``. Failed/non-finite batches are zeroed.
    """

    if jacobian.ndim != 3:
        raise ValueError(f"jacobian must be [B, M, N], got {tuple(jacobian.shape)}")
    batch_size, task_dim, _ = jacobian.shape
    if task_velocity.shape != (batch_size, task_dim):
        raise ValueError(
            f"task_velocity must be {(batch_size, task_dim)}, got {tuple(task_velocity.shape)}"
        )
    if eye is None:
        eye = torch.eye(task_dim, device=jacobian.device, dtype=jacobian.dtype).expand(
            batch_size, task_dim, task_dim
        )
    elif eye.shape != (batch_size, task_dim, task_dim):
        raise ValueError(f"eye must be {(batch_size, task_dim, task_dim)}, got {tuple(eye.shape)}")
    if torch.is_tensor(damping):
        damping_sq = damping.to(device=jacobian.device, dtype=jacobian.dtype).square().reshape(batch_size, 1, 1)
    else:
        damping_sq = float(damping) ** 2
    lhs = jacobian @ jacobian.transpose(-1, -2) + damping_sq * eye
    solved, info = torch.linalg.solve_ex(lhs, task_velocity.unsqueeze(-1), check_errors=False)
    delta_q = (jacobian.transpose(-1, -2) @ solved).squeeze(-1)
    failed = info.ne(0) | ~torch.isfinite(delta_q).all(dim=-1)
    return torch.where(failed.unsqueeze(-1), torch.zeros_like(delta_q), delta_q), failed


def damped_least_squares(
    jacobian: torch.Tensor,
    task_velocity: torch.Tensor,
    damping: torch.Tensor | float,
) -> torch.Tensor:
    """Public DLS helper returning a finite joint correction ``[B, N]``."""

    delta_q, _ = _damped_least_squares_with_status(jacobian, task_velocity, damping)
    return delta_q


class HandJacobianController:
    """Stateful, GPU-batched right-arm DLS controller."""

    def __init__(
        self,
        cfg: HandJacobianControllerConfig,
        num_envs: int,
        device: torch.device | str,
        right_arm_dof_indices: torch.Tensor,
    ) -> None:
        cfg.validate()
        indices = torch.as_tensor(right_arm_dof_indices, device=device, dtype=torch.long)
        if indices.shape != (7,) or torch.unique(indices).numel() != 7:
            raise ValueError(f"right_arm_dof_indices must contain 7 unique indices, got {indices.tolist()}")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.right_arm_dof_indices = indices
        self._max_action_index = int(indices.max())
        self.prev_delta_q = torch.zeros(self.num_envs, 7, device=self.device)
        self._task_eye = torch.eye(3, device=self.device).expand(self.num_envs, 3, 3)
        self._joint_eye = torch.eye(7, device=self.device).expand(self.num_envs, 7, 7)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear low-pass state for all or selected environments."""

        if env_ids is None:
            self.prev_delta_q.zero_()
            return
        ids_or_mask = torch.as_tensor(env_ids, device=self.device)
        if ids_or_mask.dtype == torch.bool:
            if ids_or_mask.shape != (self.num_envs,):
                raise ValueError(f"boolean reset mask must be [B], got {tuple(ids_or_mask.shape)}")
            self.prev_delta_q.masked_fill_(ids_or_mask[:, None], 0.0)
        else:
            self.prev_delta_q[ids_or_mask.long()] = 0.0

    @staticmethod
    def _as_batch_7(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if value.shape == (7,):
            return value.unsqueeze(0).expand(batch_size, -1)
        if value.shape == (batch_size, 7):
            return value
        raise ValueError(f"{name} must be [7] or [B, 7], got {tuple(value.shape)}")

    @torch.no_grad()
    def compute(
        self,
        *,
        action_bfm: torch.Tensor,
        current_wrist_pos_root: torch.Tensor,
        target_wrist_pos_root: torch.Tensor,
        wrist_linear_vel_root: torch.Tensor,
        jacobian_pos_root: torch.Tensor,
        current_right_arm_q: torch.Tensor,
        default_right_arm_q: torch.Tensor,
        lower_joint_limits: torch.Tensor,
        upper_joint_limits: torch.Tensor,
        action_scale: torch.Tensor,
        action_lower: torch.Tensor,
        action_upper: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply the residual to an actor action.

        All Cartesian quantities are in the pelvis/root frame. ``action_scale``
        is radians per *normalized actor action*, including any normalization
        and per-joint action-rescale factors used by the environment.
        """

        batch_size, action_dim = action_bfm.shape
        if batch_size != self.num_envs:
            raise ValueError(f"controller has B={self.num_envs}, action has B={batch_size}")
        expected_3 = (batch_size, 3)
        for name, value in (
            ("current_wrist_pos_root", current_wrist_pos_root),
            ("target_wrist_pos_root", target_wrist_pos_root),
            ("wrist_linear_vel_root", wrist_linear_vel_root),
        ):
            if value.shape != expected_3:
                raise ValueError(f"{name} must be {expected_3}, got {tuple(value.shape)}")
        if jacobian_pos_root.shape != (batch_size, 3, 7):
            raise ValueError(f"jacobian_pos_root must be [B, 3, 7], got {tuple(jacobian_pos_root.shape)}")
        if current_right_arm_q.shape != (batch_size, 7):
            raise ValueError(f"current_right_arm_q must be [B, 7], got {tuple(current_right_arm_q.shape)}")
        if self._max_action_index >= action_dim:
            raise ValueError("right-arm index is outside actor action dimension")

        default_q = self._as_batch_7(default_right_arm_q, batch_size, "default_right_arm_q")
        lower = self._as_batch_7(lower_joint_limits, batch_size, "lower_joint_limits")
        upper = self._as_batch_7(upper_joint_limits, batch_size, "upper_joint_limits")
        scale = self._as_batch_7(action_scale, batch_size, "action_scale")
        action_lo = self._as_batch_7(action_lower, batch_size, "action_lower")
        action_hi = self._as_batch_7(action_upper, batch_size, "action_upper")
        hand_action_bfm = action_bfm.index_select(-1, self.right_arm_dof_indices)
        safe_lower = lower + self.cfg.joint_limit_margin
        safe_upper = upper - self.cfg.joint_limit_margin
        valid_scale = torch.isfinite(scale).all(dim=-1) & scale.abs().ge(1e-8).all(dim=-1)
        valid_joint_range = (
            torch.isfinite(safe_lower).all(dim=-1)
            & torch.isfinite(safe_upper).all(dim=-1)
            & safe_lower.lt(safe_upper).all(dim=-1)
        )
        safe_scale = torch.where(valid_scale[:, None], scale, torch.ones_like(scale))

        position_error = target_wrist_pos_root - current_wrist_pos_root
        error_norm = torch.linalg.vector_norm(position_error, dim=-1, keepdim=True)
        finite_input = (
            torch.isfinite(jacobian_pos_root).all(dim=(-1, -2))
            & torch.isfinite(position_error).all(dim=-1)
            & torch.isfinite(wrist_linear_vel_root).all(dim=-1)
            & torch.isfinite(current_right_arm_q).all(dim=-1)
            & torch.isfinite(default_q).all(dim=-1)
            & torch.isfinite(hand_action_bfm).all(dim=-1)
            & torch.isfinite(action_lo).all(dim=-1)
            & torch.isfinite(action_hi).all(dim=-1)
            & action_lo.le(action_hi).all(dim=-1)
            & valid_scale
            & valid_joint_range
        )
        valid = finite_input & (torch.nan_to_num(error_norm.squeeze(-1), nan=float("inf")) < self.cfg.max_valid_error)
        if valid_mask is not None:
            if valid_mask.shape != (batch_size,):
                raise ValueError(f"valid_mask must be [B], got {tuple(valid_mask.shape)}")
            valid &= valid_mask.bool()

        safe_jacobian = torch.where(valid[:, None, None], jacobian_pos_root, torch.zeros_like(jacobian_pos_root))
        safe_error = torch.where(valid[:, None], position_error, torch.zeros_like(position_error))
        safe_velocity = torch.where(valid[:, None], wrist_linear_vel_root, torch.zeros_like(wrist_linear_vel_root))
        task_velocity = self.cfg.kp_position * safe_error - self.cfg.kd_position * safe_velocity
        velocity_norm = torch.linalg.vector_norm(task_velocity, dim=-1, keepdim=True)
        task_velocity *= (self.cfg.max_task_velocity / velocity_norm.clamp_min(1e-8)).clamp_max(1.0)

        sigma_min = torch.full((batch_size,), float("nan"), device=action_bfm.device, dtype=action_bfm.dtype)
        if self.cfg.adaptive_damping:
            sigma_min = torch.linalg.svdvals(safe_jacobian)[..., -1]
            ratio = ((self.cfg.singularity_threshold - sigma_min) / self.cfg.singularity_threshold).clamp(0.0, 1.0)
            damping: torch.Tensor | float = self.cfg.min_damping + ratio * (
                self.cfg.max_damping - self.cfg.min_damping
            )
        else:
            damping = self.cfg.damping

        delta_q_raw, solve_failed = _damped_least_squares_with_status(
            safe_jacobian, task_velocity, damping, self._task_eye
        )
        dls_failed = valid & solve_failed
        delta_q_raw = torch.nan_to_num(delta_q_raw, nan=0.0, posinf=0.0, neginf=0.0)

        q_target_bfm = default_q + hand_action_bfm * safe_scale
        nullspace_delta = torch.zeros_like(delta_q_raw)
        if self.cfg.use_nullspace:
            if torch.is_tensor(damping):
                damping_sq = damping.square().reshape(batch_size, 1, 1)
            else:
                damping_sq = float(damping) ** 2
            pinv = safe_jacobian.transpose(-1, -2) @ torch.linalg.solve(
                safe_jacobian @ safe_jacobian.transpose(-1, -2) + damping_sq * self._task_eye,
                self._task_eye,
            )
            null_projector = self._joint_eye - pinv @ safe_jacobian
            posture_error = self.cfg.nullspace_gain * (q_target_bfm - current_right_arm_q)
            nullspace_delta = (null_projector @ posture_error.unsqueeze(-1)).squeeze(-1)

        delta_q_clipped = delta_q_raw.clamp(-self.cfg.max_delta_q, self.cfg.max_delta_q)
        delta_q_filtered = self.cfg.lowpass_beta * self.prev_delta_q + (1.0 - self.cfg.lowpass_beta) * delta_q_clipped
        if self.cfg.gate_mode == "linear":
            gate = compute_linear_gate(error_norm, self.cfg.gate_near, self.cfg.gate_far)
        else:
            gate = torch.ones_like(error_norm)
        gate = torch.where(valid[:, None], gate, torch.zeros_like(gate))
        self.prev_delta_q.copy_(torch.where(valid[:, None], delta_q_filtered, torch.zeros_like(delta_q_filtered)))

        if self.cfg.composition_mode == "task_priority":
            # Differential IK is a desired joint target relative to the current
            # configuration. Blending this target with BFM lets repeated steps
            # converge instead of applying the same small instantaneous bias to
            # a freshly regenerated BFM target on every control step.
            q_target_ik = current_right_arm_q + delta_q_filtered + nullspace_delta
            joint_correction = gate * (q_target_ik - q_target_bfm)
        else:
            joint_correction = gate * (delta_q_filtered + nullspace_delta)
        delta_action = (joint_correction / safe_scale).clamp(
            -self.cfg.max_delta_action, self.cfg.max_delta_action
        )
        q_target_unprojected = q_target_bfm + delta_action * safe_scale
        q_target_final = torch.maximum(torch.minimum(q_target_unprojected, safe_upper), safe_lower)
        projected = (q_target_final - q_target_unprojected).abs().gt(1e-7).any(dim=-1)
        hand_action_candidate = ((q_target_final - default_q) / safe_scale).clamp(action_lo, action_hi)
        hand_action_final = torch.where(valid[:, None], hand_action_candidate, hand_action_bfm)
        projected &= valid

        action_final = action_bfm.clone()
        action_final[:, self.right_arm_dof_indices] = hand_action_final
        metrics = {
            "wrist_error": torch.nan_to_num(error_norm.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0),
            "wrist_error_xyz": torch.nan_to_num(
                position_error, nan=0.0, posinf=0.0, neginf=0.0
            ),
            "jacobian_gate": gate.squeeze(-1),
            "jacobian_active": gate.squeeze(-1).gt(0.0),
            "delta_q_raw_norm": torch.linalg.vector_norm(delta_q_raw, dim=-1),
            "delta_q_filtered_norm": torch.linalg.vector_norm(delta_q_filtered, dim=-1),
            "delta_action_norm": torch.linalg.vector_norm(delta_action, dim=-1),
            "joint_target_correction_norm": torch.linalg.vector_norm(
                delta_action * safe_scale, dim=-1
            ),
            "delta_action_saturated": delta_action.abs().ge(
                self.cfg.max_delta_action - 1e-7
            ).any(dim=-1),
            "joint_limit_projected": projected,
            "invalid_input": ~valid,
            "dls_failure": dls_failed,
            "sigma_min": sigma_min,
        }
        return action_final, metrics
