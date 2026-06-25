from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .actions import RightArmJointIndices, resolve_right_arm_joint_indices
from .kinematics import (
    get_isaacsim_root_physx_jacobians,
    resolve_body_index,
    rotate_position_jacobian_world_to_heading,
    select_isaacsim_active_position_jacobian,
    world_to_heading_frame,
)


@dataclass(frozen=True)
class Stage4EnvSnapshot:
    indices: RightArmJointIndices
    end_effector_body_index: int
    root_pos_world: torch.Tensor
    root_quat_xyzw: torch.Tensor
    end_effector_pos_world: torch.Tensor
    end_effector_pos_heading: torch.Tensor
    active_q: torch.Tensor
    active_lower: torch.Tensor
    active_upper: torch.Tensor
    default_active_joint_pos: torch.Tensor
    action_scale: float
    active_position_jacobian_world: torch.Tensor
    active_position_jacobian_heading: torch.Tensor


def _cfg_value(stage4_cfg: Any, key: str, default: Any = None) -> Any:
    if stage4_cfg is None:
        return default
    if isinstance(stage4_cfg, dict):
        return stage4_cfg.get(key, default)
    return getattr(stage4_cfg, key, default)


def _as_action_scale(action_scale: Any) -> float:
    if isinstance(action_scale, torch.Tensor):
        if action_scale.numel() != 1:
            raise ValueError("Stage 4 currently expects scalar robot.control.action_scale")
        return float(action_scale.item())
    return float(action_scale)


def build_stage4_env_snapshot(base_env, stage4_cfg: Any = None) -> Stage4EnvSnapshot:
    """Read the current env tensors needed by the Stage 4 DLS hand controller."""

    if stage4_cfg is None and hasattr(base_env.config, "get"):
        stage4_cfg = base_env.config.get("stage4", None)
    active_names = tuple(
        _cfg_value(
            stage4_cfg,
            "active_right_arm_joint_names",
            (
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
            ),
        )
    )
    wrist_names = tuple(
        _cfg_value(
            stage4_cfg,
            "locked_wrist_joint_names",
            (
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ),
        )
    )
    end_effector_body_name = _cfg_value(stage4_cfg, "end_effector_body_name", "right_wrist_yaw_link")

    indices = resolve_right_arm_joint_indices(
        dof_names=base_env.dof_names,
        active_joint_names=active_names,
        wrist_joint_names=wrist_names,
    )
    body_index = resolve_body_index(base_env.body_names, end_effector_body_name)

    simulator = base_env.simulator
    root_states = simulator.robot_root_states
    root_pos_world = root_states[:, :3]
    root_quat_xyzw = root_states[:, 3:7]
    end_effector_pos_world = simulator._rigid_body_pos[:, body_index]
    end_effector_pos_heading = world_to_heading_frame(
        end_effector_pos_world,
        root_pos_world=root_pos_world,
        root_quat_xyzw=root_quat_xyzw,
    )

    active_idx = torch.tensor(indices.active_dof_indices, device=simulator.dof_pos.device, dtype=torch.long)
    active_q = simulator.dof_pos.index_select(-1, active_idx)
    active_limits = base_env.dof_pos_limits.index_select(0, active_idx)
    active_lower = active_limits[:, 0].unsqueeze(0).expand_as(active_q)
    active_upper = active_limits[:, 1].unsqueeze(0).expand_as(active_q)
    default_active_joint_pos = base_env.default_dof_pos.index_select(-1, active_idx)

    jacobians = get_isaacsim_root_physx_jacobians(simulator)
    active_position_jacobian_world = select_isaacsim_active_position_jacobian(
        jacobians,
        body_index=body_index,
        active_dof_indices=indices.active_dof_indices,
        num_dofs=len(base_env.dof_names),
    )
    active_position_jacobian_heading = rotate_position_jacobian_world_to_heading(
        active_position_jacobian_world,
        root_quat_xyzw=root_quat_xyzw,
    )

    return Stage4EnvSnapshot(
        indices=indices,
        end_effector_body_index=body_index,
        root_pos_world=root_pos_world,
        root_quat_xyzw=root_quat_xyzw,
        end_effector_pos_world=end_effector_pos_world,
        end_effector_pos_heading=end_effector_pos_heading,
        active_q=active_q,
        active_lower=active_lower,
        active_upper=active_upper,
        default_active_joint_pos=default_active_joint_pos,
        action_scale=_as_action_scale(base_env.config.robot.control.action_scale),
        active_position_jacobian_world=active_position_jacobian_world,
        active_position_jacobian_heading=active_position_jacobian_heading,
    )
