from typing import Literal

import pydantic

from humanoidverse.agents.base import BaseConfig

HandControlMode = Literal["legacy_fb", "legacy_mse", "task_space_4dof"]


class Stage4Config(BaseConfig):
    """Configuration guardrails for mutually exclusive hand-control modes.

    This config is intentionally not wired into the training entry yet. Commit 1
    only makes the modes explicit so later commits can add behavior without
    silently mixing legacy hand losses with task-space control.
    """

    name: Literal["Stage4Config"] = "Stage4Config"
    hand_control_mode: HandControlMode = "legacy_mse"

    active_right_arm_joint_names: tuple[str, str, str, str] = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    )
    locked_wrist_joint_names: tuple[str, str, str] = (
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    wrist_absolute_target: float = 0.0
    enforce_wrist_zero_before_env_step: bool = True
    enforce_wrist_zero_after_pd_target_build: bool = True

    legacy_hand_mse_enabled: bool = True
    legacy_hand_fb_enabled: bool = False
    task_space_hand_enabled: bool = False
    z_hand_enabled: bool = True
    b_hand_enabled: bool = True
    f_hand_enabled: bool = True

    @pydantic.model_validator(mode="after")
    def validate_mutually_exclusive_mode(self) -> "Stage4Config":
        if self.hand_control_mode == "legacy_mse":
            expected = {
                "legacy_hand_mse_enabled": True,
                "legacy_hand_fb_enabled": False,
                "task_space_hand_enabled": False,
                "z_hand_enabled": True,
                "b_hand_enabled": True,
                "f_hand_enabled": True,
            }
        elif self.hand_control_mode == "legacy_fb":
            expected = {
                "legacy_hand_mse_enabled": False,
                "legacy_hand_fb_enabled": True,
                "task_space_hand_enabled": False,
                "z_hand_enabled": True,
                "b_hand_enabled": True,
                "f_hand_enabled": True,
            }
        else:
            expected = {
                "legacy_hand_mse_enabled": False,
                "legacy_hand_fb_enabled": False,
                "task_space_hand_enabled": True,
                "z_hand_enabled": False,
                "b_hand_enabled": False,
                "f_hand_enabled": False,
            }

        mismatches = {
            key: (getattr(self, key), value)
            for key, value in expected.items()
            if getattr(self, key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}=actual:{actual}/expected:{expected_value}"
                for key, (actual, expected_value) in sorted(mismatches.items())
            )
            raise ValueError(f"Invalid Stage4Config for hand_control_mode={self.hand_control_mode}: {details}")

        active = set(self.active_right_arm_joint_names)
        wrist = set(self.locked_wrist_joint_names)
        if len(active) != 4 or len(wrist) != 3:
            raise ValueError("Stage 4 requires exactly 4 active right-arm joints and 3 locked wrist joints")
        overlap = active & wrist
        if overlap:
            raise ValueError(f"Active and locked wrist joint names overlap: {sorted(overlap)}")
        if self.wrist_absolute_target != 0.0:
            raise ValueError("Stage 4 currently requires wrist_absolute_target == 0.0")
        return self
