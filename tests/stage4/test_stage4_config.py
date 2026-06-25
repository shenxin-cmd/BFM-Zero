import pytest

from humanoidverse.agents.stage4 import Stage4Config


def test_default_stage4_config_keeps_legacy_mse_mode():
    cfg = Stage4Config()

    assert cfg.hand_control_mode == "legacy_mse"
    assert cfg.legacy_hand_mse_enabled
    assert not cfg.legacy_hand_fb_enabled
    assert not cfg.task_space_hand_enabled
    assert cfg.z_hand_enabled
    assert cfg.b_hand_enabled
    assert cfg.f_hand_enabled


def test_task_space_mode_disables_legacy_hand_latents_and_losses():
    cfg = Stage4Config(
        hand_control_mode="task_space_4dof",
        legacy_hand_mse_enabled=False,
        legacy_hand_fb_enabled=False,
        task_space_hand_enabled=True,
        z_hand_enabled=False,
        b_hand_enabled=False,
        f_hand_enabled=False,
    )

    assert cfg.hand_control_mode == "task_space_4dof"
    assert len(cfg.active_right_arm_joint_names) == 4
    assert len(cfg.locked_wrist_joint_names) == 3
    assert set(cfg.active_right_arm_joint_names).isdisjoint(cfg.locked_wrist_joint_names)


def test_stage4_config_rejects_mixed_legacy_and_task_space_modes():
    with pytest.raises(ValueError, match="Invalid Stage4Config"):
        Stage4Config(
            hand_control_mode="task_space_4dof",
            legacy_hand_mse_enabled=True,
            task_space_hand_enabled=True,
            z_hand_enabled=False,
            b_hand_enabled=False,
            f_hand_enabled=False,
        )


def test_stage4_config_rejects_active_wrist_overlap():
    with pytest.raises(ValueError, match="overlap"):
        Stage4Config(
            active_right_arm_joint_names=(
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_wrist_roll_joint",
            ),
        )
