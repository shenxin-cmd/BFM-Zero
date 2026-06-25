from types import SimpleNamespace

import torch

from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase

G1_29DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class _Config(dict):
    def __getattr__(self, item):
        return self[item]


def _make_env_with_stage4(enabled: bool = True):
    env = object.__new__(LeggedRobotBase)
    env.dof_names = list(G1_29DOF_NAMES)
    env._stage4_right_arm_indices = None
    env.config = _Config(
        stage4={
            "hand_control_mode": "task_space_4dof" if enabled else "legacy_mse",
            "active_right_arm_joint_names": (
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
            ),
            "locked_wrist_joint_names": (
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ),
            "enforce_wrist_zero_before_env_step": True,
            "enforce_wrist_zero_after_pd_target_build": True,
            "wrist_absolute_target": 0.0,
        }
    )
    env.device = "cpu"
    env.simulator = SimpleNamespace(dof_ids=list(range(29)))
    env.num_dof = 29
    env.num_dofs = 29
    return env


def test_env_pre_step_hook_zeroes_wrist_actions_only_in_stage4_mode():
    env = _make_env_with_stage4(enabled=True)
    action = torch.ones(2, 29)

    locked = env._stage4_zero_wrist_actions_before_env_step(action)

    assert torch.all(locked[:, [26, 27, 28]] == 0.0)
    assert torch.all(locked[:, :26] == 1.0)


def test_env_pd_target_hook_clamps_absolute_wrist_target_to_zero():
    env = _make_env_with_stage4(enabled=True)
    target = torch.full((2, 29), 0.25)

    clamped = env._stage4_clamp_wrist_pd_target(target)

    assert torch.all(clamped[:, [26, 27, 28]] == 0.0)
    assert torch.all(clamped[:, :26] == 0.25)


def test_env_stage4_hooks_are_noops_in_legacy_mode():
    env = _make_env_with_stage4(enabled=False)
    action = torch.ones(2, 29)
    target = torch.full((2, 29), 0.25)

    assert torch.all(env._stage4_zero_wrist_actions_before_env_step(action) == action)
    assert torch.all(env._stage4_clamp_wrist_pd_target(target) == target)


def test_stage4_default_reset_hard_locks_wrist_q_and_dq_to_zero():
    env = _make_env_with_stage4(enabled=True)
    env.default_dof_pos = torch.full((1, 29), 0.2)
    env.default_dof_pos_offset = torch.full((3, 29), 0.3)
    env.target_robot_dof_state = torch.empty(3, 29, 2)

    env._reset_dofs(torch.tensor([0, 2]))

    assert torch.all(env.target_robot_dof_state[[0, 2]][:, [26, 27, 28], 0] == 0.0)
    assert torch.all(env.target_robot_dof_state[[0, 2]][:, [26, 27, 28], 1] == 0.0)
    assert torch.all(env.target_robot_dof_state[[0, 2]][:, 25, 0] != 0.0)


def test_stage4_target_state_reset_hard_locks_wrist_q_and_dq_to_zero():
    env = _make_env_with_stage4(enabled=True)
    env.target_robot_dof_state = torch.empty(2, 29, 2)
    target_state = torch.ones(2, 29, 2)
    target_state[:, [26, 27, 28], 0] = 0.7
    target_state[:, [26, 27, 28], 1] = -0.4

    env._reset_dofs(torch.tensor([0, 1]), target_state=target_state)

    assert torch.all(env.target_robot_dof_state[:, [26, 27, 28], 0] == 0.0)
    assert torch.all(env.target_robot_dof_state[:, [26, 27, 28], 1] == 0.0)
    assert torch.all(env.target_robot_dof_state[:, 25, 0] == 1.0)


def test_stage4_domain_randomization_keeps_wrist_default_offsets_zero():
    env = _make_env_with_stage4(enabled=True)
    env.default_dof_pos_offset = torch.zeros(3, 29)
    env.config.domain_rand = _Config(
        randomize_pd_gain=False,
        randomize_rfi_lim=False,
        randomize_ctrl_delay=False,
        randomize_default_dof_pos=True,
        default_dof_pos_noise_range=(0.5, 0.5),
    )

    env._episodic_domain_randomization(torch.tensor([0, 1, 2]))

    assert torch.all(env.default_dof_pos_offset[:, [26, 27, 28]] == 0.0)
    assert torch.all(env.default_dof_pos_offset[:, 25] == 0.5)


def test_legacy_domain_randomization_keeps_existing_wrist_offset_behavior():
    env = _make_env_with_stage4(enabled=False)
    env.default_dof_pos_offset = torch.zeros(1, 29)
    env.config.domain_rand = _Config(
        randomize_pd_gain=False,
        randomize_rfi_lim=False,
        randomize_ctrl_delay=False,
        randomize_default_dof_pos=True,
        default_dof_pos_noise_range=(0.5, 0.5),
    )

    env._episodic_domain_randomization(torch.tensor([0]))

    assert torch.all(env.default_dof_pos_offset[:, [26, 27, 28]] == 0.5)
