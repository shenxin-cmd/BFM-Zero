import os

import numpy as np
import pytest
import torch


def _stage4_env_smoke_enabled() -> bool:
    return os.environ.get("RUN_STAGE4_ISAAC_SMOKE", "0") == "1"


@pytest.mark.skipif(not _stage4_env_smoke_enabled(), reason="Set RUN_STAGE4_ISAAC_SMOKE=1 to run IsaacSim smoke")
def test_stage4_isaac_env_wrist_lock_smoke():
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

    num_envs = int(os.environ.get("STAGE4_SMOKE_NUM_ENVS", "4"))
    steps = int(os.environ.get("STAGE4_SMOKE_STEPS", "100"))
    wrist_abs_limit = float(os.environ.get("STAGE4_SMOKE_WRIST_ABS_LIMIT", "1.0"))
    lafan_tail_path = os.environ.get("STAGE4_SMOKE_MOTION_FILE", "humanoidverse/data/lafan_29dof_10s-clipped.pkl")

    cfg = HumanoidVerseIsaacConfig(
        name="humanoidverse_isaac",
        device=os.environ.get("STAGE4_SMOKE_DEVICE", "cuda:0"),
        lafan_tail_path=lafan_tail_path,
        enable_cameras=False,
        max_episode_length_s=2.0,
        disable_obs_noise=True,
        disable_domain_randomization=True,
        relative_config_path="exp/bfm_zero/bfm_zero",
        include_last_action=True,
        include_history_actor=True,
        root_height_obs=True,
        hydra_overrides=[
            "robot=g1/g1_29dof_hard_waist",
            "robot.control.action_scale=0.25",
            "robot.control.action_clip_value=5.0",
            "robot.control.normalize_action_to=5.0",
            "env.config.lie_down_init=False",
            "env.config.lie_down_init_prob=0.0",
        ],
    )

    env = None
    try:
        env, _ = cfg.build(num_envs=num_envs)
        base_env = env.unwrapped
        base_env.config.stage4 = {
            "hand_control_mode": "task_space_4dof",
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
        obs, info = env.reset()
        assert "state" in obs

        wrist_indices = torch.tensor([26, 27, 28], device=base_env.device)
        for _ in range(steps):
            action = torch.randn(num_envs, env.single_action_space.shape[0], device=base_env.device)
            action[:, wrist_indices] = 5.0
            obs, reward, terminated, truncated, info = env.step(action)
            reward_tensor = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(np.asarray(reward))
            assert torch.isfinite(reward_tensor).all()
            assert torch.isfinite(base_env.simulator.dof_pos).all()
            assert torch.all(base_env.actions[:, wrist_indices] == 0.0)

        wrist_abs_max = base_env.simulator.dof_pos[:, wrist_indices].abs().max().item()
        assert wrist_abs_max < wrist_abs_limit, f"wrist_abs_max={wrist_abs_max:.4f} exceeds {wrist_abs_limit}"
    finally:
        if env is not None:
            env.close()
