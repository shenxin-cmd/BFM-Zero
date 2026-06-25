import os

import numpy as np
import pytest
import torch


def _stage4_env_smoke_enabled() -> bool:
    return os.environ.get("RUN_STAGE4_ISAAC_SMOKE", "0") == "1"


def _stage4_isaac_fd_enabled() -> bool:
    return os.environ.get("RUN_STAGE4_ISAAC_FD_JACOBIAN", "0") == "1"


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


@pytest.mark.skipif(not _stage4_env_smoke_enabled(), reason="Set RUN_STAGE4_ISAAC_SMOKE=1 to run IsaacSim smoke")
def test_stage4_isaac_env_reset_zeroes_wrist_state_and_default_offsets():
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.agents.stage4 import Stage4Config, resolve_right_arm_joint_indices

    num_envs = int(os.environ.get("STAGE4_SMOKE_NUM_ENVS", "2"))
    atol = float(os.environ.get("STAGE4_RESET_WRIST_ATOL", "1e-6"))
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
        stage4_cfg = Stage4Config(hand_control_mode="task_space_4dof")
        base_env.config.stage4 = {
            "hand_control_mode": "task_space_4dof",
            "active_right_arm_joint_names": stage4_cfg.active_right_arm_joint_names,
            "locked_wrist_joint_names": stage4_cfg.locked_wrist_joint_names,
            "enforce_wrist_zero_before_env_step": True,
            "enforce_wrist_zero_after_pd_target_build": True,
            "wrist_absolute_target": 0.0,
        }
        base_env.config.domain_rand.randomize_default_dof_pos = True
        base_env.config.domain_rand.default_dof_pos_noise_range = (0.5, 0.5)

        env.reset()

        indices = resolve_right_arm_joint_indices(
            dof_names=base_env.simulator.dof_names,
            active_joint_names=stage4_cfg.active_right_arm_joint_names,
            wrist_joint_names=stage4_cfg.locked_wrist_joint_names,
        )
        wrist_idx = torch.tensor(indices.wrist_dof_indices, device=base_env.device, dtype=torch.long)
        non_wrist_idx = torch.tensor(
            [idx for idx in range(base_env.num_dof) if idx not in indices.wrist_dof_indices],
            device=base_env.device,
            dtype=torch.long,
        )

        wrist_q_abs_max = base_env.simulator.dof_pos[:, wrist_idx].abs().max().item()
        wrist_dq_abs_max = base_env.simulator.dof_vel[:, wrist_idx].abs().max().item()
        wrist_offset_abs_max = base_env.default_dof_pos_offset[:, wrist_idx].abs().max().item()
        non_wrist_offset_mean = base_env.default_dof_pos_offset[:, non_wrist_idx].mean().item()
        print(
            "reset wrist q/dq/offset abs max = "
            f"{wrist_q_abs_max:.6e}, {wrist_dq_abs_max:.6e}, {wrist_offset_abs_max:.6e}; "
            f"non_wrist_offset_mean={non_wrist_offset_mean:.6f}"
        )

        assert wrist_q_abs_max <= atol
        assert wrist_dq_abs_max <= atol
        assert wrist_offset_abs_max <= atol
        assert torch.allclose(
            base_env.default_dof_pos_offset[:, non_wrist_idx],
            torch.full_like(base_env.default_dof_pos_offset[:, non_wrist_idx], 0.5),
        )
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(
    not (_stage4_env_smoke_enabled() and _stage4_isaac_fd_enabled()),
    reason="Set RUN_STAGE4_ISAAC_SMOKE=1 and RUN_STAGE4_ISAAC_FD_JACOBIAN=1 to run real IsaacSim Jacobian finite differences",
)
def test_stage4_isaacsim_multi_pose_active_arm_jacobian_finite_difference():
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.agents.stage4 import (
        Stage4Config,
        get_isaacsim_root_physx_jacobians,
        resolve_body_index,
        resolve_isaacsim_physx_body_index,
        resolve_isaacsim_physx_dof_indices,
        resolve_right_arm_joint_indices,
        select_isaacsim_active_position_jacobian,
    )

    num_envs = 1
    eps = float(os.environ.get("STAGE4_FD_EPS", "0.001"))
    atol = float(os.environ.get("STAGE4_FD_ATOL", "0.08"))
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
        stage4_cfg = Stage4Config()
        base_env.config.stage4 = {
            "hand_control_mode": "task_space_4dof",
            "end_effector_body_name": "right_wrist_yaw_link",
            "enforce_wrist_zero_before_env_step": True,
            "enforce_wrist_zero_after_pd_target_build": True,
            "wrist_absolute_target": 0.0,
        }
        env.reset()

        indices = resolve_right_arm_joint_indices(
            dof_names=base_env.simulator.dof_names,
            active_joint_names=stage4_cfg.active_right_arm_joint_names,
            wrist_joint_names=stage4_cfg.locked_wrist_joint_names,
        )
        active_idx = torch.tensor(indices.active_dof_indices, device=base_env.device, dtype=torch.long)
        wrist_idx = torch.tensor(indices.wrist_dof_indices, device=base_env.device, dtype=torch.long)
        body_index = resolve_body_index(base_env.simulator.body_names, stage4_cfg.end_effector_body_name)
        physx_body_index = resolve_isaacsim_physx_body_index(base_env.simulator, body_index)
        physx_active_dof_indices = resolve_isaacsim_physx_dof_indices(base_env.simulator, indices.active_dof_indices)
        env_ids = torch.arange(num_envs, device=base_env.device)

        def sync_pose(active_q: torch.Tensor) -> torch.Tensor:
            dof_state = base_env.simulator.dof_state.clone()
            dof_state[:, :, 1] = 0.0
            dof_state[:, active_idx, 0] = active_q.to(device=base_env.device, dtype=dof_state.dtype)
            dof_state[:, wrist_idx, 0] = 0.0
            dof_state[:, wrist_idx, 1] = 0.0
            base_env.simulator.set_dof_state_tensor(env_ids, dof_state)
            base_env.simulator.scene.update(dt=0.0)
            return base_env.simulator._rigid_body_pos[:, body_index].clone()

        default_active = base_env.default_dof_pos[:, active_idx].clone().to(base_env.device)
        pose_specs = {
            "default": default_active,
            "elbow_bent": default_active + torch.tensor([[0.0, 0.0, 0.0, 0.45]], device=base_env.device),
            "shoulder_forward": default_active + torch.tensor([[0.35, 0.0, 0.0, 0.20]], device=base_env.device),
            "random_safe": default_active + torch.tensor([[0.18, -0.16, 0.14, 0.32]], device=base_env.device),
        }

        for pose_name, active_q in pose_specs.items():
            sync_pose(active_q)
            jacobians = get_isaacsim_root_physx_jacobians(base_env.simulator)
            physx_jac = select_isaacsim_active_position_jacobian(
                jacobians,
                body_index=physx_body_index,
                active_dof_indices=physx_active_dof_indices,
                num_dofs=base_env.num_dof,
            )

            fd_columns = []
            for col in range(active_idx.numel()):
                delta = torch.zeros_like(active_q)
                delta[:, col] = eps
                pos_plus = sync_pose(active_q + delta)
                pos_minus = sync_pose(active_q - delta)
                fd_columns.append(((pos_plus - pos_minus) / (2.0 * eps)).unsqueeze(-1))
            fd_jac = torch.cat(fd_columns, dim=-1)
            sync_pose(active_q)

            max_abs_err = (physx_jac - fd_jac).abs().max().item()
            sigma_min = torch.linalg.svdvals(physx_jac).amin().item()
            print(f"{pose_name}: fd_max_abs_err={max_abs_err:.6f}, sigma_min={sigma_min:.6e}")
            assert torch.isfinite(physx_jac).all()
            assert torch.isfinite(fd_jac).all()
            assert max_abs_err < atol, f"{pose_name} fd_max_abs_err={max_abs_err:.6f} >= {atol}"
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _stage4_env_smoke_enabled(), reason="Set RUN_STAGE4_ISAAC_SMOKE=1 to run IsaacSim smoke")
def test_stage4_isaac_env_dls_action_assembly_smoke():
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.agents.stage4 import HandTaskCommand, build_stage4_env_snapshot, dls_hand_action_from_snapshot

    num_envs = int(os.environ.get("STAGE4_SMOKE_NUM_ENVS", "2"))
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
            "end_effector_body_name": "right_wrist_yaw_link",
            "enforce_wrist_zero_before_env_step": True,
            "enforce_wrist_zero_after_pd_target_build": True,
            "wrist_absolute_target": 0.0,
        }
        obs, info = env.reset()
        assert "state" in obs

        snapshot = build_stage4_env_snapshot(base_env)
        body_action = torch.zeros(num_envs, 22, device=base_env.device)
        target_offset = torch.zeros(num_envs, 3, device=base_env.device)
        target_offset[:, 0] = 0.01
        command = HandTaskCommand(
            target_pos_root=snapshot.end_effector_pos_heading + target_offset,
            target_lin_vel_root=torch.zeros(num_envs, 3, device=base_env.device),
            position_mask=torch.ones(num_envs, 1, device=base_env.device),
            velocity_mask=torch.zeros(num_envs, 1, device=base_env.device),
            command_id=torch.zeros(num_envs, dtype=torch.long, device=base_env.device),
            command_done=torch.zeros(num_envs, 1, dtype=torch.bool, device=base_env.device),
        )

        out = dls_hand_action_from_snapshot(
            body_action=body_action,
            snapshot=snapshot,
            command=command,
            action_dim=env.single_action_space.shape[0],
        )

        wrist_indices = torch.tensor([26, 27, 28], device=base_env.device)
        assert out.full_action.shape == (num_envs, 29)
        assert out.active_hand_action.shape == (num_envs, 4)
        assert torch.isfinite(out.active_joint_delta).all()
        assert torch.all(out.full_action[:, wrist_indices] == 0.0)

        obs, reward, terminated, truncated, info = env.step(out.full_action)
        reward_tensor = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(np.asarray(reward))
        assert torch.isfinite(reward_tensor).all()
        assert torch.isfinite(base_env.simulator.dof_pos).all()
        assert torch.all(base_env.actions[:, wrist_indices] == 0.0)
    finally:
        if env is not None:
            env.close()
