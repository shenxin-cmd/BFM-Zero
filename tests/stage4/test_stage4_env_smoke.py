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
        stage4_cfg = Stage4Config()
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
            target_pos_heading=snapshot.end_effector_pos_heading + target_offset,
            target_lin_vel_heading=torch.zeros(num_envs, 3, device=base_env.device),
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


@pytest.mark.skipif(not _stage4_env_smoke_enabled(), reason="Set RUN_STAGE4_ISAAC_SMOKE=1 to run IsaacSim smoke")
def test_stage4_isaac_env_dls_limiter_rollout_smoke():
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.agents.stage4 import (
        HandTaskCommand,
        JointCommandLimiter,
        build_stage4_env_snapshot,
        dls_hand_action_from_snapshot,
    )

    num_envs = int(os.environ.get("STAGE4_SMOKE_NUM_ENVS", "2"))
    steps = int(os.environ.get("STAGE4_DLS_LIMITER_STEPS", "8"))
    max_joint_velocity = float(os.environ.get("STAGE4_DLS_LIMITER_MAX_JOINT_VEL", "0.30"))
    jump_tol = float(os.environ.get("STAGE4_DLS_LIMITER_JUMP_TOL", "1e-5"))
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

        initial_snapshot = build_stage4_env_snapshot(base_env)
        target_offset = torch.zeros(num_envs, 3, device=base_env.device)
        target_offset[:, 0] = 0.08
        target_offset[:, 2] = 0.02
        target_pos = initial_snapshot.end_effector_pos_heading + target_offset
        limiter = JointCommandLimiter(
            action_dim=4,
            num_envs=num_envs,
            device=base_env.device,
            ema_alpha=1.0,
            max_joint_velocity=max_joint_velocity,
        )
        limiter.step(initial_snapshot.active_q, dt=base_env.dt)

        wrist_indices = torch.tensor([26, 27, 28], device=base_env.device)
        previous_active_target = initial_snapshot.active_q.clone()
        previous_active_action = None
        max_active_target_jump = 0.0
        max_action_jump = 0.0
        final_error = None
        sigma_min_last = None

        for _ in range(steps):
            snapshot = build_stage4_env_snapshot(base_env)
            body_action = torch.zeros(num_envs, 22, device=base_env.device)
            command = HandTaskCommand(
                target_pos_heading=target_pos,
                target_lin_vel_heading=torch.zeros(num_envs, 3, device=base_env.device),
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
                max_joint_delta=0.20,
                comfortable_q=torch.zeros(num_envs, 4, device=base_env.device),
                nullspace_gain=0.2,
                max_nullspace_delta=0.03,
                joint_command_limiter=limiter,
                dt=base_env.dt,
            )

            active_target_jump = (out.active_joint_target - previous_active_target).abs().max().item()
            max_active_target_jump = max(max_active_target_jump, active_target_jump)
            if previous_active_action is not None:
                action_jump = (out.active_hand_action - previous_active_action).abs().max().item()
                max_action_jump = max(max_action_jump, action_jump)
            previous_active_target = out.active_joint_target.detach().clone()
            previous_active_action = out.active_hand_action.detach().clone()

            assert torch.isfinite(out.full_action).all()
            assert torch.all(out.full_action[:, wrist_indices] == 0.0)
            obs, reward, terminated, truncated, info = env.step(out.full_action)
            reward_tensor = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(np.asarray(reward))
            assert torch.isfinite(reward_tensor).all()
            assert torch.isfinite(base_env.simulator.dof_pos).all()
            final_error = out.position_error_heading.norm(dim=-1).mean().item()
            sigma_min_last = out.sigma_min.amin().item()

        allowed_target_jump = max_joint_velocity * base_env.dt + jump_tol
        print(
            f"limiter rollout max_active_target_jump={max_active_target_jump:.6f}, "
            f"allowed={allowed_target_jump:.6f}, max_action_jump={max_action_jump:.6f}, "
            f"final_error={final_error:.6f}, sigma_min_last={sigma_min_last:.6e}"
        )
        assert max_active_target_jump <= allowed_target_jump
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _stage4_env_smoke_enabled(), reason="Set RUN_STAGE4_ISAAC_SMOKE=1 to run IsaacSim smoke")
def test_stage4_isaac_env_static_reach_evaluation_smoke():
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
    from humanoidverse.agents.stage4 import evaluate_static_reach_dls

    num_envs = int(os.environ.get("STAGE4_SMOKE_NUM_ENVS", "2"))
    steps_per_target = int(os.environ.get("STAGE4_STATIC_REACH_STEPS", "8"))
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

        target_offsets = torch.tensor(
            [
                [0.05, 0.00, 0.02],
                [0.02, 0.04, 0.00],
                [-0.02, 0.02, 0.03],
            ],
            device=base_env.device,
        )
        result = evaluate_static_reach_dls(
            env,
            target_offsets_heading=target_offsets,
            steps_per_target=steps_per_target,
            max_joint_velocity=0.30,
            comfortable_q=torch.zeros(num_envs, 4, device=base_env.device),
            nullspace_gain=0.2,
            max_nullspace_delta=0.03,
        )

        print(
            "static reach eval "
            f"steady_error_mean={result.steady_state_error.mean().item():.6f}, "
            f"steady_error_max={result.steady_state_error.max().item():.6f}, "
            f"categories={result.target_categories}, "
            f"reachable_mean={result.grouped_metrics['reachable/steady_error_mean']:.6f}, "
            f"reachable_success_5cm={result.grouped_metrics['reachable/success_5cm']:.3f}, "
            f"boundary_mean={result.grouped_metrics['boundary/steady_error_mean']:.6f}, "
            f"coord_required_mean={result.grouped_metrics['coordination_required/steady_error_mean']:.6f}, "
            f"max_action_jump={result.global_metrics['max_action_jump']:.6f}, "
            f"active_vel_max={result.global_metrics['active_joint_velocity_max']:.6f}, "
            f"active_accel_max={result.global_metrics['active_joint_acceleration_max']:.6f}, "
            f"wrist_q_abs_max={result.global_metrics['wrist_q_abs_max']:.6e}, "
            f"wrist_dq_abs_max={result.global_metrics['wrist_dq_abs_max']:.6e}, "
            f"min_sigma={result.global_metrics['minimum_sigma']:.6e}, "
            f"min_joint_margin={result.global_metrics['minimum_joint_margin']:.6f}"
        )
        assert torch.isfinite(result.steady_state_error).all()
        assert torch.isfinite(result.max_action_jump).all()
        assert torch.isfinite(result.min_sigma_min).all()
        assert torch.isfinite(result.min_joint_margin).all()
        assert all(
            diagnostic["failure_mode"]
            in (
                "success",
                "horizon_too_short",
                "plateau_or_model_mismatch",
                "joint_limit_pressure",
                "low_manipulability",
                "limiter_velocity_saturated",
                "limiter_acceleration_saturated",
                "target_unreachable_by_arm_only_presolve",
            )
            for diagnostic in result.target_diagnostics
        )
    finally:
        if env is not None:
            env.close()
