from types import SimpleNamespace

import torch

from humanoidverse.agents.stage4 import (
    Stage4CommandSampler,
    Stage4CommandSamplerConfig,
    Stage4EnvSnapshot,
    make_hand_task_command_root_heading,
)


def _snapshot_for_sampler(*, jac_scale: float = 1.0, active_q: torch.Tensor | None = None) -> Stage4EnvSnapshot:
    num_envs = 3
    if active_q is None:
        active_q = torch.zeros(num_envs, 4)
    jac = torch.zeros(num_envs, 3, 4)
    jac[:, :3, :3] = torch.eye(3).unsqueeze(0) * jac_scale
    return Stage4EnvSnapshot(
        indices=SimpleNamespace(),
        end_effector_body_index=0,
        root_pos_world=torch.zeros(num_envs, 3),
        root_quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_envs),
        end_effector_pos_world=torch.zeros(num_envs, 3),
        end_effector_pos_heading=torch.zeros(num_envs, 3),
        active_q=active_q,
        active_lower=torch.full((num_envs, 4), -1.0),
        active_upper=torch.full((num_envs, 4), 1.0),
        active_pd_reference_pos=torch.zeros(num_envs, 4),
        action_scale=1.0,
        active_position_jacobian_world=jac,
        active_position_jacobian_heading=jac,
    )


def test_make_hand_task_command_root_heading_shapes_and_masks():
    target = torch.tensor([[0.1, 0.0, 0.2], [0.2, 0.1, 0.0]])

    command = make_hand_task_command_root_heading(target_pos_heading=target)

    assert torch.allclose(command.target_pos_heading, target)
    assert command.target_lin_vel_heading.shape == target.shape
    assert torch.all(command.position_mask == 1.0)
    assert torch.all(command.velocity_mask == 0.0)
    assert command.command_done.shape == (2, 1)


def test_stage4_command_sampler_static_reach_targets_are_root_heading_offsets():
    snapshot = _snapshot_for_sampler()
    snapshot.end_effector_pos_heading[:, 0] = torch.tensor([0.1, 0.2, 0.3])
    sampler = Stage4CommandSampler()

    command = sampler.make_static_reach_command(
        snapshot,
        target_offsets_heading=torch.tensor([[0.01, 0.02, 0.03]]),
    )

    expected = snapshot.end_effector_pos_heading + torch.tensor([[0.01, 0.02, 0.03]]).expand(3, 3)
    assert torch.allclose(command.target_pos_heading, expected)


def test_stage4_command_sampler_classifies_reachable_boundary_and_coordination_required():
    snapshot = _snapshot_for_sampler()
    cfg = Stage4CommandSamplerConfig(
        reachable_threshold=0.02,
        coordination_required_threshold=0.08,
        boundary_sigma_threshold=0.01,
        boundary_joint_margin_threshold=0.20,
        presolve_steps=4,
        presolve_max_joint_delta=0.03,
        presolve_damping_min=0.0,
        presolve_damping_max=0.0,
    )
    sampler = Stage4CommandSampler(cfg)
    targets = torch.tensor(
        [
            [0.03, 0.00, 0.00],
            [0.15, 0.00, 0.00],
            [0.40, 0.00, 0.00],
        ]
    )

    classification = sampler.classify_static_targets(snapshot, target_pos_heading=targets)

    assert classification.category == ("reachable", "boundary", "coordination_required")
    assert classification.final_ik_residual.shape == (3,)
    assert torch.isfinite(classification.min_sigma).all()
    assert torch.isfinite(classification.min_joint_margin).all()


def test_stage4_command_sampler_keeps_non_static_modes_explicitly_unimplemented():
    snapshot = _snapshot_for_sampler()
    sampler = Stage4CommandSampler(Stage4CommandSamplerConfig(command_mode="motion_reference"))

    try:
        sampler.make_static_reach_command(snapshot, target_offsets_heading=torch.zeros(1, 3))
    except NotImplementedError as exc:
        assert "motion_reference" in str(exc)
    else:
        raise AssertionError("motion_reference mode should not silently behave like static_reach")
