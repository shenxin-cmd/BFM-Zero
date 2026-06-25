# Stage 4 Audit: Body-BFM + 4DoF Task-Space Hand Controller

This audit is based on the current repository state on branch `exp1-mse`, before any Stage 4 behavior is wired into training.

## Current Training Entry

- Actual default CLI entry: `humanoidverse/train.py`, `if __name__ == "__main__": tyro.cli(train_bfm_zero_split_z)`.
- The Hydra config at `humanoidverse/config/exp/bfm_zero/bfm_zero.yaml` exists, but the current default Python entry is the Tyro function above.
- `train_bfm_zero()` is still present as the original non-split baseline.
- `train_bfm_zero_split_z()` is the current default split-z experiment.

## Current Agent And Model Classes

`train_bfm_zero_split_z()` builds:

- Agent: `FBcprAuxAgent` from `humanoidverse/agents/fb_cpr_aux/agent.py`.
- Model: `FBcprAuxModel` from `humanoidverse/agents/fb_cpr_aux/model.py`.
- Architecture config: `FBcprAuxModelArchiConfig`.

Default split-z submodules:

- B: `SplitBackwardMap`.
- F: `SplitForwardMap`.
- Actor: `SplitActor`.
- Discriminator: `SplitDiscriminator` by default, body-only unless `disc_include_hand=True`.
- Critic: `ForwardMap` or `ResidualForwardMap` through `ForwardArchiConfig`, not split.
- AuxCritic: also `ForwardArchiConfig`, not split.

The `split_network_variant` argument selects `simple` or `residual` for split F and split Actor. Both paths still exist.

## Current Split-Z Data Flow

Training loop:

1. Environment observation is converted to torch.
2. `agent.maybe_update_rollout_context(...)` samples or refreshes rollout `z`.
3. `agent.act(obs, z)` calls `FBModel.act`.
4. `FBModel.actor` normalizes obs and calls `SplitActor`.
5. The environment receives a 29D action.
6. Replay buffer stores observation, action, z, terminated/truncated, reward, step_count, and aux_rewards.
7. Agent update samples expert and train batches.
8. Expert next observations are encoded with `SplitBackwardMap` into `[z_body, z_hand]`.
9. F/B update uses body FB and hand legacy mode:
   - `fb_hand_loss_mode="mse"`: F_hand regresses B_hand/z_hand with MSE.
   - `fb_hand_loss_mode="fb"`: F_hand uses a short-discount bilinear FB loss.
10. Actor update uses body FB Q plus the selected legacy hand term.

Current split-z assumes the right arm is the last 7 action dimensions. Stage 4 must replace this assumption with name-driven indices in the new task-space path.

## Current Right Arm And Wrist Names

The current default split-z entry overrides the robot to `g1/g1_29dof_hard_waist`.

In `humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml`, the 29 action/DOF order ends with:

- `right_shoulder_pitch_joint`
- `right_shoulder_roll_joint`
- `right_shoulder_yaw_joint`
- `right_elbow_joint`
- `right_wrist_roll_joint`
- `right_wrist_pitch_joint`
- `right_wrist_yaw_joint`

Expected indices in the 29D action/DOF vector are:

- Active right arm: 22, 23, 24, 25.
- Locked wrist: 26, 27, 28.

Stage 4 must verify these by name at runtime and fail if the names are absent or reordered unexpectedly.

## Current Action And PD Target Semantics

Environment path:

- `LeggedRobotBase.step(actor_state)` reads `actor_state["actions"]`.
- `_pre_physics_step` optionally normalizes, clips, and applies control delay.
- For IsaacSim, `_apply_force_in_physics_step` computes:
  - `actions_scaled = actions_after_delay * action_scale`
  - `jpos_target = actions_scaled + (default_dof_pos + default_dof_pos_offset)`
  - `set_joint_position_target(jpos_target, joint_ids=...)`
- For IsaacGym/MuJoCo-style torque control, `_compute_torques` computes a P controller target with the same offset semantics.

Important consequence:

- `action[..., wrist_indices] = 0` means "target default wrist pose", not necessarily "absolute wrist target is zero".
- Current hard-waist config has all three right wrist default angles equal to `0.0`, but Stage 4 still needs a final PD-target clamp to absolute `0.0` after target construction.

## Current Checkpoint Logic

Training checkpoint:

- `Workspace.save(...)` calls `agent.save(work_dir/checkpoint)`.
- It optionally saves `replay_buffer["train"]`.
- It writes `checkpoint/train_status.json` containing `{"time": time}`.

Agent checkpoint:

- `FBAgent.save(...)` writes:
  - `checkpoint/config.json`
  - `checkpoint/optimizers.pth`
  - `checkpoint/model/model.safetensors`
- `FBAgent.load(...)` rebuilds config and init args, loads optimizers, then loads model tensors with `strict=False`.

Stage 4 must not silently ignore shape mismatches when migrating body policy weights. Any migration path needs an explicit report and assertions.

## Current Replay Buffer Fields

Trajectory buffer mode stores:

- `observation`
- `action`
- `z`
- `terminated`
- `truncated`
- `step_count`
- `reward`
- `aux_rewards` for `FBcprAuxAgent`

Transition buffer mode stores the same core fields plus `next`.

No current buffer field stores hand command, wrist position, sigma_min, coordination gate, or command_done. Stage 4 can keep the first DLS-only controller outside the critic path, but task-space critic training will require new command fields.

## Current Environment State Availability

Available now:

- DOF names and body names are exposed through `BaseTask` from the simulator.
- DOF position and velocity are available through `simulator.dof_pos` and `simulator.dof_vel`.
- Rigid body world position, rotation, linear velocity, and angular velocity are available through simulator properties:
  - `_rigid_body_pos`
  - `_rigid_body_rot`
  - `_rigid_body_vel`
  - `_rigid_body_ang_vel`
- Root pose and velocities are available through `simulator.robot_root_states`.
- Heading-frame transforms already exist in `humanoidverse/utils/torch_utils.py` via `calc_heading_quat_inv`, `calc_heading_quat`, and quaternion rotation helpers.

Unconfirmed:

- IsaacSim path currently exposes body pose/velocity but no direct Jacobian property in `humanoidverse/simulator/isaacsim/isaacsim.py`.
- IsaacGym path has a `jacobian` tensor and refresh call, but the current default training config uses IsaacSim.
- Stage 4 must add a tested Jacobian access path for the active right arm. If direct IsaacSim Jacobian is not available, tests should use finite differences or an explicitly documented fallback.

## Existing IK, Task-Space Reward, Residual Policy

Repository search found no current Stage 4 IK, DLS, task-space hand command, task-space critic, or gated coordination residual implementation.

## Legacy Path Compatibility

Legacy paths that must remain runnable:

- `train_bfm_zero()` original unified-z baseline.
- `train_bfm_zero_split_z(..., fb_hand_loss_mode="mse")`.
- `train_bfm_zero_split_z(..., fb_hand_loss_mode="fb")`.
- `split_network_variant="simple"` and `"residual"`.

Stage 4 must add a mutually exclusive mode rather than mixing task-space hand control into these legacy hand losses.

## Stage 4 Implementation Guardrails

Required new mode:

- `legacy_mse`: current split-z hand MSE path.
- `legacy_fb`: current split-z hand FB path.
- `task_space_4dof`: new Stage 4 path.

In `task_space_4dof` mode:

- Legacy hand FB loss disabled.
- Legacy hand MSE loss disabled.
- Hand controller outputs exactly 4 active right-arm dimensions.
- Actor never outputs wrist dimensions for the task-space hand head.
- Wrist action values are hard-set to zero during action assembly and before environment step.
- Final PD target values for the three right wrist joints are hard-set to absolute zero after target construction.

## First Implementation Plan

Commit 1:

- Add this audit.
- Add isolated Stage 4 config with mode validation.
- No training behavior change.

Commit 2:

- Add name-driven right-arm index resolution.
- Add full-action assembly and wrist action lock.
- Add final PD-target wrist clamp hooks.
- Add unit tests for mapping, wrist lock, and action assembly.

Commit 3:

- Add heading/root frame transforms and active-arm Jacobian selection interface.
- Add finite-difference test coverage.

Commit 4:

- Add DLS, adaptive damping, joint-limit scaling, and command limiter.
- Add numerical tests.

Commit 5:

- Add Stage 4 controller wrapper and smoke-test scaffolding.

Later commits:

- Add hand residual only after DLS baseline passes.
- Add body coordination residual only after hand-only control passes.
