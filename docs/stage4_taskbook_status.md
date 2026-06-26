# Stage4 Taskbook Status Check

This document checks the current BFM-Zero branch against the taskbook.

Taskbook PDF location:

`C:/Users/25878/Desktop/paper reading/`

Taskbook PDF title:

`Codex task: Stage4 Body-BFM + 4DoF Task-Space Hand Controller + Gated Body Coordination Residual`

Current branch:

`codex-stage4-body-bfm-4dof-hand`

Current status at the time of writing:

- Working tree was clean before this document was added.
- Latest implementation commit checked: `c8ace28 stage4 opt-in dls env action smoke`.
- Stage4 ordinary tests on server: `42 passed, 2 skipped`.
- Opt-in IsaacSim Stage4 smoke tests on server: both passed.

Update on 2026-06-25:

- Latest Stage4 commit checked in this update: `d942bde stage4 add static reach dls evaluation smoke`.
- Latest ordinary Stage4 tests on server: `55 passed, 6 skipped`.
- Current status should be read as: Stage4A safety/tooling foundation complete; Stage4B DLS-only validation is in progress and has passed the current smoke gates, but full Stage4B evaluation/training is not complete.
- No formal long training has been run.

## 1. Taskbook Requirements Interpreted From The PDF

The taskbook describes Stage4 as:

1. Preserve Body-BFM as the base policy.
2. Replace legacy right-hand precision control with a 4DoF task-space hand controller.
3. Lock the three right wrist joints at absolute zero.
4. Add a gated body coordination residual only when the 4DoF arm is insufficient.
5. Keep legacy hand FB and legacy hand MSE paths runnable and mutually exclusive from the new task-space path.
6. Do not silently patch checkpoint shape mismatches or replay buffer schema mismatches.
7. Add tests before formal training.

Important explicit constraints from the taskbook:

- Stage4 main mode should be `hand_control_mode="task_space_4dof"`.
- Stage4 must not use `z_hand`, `B_hand`, or `F_hand` for precise right-hand control.
- Active right arm is exactly 4DoF: three shoulder joints plus one elbow joint.
- Locked wrist is exactly 3DoF: roll, pitch, yaw.
- The hand controller should output only 4 active joint targets/actions.
- The final 29D robot action should contain the 4 active right-arm values and three wrist zeros.
- Wrist zero must be enforced in multiple layers, including final PD target after `action * action_scale + default_dof_pos`.
- First Stage4 version should prioritize DLS-only hand control before learned hand residual.
- A small hand residual is a later stage.
- A gated body coordination residual is later than basic DLS.
- If no task-space critic is used in the first version, replay-buffer command fields may be deferred, but schema should be considered early.

## 2. What Has Been Completed

### 2.1 Audit Before Modification

Completed.

File:

- `docs/stage4_audit.md`

Confirmed from current code rather than old documentation:

- Actual training entry is `humanoidverse/train.py::train_bfm_zero_split_z`.
- Current main agent path is `FBcprAuxAgent`.
- Current model is `FBcprAuxModel`.
- Current split modules are `SplitBackwardMap`, `SplitForwardMap`, `SplitActor`, and split discriminator variants.
- Replay buffer and checkpoint behavior were audited.
- Current action/PD target semantics were audited.
- Existing code did not already contain a complete Stage4 IK/Jacobian/task-space residual path.

Taskbook status:

Completed.

### 2.2 Mutually Exclusive Stage4 Config

Completed.

Files:

- `humanoidverse/agents/stage4/config.py`
- `tests/stage4/test_stage4_config.py`

Implemented modes:

- `legacy_mse`
- `legacy_fb`
- `task_space_4dof`

In `task_space_4dof`, config validation requires:

- legacy hand MSE disabled
- legacy hand FB disabled
- task-space hand enabled
- `z_hand_enabled=False`
- `b_hand_enabled=False`
- `f_hand_enabled=False`

Taskbook status:

Mostly completed at isolated config level.

Not yet completed:

- `train_bfm_zero_split_z` does not yet expose or consume this mode as the formal training switch.

### 2.3 Name-Based Right Arm And Wrist Mapping

Completed.

Files:

- `humanoidverse/agents/stage4/actions.py`
- `tests/stage4/test_right_arm_joint_name_mapping.py`

Confirmed on real IsaacSim env:

- active right arm:
  - action/dof index 22 -> `right_shoulder_pitch_joint`
  - action/dof index 23 -> `right_shoulder_roll_joint`
  - action/dof index 24 -> `right_shoulder_yaw_joint`
  - action/dof index 25 -> `right_elbow_joint`
- locked wrist:
  - action/dof index 26 -> `right_wrist_roll_joint`
  - action/dof index 27 -> `right_wrist_pitch_joint`
  - action/dof index 28 -> `right_wrist_yaw_joint`

The implementation resolves by joint names and fails on missing names.

Taskbook status:

Completed.

### 2.4 Wrist Hard Lock

Completed for the safety layers implemented so far.

Files:

- `humanoidverse/agents/stage4/actions.py`
- `humanoidverse/envs/legged_base_task/legged_robot_base.py`
- `tests/stage4/test_wrist_lock.py`
- `tests/stage4/test_env_wrist_hooks.py`
- `tests/stage4/test_stage4_env_smoke.py`

Implemented layers:

1. Full-action assembly writes wrist action values to zero.
2. Env `_pre_physics_step` writes wrist action values to zero again when `task_space_4dof` is enabled.
3. IsaacSim final `jpos_target` is clamped after:
   - `actions_scaled = actions_after_delay * action_scale`
   - `jpos_target = actions_scaled + default_dof_pos + default_dof_pos_offset`
4. P-controller final `jpos_target` is also clamped.

Important semantic point:

- The taskbook warns that `wrist action == 0` is not the same as `wrist absolute joint target == 0`.
- Current code handles this by explicitly clamping final wrist PD targets to absolute `0.0`.

Taskbook status:

Completed for current env action and PD target path.

### 2.5 Heading-Frame Coordinate And Jacobian Utilities

Completed.

Files:

- `humanoidverse/agents/stage4/kinematics.py`
- `tests/stage4/test_hand_frame_transform.py`
- `tests/stage4/test_active_arm_jacobian.py`

Implemented:

- world point to root heading frame
- heading-frame point back to world
- vector-only heading rotations
- position-Jacobian column rotation to heading frame
- finite-difference Jacobian test
- active 4DoF Jacobian selection
- IsaacSim/PhysX Jacobian selection

Taskbook status:

Completed for the DLS-first implementation.

### 2.6 IsaacSim Jacobian Access

Completed.

Files:

- `humanoidverse/agents/stage4/kinematics.py`
- `tests/stage4/test_active_arm_jacobian.py`
- `tests/stage4/test_stage4_env_smoke.py`

Real env introspection showed:

- No top-level `simulator.jacobian`.
- Jacobian is available at `simulator._robot.root_physx_view.get_jacobians()`.
- Real shape is `(num_envs, 30, 6, 35)`.
- 35 columns mean `6 floating-base columns + 29 actuated DOF columns`.

Implemented selection handles both:

- `num_dofs` columns
- `6 + num_dofs` columns

Taskbook status:

Completed.

### 2.7 DLS, Adaptive Damping, Joint Limits, Smoother

Completed for DLS-first baseline.

Files:

- `humanoidverse/agents/stage4/control.py`
- `tests/stage4/test_dls_solver.py`
- `tests/stage4/test_adaptive_damping.py`
- `tests/stage4/test_joint_limit_scaling.py`
- `tests/stage4/test_joint_command_limiter.py`

Implemented:

- damped least squares
- adaptive damping from singular values
- direction-aware joint-limit scaling
- joint-margin scale
- command/action limiter with EMA and velocity/acceleration limits

Taskbook status:

Completed for DLS-only Stage4A/Stage4B.

### 2.8 Stage4 Env Snapshot

Completed.

Files:

- `humanoidverse/agents/stage4/env_adapter.py`
- `tests/stage4/test_env_snapshot.py`
- `tests/stage4/test_stage4_env_smoke.py`

Snapshot contains:

- right-arm indices
- end-effector body index
- root position
- root heading quaternion
- end-effector world position
- end-effector heading-frame position
- active joint q
- active lower/upper limits with shape `(num_envs, 4)`
- default active joint position
- action scale
- world-frame active Jacobian
- heading-frame active Jacobian

Real env smoke result:

- `active_q_shape = (2, 4)`
- `ee_heading_shape = (2, 3)`
- `jac_heading_shape = (2, 3, 4)`
- `jac_heading_finite = True`
- `action_scale = 0.25`

Taskbook status:

Completed for DLS input extraction.

### 2.9 DLS Hand Controller And Full Action Assembly

Completed for DLS-first rollout assembly.

Files:

- `humanoidverse/agents/stage4/controller.py`
- `tests/stage4/test_stage4_controller.py`
- `tests/stage4/test_env_snapshot.py`
- `tests/stage4/test_stage4_env_smoke.py`

Implemented:

- `HandTaskCommand`
- `DLSHandController`
- `DLSHandControllerOutput`
- `body_action_indices_for_stage4`
- `dls_hand_action_from_snapshot`
- `CoordinationGate`
- `coordination_gate_raw`

Current action assembly:

```text
body_action(22D)
+ DLS active right-arm action(4D)
+ wrist zeros(3D)
-> full_action(29D)
```

Real env smoke result:

- `full_action_shape = (2, 29)`
- `active_hand_action_shape = (2, 4)`
- `wrist_action_values = 0`
- `dq_finite = True`
- Superseded observation: an earlier run reported `sigma_min` around
  `1e-12` to `1e-11`.

Conclusion:

- DLS action generation is numerically finite.
- The earlier near-zero `sigma_min` conclusion is superseded by the later
  IsaacSim/PhysX body and DOF index mapping fix and multi-pose finite-difference
  Jacobian validation in section 8.2.
- Body coordination must not be justified from that pre-fix `sigma_min`.
  Whether waist/body coordination is necessary must be decided from stratified
  Stage4B evidence: arm-only IK residual, real manipulability after correct
  indexing, joint-limit pressure, and whether targets lie outside the 4DoF
  right-arm workspace.

Taskbook status:

Completed for Stage4 DLS-only action assembly.

### 2.10 Tests And Smoke Tests

Completed up to the DLS-only real-env stage.

Server results:

```text
../../uv_binary run --with pytest python -m pytest -q tests/stage4
42 passed, 2 skipped
```

Opt-in IsaacSim smoke:

```text
RUN_STAGE4_ISAAC_SMOKE=1 \
STAGE4_SMOKE_NUM_ENVS=2 \
STAGE4_SMOKE_STEPS=20 \
../../uv_binary run --with pytest python -m pytest -q tests/stage4/test_stage4_env_smoke.py -s
```

Result:

```text
..
```

Taskbook status:

Completed for current implemented scope.

## 3. What Has Not Been Done Yet

### 3.1 Formal Training Entry Integration

Not completed.

Still not wired:

- `train_bfm_zero_split_z(..., hand_control_mode="task_space_4dof")`
- formal Stage4 training path
- Agent rollout integration
- checkpoint migration for Stage4 body policy
- replay buffer integration for Stage4 command data

Current Stage4 code is a tested control/safety/tooling layer, not yet a full training mode.

### 3.2 Actor Architecture Change

Not completed.

Taskbook says the task-space hand head should not output wrist dimensions.

Current legacy `SplitActor` still belongs to the legacy split-z route.

Implemented helper assumes the intended Stage4 shape:

```text
body actor output: 22D
DLS hand output: 4D
wrist: 3D hard zero
full env action: 29D
```

But the actual training actor has not been changed or wrapped yet.

Need to decide:

1. Add a new Stage4 actor/wrapper that emits 22D body action only.
2. Reuse existing body branch from `SplitActor` and ignore legacy hand branch.
3. Migrate body branch weights from Stage3 checkpoint.

### 3.3 Task-Space Command Source

Not completed and currently the main information blocker.

The taskbook defines task-space hand command conceptually, including fields similar to:

- target position
- target velocity
- position mask
- velocity mask
- command id
- command done

Current code has `HandTaskCommand`, but there is no repository-integrated command sampler or motion-reference extractor yet.

Need to confirm from the user/task design:

1. Should target hand position come from motion reference wrist trajectory?
2. Should targets be random heading-frame reaching goals?
3. Should targets come from a dataset field that already exists?
4. Should first implementation use static-base or moving-base targets?
5. Should command be generated online in env or sampled in agent/trainer?
6. Should command be part of observation?
7. Should command be stored in replay buffer?

### 3.4 Hand Residual

Not completed.

Taskbook says first implement DLS and later add small learned hand residual.

Current code has no learned hand residual policy.

Need to confirm:

- residual input features
- residual output dimension, likely 4D active arm only
- residual scale, taskbook example suggests max residual around `0.05`
- whether residual loss is supervised, RL, or both
- whether residual gradients should be isolated from Body-BFM

### 3.5 Gated Body Coordination Residual

Partially completed only as a rule gate utility.

Current code has:

- `coordination_gate_raw`
- `CoordinationGate`

Not yet implemented:

- body residual network
- body residual output dimensions
- integration with body action
- training objective for residual
- replay/logging for gate/residual
- saturation monitoring

Need to confirm:

1. Which body DOF can receive residual?
2. Should waist be included?
3. Should legs be included or excluded?
4. Is residual added in action space or latent/policy space?
5. Is gate rule-based only at first, or learned?
6. Should the gate be differentiable and trained?

### 3.6 Replay Buffer Changes

Not completed.

Taskbook mentions Stage4 may need fields such as:

- hand command
- command done
- wrist position
- next wrist position
- command id

Current code has not modified replay buffer fields.

Need to confirm:

- For DLS-only training, can replay schema remain unchanged?
- For hand residual or task-space critic, which exact fields are required?
- Should fields be added now as optional/reserved, or only when residual/critic is implemented?

### 3.7 Checkpoint Migration

Not completed.

Taskbook requires no silent shape mismatch and no blind `strict=False` migration.

Need to confirm:

1. Which checkpoint should Stage4 initialize from?
2. Should only body branch weights be migrated?
3. What exact old/new key mapping is expected?
4. Should migration be a one-time script or built into training startup?

## 4. Current Position In The Taskbook Roadmap

Based on the taskbook commit/stage order:

Completed:

- Audit.
- Mode config and mutual-exclusion guard.
- Joint name/index mapping.
- Wrist hard lock.
- Heading frame and Jacobian utilities.
- IsaacSim Jacobian selection.
- DLS, adaptive damping, joint limits, smoother.
- DLS-only controller.
- Env snapshot adapter.
- Real-env DLS full-action smoke.
- Tests required for current implemented scope.

Current stage:

```text
End of DLS-only control infrastructure.
Ready to start Stage4 training-path design, but blocked on command source and actor/checkpoint decisions.
```

Not yet at:

- formal Stage4 training mode
- hand residual
- gated body coordination residual
- task-space critic
- formal Stage4 checkpoint migration

## 5. Precise Information Needed Before Continuing

### Required Decision 1: Task-Space Command Source

Need answer:

```text
Where does `HandTaskCommand.target_pos_heading` come from in Stage4 training?
```

Options:

1. Motion reference `right_wrist_yaw_link` trajectory.
2. Random heading-frame reaching target.
3. A dataset-provided command field.
4. A new environment-side command sampler.
5. Other.

Also need:

- target coordinate frame
- whether target velocity is used in first version
- command reset/done semantics
- command horizon/duration

### Required Decision 2: Observation And Replay Schema

Need answer:

```text
Should hand command be part of actor observation and replay buffer in the first Stage4 training mode?
```

If yes, specify exact fields and shapes.

Likely candidate fields:

- `hand_target_pos_heading`: `(num_envs, 3)`
- `hand_target_vel_heading`: `(num_envs, 3)`
- `hand_command_id`: `(num_envs,)`
- `hand_command_done`: `(num_envs, 1)`
- `wrist_pos_heading`: `(num_envs, 3)`
- `next_wrist_pos_heading`: `(num_envs, 3)`
- `sigma_min`: `(num_envs,)`
- `joint_margin`: `(num_envs,)`
- `coordination_gate`: `(num_envs,)`

But these should not be added without confirming the training objective.

### Required Decision 3: Stage4 Actor Structure

Need answer:

```text
How should Stage4 produce the 22D body action?
```

Options:

1. New actor wrapper using old body branch only.
2. Modify `SplitActor` under `task_space_4dof`.
3. New Stage4 actor class.
4. Temporary body action from frozen checkpoint during DLS smoke.

Must not let Stage4 actor output wrist dimensions.

### Required Decision 4: Checkpoint Source And Migration

Need answer:

```text
Which checkpoint should Stage4 inherit from, and which weights should be migrated?
```

Likely desired from taskbook:

- inherit Body-BFM/body policy from Stage3 checkpoint
- do not migrate legacy hand branch into task-space controller
- print migration report and assert dimensions

Need exact checkpoint path and expected migration mapping.

### Required Decision 5: Body Coordination Residual Scope

Need answer:

```text
Which body joints can the coordination residual change?
```

Need exact allowed indices/names and max residual scale.

Taskbook intent:

- use body residual only when target is hard, near singular, near limit, or large IK residual
- avoid rewriting whole Body-BFM policy

But exact implementation details still need user confirmation.

## 6. Recommended Next Implementation Step

Do not start formal long training yet.

Recommended next commit:

```text
Add Stage4 command source module and tests.
```

Only after command source is confirmed.

Suggested shape:

```python
class Stage4CommandSampler:
    def reset(env_ids): ...
    def sample_or_update(snapshot): -> HandTaskCommand
```

Tests should cover:

- command shape
- heading-frame semantics
- command reset/done
- finite targets
- no wrist orientation target in first version

Then add an opt-in IsaacSim smoke:

```text
env -> command sampler -> snapshot -> DLS action -> env.step
```

Only after this passes should `train_bfm_zero_split_z` be wired to `task_space_4dof`.

## 7. One-Sentence Summary

The code has completed and tested the Stage4 DLS-only task-space control foundation: name-based 4DoF right-arm mapping, multi-layer wrist absolute-zero locking, heading-frame IsaacSim Jacobian extraction, DLS full-action assembly, and real-env smoke tests. The next blocker is not numerical control but deciding the formal task-space command source, replay/observation schema, Stage4 body actor structure, checkpoint migration plan, and allowed body coordination residual scope.

## 8. Stage4B Update After Real IsaacSim Validation

This section records the additional validation completed after the initial status document.

### 8.1 DLS Joint Target To Env Action Round Trip

Completed.

Implemented and tested the correct PD semantics:

```text
q_cmd = q_current + dq
action = (q_cmd - default_dof_pos - default_dof_pos_offset) / action_scale
```

The test verifies that the env-reconstructed PD target equals the intended absolute `q_cmd`.

### 8.2 Real IsaacSim Multi-Pose Jacobian Finite Difference

Completed as an opt-in smoke.

Important conclusion:

- Before mapping fix, default-pose FD error was about `0.195688` and `sigma_min` was about `1e-10`.
- Root cause was mixing humanoidverse/config body and DOF indices with raw IsaacSim/PhysX Jacobian indices.
- After mapping `body_ids` and `dof_ids`, real IsaacSim multi-pose FD passed:
  - default error about `0.069585`, `sigma_min about 0.1215`
  - elbow-bent error about `0.063420`, `sigma_min about 0.0914`
  - shoulder-forward error about `0.068925`, `sigma_min about 0.1095`
  - random-safe error about `0.063673`, `sigma_min about 0.1024`

Therefore, the earlier near-zero singular value should not be interpreted as proof that body coordination is inherently required.

### 8.3 Wrist Reset And Domain Randomization

Completed for current Stage4 task-space path.

Implemented:

- Stage4 default reset locks right wrist `q=0` and `dq=0`.
- Stage4 target-state reset locks right wrist `q=0` and `dq=0`.
- `LeggedRobotMotions` motion-reference reset also locks right wrist after motion initialization.
- Stage4 domain randomization clears the three right-wrist `default_dof_pos_offset` entries after default-offset randomization.
- Legacy modes keep previous behavior.

Real IsaacSim reset smoke passed:

```text
reset wrist q/dq/offset abs max = 0.000000e+00, 0.000000e+00, 0.000000e+00
non_wrist_offset_mean=0.500000
```

### 8.4 Null-Space Comfortable Posture And Limiter

Completed at DLS controller/tooling level and real rollout smoke level.

Implemented:

- 4DoF null-space projector for the `3x4` active arm Jacobian.
- Comfortable-posture null-space update.
- `JointCommandLimiter` integration in the DLS controller at the absolute active joint target layer, before normalized action conversion.

Real IsaacSim limiter rollout smoke passed:

```text
max_active_target_jump=0.006000
allowed=0.006010
max_action_jump=0.024000
final_error=0.391508
sigma_min_last=1.038025e-04
```

Interpretation:

- Limiter is active in the real `snapshot -> DLS -> limiter -> env.step` path.
- This smoke is not a reaching-performance pass. The low `sigma_min` and high final error indicate this random motion/target scene can be near singular.

### 8.5 Static Reach DLS Evaluation

Completed as an opt-in evaluation smoke and reusable evaluation helper.

Implemented:

- `humanoidverse/agents/stage4/evaluation.py`
- `evaluate_static_reach_dls`
- Per-target metrics:
  - steady-state error
  - max action jump
  - minimum sigma
  - minimum joint margin

Real IsaacSim static reach evaluation smoke passed:

```text
steady_error_mean=0.265757
steady_error_max=0.346388
max_action_jump=0.024000
min_sigma=3.617078e-03
min_joint_margin=0.000000
```

Interpretation:

- The evaluation path is now runnable and records useful Stage4B metrics.
- DLS-only static reach is not yet performance-complete; current metrics show near-limit and near-singular cases.
- This supports the next decision point: whether to add body coordination residual, learned hand residual, or revise command/task distribution.

### 8.6 Current Remaining Stage4B Work

Still not complete:

- Formal command sampler module with `static_reach`, `motion_reference`, and `external_task` modes.
- Motion-reference command mode with lookahead 1 or 2 frames.
- Snapshot end-effector linear velocity.
- Body actor wrapper path with `forward_body(obs, z_body)`.
- Checkpoint migration report from Stage3 to Stage4.
- Rule-based waist-only body coordination residual.
- Full Stage4B evaluation report over a larger target set.
- Any learned hand residual or learned body coordination residual.
- Replay buffer schema changes; these remain intentionally deferred during DLS-only validation.
