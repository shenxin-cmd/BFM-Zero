# Stage4B Static Reach Smoke Feedback

## 1. Last Server Result

Ordinary Stage4 tests passed:

```text
60 passed, 7 skipped
```

This means the current unit/mock-level Stage4 code paths are wired correctly:

- `Stage4CommandSampler`
- static reach command creation
- `root_heading_frame` naming in the new Stage4 path
- reachable / boundary / coordination_required classification scaffolding
- stratified static reach evaluator
- diagnostic trajectory fields
- DLS parameter scan interface
- null-space diagnostic outputs

The real IsaacSim static reach smoke also passed as a pytest test, but its
metrics exposed two lower-level problems that made the DLS accuracy result
not yet trustworthy.

Important output:

```text
steady_error_mean=0.351897
steady_error_max=0.369390
categories=('reachable', 'boundary', 'boundary')
reachable_mean=0.369390
reachable_success_5cm=0.000
max_action_jump=0.024000
active_vel_max=90.647636
active_accel_max=4547.381836
wrist_q_abs_max=6.061761e-01
wrist_dq_abs_max=1.974513e+01
min_sigma=1.101052e-03
min_joint_margin=0.000000
```

## 2. What This Result Means

### 2.1 Wrist Was Not Hard-Locked At The Real State Layer

The values:

```text
wrist_q_abs_max ~= 0.606 rad
wrist_dq_abs_max ~= 19.7 rad/s
```

show that locking wrist action to zero and clamping the final wrist PD target
to absolute zero were not sufficient in the real IsaacSim rollout.

The wrist state could still drift due to simulator dynamics and coupling.
Therefore, Stage4 task-space mode needs an additional state-layer hard lock
after physics, not only action-layer and PD-target-layer locks.

This directly confirms the original requirement:

```text
"wrist action equals zero" is not the same as "wrist absolute joint target/state is locked at zero".
```

### 2.2 Velocity And Acceleration Metrics Were Using The Wrong Quantity

The reported:

```text
active_vel_max ~= 90 rad/s
active_accel_max ~= 4547 rad/s^2
```

were not true command-layer velocity and acceleration.

The evaluator was effectively mixing in PD tracking error by comparing the
new active joint target against the current actual joint position. The correct
metric for command smoothness is based on consecutive active joint targets:

```text
command_velocity = (q_cmd[t] - q_cmd[t-1]) / dt
command_acceleration = (command_velocity[t] - command_velocity[t-1]) / dt
```

Therefore, the previous velocity and acceleration values should be treated as
invalid diagnostics.

### 2.3 DLS Accuracy Is Not Yet A Valid Conclusion

The apparent reachable-group result:

```text
reachable_mean ~= 0.369 m
reachable_success_5cm = 0.0
```

is not yet a trustworthy measure of 4DoF DLS capability because the rollout
was contaminated by:

- wrist state drift;
- incorrect velocity / acceleration reporting;
- low `min_sigma`;
- `min_joint_margin = 0`, indicating limit pressure in at least part of the rollout.

At this point, we should not conclude:

```text
4DoF DLS has failed.
Body coordination is definitely required.
Learned hand residual is needed.
```

The correct conclusion is:

```text
The real closed-loop evaluation path runs, but the instrumentation and wrist
state enforcement needed one more correction before DLS accuracy can be judged.
```

## 3. Fix Implemented After This Result

Commit:

```text
b02e36a stage4 hard lock wrist state after physics
```

Implemented fixes:

- Added Stage4 task-space post-physics wrist state clamp.
- In `_post_physics_step()`, after refreshing simulator tensors, Stage4
  task-space mode now zeroes wrist q/dq and refreshes tensors again.
- Legacy modes remain unaffected.
- The static reach evaluator now computes active joint velocity and acceleration
  from consecutive `active_joint_target` values.
- Wrist q/dq metrics are now read after `env.step()`, so they reflect the real
  post-step simulator state.
- The real static reach smoke now asserts:

```text
wrist_q_abs_max <= 1e-6
wrist_dq_abs_max <= 1e-6
```

If the real smoke passes after this fix, the wrist state lock is verified at:

- action assembly;
- env pre-step action;
- final PD target;
- reset / motion initialization / domain randomization;
- post-physics simulator state.

## 4. Current Stage Definition

Current status should be described as:

```text
Stage4A safety, indexing, Jacobian, wrist lock, and controller tooling are
mostly complete. The new post-physics wrist state clamp still needs real
IsaacSim smoke confirmation.

Stage4B DLS closed-loop rollout, command sampler, and stratified static reach
evaluation framework are implemented.

Stage4B control accuracy is not yet established. The previous static reach
accuracy result must be rerun after the wrist-state and metric fixes.
```

Not started yet:

- formal long training;
- learned hand residual;
- waist/body coordination residual;
- checkpoint migration;
- Replay Buffer schema changes.

## 5. Next Back-To-Back Check

After pulling commit `b02e36a` on the server, run:

```bash
../../uv_binary run --with pytest python -m pytest -q tests/stage4
```

Then run the real IsaacSim static reach smoke:

```bash
RUN_STAGE4_ISAAC_SMOKE=1 \
STAGE4_SMOKE_NUM_ENVS=2 \
STAGE4_STATIC_REACH_STEPS=8 \
../../uv_binary run --with pytest python -m pytest -q \
tests/stage4/test_stage4_env_smoke.py::test_stage4_isaac_env_static_reach_evaluation_smoke -s
```

The first required checks are:

```text
wrist_q_abs_max <= 1e-6
wrist_dq_abs_max <= 1e-6
active_vel_max is near the configured command limiter scale
active_accel_max is consistent with the target-difference metric
```

Only after those are clean should the reachable-group error be interpreted.

## 6. Main Question Still Open

The core Stage4B question remains:

```text
For targets that are confirmed to be inside the 4DoF right-arm reachable set,
what final position error can the current DLS controller achieve in centimeters?
```

The answer is still pending because the last real rollout metrics were not yet
clean enough to support a control-quality conclusion.

