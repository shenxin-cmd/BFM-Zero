# Stage4 Taskbook Requirements Backcheck

本文档用于背靠背核对：我从任务书 `Codex任务：实现第四阶段 Body-BFM + 4DoF Task-Space Hand Controller + Gated Body Coordination Residual.pdf` 中识别到了哪些需求、这些需求意味着需要完成哪些工程任务、需要修改哪些代码部分、以及有哪些必须额外注意的控制逻辑风险。

说明：

- PDF 文本抽取后，在 Windows 终端中显示时曾出现中文乱码。
- 重新检查后发现，抽取文本中的章节标题和大部分中文要求可以通过检索正常识别；乱码主要来自终端显示编码。
- 本文档不是逐字转录 PDF，而是对任务书需求的结构化复述和工程拆解，方便用户检查我是否真正理解需求。
- 如果本文档中有任何任务书含义被我误读，应优先以用户纠正和原始任务书为准。

## 0. 最高优先级工作原则

### 0.1 先审计，后修改

任务书要求：

- 不能直接根据旧文档或记忆修改代码。
- 必须先完整检查当前仓库，输出审计报告。
- 审计必须基于实际代码，而不是假设类名、文件结构或旧实验文档。

需要审计的内容包括：

- 当前实际训练入口。
- 当前默认使用的 Agent 类。
- 当前 Actor、F、B、Discriminator、Critic、AuxCritic 的具体类。
- 当前 split-z 的实际数据流。
- 当前右臂动作索引和关节名称。
- 当前 right wrist 三个关节的准确名称和索引。
- 当前 hand MSE loss 实际实现位置。
- 当前 checkpoint 保存与加载逻辑。
- 当前 Replay Buffer 字段。
- 当前环境中能获得的 wrist link 位置、wrist link 旋转、Jacobian、root/pelvis 位姿、link 名称与索引。
- 当前代码中是否已经存在 IK、Jacobian、task-space reward 或 residual policy 实现。
- 当前 simple 和 residual 两套 split 网络是否仍然可用。

当前进展：

- 已完成审计，文档为 `docs/stage4_audit.md`。
- 已确认实际训练入口是 `humanoidverse/train.py::train_bfm_zero_split_z`。
- 已确认当前主 Agent 是 `FBcprAuxAgent`。
- 已确认当前模型主链路仍是 `FBcprAuxModel`、`SplitBackwardMap`、`SplitForwardMap`、`SplitActor` 等。
- 已确认当前 Replay Buffer、checkpoint 和环境 action/PD target 语义。

仍需注意：

- 后续任何接训练主路径的修改，仍然要先再次读当前代码，而不是仅凭本文档。

### 0.2 禁止混合旧实验逻辑

任务书要求：

- 不得将第四阶段代码直接混入旧 hand-MSE 路径。
- 不允许一个训练过程同时执行：
  - hand FB loss
  - hand latent MSE
  - task-space hand loss
  - Jacobian residual
- 除非配置明确指定，否则不得混用。

必须新增清晰模式开关：

```python
hand_control_mode: Literal[
    "legacy_fb",
    "legacy_mse",
    "task_space_4dof",
]
```

第四阶段主模式：

```python
hand_control_mode = "task_space_4dof"
```

每种模式必须有互斥的数据流和 loss 逻辑。

当前进展：

- 已新增 `Stage4Config`。
- 已定义三种模式：
  - `legacy_fb`
  - `legacy_mse`
  - `task_space_4dof`
- 已在配置验证层强制互斥。

未完成：

- `train_bfm_zero_split_z` 尚未正式接入 `hand_control_mode`。
- Agent/Actor/update loss 尚未根据 `task_space_4dof` 切换训练路径。

### 0.3 不要删除旧版本

任务书要求：

- 旧 split-z、hand FB、hand MSE 实现必须保留，用于消融实验。
- 新增第四阶段类或清晰分支，不要通过大量就地修改导致旧实验无法复现。

当前进展：

- 当前实现基本采用新增 `humanoidverse/agents/stage4/` 工具模块。
- legacy env 路径默认不启用 Stage4 hook。
- `legacy_mse` 和 `legacy_fb` 仍保留。

未完成：

- 训练入口正式接入时仍需保证旧模式不受影响。

### 0.4 不允许静默兼容

任务书要求：

- 遇到维度、索引、checkpoint 字段或网络结构不一致时：
  - 不允许静默裁剪。
  - 不允许随意补零。
  - 不允许只用 `strict=False` 然后忽略缺失项。
  - 必须打印详细迁移报告。
  - 必须通过断言检查维度。

当前进展：

- Stage4 joint/body/Jacobian mapping 通过名称和 shape 检查。

未完成：

- checkpoint 迁移策略尚未实现。
- Stage3/legacy body actor 权重如何迁移到 Stage4 body actor 尚未确定。

## 1. 第四阶段最终方法定义

任务书的总体定义：

- 第四阶段不再使用 `z_hand`、`B_hand`、`F_hand` 完成右手精确控制。
- 系统拆为三部分：
  - Body-BFM base policy
  - 4DoF Task-Space Hand Controller
  - Gated Body Coordination Residual

### 1.1 Body BFM Base Policy

任务书要求：

- 保留原 BFM 的 body 表示和策略能力。
- Body-BFM 负责：
  - 身体行为语义
  - 腿部、腰部、左臂动作
  - 平衡
  - locomotion
  - 全身动作风格
  - 恢复能力
  - body tracking、goal reaching、reward inference
- 第四阶段第一版中：
  - `B_body` 不读取右臂状态。
  - `F_body` 保持原有 FB 语义。
  - `F_body` 第一版不额外接收动态 hand command。
  - 原 body 策略尽量从第三阶段 checkpoint 继承。
- 这样做是为了避免动态 hand command 破坏 FB Bellman 递推的一致性。

当前进展：

- 尚未修改 Body-BFM 训练主路径。
- 尚未实现 Stage3 checkpoint 到 Stage4 body policy 的迁移。

需要完成：

- 明确 Stage4 body actor 如何产生 22D body action。
- 明确是否冻结 body base actor。
- 明确何时、如何小学习率解冻 body actor 后几层。

### 1.2 4DoF Task-Space Hand Controller

任务书要求：

- 右臂原有 7 个关节中只控制：
  - 3 个 shoulder 关节
  - 1 个 elbow 关节
- 三个 wrist 关节始终保持 0。
- hand controller 直接接收任务空间目标。
- 第一版只要求位置跟踪，不要求 wrist 姿态跟踪，因为 wrist 三自由度被锁定。
- Hand controller 输出 4 维主动关节目标。
- 随后组装成原机器人 29 维动作，其中三个 wrist 动作恒为 0。

当前进展：

- 已实现 4DoF DLS controller。
- 已实现 heading-frame wrist/end-effector position 和 Jacobian 输入。
- 已实现 29D action 组装。
- 已通过真实 IsaacSim smoke。

未完成：

- 训练过程中的 task-space hand command 来源尚未确定。
- 训练 actor 尚未正式切换为 body 22D + DLS hand 4D。

### 1.3 Gated Body Coordination Residual

任务书要求：

- Body 不是完全固定不动。
- 对于右臂单独无法到达或接近奇异位形的目标，新增小型协调残差。
- 原则：
  - 目标容易到达时，主要使用 4DoF 右臂。
  - 目标较远、接近关节限位或接近奇异位形时，允许腰和身体小幅协同。
  - 不直接让 hand task loss 重写整个 body BFM policy。
  - body 协调通过有幅值限制的小残差完成。

当前进展：

- 已实现规则 gate 原型：
  - `coordination_gate_raw`
  - `CoordinationGate`
- 已使用 DLS 输出中的 `sigma_min` 和 `joint_margin` 等信号。
- 真实 smoke 显示初始位形 `sigma_min` 很小，说明 coordination residual 的确必要。

未完成：

- 没有实现 body coordination residual network。
- 没有定义 residual 作用关节。
- 没有接训练主路径。

## 2. 右臂关节与动作索引审计

任务书要求：

- 不要直接假设当前右臂索引仍然是 22-28。
- 必须从当前实际机器人配置读取 `dof_names`。
- 建立名称驱动的索引映射。
- 如果找不到准确名称，直接抛异常，不允许退化成硬编码索引。
- 启动时必须打印：
  - Active right-arm joints
  - Wrist locked joints

任务书中预期右臂 7DoF 名称类似：

- `right_shoulder_pitch`
- `right_shoulder_roll`
- `right_shoulder_yaw`
- `right_elbow`
- `right_wrist_roll`
- `right_wrist_pitch`
- `right_wrist_yaw`

当前实际仓库中确认的名称为：

- `right_shoulder_pitch_joint`
- `right_shoulder_roll_joint`
- `right_shoulder_yaw_joint`
- `right_elbow_joint`
- `right_wrist_roll_joint`
- `right_wrist_pitch_joint`
- `right_wrist_yaw_joint`

当前实际索引：

- active action/dof indices: `(22, 23, 24, 25)`
- wrist action/dof indices: `(26, 27, 28)`

当前进展：

- 已实现 `RightArmJointIndices`。
- 已实现 `resolve_right_arm_joint_indices`。
- 已通过单元测试和真实 env print 验证。

## 3. Wrist 三个关节锁零

任务书要求：

- 三个 wrist 关节必须在多层防护下保持为 0，而不是只依赖 loss 学成 0。

### 3.1 Actor 输出层不输出 wrist 动作

任务书要求：

- Task-space hand head 只输出 4 维：

```python
active_hand_action.shape[-1] == 4
```

- 不应先输出 7 维再覆盖 wrist，因为那会浪费输出并可能产生无效梯度。

当前进展：

- 工具层已经要求 `active_hand_action` 为 4D。
- 当前正式训练 Actor 尚未改造，因此这一条在训练主路径还未完成。

### 3.2 动作组装时显式写零

任务书要求：

- 新增统一函数 `assemble_full_action`。
- 组装逻辑：
  - body indices 写入 body action
  - active hand indices 写入 4D active hand action
  - wrist indices 写入 0
  - assert wrist 全部为 0

当前进展：

- 已实现 `assemble_full_action`。
- 已通过动作拼接测试。

### 3.3 环境执行前再次安全覆盖

任务书要求：

- 在将动作送入 PD 控制器或环境之前，再执行一次：

```python
action[..., wrist_indices] = 0.0
```

- 这是为了防止下游噪声、残差或旧代码重新写入 wrist 动作。

当前进展：

- 已在 `LeggedRobotBase._pre_physics_step` 中实现 Stage4 条件 hook。
- 仅在 `hand_control_mode == "task_space_4dof"` 时生效。

### 3.4 默认姿态必须确认

任务书要求：

- 如果 action 是相对于 default joint position 的 PD 偏移，则 action=0 只会让 wrist 保持在 default_joint_pos，不一定是绝对关节角 0。
- 必须区分：
  - action = 0
  - absolute joint target = 0
- 如果目标是 wrist 绝对关节角始终为 0，则需要确保：
  - `default_joint_pos[wrist_indices] = 0.0`
  - 或者在生成 PD target 后强制：

```python
joint_target[..., wrist_indices] = 0.0
```

当前进展：

- 已审计当前 PD target 语义：

```python
jpos_target = action * action_scale + default_dof_pos + default_dof_pos_offset
```

- 已在 final PD target 层 clamp wrist absolute target 到 `0.0`。
- IsaacSim 和 P-controller 分支都做了 clamp。

### 3.5 Wrist 状态惩罚

任务书要求：

- 即使目标为 0，动态耦合也可能使实际关节产生偏差，因此可加入 wrist position/velocity 惩罚。
- 但 reward 只是辅助约束，主保证仍是硬锁零。

当前进展：

- 硬锁零已完成。
- 尚未新增正式训练 reward 中的 wrist position/velocity penalty。

## 4. Task-Space Hand Command

任务书要求：

- 新增统一命令结构 `HandTaskCommand`，包括：
  - `target_pos_root`
  - `target_lin_vel_root`
  - `position_mask`
  - `velocity_mask`
  - `command_id`
  - `command_done`
- 所有目标转换到 root/pelvis facing 坐标系。
- 第一版只做位置跟踪。
- 需要能获取：
  - `wrist_pos_root`
  - `wrist_lin_vel_root`

当前进展：

- 已实现 `HandTaskCommand`。
- 已实现 end-effector position 从 world 到 root heading frame 的转换。
- 已实现 snapshot 中的 `end_effector_pos_heading`。

未完成：

- 尚未实现正式 command sampler。
- 尚未确定 command 来源。
- 尚未确定是否使用 `target_lin_vel_root`。
- 尚未将 command 放入 observation/replay buffer。

必须向用户确认：

- command 是来自 motion reference，还是随机 reaching target，还是数据集中已有字段。
- 坐标系是否就是任务书写的 root/pelvis facing frame；当前实现是 heading frame。
- command reset/done 语义。
- command horizon 和采样频率。

## 5. Jacobian 必须只使用 4 个主动关节

任务书要求：

- 环境可能返回完整 6x29 Jacobian 或 3x29 位置 Jacobian。
- 三个 wrist 列必须完全排除。
- 不能先算 7DoF 解再把 wrist 清零，因为这样得到的 4DoF 剩余动作一般不再是原问题的解。

### 5.1 Damped Least Squares

任务书要求：

- 用 4DoF active arm 的 3x4 position Jacobian 做 DLS。
- DLS 应对 normal、rank-deficient、zero、batch Jacobian 都稳定。

当前进展：

- 已实现 `damped_least_squares`。
- 已实现 `select_active_position_jacobian`。
- 已实现 IsaacSim/PhysX Jacobian 35 列 offset 处理。
- 已通过 DLS 数值测试和真实 env smoke。

### 5.2 自适应阻尼

任务书给出配置建议：

- `damping_min = 0.02`
- `damping_max = 0.20`
- `singular_value_threshold = 0.08`

当前进展：

- 已实现 `adaptive_damping`。
- 已测试奇异值越小阻尼越大。

### 5.3 最大关节增量限制

任务书给出配置建议：

- `max_joint_delta = 0.05`
- 单位需要根据控制周期和动作定义确认。

当前进展：

- DLS 输出中已实现 `max_joint_delta` clamp。

仍需确认：

- `0.05` 对当前 env control frequency 和 action_scale 是否合适。

### 5.4 关节限位权重

任务书要求：

- 避免 IK 把肩肘推向极限。
- 完整方案可以是 weighted DLS。
- 第一版至少应输出后加入 joint-limit scaling。

当前进展：

- 已实现 `joint_margin_scale`。
- 已实现 `apply_joint_limit_scaling`。
- 已修复 env snapshot 中 active limits shape 为 `(num_envs, 4)`。

## 6. 防止跳变与 IK 分支切换

任务书关注：

- 动作变化。
- 关节目标变化。
- 末端速度连续性。
- 关节限位。
- 4DoF 控制 3D 位置存在 1D 冗余。

### 6.1 关节目标连续性

要求：

- 避免 DLS 解在冗余自由度或奇异附近出现跳变。

当前进展：

- 已实现 `JointCommandLimiter`，可做 EMA、速度限制、加速度限制。

未完成：

- 尚未把 limiter 接入真实 rollout 主路径。

### 6.2 动作变化惩罚

要求：

- 在训练中应有 action rate / action acceleration 等正则。

当前进展：

- 仅有工具/测试，没有正式 reward 接入。

### 6.3 二阶平滑

要求：

- 不要一开始把二阶权重设得过大，以免快速击球动作被严重抑制。

当前进展：

- limiter 支持 acceleration 限制。
- 权重和接入点未定。

### 6.4 末端速度连续性

要求：

- 可考虑 wrist/end-effector velocity continuity。

当前进展：

- snapshot 目前未暴露 wrist linear velocity。
- env 中有 rigid body velocity 可取，但尚未接入 Stage4 snapshot。

### 6.5 零空间舒适姿态

要求：

- 4DoF 位置控制有 1D nullspace。
- 应避免关节限位，倾向舒适姿态。

当前进展：

- 当前只做 joint-limit scaling。
- 没有 nullspace posture term。

### 6.6 与上一时刻解的软约束

要求：

- 需要约束解不要大幅偏离上一时刻。

当前进展：

- `JointCommandLimiter` 可作为基础，但尚未接真实 rollout。

### 6.7 低通滤波

任务书示例：

- `alpha = 0.6`
- 必须可配置。
- 必须测试高速动作下的滞后。

当前进展：

- `JointCommandLimiter` 支持 `ema_alpha`。
- 已测试基本行为。

### 6.8 速度与加速度限制器

当前进展：

- 已实现。
- 尚未接真实训练主路径。

## 7. Hand Controller 最终结构

任务书要求：

- 第四阶段优先使用 DLS。
- DLS 提供明确、稳定、可解释的基础控制。
- learned residual 用于补偿动态基座、PD 执行器、惯性和模型误差。

### 7.1 第一阶段只实现 DLS

要求：

- 先验证基本末端控制和 wrist 锁零。

当前进展：

- 已完成。
- 已通过真实 env DLS action smoke。

### 7.2 第二阶段增加小型 Residual

任务书要求：

- 增加 `hand_residual_policy`。
- `delta_hand` 受 `max_hand_residual` 限制。
- 示例 `max_hand_residual = 0.05`。
- residual 不能直接覆盖 DLS，而应以残差方式叠加。

当前进展：

- 未实现 hand residual。

需要明确：

- residual 输入。
- residual 输出是 4D active arm action/target residual。
- residual 是否乘 gate。
- residual 训练损失。
- residual 是否允许反传到 body base。

### 7.3 Hand Residual 输入

任务书中提到的输入包括类似：

- observation
- command
- DLS action
- wrist position/root frame
- wrist velocity/root frame
- Jacobian/reachability features

当前进展：

- 部分底层数据已有：
  - snapshot
  - target command dataclass
  - DLS output
  - sigma_min
  - joint_margin
- 未实现 residual input builder。

## 8. Body Coordination Residual

任务书要求：

- 不要让 hand loss 直接更新完整 body Actor。
- body residual 是小幅协调，不是重写 Body-BFM。

### 8.1 第一版允许协调的关节

任务书要求：

- 第一版不要直接允许全部 22 个 body 动作都被修改。
- 建议按阶段开放。
- 文档中明确提到后续 Stage 4D 可只开放 3 个 waist 关节。

当前进展：

- 未实现 body residual。

必须确认：

- 第一版 coordination residual 是否只允许 waist 3DoF。
- 是否允许 torso/legs/left arm。
- 每个关节的 residual scale。

### 8.2 残差幅值限制

任务书要求：

- 不同关节使用不同最大幅值。

当前进展：

- 未实现 body residual scale。

### 8.3 Base Policy 初期冻结

任务书要求：

- 初期冻结：
  - body base Actor
  - body F/B
- 只训练：
  - hand residual
  - coordination residual
- 稳定后再小学习率解冻 body Actor 后几层。

当前进展：

- 尚未接训练。
- 尚未定义 freeze/unfreeze schedule。

## 9. Coordination Gate

任务书要求：

- gate 应根据 reachability features 触发。
- 特征包括：
  - arm-only 可达性残差
  - 奇异性指标
  - 关节裕量指标
  - 位置误差等
- 使用 4DoF DLS 预测若干步但不实际执行，计算剩余误差。
- 最终 gate 输出应控制 coordination residual 的使用强度。

当前进展：

- 已实现规则 gate：
  - `coordination_gate_raw`
  - `CoordinationGate`
- 当前 gate 使用：
  - `ik_residual`
  - `sigma_min`
  - `joint_margin`
  - `position_error_norm`
- 已实现 EMA smoothing。

未完成：

- 未实现 arm-only 多步 rollout residual。
- 未接 body coordination residual。
- 未接训练日志。

## 10. Reward 与正则项

任务书要求：

- 如果第一版使用解析 DLS，不需要为了 hand 基础控制增加完整 Task-Space Critic。
- 若训练 hand residual 或 coordination residual，可使用 RL critic。
- reward 必须基于动作执行后的下一状态。

任务书列出的 reward/regularization 类别包括：

- 位置进步奖励。
- 绝对位置误差。
- 近目标精度奖励。
- 稳态保持。
- 主动右臂平滑正则。
- 关节限位。
- 奇异性惩罚。
- Wrist 锁定惩罚。
- Body 协调代价。
- Base policy 偏离代价。
- 足底与稳定性保留原 AuxCritic 相关内容。

当前进展：

- 尚未新增 Stage4 reward。
- 当前只完成控制工具和 smoke tests。

需要确认：

- Stage4A/B 是否完全无学习，仅 DLS smoke。
- Stage4C/D 是否开始引入 reward。
- reward 权重和日志指标。

## 11. 如果新增 Task-Space Critic

任务书要求：

- 只有在解析 DLS + 监督 residual 不足时才启用。
- 如果启用 task-space critic，Replay Buffer 必须保存：
  - `hand_command_t`
  - `hand_command_tp1`
  - `command_done`
- 当 `command_done=True` 时，下一目标不能跨 command 错配。

当前进展：

- 未实现 task-space critic。
- 未修改 replay buffer。

需要确认：

- 第一版是否完全不使用 task-space critic。
- 如果不用，是否仍预留 replay schema。

## 12. Replay Buffer 修改

任务书要求：

- 第四阶段至少增加若干 command/hand 状态字段。
- 任务书明确列出候选字段：
  - `command_id`
  - `command_done`
  - `wrist_pos_root`
  - `next_wrist_pos_root`
- 如果第一版不使用 Task-Space Critic，可以暂时不保存全部字段，但数据结构最好预留。
- 命令不能在采样 batch 时随意重新生成，否则动作、状态和目标不对应。

当前进展：

- 未修改 Replay Buffer。

必须确认：

- DLS-only rollout 是否需要写 replay。
- hand residual 训练是否要保存 command。
- coordination residual 训练是否要保存 gate/reachability features。

## 13. 第四阶段训练流程

任务书阶段划分：

### Stage 4A：离线索引与数值验证

要求：

- 正确提取 4 个主动关节。
- 正确锁定 3 个 wrist。
- 正确提取 3x4 Jacobian。
- DLS 数值稳定。
- 动作组装正确。
- 坐标变换正确。

当前进展：

- 已完成。

### Stage 4B：静态 Base + 4DoF DLS

要求：

- Body 使用稳定站立或已有 body policy。
- 测试 4DoF DLS 在可达空间内的精度。
- wrist 实际角度接近 0。
- 动作无大幅跳变。

当前进展：

- 已完成基本真实 env DLS action smoke。
- 尚未做长期/多目标精度评估。
- 尚未接 command sampler。

### Stage 4C：DLS 固定 + hand residual

要求：

- DLS 固定。
- 训练小型 hand residual，补偿动态误差。

当前进展：

- 未开始。

### Stage 4D：Body Coordination Residual

任务书提到：

- 只开放 3 个 waist 关节。
- Body 执行 BFM 动作，hand 跟踪独立动态目标。

当前进展：

- 未开始。

## 14. Checkpoint 策略

任务书要求：

- 不允许静默 shape mismatch。
- 不要为了删除 `z_hand` 立刻重构旧 body Actor 内部。
- 如果旧 body Actor 必须输入完整 `[z_body, z_hand]`，需要谨慎处理。
- 更好的方式是提取 stage3 Actor 的 body branch，保留其输入和权重映射。

当前进展：

- 尚未实现 checkpoint migration。

必须确认：

- Stage3 checkpoint 路径。
- 迁移哪些模块：
  - body actor branch
  - F_body
  - B_body
  - critic/aux critic
- 是否冻结迁移后的 body branch。
- 如何报告 missing/unexpected keys。

## 15. 必须编写的测试代码

任务书要求在正式训练前新增独立测试目录：

```text
tests/stage4/
```

### 15.1 关节名称与索引测试

要求：

- 测试 4 个主动关节。
- 测试 3 个 wrist 关节。
- 测试名称缺失时抛错。

当前进展：

- 已完成。

### 15.2 Wrist 硬锁零测试

要求：

- 随机生成 body 和 hand 动作，组装后检查：

```python
full_action[..., wrist_indices] == 0
```

- 经过环境动作预处理后仍为 0。

当前进展：

- 已完成。

### 15.3 动作拼接测试

要求：

- 给每组关节写入不同常数，验证没有错位。

当前进展：

- 已完成。

### 15.4 Jacobian 列选择测试

要求：

- 确保只选 4 个 active arm columns。
- wrist columns 完全排除。

当前进展：

- 已完成。

### 15.5 DLS 数值测试

要求覆盖：

- normal Jacobian
- rank-deficient Jacobian
- zero Jacobian
- batch Jacobian

当前进展：

- 已完成。

### 15.6 奇异值阻尼测试

当前进展：

- 已完成。

### 15.7 关节限位测试

当前进展：

- 已完成。

### 15.8 平滑器测试

当前进展：

- 已完成。

### 15.9 坐标变换测试

当前进展：

- 已完成。

### 15.10 Gate 测试

要求：

- 关节接近极限时 gate 提高。
- 奇异值低时 gate 提高。

当前进展：

- 已完成基础规则 gate 测试。

### 15.11 梯度隔离测试

任务书要求示例：

- 只反传 hand residual loss。
- `hand_residual_grad > 0`
- body/legacy hand 梯度不能错误混入。

当前进展：

- 已有 gradient isolation 测试覆盖当前 Stage4 工具层。
- 但真实 hand residual 尚未实现，因此 hand residual 专项梯度隔离仍未完成。

### 15.12 短仿真 Smoke Test

要求：

- wrist command 始终为 0。
- wrist 实际角度保持在容许范围。
- Jacobian 可计算。

当前进展：

- 已完成 opt-in IsaacSim smoke。
- 已验证 DLS action assembly + env.step。

### 15.13 短训练 Smoke Test

要求：

- checkpoint 保存。
- checkpoint 重新加载。
- 不通过这些测试不允许启动正式训练。

当前进展：

- 已有 tiny agent save/load smoke。
- 尚未有正式 Stage4 training path 的 short train save/load。

## 16. 正式训练前必须生成的审计输出

任务书要求正式训练前生成：

### 16.1 文件变更表

需要列出所有新增/修改文件。

当前进展：

- 尚未生成最终正式训练前变更表。

### 16.2 数据流图

必须覆盖：

- hand command
- DLS
- hand residual
- coordination residual
- wrist hard lock

当前进展：

- DLS 和 wrist hard lock 已有文字说明。
- hand residual 和 coordination residual 未实现，尚不能生成最终图。

### 16.3 Loss 表

任务书要求明确不同 loss 更新哪些模块，例如：

- Body FB 是否更新 Body Base。
- Hand residual 是否只更新 Hand Residual。
- Coordination residual 是否只更新 Coordination Residual。
- Critic 是否参与。

当前进展：

- 尚未接训练 loss，因此最终 loss 表未完成。

### 16.4 模式互斥检查

任务书要求：

- `task_space_4dof` 中 legacy hand MSE/FB 不运行。
- wrist output dimensions = 0。

当前进展：

- 配置层已实现互斥。
- 训练主路径尚未接入，仍需进一步检查。

### 16.5 测试报告

当前进展：

- 当前测试状态已记录在 `docs/stage4_taskbook_status.md`。
- 正式训练前还需更新最终测试报告。

## 17. 日志与监控指标

任务书提到的监控指标包括：

- `hand/wrist_position_abs_mean`
- `hand/wrist_velocity_abs_mean`
- `hand/residual_action_norm`
- `hand/residual_to_dls_ratio`
- `coord/residual_norm`
- `coord/waist_residual_norm`

风险监控包括：

- wrist 实际角度超过阈值。
- 连续动作跳变超过阈值。
- coordination residual 长期饱和。
- hand residual 远大于 DLS。

当前进展：

- 尚未接训练日志。
- 尚未添加这些 logger 指标。

## 18. 推荐配置

任务书给出过 Stage4 配置建议，识别到的关键项包括：

```yaml
locked_wrist_joint_names:
  - right_wrist_roll
  - right_wrist_pitch
  - right_wrist_yaw
wrist_absolute_target: 0.0
enforce_wrist_zero_before_env_step: true
enforce_wrist_zero_after_pd_target_build: true
```

DLS/控制建议：

```yaml
kp: 1.0
damping_min: 0.02
damping_max: 0.20
singular_value_threshold: 0.08
max_joint_delta: 0.05
nullspace_gain: 0.05
ema_alpha: 0.6
```

Residual 建议：

```yaml
hand_residual:
  max_residual: 0.05
gate_ema_alpha: 0.8
residual_scale: 0.05
```

Reward/regularization 权重建议：

```yaml
wrist_position_weight: 1.0
wrist_velocity_weight: 0.1
action_rate_weight: 0.01
action_acceleration_weight: 0.005
joint_limit_weight: 1.0
singularity_weight: 0.1
coordination_magnitude_weight: 0.05
```

当前进展：

- 部分参数已作为函数默认值实现。
- 尚未做统一 YAML/CLI 配置接入。

## 19. 推荐实现顺序

任务书推荐顺序识别如下：

1. Commit 1：审计和模式配置。
2. Commit 2：关节索引和 wrist 锁定。
3. Commit 3：坐标系和 4DoF Jacobian。
4. Commit 4：DLS 与安全限制。
5. Commit 5：DLS hand 和 smoke test。
6. 后续：只有 DLS 基线通过后实施 hand residual。
7. Commit 8：Body Coordination Residual。
8. Commit 9：短训练和 checkpoint 测试。

当前进展对应：

- 已完成 1-5 的大部分内容，并额外完成 IsaacSim env snapshot 和 DLS action assembly smoke。
- 尚未进入 hand residual、body coordination residual、正式训练 checkpoint 阶段。

## 20. 完成标准

任务书中识别到的完成标准包括：

- 4 个主动右臂关节名称和索引正确。
- 3 个 wrist 关节动作和最终 PD target 均为 0。
- 4DoF Jacobian 维度正确。
- DLS 在奇异输入下无 NaN。
- 关节限位和零空间项生效。
- 动作平滑器生效。
- 所有 Stage4 单元测试通过。

当前状态：

- 名称和索引：已完成。
- wrist action 和 final PD target 硬锁零：已完成。
- 4DoF Jacobian：已完成。
- DLS singular input finite：已完成。
- 关节限位：已完成基础 scaling。
- 零空间项：未完成完整 nullspace posture term。
- 动作平滑器：工具已完成，未接训练主路径。
- Stage4 单元测试：当前服务器结果 `42 passed, 2 skipped`。

## 21. 当前已完成代码修改总表

### 新增 Stage4 模块

- `humanoidverse/agents/stage4/__init__.py`
- `humanoidverse/agents/stage4/config.py`
- `humanoidverse/agents/stage4/actions.py`
- `humanoidverse/agents/stage4/kinematics.py`
- `humanoidverse/agents/stage4/control.py`
- `humanoidverse/agents/stage4/controller.py`
- `humanoidverse/agents/stage4/env_adapter.py`

### 修改环境

- `humanoidverse/envs/legged_base_task/legged_robot_base.py`

修改内容：

- Stage4 wrist action pre-step zero hook。
- Stage4 final PD target absolute zero hook。
- Stage4 right-arm index runtime print and cache。

### 新增/修改测试

- `tests/stage4/test_stage4_config.py`
- `tests/stage4/test_right_arm_joint_name_mapping.py`
- `tests/stage4/test_wrist_lock.py`
- `tests/stage4/test_action_assembly.py`
- `tests/stage4/test_active_arm_jacobian.py`
- `tests/stage4/test_hand_frame_transform.py`
- `tests/stage4/test_dls_solver.py`
- `tests/stage4/test_adaptive_damping.py`
- `tests/stage4/test_joint_limit_scaling.py`
- `tests/stage4/test_joint_command_limiter.py`
- `tests/stage4/test_coordination_gate.py`
- `tests/stage4/test_stage4_controller.py`
- `tests/stage4/test_cuda_tensor_smoke.py`
- `tests/stage4/test_env_wrist_hooks.py`
- `tests/stage4/test_gradient_isolation.py`
- `tests/stage4/test_stage4_env_smoke.py`
- `tests/stage4/test_short_save_load.py`
- `tests/stage4/test_env_snapshot.py`

### 新增文档

- `docs/stage4_audit.md`
- `docs/stage4_taskbook_status.md`
- `docs/stage4_taskbook_requirements_backcheck.md`

## 22. 当前仍需确认的信息

### 22.1 Task-space hand command 来源

必须确认：

- command 是 motion reference 的 wrist trajectory 吗？
- command 是随机 heading-frame reaching target 吗？
- command 是数据集中已有字段吗？
- command 是 env 内部 sampler 生成，还是 trainer/agent 生成？
- command 是否进入 observation？
- command 是否进入 replay buffer？

这是当前最重要阻塞点。

### 22.2 End-effector 定义

当前实际 body names 中没有 `right_hand_link`。

当前默认使用：

- `right_wrist_yaw_link`

需要用户确认：

- Stage4 末端是否就应该是 `right_wrist_yaw_link`。
- 是否需要在模型/URDF/USD 中新增或查找手掌 link。

### 22.3 Stage4 actor 结构

必须确认：

- Stage4 body actor 是否只输出 22D。
- 是否复用 legacy `SplitActor` body branch。
- 是否新建 Stage4 actor wrapper。
- 是否完全废弃 hand branch 输出。

### 22.4 Checkpoint 迁移

必须确认：

- 从哪个 Stage3 checkpoint 初始化。
- 迁移哪些权重。
- 如何处理旧 `z_hand` 输入。
- 是否冻结 body actor。
- 如何打印 migration report。

### 22.5 Replay Buffer 字段

必须确认：

- 第一版 DLS-only 是否不改 replay。
- hand residual 是否需要 command fields。
- coordination residual 是否需要 gate/reachability fields。
- 是否现在就预留字段。

### 22.6 Body Coordination Residual 作用范围

必须确认：

- 第一版 residual 是否只作用 3 个 waist joints。
- 是否允许 legs。
- 是否允许 left arm。
- 每个关节 residual scale。
- gate 是规则还是学习。

### 22.7 Reward 和日志指标

必须确认：

- DLS-only 是否无新 reward。
- hand residual / coordination residual 的 reward 权重。
- 需要记录哪些 logger keys。

## 23. 我对当前阶段的判断

根据任务书，当前项目已经完成：

```text
Stage 4A: 离线索引与数值验证
Stage 4B 的前半部分: 静态/真实 env 下 DLS-only action assembly 和 wrist lock smoke
```

当前尚未完成：

```text
正式 Stage4 训练入口
Task-space command sampler
Stage4 replay/observation schema
Stage4 actor/checkpoint migration
hand residual
gated body coordination residual
Task-space critic
正式短训练 save/load
正式训练前完整 loss 表和数据流图
```

最安全的下一步不是直接改训练，而是先确认并实现：

```text
Stage4CommandSampler / command source
```

然后通过：

```text
command -> snapshot -> DLS -> env.step
```

的 opt-in IsaacSim smoke 后，再接 `train_bfm_zero_split_z(..., hand_control_mode="task_space_4dof")`。

