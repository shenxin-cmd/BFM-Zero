# BFM-Zero Z 空间解耦：结构化潜空间分离方案

**版本**：v1.0  
**日期**：2026-05-06  
**状态**：待确认，未修改代码

---

## 一、目标摘要

将原来统一的 `z` 向量（原始 `z_dim` 维，例如 100 维）拆分为：

| 子空间 | 维度 | 控制范围 |
|--------|------|----------|
| `z_body` | 225 | 除右手外的所有身体部位（左腿/右腿/腰/左臂） |
| `z_hand` | 36  | 右手 7 个关节（右肩到右腕） |
| **合计** | **261** | — |

**设计约束**（根据用户确认）：
1. **不引入** 跨子空间正交损失（cross-orthogonality loss）
2. Critic 网络 **不拆分**，仍使用完整 `z`（261 维）作为输入
3. Discriminator 仅使用 `z_body`（225 维）和 body-only 观测
4. 所有配置可通过 YAML 文件中的 `z_hand_dim` 和 `z_body_dim` 控制

---

## 二、维度基础：观测空间分析

### 2.1 动作空间（action，29 维）

来源：`g1_29dof.yaml` 的 `dof_names` 字段（顺序即 action 输出顺序）：

| action 索引 | 关节 | 归属 |
|-------------|------|------|
| 0–5   | 左腿（left_hip_pitch ~ left_ankle_roll） | body |
| 6–11  | 右腿（right_hip_pitch ~ right_ankle_roll） | body |
| 12–14 | 腰部（waist_yaw/roll/pitch） | body |
| 15–21 | 左臂（left_shoulder_pitch ~ left_wrist_yaw） | body |
| **22–28** | **右臂（right_shoulder_pitch ~ right_wrist_yaw）** | **hand** |

```python
ACTION_BODY_IDX = slice(0, 22)   # a_body: 22 维
ACTION_HAND_IDX = slice(22, 29)  # a_hand:  7 维
```

### 2.2 观测空间各 key 的维度

训练时使用 Isaac Sim 环境，观测 dict 各 key 维度如下：

| Key | 构成 | 维度 |
|-----|------|------|
| `state` | `[dof_pos(29), dof_vel(29), projected_gravity(3), base_ang_vel(3)]` | **64** |
| `privileged_state` | `max_local_self`（见下） | **463** |
| `last_action` | 上一步 action | **29** |
| `history_actor` | 4 步历史×(3+3+29+29+29)=4×93 | **372** |

`privileged_state`（463 维）内部结构（`root_height_obs: True`，31 个 rigid body）：

| 字段 | 维度 | 偏移（privstate 内） |
|------|------|---------------------|
| root_height | 1 | 0:1 |
| local_body_pos（30 体，排除 root） | 90 | 1:91 |
| local_body_rot（31 体，6D） | 186 | 91:277 |
| local_body_vel（31 体） | 93 | 277:370 |
| local_body_ang_vel（31 体） | 93 | 370:463 |

### 2.3 Body 名称与索引

`body_names`（30 个物理 link）在 rigid body 数组中的顺序：

| rb 索引 | link 名称 | 归属 |
|---------|-----------|------|
| 0 | pelvis（root，排除在 local_body_pos 外） | — |
| 1–6 | left leg | body |
| 7–12 | right leg | body |
| 13–15 | waist（waist_yaw/roll，torso） | body |
| 16–22 | left arm | body |
| **23–29** | **right arm（right_shoulder_pitch ~ right_wrist_yaw）** | **hand** |
| 30 | head_link（扩展 body） | body |

---

## 三、B 网络输入切分索引

B 网络使用 `DictInputConcatFilter(key=["state", "privileged_state"])`，输入维度 = 64 + 463 = **527**。

拼接顺序：`[state(0:64), privstate(64:527)]`

### 3.1 右手（HAND）索引（119 维）

| 来源 | 区间 | 维度 | 说明 |
|------|------|------|------|
| state: dof_pos[22:29] | `[22, 29)` | 7 | 右臂关节位置 |
| state: dof_vel[22:29] | `[51, 58)` | 7 | 右臂关节速度 |
| priv: local_body_pos 右臂体（slot 22–28） | `[131, 152)` | 21 | 右臂 link 局部位置 |
| priv: local_body_rot 右臂体（entry 23–29） | `[293, 335)` | 42 | 右臂 link 旋转 |
| priv: local_body_vel 右臂体（entry 23–29） | `[410, 431)` | 21 | 右臂 link 速度 |
| priv: local_body_ang_vel 右臂体（entry 23–29） | `[503, 524)` | 21 | 右臂 link 角速度 |
| **合计** | | **119** | |

> 偏移计算（以 `local_body_pos` 为例）：
> - priv 内偏移 = 1（root_height）+ 22×3 = 67，end = 1+29×3 = 88
> - 在 527 维向量中 = 64+67=131 到 64+88=152

### 3.2 Body 索引（408 维）

```python
HAND_IDX_B = (
    list(range(22, 29))   +  # state: dof_pos 右臂
    list(range(51, 58))   +  # state: dof_vel 右臂
    list(range(131, 152)) +  # privstate: local_body_pos 右臂
    list(range(293, 335)) +  # privstate: local_body_rot 右臂
    list(range(410, 431)) +  # privstate: local_body_vel 右臂
    list(range(503, 524))    # privstate: local_body_ang_vel 右臂
)
BODY_IDX_B = [i for i in range(527) if i not in set(HAND_IDX_B)]
# len(HAND_IDX_B) = 119,  len(BODY_IDX_B) = 408
```

---

## 四、新网络架构设计

### 4.1 SplitBackwardMap

```
B_hand(s_f)  ← 只看右手观测切片（119 维）→ z_hand（36 维，Norm 归一化）
B_body(s_f)  ← 只看 body 观测切片（408 维） → z_body（225 维，Norm 归一化）
```

各自独立 MLP（结构与原 `BackwardMap` 一致）：
- `[input_dim → hidden_dim(256) → hidden_dim(256) → z_dim] + Norm`

### 4.2 SplitForwardMap

```
共享 Trunk:  obs_full(928 维) → e_context(256 维)
Hand Head:   cat(e_context, a_hand(7),  z_hand(36))  → F_hand(36  维)
Body Head:   cat(e_context, a_body(22), z_body(225)) → F_body(225 维)
输出: cat([F_body, F_hand], dim=-1) → 261 维
```

其中 `obs_full = [state, privileged_state, last_action, history_actor]` = 928 维。

Trunk 结构建议（与原 ForwardMap 的 `simple_embedding` 对齐）：
```
Linear(928 → 512) → LayerNorm(512) → Tanh
Linear(512 → 256) → ReLU
```

Head 结构（建议与原 hidden_dim 对齐）：
```
# Hand Head
Linear(256+7+36=299   → 512) → ReLU → Linear(512 → 36)
# Body Head
Linear(256+22+225=503 → 512) → ReLU → Linear(512 → 225)
```

> **关于 num_parallel（集成）**：  
> ForwardMap 原本使用 2 个并行集成模型（`DenseParallel`）。SplitForwardMap 同样需要支持 `num_parallel=2`。推荐做法：Trunk 和两个 Head 都使用 `DenseParallel`，或者在最外层包裹 N 个独立 SplitForwardMap 实例（参考 `SequetialFMap`）。建议初始实现用 2 个独立实例的方式（更简单，不需要修改并行层）。

### 4.3 SplitActor

```
共享 Trunk:  obs_full(928 维) → e_context(256 维)
Hand Policy: cat(e_context, z_hand(36))  → mu_hand(7  维)
Body Policy: cat(e_context, z_body(225)) → mu_body(22 维)
输出: cat([mu_body(22), mu_hand(7)], dim=-1) = 29 维  ← 顺序与 dof_names 一致
```

Trunk 与 Head 结构建议（同 SplitForwardMap）。

> **关键**：输出拼接顺序为 `[mu_body(0:22), mu_hand(22:29)]`，对应 dof_names 中 body 在前、right arm 在后的顺序。

### 4.4 SplitDiscriminator（包装器）

Discriminator 使用完整的 body-only 观测（不含右手）和 `z_body`。可参考现有 `FilterDiscriminator` 的模式，新建 `SplitDiscriminator`：

```python
class SplitDiscriminator(nn.Module):
    """包装现有 Discriminator，自动切分 obs（body only）和 z（z_body）"""
    def __init__(self, base_discriminator, body_obs_idx, z_body_dim):
        ...
    def forward(self, obs, z):
        # obs_body = obs[..., body_obs_idx]
        # z_body   = z[..., :z_body_dim]
        return self._base(obs_body, z_body)
    def compute_logits(...): ...
    def compute_reward(...): ...
```

底层 `Discriminator` 使用：
- `obs_dim = 408`（body-only 的 B 输入维度）
- `z_dim = z_body_dim = 225`

---

## 五、project_z / sample_z 修改（`fb/model.py`）

### 5.1 split 模式下的 project_z

```python
def project_z(self, z):
    if self.cfg.archi.norm_z:
        z_body = z[..., :self.cfg.archi.z_body_dim]
        z_hand = z[..., self.cfg.archi.z_body_dim:]
        # 独立归一化，各自缩放到 sqrt(dim)
        z_body = math.sqrt(z_body.shape[-1]) * F.normalize(z_body, dim=-1)  # ×15
        z_hand = math.sqrt(z_hand.shape[-1]) * F.normalize(z_hand, dim=-1)  # ×6
        z = torch.cat([z_body, z_hand], dim=-1)
    return z
```

- `sqrt(225) = 15.0`，`sqrt(36) = 6.0`

### 5.2 split 模式下的 sample_z

```python
def sample_z(self, size, device="cpu"):
    z_dim = self.cfg.archi.z_body_dim + self.cfg.archi.z_hand_dim
    z = torch.randn((size, z_dim), dtype=torch.float32, device=device)
    return self.project_z(z)
```

---

## 六、损失函数修改（`fb/agent.py` → `update_fb`）

### 6.1 原始 FB 损失

```python
# 原始（单一 z）
Ms = torch.matmul(Fs, B.T)   # num_parallel × batch × batch
diff = Ms - discount * target_M
fb_loss = 0.5 * (diff * off_diag).pow(2).sum() / off_diag_sum - diagonal(diff).mean() * Fs.shape[0]
```

### 6.2 拆分后的 FB 损失

```python
z_body_dim = self.cfg.model.archi.z_body_dim
z_hand_dim = self.cfg.model.archi.z_hand_dim

# 前向网络输出（num_parallel × batch × 261）
Fs_all = self._model._forward_map(obs, z, action)   # F 输出 [F_body(225), F_hand(36)]
F_body = Fs_all[..., :z_body_dim]                   # num_parallel × batch × 225
F_hand = Fs_all[..., z_body_dim:]                   # num_parallel × batch × 36

# 后向网络输出（batch × 261）
B_all  = self._model._backward_map(goal)             # B 输出 [B_body(225), B_hand(36)]
B_body = B_all[:, :z_body_dim]                       # batch × 225
B_hand = B_all[:, z_body_dim:]                       # batch × 36

# target（同上，用 target 网络）
target_Fs_all = ...
target_B_all  = ...
target_F_body = target_Fs_all[..., :z_body_dim]
target_F_hand = target_Fs_all[..., z_body_dim:]
target_B_body = target_B_all[:, :z_body_dim]
target_B_hand = target_B_all[:, z_body_dim:]

# 分别计算 M（batch×batch 矩阵）
Ms_body = torch.matmul(F_body, B_body.T)  # num_parallel × batch × batch
Ms_hand = torch.matmul(F_hand, B_hand.T)

target_Ms_body = torch.matmul(target_F_body, target_B_body.T)
target_Ms_hand = torch.matmul(target_F_hand, target_B_hand.T)
_, _, target_M_body = self.get_targets_uncertainty(target_Ms_body, pessimism)
_, _, target_M_hand = self.get_targets_uncertainty(target_Ms_hand, pessimism)

# Body FB 损失
diff_body = Ms_body - discount * target_M_body
fb_body_offdiag = 0.5 * (diff_body * self.off_diag).pow(2).sum() / self.off_diag_sum
fb_body_diag    = -torch.diagonal(diff_body, dim1=1, dim2=2).mean() * Ms_body.shape[0]
fb_loss_body = fb_body_offdiag + fb_body_diag

# Hand FB 损失
diff_hand = Ms_hand - discount * target_M_hand
fb_hand_offdiag = 0.5 * (diff_hand * self.off_diag).pow(2).sum() / self.off_diag_sum
fb_hand_diag    = -torch.diagonal(diff_hand, dim1=1, dim2=2).mean() * Ms_hand.shape[0]
fb_loss_hand = fb_hand_offdiag + fb_hand_diag

# 加权合并（hand 维度远小于 body，加大 hand 权重使损失幅度均衡）
HAND_LOSS_WEIGHT = 6.25   # (z_body_dim / z_hand_dim) ≈ 225/36 ≈ 6.25
fb_loss = HAND_LOSS_WEIGHT * fb_loss_hand + 1.0 * fb_loss_body
```

### 6.3 正交归一损失（保持不变）

正交损失分别对 `B_body` 和 `B_hand` 独立计算：

```python
# B_body 正交损失
Cov_body = torch.matmul(B_body, B_body.T)
orth_body = (0.5*(Cov_body*off_diag).pow(2).sum()/off_diag_sum) - Cov_body.diag().mean()

# B_hand 正交损失
Cov_hand = torch.matmul(B_hand, B_hand.T)
orth_hand = (0.5*(Cov_hand*off_diag).pow(2).sum()/off_diag_sum) - Cov_hand.diag().mean()

fb_loss += ortho_coef * (orth_body + orth_hand)
```

### 6.4 Q-Loss（`q_loss_coef` 分支，可选）

如果启用 q_loss（`q_loss_coef > 0`），同样按 body/hand 分别计算并加权：

```python
z_body = z[:, :z_body_dim]
z_hand = z[:, z_body_dim:]

next_Qs_body = (target_F_body * z_body).sum(dim=-1)  # num_parallel × batch
next_Qs_hand = (target_F_hand * z_hand).sum(dim=-1)
_, _, next_Q_body = get_targets_uncertainty(next_Qs_body, pessimism)
_, _, next_Q_hand = get_targets_uncertainty(next_Qs_hand, pessimism)

# implicit reward 使用全 B / 全 z 计算（或分 body/hand，建议保持统一）
# ... 同理加权

Qs_body = (F_body * z_body).sum(dim=-1)
Qs_hand = (F_hand * z_hand).sum(dim=-1)
q_loss = 0.5 * num_parallel * (
    F.mse_loss(Qs_body, targets_body) +
    HAND_LOSS_WEIGHT * F.mse_loss(Qs_hand, targets_hand)
)
fb_loss += q_loss_coef * q_loss
```

---

## 七、Actor 损失修改（`fb/agent.py` → `update_td3_actor`）

```python
z_body_dim = self.cfg.model.archi.z_body_dim

dist = self._model._actor(obs, z, self._model.cfg.actor_std)
action = dist.sample(clip=self.cfg.train.stddev_clip)

Fs_all = self._model._forward_map(obs, z, action)  # num_parallel × batch × 261
F_body = Fs_all[..., :z_body_dim]
F_hand = Fs_all[..., z_body_dim:]

z_body = z[:, :z_body_dim]
z_hand = z[:, z_body_dim:]

Qs_body = (F_body * z_body).sum(-1)  # num_parallel × batch
Qs_hand = (F_hand * z_hand).sum(-1)

_, _, Q_body = self.get_targets_uncertainty(Qs_body, self.cfg.train.actor_pessimism_penalty)
_, _, Q_hand = self.get_targets_uncertainty(Qs_hand, self.cfg.train.actor_pessimism_penalty)

HAND_Q_WEIGHT = 2.5
actor_loss = -(HAND_Q_WEIGHT * Q_hand + 1.0 * Q_body).mean()
```

---

## 八、fb_cpr 路径修改

### 8.1 `fb_cpr/model.py`

在 `FBcprModel.__init__` 中，Discriminator 构建改为：

```python
# 原来：
self._discriminator = cfg.archi.discriminator.build(obs_space, cfg.archi.z_dim)

# 改为（自动适配拆分模式）：
z_disc_dim = getattr(cfg.archi, 'z_body_dim', cfg.archi.z_dim)
self._discriminator = cfg.archi.discriminator.build(obs_space, z_disc_dim)
```

Discriminator 的 obs_space 需要对应 body-only 的 408 维输入。推荐通过 `SplitDiscriminatorArchiConfig`（新增）配置，在 `build` 内部自动创建 `SplitDiscriminator`。

Critic 保持不变，使用完整 z（261 维）：
```python
self._critic = cfg.archi.critic.build(obs_space, cfg.archi.z_body_dim + cfg.archi.z_hand_dim, action_dim, output_dim=1)
```

### 8.2 `fb_cpr/agent.py` → `update_actor`

```python
# 拆分 z
z_body_dim = self.cfg.model.archi.z_body_dim
z_body = z[:, :z_body_dim]
z_hand = z[:, z_body_dim:]

# Discriminator 的 compute_reward 内部通过 SplitDiscriminator 包装自动处理 obs 和 z 的切分
Q_discriminator_val = ...（与原来相同，SplitDiscriminator 内部切分）

# FB Q（加权）
Fs_all = self._model._forward_map(obs, z, action)
F_body = Fs_all[..., :z_body_dim]
F_hand = Fs_all[..., z_body_dim:]
Qs_fb_body = (F_body * z_body).sum(-1)
Qs_fb_hand = (F_hand * z_hand).sum(-1)
_, _, Q_fb_body = get_targets(Qs_fb_body, pessimism)
_, _, Q_fb_hand = get_targets(Qs_fb_hand, pessimism)

HAND_Q_WEIGHT = 2.5
Q_fb = HAND_Q_WEIGHT * Q_fb_hand + Q_fb_body

weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
actor_loss = -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight - Q_fb.mean()
```

### 8.3 `update_discriminator`（无需修改）

`SplitDiscriminator` 包装器透明地处理观测和 z 的切分，`update_discriminator` 代码无需修改。

### 8.4 `update_critic`（无需修改）

Critic 不拆分，接收完整 z（261 维），代码无需修改。

---

## 九、fb_cpr_aux 路径修改

### 9.1 `fb_cpr_aux/agent.py` → `update_actor`

`FBcprAuxAgent.update_actor` 中的 `Q_fb` 部分需与 `FBcprAgent.update_actor` 同步修改（加权 body/hand Q 值）。`aux_critic` 相关逻辑保持不变。

---

## 十、配置修改

### 10.1 `FBModelArchiConfig`（`fb/model.py`）

```python
class FBModelArchiConfig(BaseConfig):
    z_dim: int = 100       # 保留向后兼容（非 split 模式使用）
    z_body_dim: int = 225  # NEW: body 子空间维度
    z_hand_dim: int = 36   # NEW: hand 子空间维度
    norm_z: bool = True

    # ... (f, b, actor 字段增加 SplitXxxArchiConfig 的 Union 类型)
    f: ForwardArchiConfig | ForwardFilterArchiConfig | SplitForwardArchiConfig = ...
    b: BackwardArchiConfig | BackwardFilterArchiConfig | SplitBackwardArchiConfig = ...
    actor: ... | SplitActorArchiConfig = ...

    @property
    def total_z_dim(self) -> int:
        """当使用 split 模式时，返回 z_body_dim + z_hand_dim"""
        return self.z_body_dim + self.z_hand_dim
```

### 10.2 YAML 配置示例（在 agent 配置的 `archi` 节点下）

```yaml
model:
  archi:
    z_body_dim: 225
    z_hand_dim: 36
    norm_z: true

    b:
      name: SplitBackwardArchi
      hidden_dim: 256
      hidden_layers: 2
      norm: true
      hand_obs_indices: [22,23,24,25,26,27,28,51,52,53,54,55,56,57,
                         131,...,151, 293,...,334, 410,...,430, 503,...,523]
      # 或者通过 split_b_hand_obs_dim: 119, split_b_body_obs_dim: 408 + 固定索引

    f:
      name: SplitForwardArchi
      hidden_dim: 1024
      trunk_hidden_dim: 256
      num_parallel: 2

    actor:
      name: SplitActorArchi
      hidden_dim: 1024
      trunk_hidden_dim: 256

    discriminator:
      name: SplitDiscriminatorArchi
      hidden_dim: 1024
      hidden_layers: 2
      # 内部自动使用 z_body_dim 和 body_obs_indices
```

---

## 十一、需要新增/修改的文件列表

### 新增内容

**`humanoidverse/agents/nn_models.py`**（在现有类之后新增）：
- `SplitBackwardArchiConfig`
- `SplitBackwardMap`（含 hand/body 两条 MLP 分支）
- `SplitForwardArchiConfig`
- `SplitForwardMap`（含共享 Trunk + Hand Head + Body Head）
- `SplitActorArchiConfig`
- `SplitActor`（含共享 Trunk + Hand Policy + Body Policy）
- `SplitDiscriminatorArchiConfig`
- `SplitDiscriminator`（包装器，obs + z 双切分）

### 修改内容

| 文件 | 修改内容 |
|------|----------|
| `agents/fb/model.py` | `FBModelArchiConfig` 增加 `z_body_dim`/`z_hand_dim`；`sample_z`/`project_z` 按 split 模式分支处理 |
| `agents/fb/agent.py` | `update_fb` 改为分 hand/body 计算加权 FB 损失；`update_td3_actor` 改为加权 Q 值 |
| `agents/fb_cpr/model.py` | `FBcprModel.__init__` 中 discriminator 使用 `z_body_dim`；discriminator Union 类型增加 `SplitDiscriminatorArchiConfig` |
| `agents/fb_cpr/agent.py` | `update_actor` 中 `Q_fb` 改为加权版本 |
| `agents/fb_cpr_aux/agent.py` | `update_actor` 中 `Q_fb` 同步加权 |
| `agents/nn_filter_models.py` | 可选：将 `SplitDiscriminator` 移至此文件，与 `FilterDiscriminator` 并列 |

---

## 十二、实现顺序建议

1. **第一步**：在 `nn_models.py` 中实现 `SplitBackwardMap` 和配套 Config
2. **第二步**：在 `nn_models.py` 中实现 `SplitForwardMap` 和配套 Config（先单 parallel，再扩展为 2 parallel）
3. **第三步**：在 `nn_models.py` 中实现 `SplitActor` 和配套 Config
4. **第四步**：在 `nn_models.py` 中实现 `SplitDiscriminator`（包装器）
5. **第五步**：修改 `fb/model.py`（Config + `project_z` + `sample_z`）
6. **第六步**：修改 `fb/agent.py`（`update_fb` + `update_td3_actor`）
7. **第七步**：修改 `fb_cpr/model.py` 和 `fb_cpr/agent.py`
8. **第八步**：修改 `fb_cpr_aux/agent.py`
9. **第九步**：更新 YAML 配置文件

---

## 十三、关键设计决策说明

| 决策 | 选择 | 理由 |
|------|------|------|
| B 网络 hand 侧是否包含 root_height/gravity | **否** | B_hand 只看纯右臂状态，避免全局信息污染 hand 的特征表示 |
| F 网络 Trunk vs. 完全独立双头 | **共享 Trunk** | 全身状态对手臂控制有依赖（重力补偿、基座运动等），共享 context 合理 |
| Actor 输出顺序 | **[body(0:22), hand(22:29)]** | 与 `dof_names` 顺序严格对齐，避免 PD 控制映射错位 |
| Discriminator z | **z_body only** | Discriminator 约束全身风格（非右手），使用 z_body 保持逻辑一致 |
| Critic z | **全 z（261 维）** | Critic 估计整体价值，不拆分更稳定 |
| 正交损失 | **各自独立** | B_hand 和 B_body 分别保持内部正交归一 |
| 跨子空间正交损失 | **暂不加入** | 优先验证基本 split 有效性，后续可按需添加 |
| hand/body 损失权重 | **6.25 : 1** | 按 z_body_dim/z_hand_dim = 225/36 ≈ 6.25，使两侧 MSE 幅度均衡 |

---

## 十四、待确认事项

1. 上述实现方案是否符合预期？
2. `SplitForwardMap` 的 `num_parallel=2` 实现方式：
   - 方案 A：对每个 head 使用 `DenseParallel`（需改写 head 的 forward）
   - 方案 B：创建 2 个独立的 `SplitForwardMap` 实例（类似 `SequetialFMap`），更简单但参数量翻倍
3. YAML 配置中 `hand_obs_indices` 是否硬编码在 Config 类中（作为默认值），还是每次通过配置文件显式传入？

---

*确认后开始逐步修改代码。*
