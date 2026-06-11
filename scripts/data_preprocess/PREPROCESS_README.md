# 新数据预处理说明（背靠背检查文档）

本目录将两类新采集数据转换为与 `lafan_29dof_10s-clipped.pkl` 完全一致的训练格式，并合并为单一数据集。所有脚本只依赖 `numpy / scipy / pandas / joblib`（`verify_with_motion_lib.py` 除外，它需要在训练环境中运行）。

## 目标格式（与 motion_lib 对齐）

`MotionLibRobot`（`humanoidverse/utils/motion_lib/motion_lib_base.py` 512–542 行）读取 joblib dict pkl，每条 motion 仅需三个字段：

| 字段 | shape | 说明 |
|------|-------|------|
| `root_trans_offset` | (T, 3) float32 | pelvis 世界系位置 [m] |
| `pose_aa` | (T, 30, 3) float32 | 轴角；`[:,0]` 为**全局** root 旋转 rotvec，`[:,1:30]` 为各关节**局部**旋转 = `angle × axis`（MJCF body 顺序 = `dof_names` 顺序，每 body 一个 hinge） |
| `fps` | int | 30 |

速度、body 位姿等全部由 motion_lib 运行时 FK + `np.gradient` 重算，**不需要**预存。

`scripts/data_preprocess/g1_kinematics.py` 按 `torch_humanoid_batch.py` 的逻辑解析 `g1_29dof.xml`（深度优先 body 树、29 个 hinge 轴），并提供 numpy FK 用于离线验证/地面对齐。脚本启动时会断言 MJCF 关节顺序与 `G1_DOF_NAMES`（即训练 yaml 的 `dof_names`）一致，顺序错位会直接报错而不是产出错误数据。

---

## 1. BONES-SEED CSV（`convert_bones_csv.py`）

输入：`selected_one_per_type/*.csv`，120 fps，root 平移单位 cm，所有角度单位**度**，root 旋转为**外旋 XYZ 欧拉角**，29 个关节列与 `dof_names` 顺序一致（已对照表头验证）。

每个 CSV 的处理步骤（顺序即代码顺序）：

1. **降采样** 120→30 fps：`df.iloc[::4]`（在单位换算之前做，纯抽帧无插值）。
2. **单位换算**：平移 cm→m；关节角与欧拉角 deg→rad。
3. **root 旋转**：`scipy Rotation.from_euler('xyz', deg, degrees=True)`（小写 = extrinsic，x→y→z 应用顺序，与数据说明文档一致）→ rotvec 存入 `pose_aa[:,0]`。
4. **关节角 → 轴角**：`pose_aa[:,1+j] = angle_j × axis_j`，`axis_j` 取自 MJCF（全部为正单位轴，故 motion_lib 用 `pose.sum(-1)` 恢复 dof_pos 时符号正确）。
5. **地面对齐**（LAFAN 数据是贴地的，BONES 重定向数据存在整体高度偏移）：
   - 用 numpy FK 计算每帧左右 `ankle_roll_link` 的世界 z，取两脚较低者；
   - 校准基准：MJCF home 位形（全零关节角、pelvis z=0.793，双脚平贴地面）下的 ankle z；
   - 对每条轨迹施加**常数** z 平移，使"每帧最低脚踝高度的 5% 分位数"等于校准基准（取分位数而非最小值，对重定向穿模帧鲁棒）；
   - 平移量逐条记录在 report 中（`ground_z_shift_m`），便于背靠背核对。
6. **裁剪**：
   - ≥300 帧的序列切成首尾相接、互不重叠的 300 帧片段；
   - 尾段以及总长 <300 帧的整条序列，若 ≥**160 帧**（5.33 s）则保留为变长片段，否则丢弃。
   - 160 帧阈值的依据：专家 buffer 以 50 Hz 重采样（`env.dt=0.02`），160 帧 ≈ 267 步 > `rollout_expert_trajectories_length=250` + 1，保证**所有**产出片段都能被三个训练组件使用（discriminator、z_expert 编码、专家 z-tracking rollout）。运行时已验证 `standardize_motion_length` 配置项实际未被代码使用，变长片段被 motion_lib / 专家 buffer 原生支持。
7. **限位检查**（仅报告不修改）：统计超出 MJCF 关节限位 ±0.05 rad 的样本占比，写入 report。

输出 key 命名：`bones_{csv文件名}_seg{i}`。

## 2. data1 画形状 NPZ（`convert_shape_npz.py`）

输入：`data1/{batch_data, batch_data_xy, batch_data_xz}/clips_obs/**/*_obs.npz`，已是 30 fps × 300 帧。

1. **分层抽样**：每个 batch 目录抽 `--per-batch`（默认 250）条，在 shape 子目录间均匀分配，`--seed 0` 固定可复现，共约 750 条（避免手画轨迹在合并数据集中占比过大）。
2. **转换**（直接用原始 MuJoCo 量，**不用** npz 里的 obs）：
   - `root_trans_offset = qpos[:, 0:3]`；
   - `pose_aa[:,0] = rotvec(qpos[:, 3:7])`（wxyz 四元数）；
   - `pose_aa[:,1:30] = qpos[:, 7:36] × axis`。
   - 不使用 `privileged_state`：其有限差分速度被错误缩放约 33 倍（BATCH_DATA_README §5.4）；motion_lib 会从 FK 重新差分出正确速度。
3. **背靠背验证**（每条都做）：从 `qpos/qvel` 重建 `state` 的 `dof_pos / dof_vel / proj_grav` 与 npz 内置 `state` 对比，容差 1e-3。该检查能捕获数据生成模型（`scene_g1_draggable.xml`）与训练模型（`g1_29dof.xml`）之间任何关节顺序 / 坐标约定不一致。`ang_vel` 的参考系在源文档中描述有歧义（`rot.apply` vs `rot.inv().apply`），脚本对三种候选变换分别计算误差并报告最优者，不作为失败条件（转换本身不依赖 ang_vel）。
4. 不做地面对齐（qpos 来自合法 MuJoCo 位形），但 report 中记录 FK 最低脚高供人工核查。

输出 key 命名：`shape_{bd|bd_xy|bd_xz}_{原文件名}`。

## 3. 合并（`merge_datasets.py`）

- LAFAN key **保持原样**（兼容评估日志/优先级采样按 key 索引），bones/shape 带前缀；
- 逐条校验：字段齐全、无 NaN/Inf、≥`--min-frames`（默认 120）、关节限位统计；
- 产出 `combined_29dof_mixed.pkl` + `combined_29dof_mixed.manifest.json`（各源条数、时长、逐条统计、被拒清单）。

数据源识别约定（训练代码同样使用）：key 以 `bones_` 开头 → bones；以 `shape_` 开头 → shape；其余 → lafan。

## 4. 训练环境内最终验证（`verify_with_motion_lib.py`，服务器运行）

用仓库真实的 `MotionLibRobot` 完整加载合并 pkl，逐条 FK，检查：无 NaN、FK 恢复的 `dof_pos` 与源 `pose_aa` 一致（<1e-3 rad）、各源 root 高度 / 关节速度统计是否合理。

---

## 服务器执行顺序

```bash
# 0) 依赖（任意 python>=3.10 环境）
pip install numpy scipy pandas joblib

# 1) BONES CSV -> pkl
python scripts/data_preprocess/convert_bones_csv.py \
    --input-dir /path/to/new_data/selected_one_per_type \
    --output-pkl humanoidverse/data/bones_29dof_clips.pkl

# 2) data1 NPZ -> pkl
python scripts/data_preprocess/convert_shape_npz.py \
    --data1-dir /path/to/new_data/data1 \
    --output-pkl humanoidverse/data/shape_29dof_clips.pkl

# 3) 合并
python scripts/data_preprocess/merge_datasets.py \
    --lafan-pkl humanoidverse/data/lafan_29dof_10s-clipped.pkl \
    --bones-pkl humanoidverse/data/bones_29dof_clips.pkl \
    --shape-pkl humanoidverse/data/shape_29dof_clips.pkl \
    --output-pkl humanoidverse/data/combined_29dof_mixed.pkl

# 4) 训练环境内终验（需要 torch + 仓库依赖）
python scripts/data_preprocess/verify_with_motion_lib.py \
    --pkl humanoidverse/data/combined_29dof_mixed.pkl
```

每步都会输出 `.report.json` / `.manifest.json`，请重点核对：

- `convert_bones_csv` report：`mean_ground_z_shift_m`（典型应在 ±0.1 m 内；普遍很大说明 root 坐标约定理解有误）、`over_limit_ratio`（普遍偏高说明重定向质量差或单位错误）；
- `convert_shape_npz` report：`n_state_check_failed` 必须为 0；`min_foot_z_m` 应接近 0；
- `verify_with_motion_lib`：必须 PASS。

## 已知影响与配套调整（详见训练代码改动）

- 合并后约 2400+ 条 motion，专家 buffer（50 Hz 展开）显存占用约为原来的 2.3 倍；若 `buffer_device='cuda'` 吃紧可改 `'cpu'`。
- tracking 评估会遍历全部 motion，单次评估耗时同比例增加。
- 训练入口已支持 `--lafan-tail-path` 指向合并 pkl，并支持按数据源设置专家采样权重（默认 lafan:bones:shape = 4:4:2）。
