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

`scripts/data_preprocess/g1_kinematics.py` 按 `torch_humanoid_batch.py` 的逻辑解析 `g1_29dof.xml`（深度优先 body 树、29 个 hinge 轴），并提供 numpy FK 用于离线验证/地面对齐。MJCF 默认自动在 `humanoidverse/data/robot/g1/` 与 `humanoidverse/data/robots/g1/` 下查找（不同 checkout 目录名可能不同）；找不到时用 `--mjcf` 显式指定。

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

## 2. data2 画形状 NPZ — V2 连续 IK（`convert_shape_npz_v2.py`，**推荐**）

**请用 data2 替代 data1。** data1（`convert_shape_npz.py`）因旧版 IK 存在大量右臂解支跳变，已弃用于训练。

输入：`data2/{batch_data_xy_v2, batch_data_xz_v2, batch_data_yz_v2}/clips_obs/{shape}/{plane}/*_obs.npz`  
详见 `new_data/data2/BATCH_DATA_V2_README.md`（30 fps × 300 帧，固定/连续 swivel IK + 质量筛选）。

1. **抽样**（默认与 data1 相同，控制三源比例）：
   - 每个平面 batch 目录分层抽 `--per-batch`（默认 250）条，在 10 种 shape 间均匀分配 → 约 **750** 条；
   - 或 `--use-all` 转换全部 `clips_obs`（约 **3000** 条，V2 已通过生成端筛选）。
2. **转换**（与 data1 相同，直接用 `qpos`，不用 `privileged_state`）：
   - `root_trans_offset = qpos[:, 0:3]`；
   - `pose_aa[:,0] = rotvec(qpos[:, 3:7])`；
   - `pose_aa[:,1:30] = qpos[:, 7:36] × axis`。
3. **背靠背验证**：重建 `state` 的 `dof_pos / dof_vel / proj_grav`，容差 1e-3，`n_state_check_failed` 必须为 0。
4. **右臂连续性报告**（V2 新增）：统计相邻帧右臂 7 关节最大 `|Δq|`，超过 `--jump-threshold`（默认 0.12 rad/帧，与生成端 `FilterThresholds` 一致）的 clip 写入 `jump_flagged`。V2 数据应接近 0 条 flagged。

输出 key 命名：`shape_{bd_xy|bd_xz|bd_yz}_{原文件名}`（仍匹配训练里 `shape_` 前缀的源权重）。

### 2c. 少量 ad-hoc clip → tracking inference（`prepare_tracking_clips.py`）

**不写入训练 pkl**，只做与 data2 相同的 **state↔qpos 背靠背校验** + 右臂连续性报告，输出 `*_obs.npz` 供 `tracking_inference_split.py` 直接读取。

```bash
python scripts/data_preprocess/prepare_tracking_clips.py \
    --obs-npz /path/to/circle_Pyz_..._seed003_F300_obs.npz \
              /path/to/circle_Pyz_..._seed007_F300_obs.npz \
    --raw-npz /path/to/seed003_L1_F300.npz \
              /path/to/seed007_L1_F300.npz \
    --output-dir humanoidverse/data/inference_clips/custom_circles

uv run -m humanoidverse.tracking_inference_split \
    --model-folder results/<your_checkpoint> \
    --traj-obs-dir humanoidverse/data/inference_clips/custom_circles \
    --traj-glob "*_obs.npz"
```

`--raw-npz` 可选；若 `*_obs.npz` 内已有 `qpos/qvel`（V2 标准格式），会自动用 obs 内字段做校验。

## 2b. data1 画形状 NPZ（`convert_shape_npz.py`，**已弃用**）

旧版 `data1/{batch_data, batch_data_xy, batch_data_xz}/clips_obs/**/*_obs.npz`。因 IK 解支跳变问题，**请勿再用于合并训练集**；脚本保留仅供对照或复现旧实验。

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

# 2) data2 NPZ -> pkl  （替代 data1）
python scripts/data_preprocess/convert_shape_npz_v2.py \
    --data2-dir /path/to/new_data/data2 \
    --output-pkl humanoidverse/data/shape_v2_29dof_clips.pkl

# 可选：使用全部 ~3000 条 V2 clip（不再分层抽样）
# python scripts/data_preprocess/convert_shape_npz_v2.py \
#     --data2-dir /path/to/new_data/data2 \
#     --output-pkl humanoidverse/data/shape_v2_29dof_clips.pkl \
#     --use-all

# 3) 合并
python scripts/data_preprocess/merge_datasets.py \
    --lafan-pkl humanoidverse/data/lafan_29dof_10s-clipped.pkl \
    --bones-pkl humanoidverse/data/bones_29dof_clips.pkl \
    --shape-pkl humanoidverse/data/shape_v2_29dof_clips.pkl \
    --output-pkl humanoidverse/data/combined_29dof_mixed.pkl

# 4) 训练环境内终验（逐条 FK，无多进程 / 无 bulk load，共享节点可跑）
python scripts/data_preprocess/verify_with_motion_lib.py \
    --pkl humanoidverse/data/combined_29dof_mixed.pkl \
    --device cpu

# 快速冒烟（只验前 100 条）：
# python scripts/data_preprocess/verify_with_motion_lib.py \
#     --pkl humanoidverse/data/combined_29dof_mixed.pkl --max-motions 100
```

每步都会输出 `.report.json` / `.manifest.json`，请重点核对：

- `convert_bones_csv` report：`mean_ground_z_shift_m`（典型应在 ±0.1 m 内；普遍很大说明 root 坐标约定理解有误）、`over_limit_ratio`（普遍偏高说明重定向质量差或单位错误）；
- `convert_shape_npz_v2` report：`n_state_check_failed` 必须为 0；`n_jump_flagged` 应接近 0（V2 连续 IK）；`continuity_max_arm_delta.p99` 应 ≤ 0.12 rad；
- `verify_with_motion_lib`：必须 PASS。

## 已知影响与配套调整（详见训练代码改动）

- 合并后约 2400+ 条 motion，专家 buffer（50 Hz 展开）显存占用约为原来的 2.3 倍；若 `buffer_device='cuda'` 吃紧可改 `'cpu'`。
- tracking 评估会遍历全部 motion，单次评估耗时同比例增加。
- 训练入口已支持 `--lafan-tail-path` 指向合并 pkl，并支持按数据源设置专家采样权重（默认 lafan:bones:shape = 4:4:2）。
