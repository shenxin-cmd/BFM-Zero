"""
tracking_inference_npz.py
=========================
从单个 .npz 轨迹文件（而非 lafan 运动库）做 tracking inference。

输入 NPZ 格式（与 g1_circle_7_obs.npz 一致）
------------------------------------------------
  state             (N, 64)   float32
      = [dof_pos_rel(29), dof_vel(29), proj_grav(3), ang_vel(3)]
      dof_pos_rel = qpos[7:36] - DEFAULT_JOINT_POS   （关节角相对默认姿态偏移）
      dof_vel     = qvel[6:35]
      proj_grav   = R⁻¹ · [0, 0, -1]                （重力在机体系投影）
      ang_vel     = R · ω_body                        （根角速度）

  last_action       (N, 29)   float32
      = 上一步 action（通常近似为 dof_pos_rel）

  privileged_state  (N, D)    float32
      = compute_humanoid_observations_max 的输出：
        [root_h(1)?, local_body_pos, local_body_rot_6d, local_body_vel, local_body_ang_vel]
        D 典型值：448（30 刚体，含 root_height）、447（不含）、
                  463（31 刚体，含 root_height）、462（不含）

关于第 0 帧速度问题
-------------------
privileged_state 中的 body_vel / body_ang_vel 是有限差分：
  frame 0: body_pos_prev = None → vel = 0（不准）
  frame 1+: (pos_t - pos_{t-1}) / dt         （正确）

因此与原始 tracking_inference.py 保持一致，backward_map 的输入跳过第 0 帧：
  z = tracking_inference(obs[1:])
frame 0 的错误速度被天然丢弃。

环境初始化
----------
npz 不含根轨迹（root pos/quat/vel），Isaac 环境初始化时使用：
  - root 状态：env.base_init_state
  - dof 状态：state[0, :29] + env.default_dof_pos（绝对关节角）、state[0, 29:58]（关节速度）
专家侧 MuJoCo 视频中根保持静止，只反映关节运动对比。

兼容性
------
若 privileged_state 维度比模型期望多一个刚体（即包含 head_link 扩展体），
脚本会自动裁剪对齐，与 tracking_inference2.py 逻辑相同。

用法
----
    python -m humanoidverse.tracking_inference_npz \\
        --model-folder  workdir/bfmzero-split-z/<run-id> \\
        --npz-path      g1_circle_7_obs.npz \\
        [--episode-len  500] \\
        [--save-mp4] \\
        [--device cuda] \\
        [--headless True]
"""
from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import joblib
import json
import numpy as np
import torch
from torch.utils._pytree import tree_map

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_isaac import (
    HumanoidVerseIsaacConfig,
    IsaacRendererWithMuJoco,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import export_meta_policy_as_onnx

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# NPZ 加载工具
# ---------------------------------------------------------------------------

def _normalize_key(k: str) -> str:
    """统一键名：去空格、小写、修正 priviledged 拼写。"""
    k = k.strip().lower().replace(" ", "_")
    if k == "priviledged_state":
        k = "privileged_state"
    return k


def load_npz_obs(path: Path) -> dict[str, np.ndarray]:
    """加载 NPZ，返回 {key: (T, D) float32 ndarray}，必须包含三个必要键。"""
    raw = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in raw.files:
        out[_normalize_key(k)] = np.asarray(raw[k], dtype=np.float32)
    raw.close()

    for req in ("state", "last_action", "privileged_state"):
        if req not in out:
            raise KeyError(
                f"NPZ 文件 {path.name!r} 缺少必须的键 '{req}'；"
                f"现有键: {list(out.keys())}"
            )
    return out


# ---------------------------------------------------------------------------
# privileged_state 维度对齐
# ---------------------------------------------------------------------------

def _get_model_priv_dim(model) -> int:
    """从模型的 obs_space 读取 privileged_state 期望维度。"""
    return int(model.obs_space.spaces["privileged_state"].shape[0])


def _align_privileged_state(
    priv: np.ndarray,
    target_dim: int,
    root_height_obs: bool,
) -> np.ndarray:
    """
    若 privileged_state 维度与模型不符，尝试自动裁剪对齐。

    支持的转换（对应 nums_extend_bodies=1 即去掉 head_link 扩展体）：
      463 → 448  含 root_height（30+1 → 30 刚体）
      462 → 447  不含 root_height

    layout: [root_h(1)?] [local_pos(N*3-3)] [local_rot_6d(N*6)]
            [local_vel(N*3)] [local_ang_vel(N*3)]
    N=31: 1+90+186+93+93=463;  N=30: 1+87+180+90+90=448
    """
    src_dim = int(priv.shape[-1])
    if src_dim == target_dim:
        return priv

    trim_map: dict[tuple[int, int], bool] = {
        (463, 448): True,
        (462, 447): False,
    }
    key = (src_dim, target_dim)
    if key not in trim_map:
        raise ValueError(
            f"privileged_state 维数 {src_dim} 与模型期望 {target_dim} 不符，"
            "且不在已知的自动裁剪映射（463→448 / 462→447）中。\n"
            "请确认训练时的 env.root_height_obs 与 NPZ 数据配置一致。"
        )

    has_root = trim_map[key]
    if has_root != root_height_obs:
        print(
            f"[警告] 由维数推断 has_root_height={has_root}，"
            f"但 config.root_height_obs={root_height_obs}，"
            "以维数推断为准。"
        )

    # 按段去掉最后一个刚体的贡献（每段各去掉 3 或 6 维）
    seg_full_and_chop = [(90, 3), (186, 6), (93, 3), (93, 3)]
    parts: list[np.ndarray] = []
    if has_root:
        parts.append(priv[:, :1])
    cur = 1 if has_root else 0
    for length, chop in seg_full_and_chop:
        parts.append(priv[:, cur: cur + length - chop])
        cur += length

    result = np.concatenate(parts, axis=-1).astype(np.float32)
    assert result.shape[-1] == target_dim, (
        f"裁剪后维数 {result.shape[-1]} ≠ 目标 {target_dim}，请检查配置。"
    )
    print(f"  privileged_state 自动对齐: {src_dim} → {target_dim} (root_height={has_root})")
    return result


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(
    model_folder: Path,
    npz_path: Path,
    data_path: Path | None = None,
    headless: bool = True,
    device: str = "cuda",
    simulator: str = "isaacsim",
    save_mp4: bool = False,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    episode_len: int | None = None,
) -> None:
    """
    参数说明
    --------
    model_folder : checkpoint 上一级目录（内含 checkpoint/ 和 config.json）
    npz_path     : 轨迹 NPZ 文件路径（如 g1_circle_7_obs.npz）
    data_path    : 覆盖 config.json 里的 lafan_tail_path（仅用于 Isaac 环境初始化，
                   不影响 tracking inference 本身）
    episode_len  : rollout 最大步数（None = 使用 z 的完整长度）
    """
    model_folder = Path(model_folder)
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ 文件不存在: {npz_path}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    dev = next(model.parameters()).device

    # ── 读取训练配置 ──────────────────────────────────────────────────────────
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs: bool = bool(config["env"].get("root_height_obs", False))
    priv_expected: int = _get_model_priv_dim(model)
    print(f"模型期望 privileged_state 维度: {priv_expected}")
    print(f"use_root_height_obs: {use_root_height_obs}")

    # 保证 Isaac 环境能正常初始化（需要一个合法的 lafan_tail_path）
    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"

    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    # ── 导出 ONNX（与原版一致）────────────────────────────────────────────────
    export_dir = model_folder / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    z_export_dim = model.cfg.archi.total_z_dim
    export_meta_policy_as_onnx(
        model,
        export_dir,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(
            1, model._actor.input_filter.output_space.shape[0] + z_export_dim
        )},
        z_dim=z_export_dim,
        history=("history_actor" in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"Exported model to {export_dir}/{model_name}.onnx")

    # ── backward_map → cumulative mean → project_z（与原版完全一致）───────────
    def tracking_inference(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """obs 已跳过第 0 帧（速度有限差分第 0 帧不准），从第 1 帧开始。"""
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    # ── 构建 Isaac 环境 ───────────────────────────────────────────────────────
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("=" * 80)
    print(env.config.simulator)
    print("-" * 80)

    # env.default_dof_pos 可能是 (29,) 或 (1, 29)
    ddp = env.default_dof_pos.detach().float()
    default_dof_np: np.ndarray = (ddp[0] if ddp.ndim >= 2 else ddp).cpu().numpy()  # (29,)

    # ── 加载 NPZ 轨迹 ─────────────────────────────────────────────────────────
    print(f"Loading trajectory: {npz_path}")
    traj = load_npz_obs(npz_path)
    N = traj["state"].shape[0]
    print(f"  Trajectory length: {N} frames")
    print(f"  state shape:             {traj['state'].shape}")
    print(f"  last_action shape:       {traj['last_action'].shape}")
    print(f"  privileged_state shape:  {traj['privileged_state'].shape}")

    # privileged_state 维度对齐（自动裁剪 head_link 扩展体）
    traj["privileged_state"] = _align_privileged_state(
        traj["privileged_state"], priv_expected, use_root_height_obs
    )

    # ── 构建 backward_map 输入（跳过第 0 帧）────────────────────────────────
    # 理由：privileged_state 的 body_vel/body_ang_vel 由有限差分计算，
    # 第 0 帧 body_pos_prev=None 导致速度为 0，从第 1 帧开始才正确。
    obs_full: dict[str, torch.Tensor] = {
        "state":            torch.from_numpy(traj["state"]).to(dev),
        "last_action":      torch.from_numpy(traj["last_action"]).to(dev),
        "privileged_state": torch.from_numpy(traj["privileged_state"]).to(dev),
    }
    # x[1:] 跳过第 0 帧，与原版 tracking_inference.py 第 101 行一致
    obs_for_bmap = tree_map(lambda x: x[1:], obs_full)

    # ── backward_map → z ──────────────────────────────────────────────────────
    print("Running backward_map to get z...")
    with torch.no_grad():
        z = tracking_inference(obs_for_bmap)   # (N-1, z_dim)
    print(f"  z shape: {z.shape}")

    # ── 保存 z ───────────────────────────────────────────────────────────────
    out_dir = model_folder / "tracking_inference_npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    z_save_path = out_dir / f"zs_{stem}.pkl"
    joblib.dump(z.cpu().numpy(), z_save_path)
    print(f"Saved z → {z_save_path}")

    # ── 初始化 Isaac 环境到轨迹第 0 帧 ──────────────────────────────────────
    # 关节角：state[:, :29] 是相对 default_dof_pos 的偏移 → 还原绝对关节角
    dof_abs_np = traj["state"][:, :29] + default_dof_np   # (N, 29)
    dof_vel_np = traj["state"][:, 29:58]                   # (N, 29)  关节速度

    # 根状态：NPZ 不含根轨迹，使用环境默认 base_init_state
    bis = env.base_init_state.detach().float().cpu()
    root7 = (bis[:7] if bis.ndim == 1 else bis[0, :7]).numpy().astype(np.float32)
    root_pos  = root7[:3]          # (3,)  xyz
    root_quat_xyzw = root7[3:7]   # (4,)  x,y,z,w
    # IsaacSim 内部约定 wxyz（与 tracking_inference.py 第 110–111 行一致）
    root_quat = (
        root_quat_xyzw[[3, 0, 1, 2]] if simulator == "isaacsim" else root_quat_xyzw
    )

    # expert_qpos 仅用于 MuJoCo 专家侧视频：根静止、只有关节运动
    T_vis = min(N, (episode_len or N) + 1)
    expert_qpos = np.zeros((T_vis, 36), dtype=np.float32)
    expert_qpos[:, :3]  = root_pos
    expert_qpos[:, 3:7] = root_quat
    expert_qpos[:, 7:]  = dof_abs_np[:T_vis]

    # 重置环境
    sim_dev = env.device
    ref_root = torch.tensor(
        np.concatenate([root_pos, root_quat, np.zeros(6)]),  # pos + quat + vel(zero)
        dtype=torch.float32,
    )
    dof_init = torch.zeros_like(
        wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0]
    )
    dof_init[..., 0] = torch.from_numpy(dof_abs_np[0]).float().to(dof_init.device)
    dof_init[..., 1] = torch.from_numpy(dof_vel_np[0]).float().to(dof_init.device)

    target_states = {
        "dof_states":  dof_init,
        "root_states": torch.stack([ref_root.clone().to(sim_dev)
                                    for _ in range(num_envs)]),
    }
    env_ids = torch.arange(num_envs, dtype=torch.long, device=sim_dev)
    wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
    wrapped_env.step(
        torch.zeros((num_envs, wrapped_env.action_space.shape[-1]),
                    dtype=torch.float32, device=sim_dev),
        to_numpy=False,
    )
    observation = wrapped_env._get_g1env_observation(to_numpy=False)

    # ── Rollout ───────────────────────────────────────────────────────────────
    n_steps = z.shape[0]                          # N-1（与 z 长度对齐）
    if episode_len is not None:
        n_steps = min(n_steps, episode_len)
    print(f"Running rollout for {n_steps} steps "
          f"(z length={z.shape[0]}, episode_len={episode_len})")

    joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]

    if save_mp4:
        try:
            import mediapy as media
        except ImportError:
            raise ImportError("save_mp4 需要 mediapy 库：pip install mediapy")
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
        expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + n_steps])
        frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

    for i in range(n_steps):
        action = model.act(observation, z[i % len(z)].repeat(num_envs, 1), mean=True)
        observation, reward, terminated, truncated, info = wrapped_env.step(
            action, to_numpy=False
        )
        joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1}/{n_steps}")

    joint_pos_arr = np.stack(joint_pos, axis=0).squeeze(1)   # (n_steps+1, 29)
    print(f"Rollout complete. joint_pos shape: {joint_pos_arr.shape}")

    if save_mp4:
        new_frames = [
            np.concatenate([a, b], axis=1)
            for a, b in zip(expert_video, frames)
        ]
        video_path = out_dir / f"tracking_{stem}.mp4"
        media.write_video(str(video_path), new_frames, fps=50)
        print(f"Saved video: {video_path}")

    print(f"\n=== Done ===")
    print(f"  z pkl  : {z_save_path}")
    if save_mp4:
        print(f"  video  : {out_dir / f'tracking_{stem}.mp4'}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
