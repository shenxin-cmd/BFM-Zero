"""
基于 tracking_inference.py 修改而来。
区别：不再从 lafan_29dof.pkl 运动库读取轨迹，而是直接从 --traj-obs-dir 指定的目录
加载 *.npz 文件（每个文件包含 state / last_action / privileged_state 三个数组）。

NPZ 格式：
  state             (T, 64)  float32  [dof_pos_rel(29), dof_vel(29), proj_grav(3), ang_vel(3)]
  last_action       (T, 29)  float32
  privileged_state  (T, 463) float32  compute_humanoid_observations_max (31 bodies + root_height)

兼容拼写：priviledged_state / "last action"（会自动归一化为 last_action）。

若 privileged_state 的维度比模型 BatchNorm 期望的多（典型：463 vs 448），
脚本会自动裁掉最后一个刚体（head_link 扩展体）的贡献，无需手动处理。
"""
from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import joblib
import json
import mediapy as media
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
# 轨迹文件加载 & 预处理
# ---------------------------------------------------------------------------

def _normalize_key(k: str) -> str:
    """统一键名：去空格、小写、修正 priviledged 拼写。"""
    k = k.strip().lower().replace(" ", "_")
    if k == "priviledged_state":
        k = "privileged_state"
    return k


def load_npz_obs(path: Path) -> dict[str, np.ndarray]:
    """加载单个 NPZ 轨迹文件，返回 {key: (T, D) float32 ndarray}。"""
    z = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in z.files:
        arr = np.asarray(z[k], dtype=np.float32)
        out[_normalize_key(k)] = arr
    z.close()
    for req in ("state", "last_action", "privileged_state"):
        if req not in out:
            raise KeyError(
                f"轨迹文件 {path.name} 缺少键 '{req}'，"
                f"现有键: {list(out.keys())}"
            )
    return out


def _get_model_priv_dim(model) -> int:
    """从 model.obs_space 读取 privileged_state 期望维度。"""
    return int(model.obs_space.spaces["privileged_state"].shape[0])


def _trim_privileged_state(
    priv: np.ndarray,
    target_dim: int,
    root_height_obs: bool,
) -> np.ndarray:
    """
    裁掉 privileged_state 中末尾刚体的贡献（对应 nums_extend_bodies=1 的 head_link）。

    仅支持以下自动转换：
      463 → 448  （含 root_height，30+1 → 30 bodies）
      462 → 447  （不含 root_height，30+1 → 30 bodies）
    """
    src_dim = int(priv.shape[-1])
    if src_dim == target_dim:
        return priv

    # 已知映射：(src, dst) -> has_root_height_in_data
    trim_map: dict[tuple[int, int], bool] = {
        (463, 448): True,
        (462, 447): False,
    }
    key = (src_dim, target_dim)
    if key not in trim_map:
        raise ValueError(
            f"privileged_state 维数 {src_dim} 与模型期望 {target_dim} 不符，"
            "且不在已知的自动裁剪映射中（支持 463→448 / 462→447）。\n"
            "请确认训练时 env.root_height_obs 与轨迹数据所用配置一致。"
        )
    has_root = trim_map[key]
    if has_root != root_height_obs:
        print(
            f"[警告] 维数推断 has_root={has_root} 与 config.root_height_obs={root_height_obs} 不一致，"
            "以维数推断为准。"
        )

    # 按 compute_humanoid_observations_max 拼接顺序裁剪
    # layout: [root_height(1)?] [local_body_pos(N*3-3)] [local_body_rot(N*6)]
    #         [local_body_vel(N*3)] [local_body_ang_vel(N*3)]
    # N=31(with extend): 1+90+186+93+93=463  ;  N=30(without): 1+87+180+90+90=448
    seg_full_and_chop = [(90, 3), (186, 6), (93, 3), (93, 3)]

    parts: list[np.ndarray] = []
    if has_root:
        parts.append(priv[:, :1])
    cur = 1 if has_root else 0
    for length, chop in seg_full_and_chop:
        parts.append(priv[:, cur : cur + length - chop])
        cur += length

    result = np.concatenate(parts, axis=-1).astype(np.float32)
    assert result.shape[-1] == target_dim, (
        f"裁剪后维数 {result.shape[-1]} ≠ 目标 {target_dim}，请检查 seg_full_and_chop 配置。"
    )
    print(f"  privileged_state 自动对齐: {src_dim} → {target_dim} (root_height={has_root})")
    return result


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(
    model_folder: Path,
    traj_obs_dir: Path,
    traj_glob: str = "*.npz",
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
    model_folder  : checkpoint 上一级目录（内含 checkpoint/ 和 config.json）
    traj_obs_dir  : 存放 *.npz 轨迹的目录（e.g. /path/to/recordings/obs）
    traj_glob     : 文件匹配模式，默认 *.npz
    data_path     : 若需覆盖 config.json 里的 lafan_tail_path（仅用于 Isaac 环境初始化）
    episode_len   : rollout 步数上限（默认用 z 的完整长度）
    """
    model_folder = Path(model_folder)
    traj_obs_dir = Path(traj_obs_dir)

    # ---- 加载模型 ----
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    dev = next(model.parameters()).device

    # ---- 读取训练配置 ----
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs: bool = bool(config["env"].get("root_height_obs", False))
    priv_expected: int = _get_model_priv_dim(model)

    # 保证 Isaac 环境能正常初始化（仍需一个合法的 lafan_tail_path）
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

    # ---- 导出 ONNX（与 tracking_inference.py 一致）----
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

    # ---- backward_map + cumulative mean + project_z（与原版相同）----
    def tracking_inference(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    # ---- 构建 Isaac 环境 ----
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("=" * 80)
    print(env.config.simulator)
    print("-" * 80)

    # default_dof_pos 可能是 (29,) 或 (1, 29)
    ddp = env.default_dof_pos.detach().float()
    default_dof_np: np.ndarray = (ddp[0] if ddp.ndim >= 2 else ddp).cpu().numpy()  # (29,)

    # ---- 扫描轨迹文件 ----
    traj_paths = sorted(traj_obs_dir.glob(traj_glob))
    if not traj_paths:
        alt = "*.pkl" if traj_glob == "*.npz" else "*.npz"
        traj_paths = sorted(traj_obs_dir.glob(alt))
        if traj_paths:
            print(f"未找到 {traj_glob!r}，自动改用 {alt!r}，共 {len(traj_paths)} 个文件")
    if not traj_paths:
        raise FileNotFoundError(
            f"在 {traj_obs_dir} 下未找到匹配 {traj_glob!r} 的文件"
        )

    out_z_dir = model_folder / "tracking_inference2"
    out_z_dir.mkdir(parents=True, exist_ok=True)

    # ---- 对每条轨迹做 backward_map → z，保存 pkl ----
    last_stem: str | None = None
    last_z: torch.Tensor | None = None
    last_traj: dict[str, np.ndarray] | None = None

    for traj_path in traj_paths:
        print(f"Load trajectory: {traj_path}")
        traj = load_npz_obs(traj_path)

        # 维度对齐
        traj["privileged_state"] = _trim_privileged_state(
            traj["privileged_state"], priv_expected, use_root_height_obs
        )

        obs: dict[str, torch.Tensor] = {
            "state":            torch.from_numpy(traj["state"]).to(dev),
            "last_action":      torch.from_numpy(traj["last_action"]).to(dev),
            "privileged_state": torch.from_numpy(traj["privileged_state"]).to(dev),
        }

        # 与原版一致：去掉第 0 帧再送 backward_map（原 tracking_inference.py 第 101 行）
        z = tracking_inference(tree_map(lambda x: x[1:], obs))

        stem = traj_path.stem
        save_path = out_z_dir / f"zs_{stem}.pkl"
        joblib.dump(z.cpu().numpy(), save_path)
        print(f"  Saved {save_path}  (z steps={z.shape[0]})")

        last_stem, last_z, last_traj = stem, z, traj

    # ---- Rollout（最后一条轨迹，与原版单段 rollout 一致）----
    assert last_traj is not None and last_z is not None and last_stem is not None

    traj = last_traj
    z    = last_z
    T    = traj["state"].shape[0]

    # 绝对关节角：state[:, :29] 是相对默认姿态的偏移
    dof_abs  = traj["state"][:, :29] + default_dof_np          # (T, 29)
    dof_vel0 = traj["state"][0, 29:58].astype(np.float32)      # 初始关节速度

    # 根状态：无根轨迹数据，取环境默认 base_init_state
    bis = env.base_init_state.detach().float().cpu()
    root7 = (bis[:7] if bis.ndim == 1 else bis[0, :7]).numpy().astype(np.float32)
    root_pos      = root7[:3]
    root_quat_xyzw = root7[3:7]
    # IsaacSim 内部约定 wxyz（与 tracking_inference.py 第 110-111 行一致）
    root_quat = root_quat_xyzw[[3, 0, 1, 2]] if simulator == "isaacsim" else root_quat_xyzw

    # expert_qpos：(T, 36) = root_pos(3) + root_quat_wxyz(4) + dof_abs(29)
    # 由于无真实根轨迹，根保持静止（仅关节运动用于视频的「专家侧」对比）
    expert_qpos = np.zeros((T, 36), dtype=np.float32)
    expert_qpos[:, :3]  = root_pos
    expert_qpos[:, 3:7] = root_quat
    expert_qpos[:, 7:]  = dof_abs

    # 重置环境到第 0 帧状态
    sim_dev = wrapped_env._env.device
    ref_root_init = torch.tensor(
        np.concatenate([root_pos, root_quat, np.zeros(6)]),
        dtype=torch.float32,
    )
    dof_init = torch.zeros_like(
        wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0]
    )
    dof_init[..., 0] = torch.from_numpy(dof_abs[0]).float().to(dof_init.device)
    dof_init[..., 1] = torch.from_numpy(dof_vel0).float().to(dof_init.device)

    target_states = {
        "dof_states":  dof_init,
        "root_states": torch.stack([ref_root_init.clone().to(sim_dev) for _ in range(num_envs)]),
    }
    env_ids = torch.arange(num_envs, dtype=torch.long, device=sim_dev)
    wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
    wrapped_env.step(
        torch.zeros((num_envs, wrapped_env.action_space.shape[-1]),
                    dtype=torch.float32, device=sim_dev),
        to_numpy=False,
    )
    observation = wrapped_env._get_g1env_observation(to_numpy=False)

    n_steps = z.shape[0]
    if episode_len is not None:
        n_steps = min(n_steps, episode_len)
    print(f"Running tracking inference for {n_steps} steps")

    joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]

    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
        expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + n_steps])
        frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

    for i in range(n_steps):
        print(f"Step {i + 1}/{n_steps}")
        action = model.act(observation, z[i % len(z)].repeat(num_envs, 1), mean=True)
        observation, reward, terminated, truncated, info = wrapped_env.step(
            action, to_numpy=False
        )
        joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])

    joint_pos = np.stack(joint_pos, axis=0).squeeze(1)

    if save_mp4:
        new_frames = [np.concatenate([a, b], axis=1) for a, b in zip(expert_video, frames)]
        video_path = out_z_dir / f"tracking_{last_stem}.mp4"
        media.write_video(str(video_path), new_frames, fps=50)
        print(f"Saved video: {video_path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
