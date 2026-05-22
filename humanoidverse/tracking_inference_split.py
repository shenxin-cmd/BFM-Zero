"""
从外部轨迹目录加载 backward 观测（``state`` / ``last_action`` / ``privileged_state``），
做与 ``tracking_inference.py`` 相同的 ``backward_map → cumulative mean → project_z``，
再在 Isaac 中 rollout 并可保存并排视频。

不依赖 ``lafan_29dof.pkl`` 的运动库回放；环境与 actor 仍需正常 ``HumanoidVerseIsaacConfig``，
``config.json`` 里仍可保留合法的 ``lafan_tail_path`` 仅用于环境初始化。

轨迹文件：
  - ``.pkl``：``joblib`` 或 pickle，内容为 dict；或长度为 T 的 list，每项为帧级 dict。
  - ``.npz``：每组一个数组，维度为 ``(T, dim)``.

默认按 ``*.npz`` 扫描；若 ``*.npz`` 无文件且模式为 ``*.npz`` 或 ``*.pkl``，会自动尝试另一种扩展名。

必备键（不区分大小写；``last action`` → ``last_action``；兼容 ``priviledged_state``）：
  - ``state``          形状 ``(T, 64)``
  - ``last_action``   形状 ``(T, 29)``
  - ``privileged_state`` 形状 ``(T, 527)`` 或 ``(T, 462/463)``（与 checkpoint 一致）

可选键（若提供则用于更准确的 MuJoCo 专家侧）：
  - ``mujoco_qpos`` 形状 ``(T, 36)``：7 自由根 + 29 关节，与 ``IsaacRendererWithMuJoco`` 一致。

若无 ``mujoco_qpos``：用每帧 ``state[:, :29] + default_dof_pos`` 作为关节绝对角，
根姿态使用环境 ``reset`` 后的默认根（仅用于专家侧渲染，可能与真实采集略有偏差）。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import json
import mediapy as media
import numpy as np
import torch
from torch.utils._pytree import tree_map

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import export_meta_policy_as_onnx

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

_BACKWARD_KEYS = ("state", "last_action", "privileged_state")


def _normalize_traj_key(name: str) -> str:
    k = name.strip().lower().replace(" ", "_")
    if k == "priviledged_state":
        k = "privileged_state"
    return k


def _stack_traj_obs(raw: object) -> dict[str, np.ndarray]:
    """统一为 {key: (T, D) float32 numpy}。"""
    if isinstance(raw, list):
        if len(raw) == 0:
            raise ValueError("轨迹 list 为空")
        keys = set()
        for row in raw:
            if not isinstance(row, dict):
                raise TypeError("轨迹 list 的每个元素应为 dict")
            keys.update(row.keys())
        out: dict[str, list[np.ndarray]] = { _normalize_traj_key(k): [] for k in keys }
        for row in raw:
            for k, v in row.items():
                nk = _normalize_traj_key(k)
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                out[nk].append(arr)
        return {k: np.stack(vs, axis=0) for k, vs in out.items()}

    if not isinstance(raw, dict):
        raise TypeError(f"不支持的轨迹类型: {type(raw)}")

    fixed: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        nk = _normalize_traj_key(k)
        fixed[nk] = np.asarray(v, dtype=np.float32)
        if fixed[nk].ndim != 2:
            raise ValueError(f"键 {k!r} 展开后应为 2维 (T,D)，got shape {fixed[nk].shape}")
    return fixed


def load_traj_obs_file(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        raw = joblib.load(path)
    elif path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        raw = {k: z[k] for k in z.files}
        z.close()
    else:
        raise ValueError(f"仅支持 .pkl / .npz: {path}")
    return _stack_traj_obs(raw)


def traj_to_backward_batch(traj: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    for k in _BACKWARD_KEYS:
        if k not in traj:
            raise KeyError(f"轨迹缺键 {k!r}（或拼写 priviledged_state）；现有: {sorted(traj.keys())}")
        if traj[k].ndim != 2:
            raise ValueError(f"{k} 应为 (T, D)，got {traj[k].shape}")
    return {
        k: torch.from_numpy(traj[k]).to(device=device, dtype=torch.float32)
        for k in _BACKWARD_KEYS
    }


def _build_expert_qpos(
    traj: dict[str, np.ndarray],
    *,
    env,
    wrapped_env,
    device: torch.device,
    num_envs: int,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    返回 expert_qpos (T,36)、ref_root (13,)、dof_init for reset。

    ``mujoco_qpos`` 若存在：每行 36 = 根位置(3)+根四元数 wxyz(4)+29 关节；与 ``IsaacRendererWithMuJoco`` / ``robot_root_states`` 拼 qpos 的约定一致。

    否则：各帧关节 = ``state[:, :29] + default_dof_pos``，根固定在 ``base_init_state`` 的首帧姿态（仅便于渲染对齐，不等价于采集真值根轨迹）。
    """
    T = traj["state"].shape[0]
    st = traj["state"]
    if st.shape[1] < 58:
        raise ValueError(
            f"state 第二维至少 58（29 相对关节 + 29 关节速度 + …），当前 {st.shape[1]}"
        )
    default_dof = env.default_dof_pos[0].detach().float().cpu().numpy()

    if "mujoco_qpos" in traj:
        q = np.asarray(traj["mujoco_qpos"], dtype=np.float64)
        if q.shape != (T, 36):
            raise ValueError(f"mujoco_qpos 期望 (T,36) T={T}, got {q.shape}")
        expert_qpos = q.astype(np.float32)
        root_pos = torch.from_numpy(expert_qpos[0, :3]).float().to(device)
        quat_wxyz = torch.from_numpy(expert_qpos[0, 3:7]).float().to(device)
        lin = torch.zeros(3, device=device, dtype=torch.float32)
        ang = torch.zeros(3, device=device, dtype=torch.float32)
        ref_root = torch.cat([root_pos, quat_wxyz, lin, ang], dim=0)
    else:
        dof_rel = st[:, :29].astype(np.float64)
        dof_abs = dof_rel + default_dof[None, :]
        root_template = env.base_init_state[0, :7].detach().cpu().numpy()
        root_pos = root_template[:3].astype(np.float32)
        root_quat = root_template[3:7].astype(np.float32)
        expert_qpos = np.zeros((T, 36), dtype=np.float32)
        expert_qpos[:, :3] = root_pos
        expert_qpos[:, 3:7] = root_quat
        expert_qpos[:, 7:] = dof_abs.astype(np.float32)
        ref_root = torch.cat(
            [
                torch.from_numpy(root_pos).float().to(device),
                torch.from_numpy(root_quat).float().to(device),
                torch.zeros(3, device=device),
                torch.zeros(3, device=device),
            ],
            dim=0,
        )

    dof_vel0 = st[0, 29:58].astype(np.float32)
    dof_template = wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0]
    dof_init = torch.zeros_like(dof_template)
    dof_init[..., 0] = torch.from_numpy(expert_qpos[0, 7:]).float().to(dof_template.device)
    dof_init[..., 1] = torch.from_numpy(dof_vel0).float().to(dof_template.device)

    return expert_qpos, ref_root, dof_init


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
    model_folder = Path(model_folder)
    traj_obs_dir = Path(traj_obs_dir)

    model = load_model_from_checkpoint_dir(str(model_folder / "checkpoint"), device=device)
    model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    dev = next(model.parameters()).device

    with open(model_folder / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

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

    export_root = model_folder / "exported"
    export_root.mkdir(parents=True, exist_ok=True)
    z_export_dim = model.cfg.archi.total_z_dim
    export_meta_policy_as_onnx(
        model,
        export_root,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + z_export_dim)},
        z_dim=z_export_dim,
        history=("history_actor" in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"Exported model to {export_root}/{model_name}.onnx")

    def tracking_inference(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("=" * 80)
    print(env.config.simulator)
    print("-" * 80)

    output_dir = model_folder / "tracking_inference_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(traj_obs_dir.glob(traj_glob))
    if not paths and traj_glob in ("*.pkl", "*.npz"):
        alt = "*.npz" if traj_glob == "*.pkl" else "*.pkl"
        paths = sorted(traj_obs_dir.glob(alt))
        if paths:
            print(f"提示: {traj_glob!r} 无匹配，已改用 {alt!r}，共 {len(paths)} 个文件。")
    if not paths:
        raise FileNotFoundError(
            f"在 {traj_obs_dir} 下未找到匹配 {traj_glob!r} 的文件"
            f"（若轨迹为另一种格式，请显式设置 --traj-glob，例如 '*.npz' 或 '*.pkl'）"
        )

    last_z: torch.Tensor | None = None
    last_traj_np: dict[str, np.ndarray] | None = None
    last_stem: str | None = None

    for traj_path in paths:
        print(f"Load trajectory: {traj_path}")
        traj_np = load_traj_obs_file(traj_path)
        obs_full = traj_to_backward_batch(traj_np, dev)
        z = tracking_inference(tree_map(lambda x: x[1:], obs_full))
        stem = re.sub(r"[^\w\-.]+", "_", traj_path.stem)
        joblib.dump(z.detach().cpu().numpy(), output_dir / f"zs_{stem}.pkl")
        print(f"Saved {output_dir / f'zs_{stem}.pkl'}  (z steps={z.shape[0]})")
        last_z = z
        last_traj_np = traj_np
        last_stem = stem

    assert last_z is not None and last_stem is not None and last_traj_np is not None

    # 环境与 rollout：默认使用排序后「最后一个」轨迹（与原版多 motion 只录一段视频类似）
    traj_np = last_traj_np
    z = last_z
    expert_qpos, ref_root, dof_init_state = _build_expert_qpos(
        traj_np, env=env, wrapped_env=wrapped_env, device=dev, num_envs=num_envs
    )

    sim_dev = wrapped_env._env.device
    env.set_is_evaluating(0)
    wrapped_env.reset(to_numpy=False)

    env_ids = torch.arange(num_envs, dtype=torch.long, device=sim_dev)
    target_states = {
        "dof_states": dof_init_state,
        "root_states": torch.stack([ref_root.clone().to(sim_dev) for _ in range(num_envs)]),
    }
    wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
    wrapped_env.step(
        torch.zeros((num_envs, wrapped_env.action_space.shape[-1]), dtype=torch.float32, device=sim_dev),
        to_numpy=False,
    )
    observation = wrapped_env._get_g1env_observation(to_numpy=False)

    Tz = z.shape[0]
    traj_T = traj_np["state"].shape[0]
    n_steps = min(Tz, traj_T - 1, expert_qpos.shape[0] - 1)
    if episode_len is not None:
        n_steps = min(n_steps, episode_len)
    print(f"Rollout steps: {n_steps} (z={Tz}, traj_T={traj_T})")

    joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]

    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
        expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + n_steps])
        frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

    for i in range(n_steps):
        print(f"Step {i + 1}/{n_steps}")
        action = model.act(observation, z[i % len(z)].unsqueeze(0).expand(num_envs, -1), mean=True)
        observation, _r, _t, _trunc, _info = wrapped_env.step(action, to_numpy=False)
        joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])

    np.stack(joint_pos, axis=0).squeeze(1)

    if save_mp4:
        new_frames = [np.concatenate([a, b], axis=1) for a, b in zip(expert_video, frames)]
        video_path = output_dir / f"tracking_{last_stem}.mp4"
        media.write_video(str(video_path), new_frames, fps=50)
        print(f"Saved video: {video_path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
