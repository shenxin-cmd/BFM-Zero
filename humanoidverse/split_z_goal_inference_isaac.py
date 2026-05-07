"""
Isaac 上 single-frame 的 goal inference（Split-z 友好），不依赖多目标 pkl。

流程（与 ``goal_inference.py`` 一致：Isaac 环境 + ``get_backward_observation`` + ``model.goal_inference``），
区别为 **只取运动库中一帧** 作为目标，并在代码顶部可编辑默认 ``motion_id`` / ``goal_frame``。

``z_hand`` 噪声：在 ``project_z`` 前只对 ``z_ref`` 的手部子向量加高斯扰动，再 ``project_z`` 拼回，
闭环 rollout 与 ``goal_inference.py`` 相同（每步用环境观测 + 固定 ``z`` 调用 ``model.act``）。

用法::

  # 默认使用文件顶部 GOAL_MOTION_ID / GOAL_FRAME_IDX；手部无噪声；保存 MP4
  python -m humanoidverse.split_z_goal_inference_isaac \\
    --model-folder workdir/bfmzero-split-z/<run> \\
    --data-path humanoidverse/data/lafan_29dof.pkl

  # 只看右手扰动：无噪与有噪各跑一次，对比两段视频
  python -m humanoidverse.split_z_goal_inference_isaac ... --hand-noise-std 0   --video-suffix clean
  python -m humanoidverse.split_z_goal_inference_isaac ... --hand-noise-std 2.0 --video-suffix zh_noisy

环境变量：沿用 Isaac / 仓库约定，见 ``goal_inference.py``。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mediapy as media
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, get_backward_observation

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 用户可编辑：当命令行未传 ``--motion-id`` / ``--goal-frame`` 时使用。
# -----------------------------------------------------------------------------
GOAL_MOTION_ID = 0
GOAL_FRAME_IDX = 0


def main(
    model_folder: Path,
    data_path: Path | None = None,
    motion_id: int | None = None,
    goal_frame: int | None = None,
    headless: bool = True,
    device: str = "cuda",
    simulator: str = "isaacsim",
    hand_noise_std: float = 0.0,
    seed: int = 0,
    episode_len: int = 500,
    save_mp4: bool = True,
    video_folder: Path | None = None,
    video_suffix: str = "",
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    export_onnx: bool = False,
) -> None:
    model_folder = Path(model_folder)
    vid_dir = Path(video_folder) if video_folder is not None else model_folder / "split_z_goal_isaac" / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()

    if not model.cfg.archi.is_split_mode:
        raise RuntimeError(
            "此脚本面向 split-z checkpoint（archi.z_body_dim>0 且 z_hand_dim>0）。"
            "非 split 模型请用 humanoidverse.goal_inference。"
        )

    with open(model_folder / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)

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

    if export_onnx:
        output_dir = model_folder / "exported"
        output_dir.mkdir(parents=True, exist_ok=True)
        z_export_dim = model.cfg.archi.total_z_dim
        model_name = model.__class__.__name__
        export_meta_policy_as_onnx(
            model,
            output_dir,
            f"{model_name}.onnx",
            {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + z_export_dim)},
            z_dim=z_export_dim,
            history=("history_actor" in model.cfg.archi.actor.input_filter.key),
            use_29dof=True,
        )
        print(f"Exported ONNX to {output_dir}/{model_name}.onnx")

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env

    mid = GOAL_MOTION_ID if motion_id is None else motion_id
    fid = GOAL_FRAME_IDX if goal_frame is None else goal_frame

    print(f"motion_id={mid}, goal_frame={fid}, use_root_height_obs={use_root_height_obs}")
    print(f"hand_noise_std={hand_noise_std} (z_hand only, before project_z)")

    env.set_is_evaluating(mid)
    gobs, _gobs_dict = get_backward_observation(env, mid, use_root_height_obs=use_root_height_obs, velocity_multiplier=0)
    n_frames = int(gobs["state"].shape[0])
    if fid < 0 or fid >= n_frames:
        raise IndexError(f"goal_frame {fid} out of range [0, {n_frames}) for motion {mid}")

    goal_observation = {k: v[fid : fid + 1] for k, v in gobs.items()}
    goal_observation = tree_map(
        lambda x: torch.tensor(x, device=model.device, dtype=torch.float32),
        goal_observation,
    )

    with torch.no_grad():
        z_ref = model.goal_inference(goal_observation)

    zb = z_ref[:, : model.cfg.archi.z_body_dim]
    zh = z_ref[:, model.cfg.archi.z_body_dim :]
    print(
        f"z_ref: total_dim={z_ref.shape[-1]}, ||z||={z_ref.norm().item():.4f}, "
        f"||z_body||={zb.norm().item():.4f}, ||z_hand||={zh.norm().item():.4f}"
    )

    rng = torch.Generator(device=z_ref.device)
    rng.manual_seed(seed)
    if hand_noise_std > 0:
        noise = torch.randn(zh.shape, device=zh.device, dtype=zh.dtype, generator=rng)
        zh = zh + float(hand_noise_std) * noise
    z = model.project_z(torch.cat([zb, zh], dim=-1))
    print(f"z after noise+project_z: ||z||={z.norm().item():.4f}")

    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)

    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)

    frames: list = []
    _pbar = tqdm(desc="steps", disable=False, leave=False, total=episode_len)
    for _counter in range(episode_len):
        action = model.act(observation, z.repeat(num_envs, 1), mean=True)
        observation, _reward, _terminated, _truncated, _info = wrapped_env.step(action, to_numpy=False)
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
        _pbar.update(1)
    _pbar.close()

    if save_mp4 and frames:
        sfx = f"_{video_suffix}" if video_suffix else ""
        noise_tag = f"hnoise{hand_noise_std:.4f}".replace(".", "p")
        out_mp4 = vid_dir / f"goal_M{mid}_f{fid}_{noise_tag}{sfx}.mp4"
        media.write_video(str(out_mp4), frames, fps=50)
        print(f"Saved video: {out_mp4}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
