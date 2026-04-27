import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import mediapy as media
import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.torch_utils import calc_heading_quat_inv, my_quat_rotate


def _parse_vec3(text: str) -> np.ndarray:
    tokens = [token for token in re.split(r"[\s,]+", text.strip()) if token]
    if len(tokens) != 3:
        raise ValueError(f"Expected 3 numbers, got: {text!r}")
    return np.asarray([float(token) for token in tokens], dtype=np.float32)


def _format_vec3(vec: np.ndarray) -> str:
    return f"[{vec[0]:+.4f}, {vec[1]:+.4f}, {vec[2]:+.4f}]"


def _slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return slug.strip("_") or "run"


def _default_deltas() -> list[np.ndarray]:
    raw = [
        (0.0, 0.0, 0.0),
        (0.08, 0.0, 0.0),
        (-0.08, 0.0, 0.0),
        (0.0, 0.08, 0.0),
        (0.0, -0.08, 0.0),
        (0.0, 0.0, 0.08),
        (0.0, 0.0, -0.08),
    ]
    return [np.asarray(v, dtype=np.float32) for v in raw]


def _load_model_and_env(
    model_folder: Path,
    device: str,
    headless: bool,
    simulator: str,
    disable_dr: bool,
    disable_obs_noise: bool,
):
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    if not getattr(model, "has_structured_z", False):
        raise ValueError("This probe requires a structured-z model, but the loaded model does not expose `has_structured_z=True`.")

    with open(model_folder / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(1)
    task_env = wrapped_env._env
    return model, wrapped_env, task_env


def _local_body_positions(task_env) -> torch.Tensor:
    body_pos = getattr(task_env, "_rigid_body_pos_extend", None)
    if body_pos is None:
        body_pos = task_env.simulator._rigid_body_pos
    body_pos = body_pos.clone().detach()
    root_pos = body_pos[:, 0:1, :]
    heading_inv = calc_heading_quat_inv(task_env.simulator.robot_root_states[:, 3:7].clone().detach(), w_last=True)
    heading_inv_expand = heading_inv.unsqueeze(1).repeat(1, body_pos.shape[1], 1).reshape(-1, 4)
    local_body_pos = body_pos - root_pos
    local_body_pos = my_quat_rotate(heading_inv_expand, local_body_pos.reshape(-1, 3)).view(body_pos.shape[0], body_pos.shape[1], 3)
    return local_body_pos


def _build_body_groups(task_env, right_hand_body: str | None) -> dict[str, Any]:
    body_names = list(task_env.simulator._body_list)
    num_local_bodies = int(_local_body_positions(task_env).shape[1])
    if len(body_names) != num_local_bodies:
        body_names = body_names[:num_local_bodies]
    if right_hand_body is None:
        for candidate in ["right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"]:
            if candidate in body_names:
                right_hand_body = candidate
                break
    if right_hand_body is None or right_hand_body not in body_names:
        raise ValueError(f"Could not resolve right-hand body from simulator body list: {body_names}")

    lower_body_names = [name for name in list(task_env.config.robot.motion.lower_body_link) if name in body_names]
    right_arm_names = [
        name
        for name in body_names
        if name.startswith("right_shoulder") or name.startswith("right_elbow") or name.startswith("right_wrist")
    ]
    non_target_names = [name for name in body_names if name not in set(right_arm_names)]

    left_arm_names = [
        name
        for name in body_names
        if name.startswith("left_shoulder") or name.startswith("left_elbow") or name.startswith("left_wrist")
    ]

    name_to_idx = {name: idx for idx, name in enumerate(body_names)}
    return {
        "body_names": body_names,
        "right_hand_body": right_hand_body,
        "right_hand_idx": name_to_idx[right_hand_body],
        "lower_body_idxs": [name_to_idx[name] for name in lower_body_names],
        "right_arm_idxs": [name_to_idx[name] for name in right_arm_names],
        "left_arm_idxs": [name_to_idx[name] for name in left_arm_names],
        "non_target_idxs": [name_to_idx[name] for name in non_target_names],
    }


def _capture_state(task_env, group_info: dict[str, Any]) -> dict[str, np.ndarray]:
    local_body = _local_body_positions(task_env)[0].cpu().numpy()
    root_pos = task_env.simulator.robot_root_states[0, :3].clone().detach().cpu().numpy()
    return {
        "local_body_pos": local_body,
        "right_hand_local": local_body[group_info["right_hand_idx"]].copy(),
        "root_pos": root_pos,
    }


def _mean_group_displacement(initial_local: np.ndarray, final_local: np.ndarray, idxs: list[int]) -> float:
    if not idxs:
        return 0.0
    disp = np.linalg.norm(final_local[idxs] - initial_local[idxs], axis=-1)
    return float(disp.mean())


def _compute_metrics(
    rollout_states: list[dict[str, np.ndarray]],
    target_hand: np.ndarray,
    group_info: dict[str, Any],
) -> dict[str, float]:
    initial_local = rollout_states[0]["local_body_pos"]
    final_local = rollout_states[-1]["local_body_pos"]
    initial_hand = rollout_states[0]["right_hand_local"]
    hand_traj = np.stack([state["right_hand_local"] for state in rollout_states], axis=0)
    root_traj = np.stack([state["root_pos"] for state in rollout_states], axis=0)

    hand_errors = np.linalg.norm(hand_traj - target_hand[None, :], axis=-1)
    hand_disp = np.linalg.norm(hand_traj - initial_hand[None, :], axis=-1)

    final_right_hand_displacement = float(hand_disp[-1])
    max_right_hand_displacement = float(hand_disp.max())
    final_right_hand_error = float(hand_errors[-1])
    min_right_hand_error = float(hand_errors.min())
    root_xy_drift = float(np.linalg.norm(root_traj[-1, :2] - root_traj[0, :2]))
    root_xyz_drift = float(np.linalg.norm(root_traj[-1] - root_traj[0]))
    lower_body_mean_displacement = _mean_group_displacement(initial_local, final_local, group_info["lower_body_idxs"])
    right_arm_mean_displacement = _mean_group_displacement(initial_local, final_local, group_info["right_arm_idxs"])
    left_arm_mean_displacement = _mean_group_displacement(initial_local, final_local, group_info["left_arm_idxs"])
    non_target_mean_displacement = _mean_group_displacement(initial_local, final_local, group_info["non_target_idxs"])
    isolation_ratio_non_target = final_right_hand_displacement / max(non_target_mean_displacement, 1e-6)
    isolation_ratio_lower_body = final_right_hand_displacement / max(lower_body_mean_displacement, 1e-6)

    return {
        "final_right_hand_displacement": final_right_hand_displacement,
        "max_right_hand_displacement": max_right_hand_displacement,
        "final_right_hand_error": final_right_hand_error,
        "min_right_hand_error": min_right_hand_error,
        "root_xy_drift": root_xy_drift,
        "root_xyz_drift": root_xyz_drift,
        "lower_body_mean_displacement": lower_body_mean_displacement,
        "right_arm_mean_displacement": right_arm_mean_displacement,
        "left_arm_mean_displacement": left_arm_mean_displacement,
        "non_target_mean_displacement": non_target_mean_displacement,
        "isolation_ratio_non_target": isolation_ratio_non_target,
        "isolation_ratio_lower_body": isolation_ratio_lower_body,
    }


def _save_rollout_artifacts(
    output_dir: Path,
    run_name: str,
    rollout_states: list[dict[str, np.ndarray]],
    z_command: np.ndarray,
    z_body: np.ndarray,
    z_hand_target: np.ndarray,
    metrics: dict[str, float],
) -> None:
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_file = run_dir / f"{run_name}.pkl"
    payload = {
        "rollout_states": rollout_states,
        "z_command": z_command,
        "z_body": z_body,
        "z_hand_target": z_hand_target,
        "metrics": metrics,
    }
    joblib.dump(payload, state_file)


def _run_rollout(
    wrapped_env,
    task_env,
    model,
    group_info: dict[str, Any],
    z_command: torch.Tensor,
    z_body: np.ndarray,
    z_hand_target: np.ndarray,
    episode_len: int,
    save_mp4: bool,
    video_path: Path | None,
):
    observation, _ = wrapped_env.reset(to_numpy=False, reset_to_default_pose=True)
    rollout_states = [_capture_state(task_env, group_info)]
    frames = []
    renderer = None
    if save_mp4:
        try:
            renderer = IsaacRendererWithMuJoco(render_size=256)
        except Exception as exc:
            print(f"Video renderer initialization failed, continuing without mp4 export: {exc}")
    if renderer is not None:
        frames.append(renderer.render(task_env, 0)[0])

    with torch.no_grad():
        for _ in range(episode_len):
            action = model.act(observation, z_command.unsqueeze(0), mean=True)
            observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
            rollout_states.append(_capture_state(task_env, group_info))
            if renderer is not None:
                try:
                    frames.append(renderer.render(task_env, 0)[0])
                except Exception as exc:
                    print(f"Video rendering failed during rollout, disabling mp4 export for this run: {exc}")
                    renderer = None

    metrics = _compute_metrics(rollout_states, z_hand_target, group_info)
    if renderer is not None and video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(str(video_path), frames, fps=50)

    return rollout_states, metrics


def _write_metrics_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_metrics(run_name: str, metrics: dict[str, float]) -> None:
    print(f"\n[{run_name}]")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def _prepare_base_context(wrapped_env, task_env, model, group_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation, _ = wrapped_env.reset(to_numpy=False, reset_to_default_pose=True)
    with torch.no_grad():
        z_base = model.goal_inference(observation)
    if z_base.ndim > 1:
        z_base = z_base[0]
    z_body = model.extract_z_body(z_base).detach().cpu().numpy().astype(np.float32)
    hand_base = model.extract_hand_pos(observation)[0].detach().cpu().numpy().astype(np.float32)
    current_hand = _capture_state(task_env, group_info)["right_hand_local"].astype(np.float32)
    return z_body, hand_base, current_hand


def _run_probe_case(
    wrapped_env,
    task_env,
    model,
    output_dir: Path,
    group_info: dict[str, Any],
    run_name: str,
    z_body: np.ndarray,
    z_hand_target: np.ndarray,
    save_mp4: bool,
    episode_len: int,
) -> dict[str, Any]:
    z_command = model.merge_z(
        torch.tensor(z_hand_target, device=model.device, dtype=torch.float32).unsqueeze(0),
        torch.tensor(z_body, device=model.device, dtype=torch.float32).unsqueeze(0),
    )[0]
    video_path = output_dir / "videos" / f"{run_name}.mp4" if save_mp4 else None
    rollout_states, metrics = _run_rollout(
        wrapped_env=wrapped_env,
        task_env=task_env,
        model=model,
        group_info=group_info,
        z_command=z_command,
        z_body=z_body,
        z_hand_target=z_hand_target,
        episode_len=episode_len,
        save_mp4=save_mp4,
        video_path=video_path,
    )
    _save_rollout_artifacts(
        output_dir=output_dir,
        run_name=run_name,
        rollout_states=rollout_states,
        z_command=z_command.detach().cpu().numpy(),
        z_body=z_body,
        z_hand_target=z_hand_target,
        metrics=metrics,
    )
    row = {
        "run_name": run_name,
        "target_hand_x": float(z_hand_target[0]),
        "target_hand_y": float(z_hand_target[1]),
        "target_hand_z": float(z_hand_target[2]),
        **metrics,
    }
    _print_metrics(run_name, metrics)
    return row


def _sweep_mode(
    wrapped_env,
    task_env,
    model,
    output_dir: Path,
    group_info: dict[str, Any],
    episode_len: int,
    save_mp4: bool,
    deltas: list[np.ndarray],
) -> None:
    z_body, hand_base_from_obs, current_hand = _prepare_base_context(wrapped_env, task_env, model, group_info)
    base_hand = current_hand
    rows = []
    print(f"Base z_hand target: {_format_vec3(base_hand)}")
    print(f"Base observed hand pose: {_format_vec3(current_hand)}")
    print(f"Base inferred hand pose: {_format_vec3(hand_base_from_obs)}")

    for idx, delta in enumerate(deltas):
        target_hand = base_hand + delta
        run_name = _slugify(f"{idx:03d}_delta_{delta[0]:+.3f}_{delta[1]:+.3f}_{delta[2]:+.3f}")
        print(f"\nRunning sweep case {idx + 1}/{len(deltas)} with delta {_format_vec3(delta)} -> target {_format_vec3(target_hand)}")
        row = _run_probe_case(
            wrapped_env=wrapped_env,
            task_env=task_env,
            model=model,
            output_dir=output_dir,
            group_info=group_info,
            run_name=run_name,
            z_body=z_body,
            z_hand_target=target_hand,
            save_mp4=save_mp4,
            episode_len=episode_len,
        )
        row["delta_x"] = float(delta[0])
        row["delta_y"] = float(delta[1])
        row["delta_z"] = float(delta[2])
        rows.append(row)

    _write_metrics_csv(rows, output_dir / "sweep_metrics.csv")


def _interactive_mode(
    wrapped_env,
    task_env,
    model,
    output_dir: Path,
    group_info: dict[str, Any],
    episode_len: int,
    save_mp4: bool,
) -> None:
    z_body, hand_base_from_obs, current_hand = _prepare_base_context(wrapped_env, task_env, model, group_info)
    base_hand = current_hand
    rows = []
    run_idx = 0

    print("Structured-z interactive probe")
    print(f"Base z_hand target: {_format_vec3(base_hand)}")
    print(f"Base observed hand pose: {_format_vec3(current_hand)}")
    print(f"Base inferred hand pose: {_format_vec3(hand_base_from_obs)}")
    print("Input format:")
    print("  dx dy dz          -> relative offset from base z_hand")
    print("  abs x y z         -> absolute z_hand target")
    print("  base              -> rerun the base hand target")
    print("  quit              -> exit")

    while True:
        raw = input("\nprobe> ").strip()
        if not raw:
            continue
        lower = raw.lower()
        if lower in {"quit", "q", "exit"}:
            break
        if lower == "base":
            target_hand = base_hand.copy()
            run_label = f"{run_idx:03d}_base"
        elif lower.startswith("abs "):
            target_hand = _parse_vec3(raw[4:])
            run_label = f"{run_idx:03d}_abs"
        else:
            delta = _parse_vec3(raw)
            target_hand = base_hand + delta
            run_label = f"{run_idx:03d}_delta_{delta[0]:+.3f}_{delta[1]:+.3f}_{delta[2]:+.3f}"

        run_name = _slugify(run_label)
        print(f"Running target {_format_vec3(target_hand)}")
        row = _run_probe_case(
            wrapped_env=wrapped_env,
            task_env=task_env,
            model=model,
            output_dir=output_dir,
            group_info=group_info,
            run_name=run_name,
            z_body=z_body,
            z_hand_target=target_hand,
            save_mp4=save_mp4,
            episode_len=episode_len,
        )
        rows.append(row)
        run_idx += 1
        _write_metrics_csv(rows, output_dir / "interactive_metrics.csv")


def main(
    model_folder: Path,
    mode: str = "sweep",
    headless: bool = True,
    device: str = "cuda",
    simulator: str = "isaacsim",
    episode_len: int = 200,
    save_mp4: bool = False,
    output_dir: str | None = None,
    disable_dr: bool = True,
    disable_obs_noise: bool = True,
    right_hand_body: str | None = "right_wrist_yaw_link",
    deltas: list[str] | None = None,
):
    model_folder = Path(model_folder)
    output_dir = Path(output_dir) if output_dir is not None else model_folder / "structured_z_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, wrapped_env, task_env = _load_model_and_env(
        model_folder=model_folder,
        device=device,
        headless=headless,
        simulator=simulator,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
    )
    group_info = _build_body_groups(task_env, right_hand_body)

    config_to_save = {
        "mode": mode,
        "episode_len": episode_len,
        "save_mp4": save_mp4,
        "device": device,
        "simulator": simulator,
        "disable_dr": disable_dr,
        "disable_obs_noise": disable_obs_noise,
        "right_hand_body": group_info["right_hand_body"],
        "body_names": group_info["body_names"],
    }
    with open(output_dir / "probe_config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    if mode == "sweep":
        parsed_deltas = _default_deltas() if not deltas else [_parse_vec3(delta) for delta in deltas]
        _sweep_mode(
            wrapped_env=wrapped_env,
            task_env=task_env,
            model=model,
            output_dir=output_dir,
            group_info=group_info,
            episode_len=episode_len,
            save_mp4=save_mp4,
            deltas=parsed_deltas,
        )
    elif mode == "interactive":
        _interactive_mode(
            wrapped_env=wrapped_env,
            task_env=task_env,
            model=model,
            output_dir=output_dir,
            group_info=group_info,
            episode_len=episode_len,
            save_mp4=save_mp4,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}. Expected 'sweep' or 'interactive'.")

    wrapped_env.close()
    print(f"Structured-z probe results saved to: {output_dir}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
