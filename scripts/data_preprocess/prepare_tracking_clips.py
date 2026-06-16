"""Prepare one or a few shape-drawing *_obs.npz clips for tracking_inference_split.py.

Same validation as ``convert_shape_npz_v2.py`` (state ↔ qpos back-to-back, right-arm
continuity), but **output stays NPZ** (not motion_lib pkl) because inference reads
``state`` / ``last_action`` / ``privileged_state`` directly.

Typical use: ad-hoc V2 IK clips saved on Desktop (clips_obs + optional raw).

Run (repo root or scripts/data_preprocess):

    python scripts/data_preprocess/prepare_tracking_clips.py \\
        --obs-npz /path/to/circle_Pyz_..._F300_obs.npz \\
                  /path/to/circle_Pyz_..._F300_obs.npz \\
        --raw-npz /path/to/seed003_L1_F300.npz \\
                  /path/to/seed007_L1_F300.npz \\
        --output-dir humanoidverse/data/inference_clips/custom_circles

Then inference (server, trained checkpoint):

    uv run -m humanoidverse.tracking_inference_split \\
        --model-folder results/<your_run> \\
        --traj-obs-dir humanoidverse/data/inference_clips/custom_circles \\
        --traj-glob \"*_obs.npz\"

Requires: numpy, scipy, joblib (no torch).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from convert_shape_npz import STATE_ATOL, validate_against_state
from convert_shape_npz_v2 import DEFAULT_JUMP_THRESHOLD_RAD, right_arm_continuity

_INFERENCE_KEYS = ("state", "last_action", "privileged_state")
_OPTIONAL_KEYS = ("qpos", "qvel", "timestamps", "fps", "waypoints", "mujoco_qpos")


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _scalar_str(arr) -> str:
    a = np.asarray(arr).reshape(-1)
    if a.size == 0:
        return ""
    return str(a[0])


def prepare_one_obs(
    obs_path: Path,
    raw_path: Path | None,
    output_dir: Path,
    jump_threshold: float,
    output_name: str | None = None,
) -> dict:
    obs_path = Path(obs_path)
    raw = _load_npz_dict(raw_path) if raw_path is not None else {}

    data = _load_npz_dict(obs_path)
    missing = [k for k in _INFERENCE_KEYS if k not in data]
    if missing:
        raise KeyError(f"{obs_path.name}: missing inference keys {missing}; have {list(data)}")

    T = int(data["state"].shape[0])
    for k in _INFERENCE_KEYS:
        if data[k].shape[0] != T:
            raise ValueError(f"{obs_path.name}: {k} length {data[k].shape[0]} != T={T}")

    # Prefer qpos inside obs; fall back to raw NPZ (V2 pipeline: obs and raw match)
    if "qpos" not in data and raw_path is not None:
        if "qpos" not in raw:
            raise KeyError(f"{raw_path.name}: no qpos for back-to-back check")
        data["qpos"] = np.asarray(raw["qpos"], dtype=np.float32)
        if "qvel" in raw:
            data["qvel"] = np.asarray(raw["qvel"], dtype=np.float32)

    stats: dict = {
        "source_obs": str(obs_path.resolve()),
        "source_raw": str(raw_path.resolve()) if raw_path else None,
        "n_frames": T,
        "fps": float(data["fps"]) if "fps" in data else 30.0,
    }

    if "qpos" in data and "qvel" in data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        qvel = np.asarray(data["qvel"], dtype=np.float64)
        state = np.asarray(data["state"], dtype=np.float64)
        errs = validate_against_state(qpos, qvel, state)
        stats["state_check_ok"] = bool(
            errs["dof_pos"] < STATE_ATOL
            and errs["dof_vel"] < STATE_ATOL
            and errs["proj_grav"] < STATE_ATOL
        )
        stats["state_errs"] = errs
        cont = right_arm_continuity(qpos, jump_threshold)
        stats["continuity"] = cont
        stats["jump_flagged"] = cont["max_arm_delta_rad"] > jump_threshold
    else:
        stats["state_check_ok"] = None
        stats["state_errs"] = "no qpos/qvel — skipped"
        stats["continuity"] = None
        stats["jump_flagged"] = None
        print(f"WARNING: {obs_path.name} has no qpos/qvel; inference still OK, no FK validation")

    # Standard inference filename: keep *_obs.npz suffix
    out_name = output_name or (
        obs_path.name if obs_path.name.endswith("_obs.npz") else f"{obs_path.stem}_obs.npz"
    )
    out_path = output_dir / out_name

    # Save float32 arrays for inference
    save_dict = {}
    for k, v in data.items():
        if k in _INFERENCE_KEYS or k in _OPTIONAL_KEYS:
            save_dict[k] = np.asarray(v, dtype=np.float32)
    # Preserve string metadata if present
    with np.load(obs_path, allow_pickle=True) as src:
        for k in ("shape_name", "shape_plane", "traj_name", "source_raw"):
            if k in src.files and k not in save_dict:
                save_dict[k] = src[k]

    np.savez_compressed(out_path, **save_dict)
    stats["output_npz"] = str(out_path)
    stats["output_keys"] = sorted(save_dict.keys())
    return stats


def _match_raw_for_obs(obs_path: Path, raw_paths: list[Path]) -> Path | None:
    """Match seed003 in obs filename to seed003_L1_F300.npz raw."""
    stem = obs_path.stem.removesuffix("_obs")
    for r in raw_paths:
        name = r.stem.lower()
        # circle_..._seed003_F300_obs -> seed003
        parts = stem.split("_")
        seeds = [p for p in parts if p.startswith("seed") and p[4:].isdigit()]
        if not seeds:
            continue
        seed_tag = seeds[-1]  # seed003
        if seed_tag in name:
            return r
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--obs-npz",
        type=Path,
        nargs="+",
        required=True,
        help="one or more *_obs.npz (clips_obs format)",
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        nargs="*",
        default=(),
        help="optional raw NPZ(s); auto-matched by seedXXX in filename",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jump-threshold", type=float, default=DEFAULT_JUMP_THRESHOLD_RAD)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="only copy obs npz without validation (not recommended)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_list = list(args.raw_npz)
    all_stats = []

    for obs_path in args.obs_npz:
        if args.copy_only:
            out = args.output_dir / obs_path.name
            shutil.copy2(obs_path, out)
            all_stats.append({"source_obs": str(obs_path), "output_npz": str(out), "copy_only": True})
            print(f"Copied -> {out}")
            continue

        raw_path = _match_raw_for_obs(obs_path, raw_list) if raw_list else None
        if raw_list and raw_path is None:
            print(f"WARNING: no raw match for {obs_path.name}; validation may be skipped")
        stats = prepare_one_obs(obs_path, raw_path, args.output_dir, args.jump_threshold)
        all_stats.append(stats)
        flag = "OK" if stats.get("state_check_ok") is not False else "FAIL state"
        jump = stats.get("jump_flagged")
        jump_s = f" jump={jump}" if jump is not None else ""
        print(f"{flag}{jump_s} -> {stats['output_npz']}")

    summary = {
        "output_dir": str(args.output_dir.resolve()),
        "n_clips": len(all_stats),
        "n_state_failed": sum(1 for s in all_stats if s.get("state_check_ok") is False),
        "clips": all_stats,
        "inference_command_hint": (
            "uv run -m humanoidverse.tracking_inference_split "
            f"--model-folder <checkpoint_dir> "
            f"--traj-obs-dir {args.output_dir} "
            "--traj-glob '*_obs.npz'"
        ),
    }
    report = args.report_json or (args.output_dir / "prepare_report.json")
    with open(report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nReport -> {report}")
    if summary["n_state_failed"]:
        raise SystemExit(f"{summary['n_state_failed']} clip(s) failed state check")


if __name__ == "__main__":
    main()
