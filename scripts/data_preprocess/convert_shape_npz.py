"""Convert data1 right-hand shape-drawing NPZ clips to a motion_lib-compatible pkl.

Input  : new_data/data1/{batch_data, batch_data_xy, batch_data_xz}/clips_obs/**/*_obs.npz
         (already 30 fps x 300 frames, see BATCH_DATA_README.md)
Output : a single joblib dict pkl  {motion_key: {root_trans_offset, pose_aa, fps}}

Processing per NPZ:
  1. take raw MuJoCo ``qpos`` (300, 36):
        root_trans_offset = qpos[:, 0:3]                (world frame, m)
        pose_aa[:, 0]     = rotvec( qpos[:, 3:7] wxyz )  (global root rotation)
        pose_aa[:, 1:30]  = qpos[:, 7:36] * joint_axis   (per-hinge axis-angle)
     The stored ``privileged_state`` is NOT used (its finite-difference velocities are
     scaled ~33x, see BATCH_DATA_README.md §5.4); motion_lib re-derives correct
     velocities from FK + np.gradient.
  2. back-to-back validation: rebuild ``state`` (dof_pos / dof_vel / proj_grav) from
     qpos / qvel and compare with the ``state`` stored in the NPZ.  This catches any
     joint-order or frame-convention mismatch between the data-generation model
     (g1/scene_g1_draggable.xml) and the training model (g1_29dof.xml).
  3. stratified subsampling: per batch directory, sample evenly across shape
     sub-directories with a fixed seed (default 250 clips / batch dir, ~750 total)
     to keep the three data sources (lafan / bones / shapes) roughly balanced.

Run (server):
    python scripts/data_preprocess/convert_shape_npz.py \
        --data1-dir /path/to/new_data/data1 \
        --output-pkl humanoidverse/data/shape_29dof_clips.pkl

Requires: numpy, scipy, joblib (no torch).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from g1_kinematics import (
    G1_DEFAULT_JOINT_POS,
    build_pose_aa,
    fk_body_positions,
    load_g1_skeleton,
    quat_wxyz_to_rotvec,
    LEFT_FOOT_BODY,
    RIGHT_FOOT_BODY,
)

BATCH_DIRS = ["batch_data", "batch_data_xy", "batch_data_xz"]
STATE_ATOL = 1e-3  # rad / unit-vector tolerance for the back-to-back check


def stratified_sample(files_by_shape: dict[str, list[Path]], n_target: int, rng: random.Random) -> list[Path]:
    """Sample ~n_target files, as evenly as possible across shapes."""
    shapes = sorted(files_by_shape.keys())
    per_shape = max(1, n_target // max(1, len(shapes)))
    selected: list[Path] = []
    leftovers: list[Path] = []
    for shape in shapes:
        files = sorted(files_by_shape[shape])
        rng.shuffle(files)
        selected.extend(files[:per_shape])
        leftovers.extend(files[per_shape:])
    # top up with leftovers if rounding left us short
    rng.shuffle(leftovers)
    selected.extend(leftovers[: max(0, n_target - len(selected))])
    return sorted(selected)


def collect_batch_files(batch_dir: Path) -> dict[str, list[Path]]:
    """Group *_obs.npz under clips_obs/ by shape (first sub-directory name)."""
    clips_root = batch_dir / "clips_obs"
    files_by_shape: dict[str, list[Path]] = defaultdict(list)
    for npz_path in clips_root.rglob("*_obs.npz"):
        shape = npz_path.relative_to(clips_root).parts[0]
        files_by_shape[shape].append(npz_path)
    return dict(files_by_shape)


def validate_against_state(qpos: np.ndarray, qvel: np.ndarray, state: np.ndarray) -> dict:
    """Rebuild state from qpos/qvel and compare with the stored 64-dim state."""
    dof_pos = qpos[:, 7:36] - G1_DEFAULT_JOINT_POS[None, :]
    dof_vel = qvel[:, 6:35]

    quat_wxyz = qpos[:, 3:7]
    rot = sRot.from_quat(np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=-1))
    proj_grav = rot.inv().apply(np.array([0.0, 0.0, -1.0]))

    errs = {
        "dof_pos": float(np.abs(dof_pos - state[:, 0:29]).max()),
        "dof_vel": float(np.abs(dof_vel - state[:, 29:58]).max()),
        "proj_grav": float(np.abs(proj_grav - state[:, 58:61]).max()),
    }
    # ang_vel frame convention is ambiguous in the source README; report best variant.
    w = qvel[:, 3:6]
    candidates = {
        "raw": w,
        "rot_apply": rot.apply(w),
        "rot_inv_apply": rot.inv().apply(w),
    }
    ang_errs = {k: float(np.abs(v - state[:, 61:64]).max()) for k, v in candidates.items()}
    errs["ang_vel_best_variant"] = min(ang_errs, key=ang_errs.get)
    errs["ang_vel_best_err"] = ang_errs[errs["ang_vel_best_variant"]]
    return errs


def convert_one_npz(npz_path: Path, skel) -> tuple[dict, dict]:
    with np.load(npz_path, allow_pickle=True) as d:
        qpos = np.asarray(d["qpos"], dtype=np.float64)
        qvel = np.asarray(d["qvel"], dtype=np.float64)
        state = np.asarray(d["state"], dtype=np.float64)
        fps = float(d["fps"])

    assert qpos.shape[1] == 36 and qvel.shape[1] == 35, f"{npz_path.name}: bad qpos/qvel shape"

    errs = validate_against_state(qpos, qvel, state)
    ok = (
        errs["dof_pos"] < STATE_ATOL
        and errs["dof_vel"] < STATE_ATOL
        and errs["proj_grav"] < STATE_ATOL
    )

    root_rotvec = quat_wxyz_to_rotvec(qpos[:, 3:7])
    pose_aa = build_pose_aa(skel, root_rotvec, qpos[:, 7:36])
    trans = qpos[:, 0:3].astype(np.float32)

    feet = [skel.body_index(LEFT_FOOT_BODY), skel.body_index(RIGHT_FOOT_BODY)]
    foot_z = fk_body_positions(skel, pose_aa, trans.astype(np.float64), feet)[:, :, 2].min(axis=1)

    motion = {"root_trans_offset": trans, "pose_aa": pose_aa, "fps": int(round(fps))}
    stats = {
        "file": str(npz_path),
        "state_check_ok": bool(ok),
        "state_errs": errs,
        "min_foot_z_m": round(float(foot_z.min()), 4),
        "root_z_mean_m": round(float(trans[:, 2].mean()), 3),
    }
    return motion, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data1-dir", type=Path, required=True, help="new_data/data1 directory")
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--per-batch", type=int, default=250, help="clips sampled per batch dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "humanoidverse/data/robot/g1/g1_29dof.xml",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    skel = load_g1_skeleton(args.mjcf)
    rng = random.Random(args.seed)

    selected: list[tuple[str, Path]] = []
    for batch_name in BATCH_DIRS:
        batch_dir = args.data1_dir / batch_name
        if not batch_dir.exists():
            print(f"WARNING: {batch_dir} not found, skipping")
            continue
        files_by_shape = collect_batch_files(batch_dir)
        n_total = sum(len(v) for v in files_by_shape.values())
        picked = stratified_sample(files_by_shape, args.per_batch, rng)
        print(f"{batch_name}: {n_total} clips, {len(files_by_shape)} shapes -> sampled {len(picked)}")
        selected.extend((batch_name, p) for p in picked)

    if not selected:
        raise FileNotFoundError(f"no *_obs.npz found under {args.data1_dir}")

    motions: dict[str, dict] = {}
    all_stats, failed_checks = [], []
    for i, (batch_name, npz_path) in enumerate(selected):
        motion, stats = convert_one_npz(npz_path, skel)
        tag = batch_name.replace("batch_data", "bd").strip("_") or "bd"
        key = f"shape_{tag}_{npz_path.stem.removesuffix('_obs')}"
        motions[key] = motion
        stats["key"] = key
        all_stats.append(stats)
        if not stats["state_check_ok"]:
            failed_checks.append(stats)
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(selected)}] converted")

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motions, args.output_pkl)

    max_errs = {
        k: max(s["state_errs"][k] for s in all_stats) for k in ("dof_pos", "dof_vel", "proj_grav")
    }
    summary = {
        "data1_dir": str(args.data1_dir),
        "output_pkl": str(args.output_pkl),
        "seed": args.seed,
        "per_batch": args.per_batch,
        "n_clips": len(motions),
        "total_frames": sum(m["pose_aa"].shape[0] for m in motions.values()),
        "n_state_check_failed": len(failed_checks),
        "max_state_errs": max_errs,
        "ang_vel_best_variants": sorted({s["state_errs"]["ang_vel_best_variant"] for s in all_stats}),
        "failed_checks": failed_checks[:20],
        "per_file": all_stats,
    }
    report_path = args.report_json or args.output_pkl.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(motions)} clips -> {args.output_pkl}")
    print(f"Back-to-back state check: {len(failed_checks)} failed, max errs = {max_errs}")
    if failed_checks:
        print("!! state check failures indicate joint-order / frame mismatch - inspect report before training")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
