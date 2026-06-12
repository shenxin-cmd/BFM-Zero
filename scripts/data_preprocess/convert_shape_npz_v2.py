"""Convert data2 (V2 continuous-IK shape-drawing) NPZ clips to motion_lib pkl.

Replaces ``convert_shape_npz.py`` (data1) for training: data1 clips often contain
right-arm IK branch jumps; data2 uses fixed/continuous swivel IK with per-clip
quality filtering (see new_data/data2/BATCH_DATA_V2_README.md).

Input  : new_data/data2/{batch_data_xy_v2, batch_data_xz_v2, batch_data_yz_v2}/
         clips_obs/{shape}/{plane}/*_obs.npz   (30 fps × 300 frames)
Output : joblib dict pkl  {motion_key: {root_trans_offset, pose_aa, fps}}

Processing (same motion_lib mapping as data1):
  * qpos -> root_trans_offset + pose_aa (via MJCF hinge axes)
  * back-to-back state check (dof_pos / dof_vel / proj_grav)
  * right-arm continuity report (max adjacent |Δq|, flags > --jump-threshold)

Sampling (default): stratified --per-batch clips per plane dir (~750 total),
matching the lafan:bones:shape balance used in training.  Pass --use-all to
convert every clip under clips_obs/ (~3000).

Run (server, repo root):
    python scripts/data_preprocess/convert_shape_npz_v2.py \\
        --data2-dir /path/to/new_data/data2 \\
        --output-pkl humanoidverse/data/shape_v2_29dof_clips.pkl

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

from convert_shape_npz import convert_one_npz, stratified_sample
from g1_kinematics import load_g1_skeleton, resolve_g1_mjcf_path

# V2 batch roots (three trajectory planes; no legacy batch_data without plane suffix)
BATCH_DIRS_V2 = ["batch_data_xy_v2", "batch_data_xz_v2", "batch_data_yz_v2"]

# qpos[29:36] = right arm 7 DOF; matches BATCH_DATA_V2_README §6.1 indices 22-28
RIGHT_ARM_QPOS_SLICE = slice(29, 36)

# Generation-side FilterThresholds use 0.12 rad for shoulder roll/pitch; we report
# the same scale so outliers are visible in the conversion report.
DEFAULT_JUMP_THRESHOLD_RAD = 0.12


def batch_dir_to_tag(batch_name: str) -> str:
    """batch_data_xy_v2 -> bd_xy, batch_data_yz_v2 -> bd_yz, ..."""
    tag = batch_name.removeprefix("batch_data_").removesuffix("_v2")
    return f"bd_{tag}" if tag else "bd"


def collect_batch_files_v2(batch_dir: Path) -> dict[str, list[Path]]:
    """Group clips_obs/**/_obs.npz by shape (first sub-directory under clips_obs/)."""
    clips_root = batch_dir / "clips_obs"
    if not clips_root.is_dir():
        return {}
    files_by_shape: dict[str, list[Path]] = defaultdict(list)
    for npz_path in sorted(clips_root.rglob("*_obs.npz")):
        rel_parts = npz_path.relative_to(clips_root).parts
        if len(rel_parts) < 2:
            print(f"WARNING: unexpected path (expected shape/plane/file): {npz_path}")
            continue
        shape = rel_parts[0]
        files_by_shape[shape].append(npz_path)
    return dict(files_by_shape)


def right_arm_continuity(qpos: np.ndarray, jump_threshold: float) -> dict:
    """Per-clip adjacent-frame right-arm joint deltas (rad @ 30 fps)."""
    arm = qpos[:, RIGHT_ARM_QPOS_SLICE]
    if len(arm) < 2:
        return {"max_arm_delta_rad": 0.0, "p99_arm_delta_rad": 0.0, "n_jump_frames": 0}
    per_frame_max = np.abs(np.diff(arm, axis=0)).max(axis=1)
    return {
        "max_arm_delta_rad": round(float(per_frame_max.max()), 5),
        "p99_arm_delta_rad": round(float(np.percentile(per_frame_max, 99)), 5),
        "n_jump_frames": int((per_frame_max > jump_threshold).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data2-dir", type=Path, required=True, help="new_data/data2 directory")
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument(
        "--per-batch",
        type=int,
        default=250,
        help="clips sampled per batch dir (ignored when --use-all)",
    )
    parser.add_argument("--use-all", action="store_true", help="convert all clips_obs NPZ (no subsampling)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=DEFAULT_JUMP_THRESHOLD_RAD,
        help="report right-arm frames with max |Δq| above this (rad/frame @30fps)",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=None,
        help="G1 MJCF (default: auto-detect under repo humanoidverse/data/)",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    mjcf = args.mjcf or resolve_g1_mjcf_path()
    print(f"Using MJCF: {mjcf}")
    skel = load_g1_skeleton(mjcf)
    rng = random.Random(args.seed)

    selected: list[tuple[str, Path]] = []
    for batch_name in BATCH_DIRS_V2:
        batch_dir = args.data2_dir / batch_name
        if not batch_dir.is_dir():
            print(f"WARNING: {batch_dir} not found, skipping")
            continue
        files_by_shape = collect_batch_files_v2(batch_dir)
        n_total = sum(len(v) for v in files_by_shape.values())
        if args.use_all:
            picked = sorted(p for files in files_by_shape.values() for p in files)
            print(f"{batch_name}: {n_total} clips, {len(files_by_shape)} shapes -> use all {len(picked)}")
        else:
            picked = stratified_sample(files_by_shape, args.per_batch, rng)
            print(f"{batch_name}: {n_total} clips, {len(files_by_shape)} shapes -> sampled {len(picked)}")
        selected.extend((batch_name, p) for p in picked)

    if not selected:
        raise FileNotFoundError(
            f"no *_obs.npz found under {args.data2_dir}/batch_data_*_v2/clips_obs"
        )

    motions: dict[str, dict] = {}
    all_stats, failed_checks, jump_flagged = [], [], []

    for i, (batch_name, npz_path) in enumerate(selected):
        with np.load(npz_path, allow_pickle=True) as d:
            qpos = np.asarray(d["qpos"], dtype=np.float64)
        cont = right_arm_continuity(qpos, args.jump_threshold)
        if cont["max_arm_delta_rad"] > args.jump_threshold:
            jump_flagged.append({"file": str(npz_path), **cont})

        motion, stats = convert_one_npz(npz_path, skel)
        tag = batch_dir_to_tag(batch_name)
        key = f"shape_{tag}_{npz_path.stem.removesuffix('_obs')}"
        motions[key] = motion
        stats["key"] = key
        stats["continuity"] = cont
        stats["shape_plane"] = str(npz_path.relative_to(args.data2_dir / batch_name / "clips_obs").parts[:2])
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
    cont_stats = [s["continuity"]["max_arm_delta_rad"] for s in all_stats]
    summary = {
        "data2_dir": str(args.data2_dir),
        "output_pkl": str(args.output_pkl),
        "version": "v2_continuous_ik",
        "use_all": args.use_all,
        "seed": args.seed,
        "per_batch": None if args.use_all else args.per_batch,
        "jump_threshold_rad": args.jump_threshold,
        "n_clips": len(motions),
        "total_frames": sum(m["pose_aa"].shape[0] for m in motions.values()),
        "n_state_check_failed": len(failed_checks),
        "n_jump_flagged": len(jump_flagged),
        "max_state_errs": max_errs,
        "continuity_max_arm_delta": {
            "mean": round(float(np.mean(cont_stats)), 5),
            "p99": round(float(np.percentile(cont_stats, 99)), 5),
            "max": round(float(max(cont_stats)), 5),
        },
        "ang_vel_best_variants": sorted({s["state_errs"]["ang_vel_best_variant"] for s in all_stats}),
        "failed_checks": failed_checks[:20],
        "jump_flagged": jump_flagged[:30],
        "per_file": all_stats,
    }
    report_path = args.report_json or args.output_pkl.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(motions)} clips -> {args.output_pkl}")
    print(f"Back-to-back state check: {len(failed_checks)} failed, max errs = {max_errs}")
    print(
        f"Right-arm continuity: p99 max|Δq|={summary['continuity_max_arm_delta']['p99']} rad, "
        f"flagged (>{args.jump_threshold})={len(jump_flagged)}"
    )
    if failed_checks:
        print("!! state check failures - inspect report before training")
    if jump_flagged:
        print(f"!! {len(jump_flagged)} clips exceed jump threshold - inspect jump_flagged in report")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
