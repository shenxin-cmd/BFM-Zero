"""Convert BONES-SEED retargeted G1 CSV trajectories to a motion_lib-compatible pkl.

Input  : selected_one_per_type/*.csv  (120 fps, root translate in cm, angles in deg,
         36 columns: Frame, root_translateXYZ, root_rotateXYZ, 29x *_joint_dof)
Output : a single joblib dict pkl  {motion_key: {root_trans_offset, pose_aa, fps}}

Processing per CSV (see PREPROCESS_README.md for the rationale):
  1. unit conversion        cm -> m, deg -> rad
  2. root rotation          extrinsic-XYZ euler (deg) -> rotvec (pose_aa[:, 0])
  3. joint angles           29 angles -> pose_aa[:, 1:30] = angle * MJCF joint axis
  4. resample 120 -> 30 fps via stride-4 decimation
  5. ground alignment       constant per-clip z shift so the 5th-percentile lowest
                            foot height equals the calibrated standing ankle height
  6. clip slicing           non-overlapping 300-frame clips; tail / short sequences
                            kept when >= 160 frames (5.33 s); shorter dropped

Run (server):
    python scripts/data_preprocess/convert_bones_csv.py \
        --input-dir /path/to/new_data/selected_one_per_type \
        --output-pkl humanoidverse/data/bones_29dof_clips.pkl

Requires: numpy, scipy, pandas, joblib (no torch).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from g1_kinematics import (
    G1_DOF_NAMES,
    build_pose_aa,
    euler_xyz_extrinsic_deg_to_rotvec,
    ground_align_translation,
    load_g1_skeleton,
    slice_clips,
)

FPS_IN = 120
FPS_OUT = 30
CLIP_LEN = 300
MIN_LEN = 160
GROUND_PERCENTILE = 5.0

EXPECTED_JOINT_COLS = [f"{name}_dof" for name in G1_DOF_NAMES]
ROOT_TRANS_COLS = ["root_translateX", "root_translateY", "root_translateZ"]
ROOT_ROT_COLS = ["root_rotateX", "root_rotateY", "root_rotateZ"]


def convert_one_csv(csv_path: Path, skel) -> tuple[list[dict], dict]:
    df = pd.read_csv(csv_path)

    missing = [c for c in ROOT_TRANS_COLS + ROOT_ROT_COLS + EXPECTED_JOINT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing[:5]} ...")

    # 120 -> 30 fps
    df = df.iloc[::FPS_IN // FPS_OUT].reset_index(drop=True)

    trans = df[ROOT_TRANS_COLS].to_numpy(dtype=np.float64) / 100.0          # cm -> m
    root_rotvec = euler_xyz_extrinsic_deg_to_rotvec(df[ROOT_ROT_COLS].to_numpy())
    dof_pos = np.deg2rad(df[EXPECTED_JOINT_COLS].to_numpy(dtype=np.float64))  # deg -> rad

    pose_aa = build_pose_aa(skel, root_rotvec, dof_pos)
    trans_aligned, z_shift = ground_align_translation(skel, pose_aa, trans, GROUND_PERCENTILE)

    # joint-limit sanity (report only - retargeted data may slightly exceed limits)
    over_limit = np.logical_or(
        dof_pos < skel.joints_range[None, :, 0] - 0.05,
        dof_pos > skel.joints_range[None, :, 1] + 0.05,
    )
    over_limit_ratio = float(over_limit.mean())

    clips = []
    for seg_idx, (s, e) in enumerate(slice_clips(len(df), CLIP_LEN, MIN_LEN)):
        clips.append(
            {
                "key": f"bones_{csv_path.stem}_seg{seg_idx}",
                "motion": {
                    "root_trans_offset": trans_aligned[s:e].astype(np.float32),
                    "pose_aa": pose_aa[s:e],
                    "fps": FPS_OUT,
                },
            }
        )

    stats = {
        "file": csv_path.name,
        "frames_120fps": int(df.shape[0] * (FPS_IN // FPS_OUT)),
        "frames_30fps": int(len(df)),
        "n_clips": len(clips),
        "ground_z_shift_m": round(z_shift, 4),
        "over_limit_ratio": round(over_limit_ratio, 5),
        "root_z_range_m": [round(float(trans_aligned[:, 2].min()), 3),
                           round(float(trans_aligned[:, 2].max()), 3)],
    }
    return clips, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="dir with BONES-SEED *.csv")
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "humanoidverse/data/robot/g1/g1_29dof.xml",
    )
    parser.add_argument("--report-json", type=Path, default=None,
                        help="default: <output-pkl>.report.json")
    args = parser.parse_args()

    skel = load_g1_skeleton(args.mjcf)
    csv_files = sorted(args.input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no .csv found in {args.input_dir}")
    print(f"Found {len(csv_files)} csv files")

    motions: dict[str, dict] = {}
    all_stats, dropped, failed = [], [], []
    for i, csv_path in enumerate(csv_files):
        try:
            clips, stats = convert_one_csv(csv_path, skel)
        except Exception as exc:  # noqa: BLE001 - report and continue over corrupt files
            failed.append({"file": csv_path.name, "error": str(exc)})
            print(f"[{i + 1}/{len(csv_files)}] FAILED {csv_path.name}: {exc}")
            continue
        all_stats.append(stats)
        if not clips:
            dropped.append(stats["file"])
        for clip in clips:
            motions[clip["key"]] = clip["motion"]
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(csv_files)}] clips so far: {len(motions)}")

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motions, args.output_pkl)

    n_frames = sum(m["pose_aa"].shape[0] for m in motions.values())
    summary = {
        "input_dir": str(args.input_dir),
        "output_pkl": str(args.output_pkl),
        "n_source_csv": len(csv_files),
        "n_failed": len(failed),
        "n_dropped_too_short": len(dropped),
        "n_clips": len(motions),
        "total_frames_30fps": n_frames,
        "total_hours": round(n_frames / FPS_OUT / 3600, 2),
        "clip_len": CLIP_LEN,
        "min_len": MIN_LEN,
        "ground_percentile": GROUND_PERCENTILE,
        "mean_ground_z_shift_m": round(float(np.mean([s["ground_z_shift_m"] for s in all_stats])), 4)
        if all_stats else None,
        "failed": failed,
        "dropped_files": dropped,
        "per_file": all_stats,
    }
    report_path = args.report_json or args.output_pkl.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(motions)} clips ({summary['total_hours']} h) -> {args.output_pkl}")
    print(f"Dropped (too short): {len(dropped)}, failed: {len(failed)}")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
