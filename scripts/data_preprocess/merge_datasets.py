"""Merge LAFAN + BONES + shape-drawing pkls into a single training dataset.

Input  : - humanoidverse/data/lafan_29dof_10s-clipped.pkl   (original, keys unchanged)
         - bones pkl from convert_bones_csv.py              (keys prefixed 'bones_')
         - shape pkl from convert_shape_npz.py              (keys prefixed 'shape_')
Output : a single joblib dict pkl with all motions + a manifest json.

Validation performed on every motion:
  * required fields present (root_trans_offset (T,3), pose_aa (T,J,3), fps)
  * no NaN / Inf
  * length >= --min-frames at the motion's own fps
  * joint angles (recovered as pose_aa[:,1:30] summed over axis components, which is
    exact for single-axis hinges) within MJCF joint limits +- tolerance (report only)

Run (server):
    python scripts/data_preprocess/merge_datasets.py \
        --lafan-pkl humanoidverse/data/lafan_29dof_10s-clipped.pkl \
        --bones-pkl humanoidverse/data/bones_29dof_clips.pkl \
        --shape-pkl humanoidverse/data/shape_29dof_clips.pkl \
        --output-pkl humanoidverse/data/combined_29dof_mixed.pkl

Requires: numpy, joblib (no torch).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from g1_kinematics import load_g1_skeleton, resolve_g1_mjcf_path

LIMIT_TOL = 0.05  # rad


def to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor without importing torch
        return x.detach().cpu().numpy()
    return np.asarray(x)


def validate_motion(key: str, motion: dict, skel, min_frames: int) -> tuple[bool, dict]:
    problems = []
    for field in ("root_trans_offset", "pose_aa", "fps"):
        if field not in motion:
            problems.append(f"missing field {field}")
    if problems:
        return False, {"key": key, "problems": problems}

    trans = to_numpy(motion["root_trans_offset"])
    pose_aa = to_numpy(motion["pose_aa"])
    T = trans.shape[0]

    if trans.shape != (T, 3):
        problems.append(f"root_trans_offset shape {trans.shape}")
    if pose_aa.ndim != 3 or pose_aa.shape[0] != T or pose_aa.shape[2] != 3:
        problems.append(f"pose_aa shape {pose_aa.shape}")
    if not np.isfinite(trans).all() or not np.isfinite(pose_aa).all():
        problems.append("NaN/Inf detected")
    if T < min_frames:
        problems.append(f"too short: {T} frames < {min_frames}")

    info = {
        "key": key,
        "frames": int(T),
        "fps": float(motion["fps"]),
        "duration_s": round(T / float(motion["fps"]), 2),
        "root_z_range": [round(float(trans[:, 2].min()), 3), round(float(trans[:, 2].max()), 3)],
    }
    if pose_aa.shape[1] >= 30 and not problems:
        # exact for single-axis hinges (axis components sum to the signed angle)
        dof_pos = pose_aa[:, 1:30].sum(axis=-1)
        over = np.logical_or(
            dof_pos < skel.joints_range[None, :, 0] - LIMIT_TOL,
            dof_pos > skel.joints_range[None, :, 1] + LIMIT_TOL,
        )
        info["over_limit_ratio"] = round(float(over.mean()), 5)

    if problems:
        info["problems"] = problems
        return False, info
    return True, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lafan-pkl", type=Path, required=True)
    parser.add_argument("--bones-pkl", type=Path, required=True)
    parser.add_argument("--shape-pkl", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=120,
                        help="reject motions shorter than this many frames")
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=None,
        help="G1 MJCF (default: auto-detect data/robot/g1 or data/robots/g1 under repo root)",
    )
    args = parser.parse_args()

    mjcf = args.mjcf or resolve_g1_mjcf_path()
    print(f"Using MJCF: {mjcf}")
    skel = load_g1_skeleton(mjcf)

    sources = {
        "lafan": args.lafan_pkl,
        "bones": args.bones_pkl,
        "shape": args.shape_pkl,
    }
    merged: dict[str, dict] = {}
    manifest = {"sources": {}, "rejected": []}

    for source, pkl_path in sources.items():
        data = joblib.load(pkl_path)
        print(f"{source}: {len(data)} motions from {pkl_path}")
        kept, infos = 0, []
        for key, motion in data.items():
            # bones_/shape_ keys are already prefixed by their converters; keep
            # original lafan keys unchanged for backward compat with eval logs.
            if source != "lafan" and not key.startswith(f"{source}_"):
                key = f"{source}_{key}"
            if key in merged:
                raise ValueError(f"duplicate key after prefixing: {key}")
            ok, info = validate_motion(key, motion, skel, args.min_frames)
            if not ok:
                manifest["rejected"].append(info)
                continue
            merged[key] = motion
            infos.append(info)
            kept += 1
        total_s = sum(i["duration_s"] for i in infos)
        manifest["sources"][source] = {
            "pkl": str(pkl_path),
            "n_motions": kept,
            "total_hours": round(total_s / 3600, 2),
            "motions": infos,
        }
        print(f"  kept {kept} ({round(total_s / 3600, 2)} h)")

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(merged, args.output_pkl)

    counts = {s: manifest["sources"][s]["n_motions"] for s in sources}
    manifest["output_pkl"] = str(args.output_pkl)
    manifest["n_total"] = len(merged)
    manifest["counts"] = counts
    manifest["source_fractions"] = {s: round(c / len(merged), 3) for s, c in counts.items()}

    manifest_path = args.output_pkl.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nMerged {len(merged)} motions -> {args.output_pkl}")
    print(f"Counts: {counts}, rejected: {len(manifest['rejected'])}")
    print(f"Manifest -> {manifest_path}")
    print("\nNext: run verify_with_motion_lib.py on the server to do a full motion_lib load test.")


if __name__ == "__main__":
    main()
