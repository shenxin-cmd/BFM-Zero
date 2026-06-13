"""Diagnose right-hand tracking jumps: is it the DATA (IK branch switching) or the
INFERENCE (z discontinuity / policy multi-solution oscillation)?

Two independent checks (run whichever inputs you have):

1. --data-dir : scan data1 NPZ clips (clips_obs/**/*_obs.npz, or raw/*.npz with qpos).
   For each clip, compute per-frame right-arm joint deltas. Large single-frame deltas
   (default > 0.3 rad @ 30 fps ~= 9 rad/s) while the wrist世界位置保持连续 are the
   signature of IK solution-branch switching (elbow-up/down flips) baked into the
   data itself. If many clips are flagged, regenerate the data with a
   continuity-constrained IK (fix/regularize the swivel angle) before retraining.

2. --z-dir : analyze z sequences saved by tracking_inference*.py (zs_*.pkl).
   Reports adjacent-step distances for z_body / z_hand separately. Spikes here mean
   the B projection of the target trajectory is discontinuous (OOD targets) ->
   mitigate with --z-window / --z-ema-alpha smoothing in tracking_inference_split.py.
   If z is smooth but the rollout still jumps, the policy itself oscillates between
   arm configurations -> data-side fix (1) is the priority.

Run (server or any machine with numpy):
    python scripts/diagnose_hand_jump.py --data-dir /path/to/new_data/data1 --report-json data_jump_report.json
    python scripts/diagnose_hand_jump.py --z-dir results/<run>/tracking_inference_split --z-body-dim 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

RIGHT_ARM_QPOS_IDX = list(range(7 + 22, 7 + 29))  # qpos[29:36] = right arm 7 dofs
RIGHT_ELBOW_QPOS_IDX = 7 + 25                       # right_elbow_joint
JUMP_THRESHOLD_RAD = 0.3                            # per-frame delta @30fps ≈ 9 rad/s


def analyze_qpos_clip(qpos: np.ndarray) -> dict:
    """Per-clip right-arm continuity statistics from (T, 36) qpos."""
    arm = qpos[:, RIGHT_ARM_QPOS_IDX]                     # (T, 7)
    deltas = np.abs(np.diff(arm, axis=0))                  # (T-1, 7)
    max_per_frame = deltas.max(axis=1)                     # (T-1,)

    jump_frames = np.where(max_per_frame > JUMP_THRESHOLD_RAD)[0]
    elbow = qpos[:, RIGHT_ELBOW_QPOS_IDX]
    # crude branch-flip proxy: elbow angle crossing through near-zero with large delta
    elbow_delta = np.abs(np.diff(elbow))
    flips = int(np.sum((np.sign(elbow[:-1]) != np.sign(elbow[1:])) & (elbow_delta > JUMP_THRESHOLD_RAD)))

    return {
        "n_frames": int(qpos.shape[0]),
        "max_arm_delta_rad": round(float(max_per_frame.max()), 4) if len(max_per_frame) else 0.0,
        "p99_arm_delta_rad": round(float(np.percentile(max_per_frame, 99)), 4) if len(max_per_frame) else 0.0,
        "n_jump_frames": int(len(jump_frames)),
        "jump_frames": jump_frames[:20].tolist(),
        "elbow_branch_flips": flips,
    }


def run_data_check(data_dir: Path, report_json: Path | None) -> None:
    npz_files = sorted(data_dir.rglob("*_obs.npz")) or sorted(data_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"no npz found under {data_dir}")
    print(f"Scanning {len(npz_files)} npz clips for right-arm discontinuities ...")

    results, flagged = [], []
    for i, path in enumerate(npz_files):
        with np.load(path, allow_pickle=True) as d:
            if "qpos" not in d:
                continue
            stats = analyze_qpos_clip(np.asarray(d["qpos"], dtype=np.float64))
        stats["file"] = str(path)
        results.append(stats)
        if stats["n_jump_frames"] > 0:
            flagged.append(stats)
        if (i + 1) % 200 == 0:
            print(f"[{i + 1}/{len(npz_files)}]")

    n = len(results)
    print("\n=== DATA-SIDE SUMMARY ===")
    print(f"clips analyzed           : {n}")
    print(f"clips with jump frames   : {len(flagged)} ({100 * len(flagged) / max(1, n):.1f}%)  "
          f"(threshold {JUMP_THRESHOLD_RAD} rad/frame)")
    print(f"clips with elbow flips   : {sum(1 for r in results if r['elbow_branch_flips'] > 0)}")
    if results:
        print(f"p99 of max arm delta     : "
              f"{np.percentile([r['max_arm_delta_rad'] for r in results], 99):.3f} rad/frame")
    if len(flagged) / max(1, n) > 0.05:
        print("\n>> VERDICT: a significant fraction of clips contain IK branch switching.")
        print(">> Fix the data generation (continuity-constrained IK / fixed swivel angle)")
        print(">> before blaming the policy - the jumps are in the expert data itself.")
    else:
        print("\n>> VERDICT: expert data looks continuous; investigate the z side (--z-dir).")

    if report_json:
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump({"threshold": JUMP_THRESHOLD_RAD, "n_clips": n,
                       "n_flagged": len(flagged), "flagged": flagged, "all": results},
                      f, indent=2, ensure_ascii=False)
        print(f"Report -> {report_json}")


def run_z_check(z_dir: Path, z_body_dim: int) -> None:
    z_files = sorted(z_dir.glob("zs_*.pkl"))
    if not z_files:
        raise FileNotFoundError(f"no zs_*.pkl found in {z_dir}")
    print(f"Analyzing {len(z_files)} z sequences (z_body_dim={z_body_dim}) ...\n")

    for path in z_files:
        z = np.asarray(joblib.load(path), dtype=np.float64)  # (T, z_dim)
        dz = np.diff(z, axis=0)
        d_body = np.linalg.norm(dz[:, :z_body_dim], axis=1)
        d_hand = np.linalg.norm(dz[:, z_body_dim:], axis=1)
        # reference scale: z sub-vectors are normalized to sqrt(dim)
        body_scale = np.sqrt(z_body_dim)
        hand_scale = np.sqrt(max(1, z.shape[1] - z_body_dim))

        def fmt(d, scale):
            rel = d / scale
            return (f"mean={rel.mean():.3f}  p99={np.percentile(rel, 99):.3f}  "
                    f"max={rel.max():.3f}  spikes(>0.5)={int((rel > 0.5).sum())}")

        print(f"{path.name} (T={z.shape[0]}, z_dim={z.shape[1]})")
        print(f"  z_body |dz|/sqrt(d): {fmt(d_body, body_scale)}")
        print(f"  z_hand |dz|/sqrt(d): {fmt(d_hand, hand_scale)}")
        spike_steps = np.where(d_hand / hand_scale > 0.5)[0]
        if len(spike_steps):
            print(f"  z_hand spike steps : {spike_steps[:20].tolist()}")
        print()

    print(">> Relative |dz| ~ O(0.05) is smooth; spikes > 0.5 mean the B projection of")
    print(">> the target is discontinuous -> use --z-window 8 --z-ema-alpha 0.6 in")
    print(">> tracking_inference_split.py. If z is smooth but the arm still jumps,")
    print(">> the policy oscillates between IK-equivalent arm configurations -> fix the data.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None, help="data1 root (scans *_obs.npz / raw npz)")
    parser.add_argument("--z-dir", type=Path, default=None, help="dir containing zs_*.pkl from tracking inference")
    parser.add_argument("--z-body-dim", type=int, default=324, help="z_body dim of the checkpoint (225 original, 256 interim, 324 main-exp)")
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    if args.data_dir is None and args.z_dir is None:
        parser.error("provide --data-dir and/or --z-dir")
    if args.data_dir:
        run_data_check(args.data_dir, args.report_json)
    if args.z_dir:
        run_z_check(args.z_dir, args.z_body_dim)


if __name__ == "__main__":
    main()
