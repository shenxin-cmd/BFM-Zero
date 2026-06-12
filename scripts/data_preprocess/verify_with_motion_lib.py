"""End-to-end load test of a merged pkl through the repo's MotionLibRobot + FK.

Must run inside the training environment (needs torch + repo deps, GPU optional).
It loads every motion exactly like training does, then checks:
  * FK output (dof_pos, body positions, velocities) contains no NaN
  * dof_pos recovered through FK matches pose_aa within tolerance
  * per-source frame statistics (root height, foot height, dof velocity range)

Run (server, repo root):
    python scripts/data_preprocess/verify_with_motion_lib.py \\
        --pkl humanoidverse/data/combined_29dof_mixed.pkl

Large merged pkls (~2600 motions) are verified in batches (default 200 at a time)
to avoid OOM during motion_lib FK preload.  Use --max-motions 100 for a quick smoke test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def source_of(key: str) -> str:
    if key.startswith("bones_"):
        return "bones"
    if key.startswith("shape_"):
        return "shape"
    return "lafan"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--robot-yaml", type=Path,
                        default=REPO_ROOT / "humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-motions", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--load-batch-size",
        type=int,
        default=200,
        help="how many motions to FK-load into motion_lib at once (avoid OOM on large pkls)",
    )
    args = parser.parse_args()

    robot_cfg = OmegaConf.load(args.robot_yaml).robot
    motion_cfg = robot_cfg.motion
    motion_cfg.motion_file = str(args.pkl)
    # asset path fix-up (configs use data/robots/, repo has data/robot/) mirroring
    # humanoidverse_isaac.py runtime replacement
    asset_root = str(motion_cfg.asset.assetRoot)
    if not (REPO_ROOT / asset_root).exists():
        motion_cfg.asset.assetRoot = asset_root.replace("data/robots/", "data/robot/")

    from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

    motion_lib = MotionLibRobot(motion_cfg, num_envs=1, device=args.device)
    n_total = motion_lib._num_unique_motions
    n = n_total if not args.max_motions else min(n_total, args.max_motions)
    print(f"motion_lib sees {n_total} motions, verifying {n}")

    raw = joblib.load(args.pkl)
    stats = defaultdict(lambda: {"n": 0, "nan": 0, "dof_mismatch": 0,
                                 "root_z": [], "max_dof_vel": []})

    # load_motions_for_training() would FK-precompute ALL motions into giant tensors
    # (~750k frames for 2599 clips) and often OOM / get SIGKILL on shared nodes.
    # Instead, load in small batches via load_motions(num_motions_to_load=...).
    batch_size = max(1, args.load_batch_size)
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        n_load = batch_end - batch_start
        print(f"\n--- batch {batch_start}-{batch_end - 1} / {n - 1} (loading {n_load} motions) ---")
        motion_lib.load_motions(
            random_sample=False,
            start_idx=batch_start,
            num_motions_to_load=n_load,
        )

        for local_i in range(n_load):
            global_i = batch_start + local_i
            key = str(motion_lib._motion_data_keys[global_i])
            src = source_of(key)
            s = stats[src]
            s["n"] += 1

            length = motion_lib._motion_lengths[local_i]
            times = torch.arange(0, float(length), 1.0 / 30.0, device=args.device)
            ids = torch.full((times.shape[0],), local_i, dtype=torch.long, device=args.device)
            res = motion_lib.get_motion_state(ids, times)

            bad = False
            for k in ("dof_pos", "dof_vel", "root_pos", "root_rot", "rg_pos_t", "body_vel_t"):
                if k in res and not torch.isfinite(res[k]).all():
                    bad = True
            if bad:
                s["nan"] += 1
                print(f"NaN in motion {key}")

            # FK dof_pos vs source pose_aa (sum of axis components == signed hinge angle)
            pose_aa = np.asarray(raw[key]["pose_aa"])
            src_dof = pose_aa[:, 1:30].sum(axis=-1)
            fk_dof = res["dof_pos"].cpu().numpy()
            m = min(len(src_dof), len(fk_dof))
            err = np.abs(src_dof[:m] - fk_dof[:m]).max()
            if err > 1e-3:
                s["dof_mismatch"] += 1
                if s["dof_mismatch"] <= 3:
                    print(f"dof mismatch {err:.4f} in {key}")

            s["root_z"].append(float(res["root_pos"][:, 2].mean()))
            s["max_dof_vel"].append(float(res["dof_vel"].abs().max()))

            if (global_i + 1) % 200 == 0:
                print(f"[{global_i + 1}/{n}]")

    print("\n=== Summary ===")
    for src, s in sorted(stats.items()):
        print(
            f"{src:6s}: n={s['n']:5d}  nan={s['nan']}  dof_mismatch={s['dof_mismatch']}  "
            f"root_z mean={np.mean(s['root_z']):.3f}  "
            f"max|dof_vel| p95={np.percentile(s['max_dof_vel'], 95):.2f} rad/s"
        )
    total_bad = sum(s["nan"] + s["dof_mismatch"] for s in stats.values())
    print("PASS" if total_bad == 0 else f"FAIL: {total_bad} motions with problems")


if __name__ == "__main__":
    main()
