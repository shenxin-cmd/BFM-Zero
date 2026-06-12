"""End-to-end load test of a merged pkl through the repo's MotionLibRobot FK.

Must run inside the training environment (needs torch + repo deps, GPU optional).
Verifies each motion **one at a time** via ``load_motion_with_skeleton`` (same FK
as training) without calling ``load_motions()`` — that path spawns multiprocessing
workers + a Manager queue ("Gathering results...") which often hangs or gets
SIGKILL on shared CPU nodes, and stacks all frames into multi-GB tensors.

Checks per motion:
  * FK output contains no NaN / Inf
  * ``dof_pos`` from FK matches ``pose_aa`` within tolerance
  * per-source root height / dof velocity statistics

Run (server, repo root):
    python scripts/data_preprocess/verify_with_motion_lib.py \\
        --pkl humanoidverse/data/combined_29dof_mixed.pkl \\
        --device cpu
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


def _fk_one_motion(motion_lib, motion_data: dict):
    """Run the same single-clip FK as motion_lib.load_motions, without mp / bulk cat."""
    motion_lib.multi_thread = False
    res = motion_lib.load_motion_with_skeleton(
        np.array([0]),
        [motion_data],
        None,
        motion_lib.fix_height,
        None,
        -1,
        None,  # queue: stay in-process
        0,
    )
    return res[0][1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument(
        "--robot-yaml",
        type=Path,
        default=REPO_ROOT / "humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-motions", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    torch.set_num_threads(1)

    robot_cfg = OmegaConf.load(args.robot_yaml).robot
    motion_cfg = robot_cfg.motion
    motion_cfg.motion_file = str(args.pkl)
    asset_root = str(motion_cfg.asset.assetRoot)
    if not (REPO_ROOT / asset_root).exists():
        motion_cfg.asset.assetRoot = asset_root.replace("data/robots/", "data/robot/")

    from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

    motion_lib = MotionLibRobot(motion_cfg, num_envs=1, device=args.device)
    motion_lib.multi_thread = False

    n_total = motion_lib._num_unique_motions
    n = n_total if not args.max_motions else min(n_total, args.max_motions)
    print(f"motion_lib sees {n_total} motions, verifying {n} (one-at-a-time FK, no mp)")

    raw = joblib.load(args.pkl)
    stats = defaultdict(
        lambda: {"n": 0, "nan": 0, "dof_mismatch": 0, "root_z": [], "max_dof_vel": []}
    )

    for i in range(n):
        key = str(motion_lib._motion_data_keys[i])
        src = source_of(key)
        s = stats[src]
        s["n"] += 1

        motion_data = motion_lib._motion_data_list[i]
        if not isinstance(motion_data, dict):
            motion_data = raw[key]

        try:
            curr = _fk_one_motion(motion_lib, motion_data)
        except Exception as exc:  # noqa: BLE001
            s["nan"] += 1
            print(f"FK failed for {key}: {exc}")
            continue

        bad = False
        for attr in ("global_translation", "global_rotation", "dof_pos", "dof_vels", "global_velocity"):
            if hasattr(curr, attr):
                t = getattr(curr, attr)
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    bad = True
        if bad:
            s["nan"] += 1
            print(f"NaN/Inf in FK tensors for {key}")

        pose_aa = np.asarray(raw[key]["pose_aa"])
        src_dof = pose_aa[:, 1:30].sum(axis=-1)
        if hasattr(curr, "dof_pos"):
            fk_dof = curr.dof_pos.detach().cpu().numpy()
            m = min(len(src_dof), len(fk_dof))
            err = float(np.abs(src_dof[:m] - fk_dof[:m]).max())
            if err > 1e-3:
                s["dof_mismatch"] += 1
                if s["dof_mismatch"] <= 3:
                    print(f"dof mismatch {err:.4f} in {key}")

        if hasattr(curr, "global_translation"):
            s["root_z"].append(float(curr.global_translation[:, 0, 2].mean()))
        if hasattr(curr, "dof_vels"):
            s["max_dof_vel"].append(float(curr.dof_vels.abs().max()))

        if (i + 1) % 200 == 0:
            print(f"[{i + 1}/{n}]")

    print("\n=== Summary ===")
    for src, s in sorted(stats.items()):
        root_z = np.mean(s["root_z"]) if s["root_z"] else float("nan")
        vel_p95 = np.percentile(s["max_dof_vel"], 95) if s["max_dof_vel"] else float("nan")
        print(
            f"{src:6s}: n={s['n']:5d}  nan={s['nan']}  dof_mismatch={s['dof_mismatch']}  "
            f"root_z mean={root_z:.3f}  "
            f"max|dof_vel| p95={vel_p95:.2f} rad/s"
        )
    total_bad = sum(s["nan"] + s["dof_mismatch"] for s in stats.values())
    print("PASS" if total_bad == 0 else f"FAIL: {total_bad} motions with problems")


if __name__ == "__main__":
    main()
