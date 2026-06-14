"""Repair ``privileged_state`` in recording NPZ files for tracking inference.

Recording format (``data/recordings/*_obs.npz``):
  state            (T, 64)  — joint angles/vels + proj_grav + ang_vel (correct)
  last_action      (T, 29)
  privileged_state (T, 463) — body velocities often ~30× too large (finite-diff bug)

This script reconstructs qpos/qvel from ``state``, reruns MuJoCo FK, and writes
``{stem}_fixed.npz`` with corrected ``privileged_state`` plus ``mujoco_qpos`` /
``mujoco_qvel`` for Isaac rollout.

Run (server):
    uv run python scripts/repair_recordings_privstate.py \\
        --recordings-dir data/recordings \\
        --out-dir data/recordings/fixed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from humanoidverse.utils.recording_obs_repair import find_g1_mjcf, repair_recording_traj


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def repair_file(
    npz_path: Path,
    out_dir: Path,
    mjcf_path: Path,
    *,
    dt: float,
    force: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{npz_path.stem}_fixed.npz"
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} exists (use --force)")
        return

    print(f"\n=== {npz_path.name} ===")
    traj = _load_npz_dict(npz_path)
    print(f"  Keys: {list(traj.keys())}")
    print(f"  T = {traj['state'].shape[0]} frames")

    repaired = repair_recording_traj(traj, mjcf_path=mjcf_path, dt=dt, verbose=True)
    np.savez(str(out_path), **{k: np.asarray(v) for k, v in repaired.items()})
    print(f"  Saved → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--recordings-dir", type=Path)
    src.add_argument("--files", nargs="+", type=Path, metavar="NPZ")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--mjcf", type=Path, default=None)
    ap.add_argument("--dt", type=float, default=1.0 / 30.0, help="Frame interval for root vel finite diff")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()

    mjcf_path = args.mjcf or find_g1_mjcf(args.repo_root)
    print(f"MJCF: {mjcf_path}")

    if args.recordings_dir:
        npz_files = sorted(args.recordings_dir.glob("*_obs.npz")) or sorted(args.recordings_dir.glob("*.npz"))
        if not npz_files:
            print(f"No NPZ in {args.recordings_dir}")
            sys.exit(1)
        out_dir = args.out_dir or (args.recordings_dir / "fixed")
    else:
        npz_files = [Path(p) for p in args.files]
        out_dir = args.out_dir or (npz_files[0].parent / "fixed")

    print(f"Processing {len(npz_files)} file(s) → {out_dir}")
    for p in npz_files:
        repair_file(p, out_dir, mjcf_path, dt=args.dt, force=args.force)
    print("\nAll done.")


if __name__ == "__main__":
    main()
