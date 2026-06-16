"""Batch-prepare V3 shape-drawing clips for tracking_inference_split.py.

Scans ``batch_data_{xy,xz,yz}_v3/clips_obs/{shape}/{plane}/*_obs.npz`` under
``--src-dir``, validates each clip like data2 (state back-to-back + arm continuity),
and writes inference-ready NPZ to a flat ``--output-dir``.

Run (repo root):

    python scripts/data_preprocess/prepare_tracking_clips_batch.py \\
        --src-dir humanoidverse/data/inference_clips/custom_circles_src \\
        --output-dir humanoidverse/data/inference_clips/shapes_v3

Requires: numpy, scipy, joblib (no torch).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from convert_shape_npz_v2 import DEFAULT_JUMP_THRESHOLD_RAD
from prepare_tracking_clips import _match_raw_for_obs, prepare_one_obs

BATCH_DIRS_V3 = ["batch_data_xy_v3", "batch_data_xz_v3", "batch_data_yz_v3"]
_PLANE_FROM_BATCH = {
    "batch_data_xy_v3": "xy",
    "batch_data_xz_v3": "xz",
    "batch_data_yz_v3": "yz",
}


def _batch_plane(batch_name: str) -> str:
    return _PLANE_FROM_BATCH.get(batch_name, batch_name.removeprefix("batch_data_").removesuffix("_v3"))


def collect_obs_files(src_dir: Path) -> list[tuple[str, Path]]:
    """Return (batch_name, obs_path) for every *_obs.npz under clips_obs/."""
    found: list[tuple[str, Path]] = []
    for batch_name in BATCH_DIRS_V3:
        batch_dir = src_dir / batch_name
        clips_root = batch_dir / "clips_obs"
        if not clips_root.is_dir():
            print(f"WARNING: {clips_root} not found, skipping")
            continue
        for npz_path in sorted(clips_root.rglob("*_obs.npz")):
            found.append((batch_name, npz_path))
    return found


def _raw_candidates(batch_dir: Path, obs_path: Path) -> list[Path]:
    """Locate raw NPZ candidates in raw/ or clips_raw/ mirroring clips_obs layout."""
    clips_root = batch_dir / "clips_obs"
    try:
        rel = obs_path.relative_to(clips_root)
    except ValueError:
        return []
    parts = rel.parts
    if len(parts) < 2:
        return []
    shape, plane = parts[0], parts[1]
    candidates: list[Path] = []
    for root_name in ("clips_raw", "raw"):
        for sub in (batch_dir / root_name / shape / plane, batch_dir / root_name / shape):
            if sub.is_dir():
                candidates.extend(sorted(sub.glob("*.npz")))
    return candidates


def _unique_out_name(out_dir: Path, batch_name: str, obs_name: str, seen: set[str]) -> str:
    base = obs_name if obs_name.endswith("_obs.npz") else f"{Path(obs_name).stem}_obs.npz"
    if base not in seen and not (out_dir / base).exists():
        seen.add(base)
        return base
    tag = batch_name.removeprefix("batch_data_").removesuffix("_v3")
    alt = f"{tag}_{base}"
    if alt not in seen and not (out_dir / alt).exists():
        seen.add(alt)
        return alt
    i = 2
    while True:
        alt2 = f"{tag}_{Path(base).stem}_{i}_obs.npz"
        if alt2 not in seen and not (out_dir / alt2).exists():
            seen.add(alt2)
            return alt2
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="directory containing batch_data_{xy,xz,yz}_v3/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jump-threshold", type=float, default=DEFAULT_JUMP_THRESHOLD_RAD)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--max-clips", type=int, default=None, help="debug: process at most N clips")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = collect_obs_files(args.src_dir)
    if args.max_clips is not None:
        pairs = pairs[: args.max_clips]
    if not pairs:
        raise FileNotFoundError(
            f"No *_obs.npz under {args.src_dir}/batch_data_*_v3/clips_obs"
        )

    all_stats: list[dict] = []
    seen_names: set[str] = set()
    by_group: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_jump": 0, "n_fail": 0})

    for i, (batch_name, obs_path) in enumerate(pairs):
        batch_dir = args.src_dir / batch_name
        plane = _batch_plane(batch_name)
        rel = obs_path.relative_to(batch_dir / "clips_obs")
        shape = rel.parts[0] if rel.parts else "unknown"
        group_key = f"{plane}/{shape}"

        raw_cands = _raw_candidates(batch_dir, obs_path)
        raw_path = _match_raw_for_obs(obs_path, raw_cands) if raw_cands else None
        if raw_cands and raw_path is None:
            print(f"WARNING: no raw match for {obs_path.name} in {batch_name}")

        out_name = _unique_out_name(args.output_dir, batch_name, obs_path.name, seen_names)
        stats = prepare_one_obs(
            obs_path,
            raw_path,
            args.output_dir,
            args.jump_threshold,
            output_name=out_name,
        )
        target = args.output_dir / out_name
        stats["output_npz"] = str(target)

        stats["batch"] = batch_name
        stats["plane"] = plane
        stats["shape"] = shape
        all_stats.append(stats)

        g = by_group[group_key]
        g["n"] += 1
        if stats.get("jump_flagged"):
            g["n_jump"] += 1
        if stats.get("state_check_ok") is False:
            g["n_fail"] += 1

        flag = "OK" if stats.get("state_check_ok") is not False else "FAIL state"
        jump = stats.get("jump_flagged")
        jump_s = f" jump={jump}" if jump is not None else ""
        if (i + 1) % 100 == 0 or i == 0 or i + 1 == len(pairs):
            print(f"[{i + 1}/{len(pairs)}] {flag}{jump_s} {plane}/{shape} -> {target.name}")

    summary = {
        "src_dir": str(args.src_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "version": "v3_inference_batch",
        "n_clips": len(all_stats),
        "n_state_failed": sum(1 for s in all_stats if s.get("state_check_ok") is False),
        "n_jump_flagged": sum(1 for s in all_stats if s.get("jump_flagged")),
        "by_plane_shape": dict(by_group),
        "clips": all_stats,
        "inference_command_hint": (
            "uv run -m humanoidverse.tracking_inference_split "
            f"--model-folder <checkpoint_dir> "
            f"--traj-obs-dir {args.output_dir} "
            "--traj-glob '*_obs.npz' --one-per-shape-plane"
        ),
    }
    report = args.report_json or (args.output_dir / "prepare_report.json")
    with open(report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_stats)} clips -> {args.output_dir}")
    print(f"state failed={summary['n_state_failed']} jump_flagged={summary['n_jump_flagged']}")
    print(f"Report -> {report}")
    if summary["n_state_failed"]:
        raise SystemExit(f"{summary['n_state_failed']} clip(s) failed state check")


if __name__ == "__main__":
    main()
