"""
Plot training curves for the BFM-Zero split-z experiment.
Modified: only plot selected metrics, tail 3/4 segment, one metric per figure.

Selected metrics:
    actor_loss, fb_loss, fb_body_loss, orth_loss, orth_loss_offdiag,
    orth_body_loss, orth_body_loss_offdiag, Q_fb, Q_fb_body
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared data utilities (unchanged)
# ---------------------------------------------------------------------------

def _as_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _smooth(y: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return y
    return y.rolling(window=window, min_periods=1, center=True).mean()


def _clean_train_df(train_log: Path) -> pd.DataFrame:
    df = pd.read_csv(train_log)
    df = _as_numeric(df)
    if "timestep" not in df.columns:
        raise ValueError(f"{train_log} does not contain 'timestep' column.")
    df = df[np.isfinite(df["timestep"])].copy()
    df = df.sort_values("timestep")
    df = df.drop_duplicates(subset=["timestep"], keep="last")
    df = df.reset_index(drop=True)
    return df


def _valid_metric_cols(
    df: pd.DataFrame, x_col: str, metrics: Iterable[str] | None = None
) -> list[str]:
    if metrics is None:
        candidates = [c for c in df.columns if c != x_col]
    else:
        candidates = [c for c in metrics if c in df.columns and c != x_col]
    return [c for c in candidates if df[c].notna().sum() >= 2]


# ---------------------------------------------------------------------------
# New single-metric plotter: only the tail 3/4 of data
# ---------------------------------------------------------------------------
def _plot_tail_single(
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    out_file: Path,
    smooth_window: int,
    tail_start_ratio: float = 0.25,
) -> None:
    """
    Plot a single metric using only the last `1 - tail_start_ratio` fraction
    of timesteps (e.g. tail 75% when tail_start_ratio = 0.25).
    """
    x = df[x_col]
    y = df[metric].astype(float)
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    if mask.sum() < 2:
        return

    # Slice to tail
    start_idx = int(len(x) * tail_start_ratio)
    start_idx = min(start_idx, len(x) - 2)  # ensure at least 2 points
    x = x[mask].iloc[start_idx:]
    y = y[mask].iloc[start_idx:]

    # Interpolate and smooth
    y = y.interpolate(limit_direction="both")
    y_smooth = _smooth(y, smooth_window)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(x, y, alpha=0.25, linewidth=1.0, label="raw")
    ax.plot(x, y_smooth, linewidth=1.8, label=f"smooth(w={smooth_window})")
    ax.set_title(f"{metric}  (last 75% of training)")
    ax.set_xlabel(x_col)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot selected BFM-Zero split-z training curves (tail 75%)."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./workdir/bfmzero-split-z",
        help="Path to the training run directory containing train_log.txt",
    )
    parser.add_argument("--train_log", type=str, default="train_log.txt")
    parser.add_argument("--out_dir", type=str, default="plots_split_z_tail")
    parser.add_argument("--smooth_window", type=int, default=7)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    train_log = result_dir / args.train_log
    out_dir = result_dir / args.out_dir

    train_df = _clean_train_df(train_log)

    # Only these metrics will be plotted
    target_metrics = [
        "actor_loss",
        "fb_loss",
        "fb_body_loss",
        "orth_loss",
        "orth_loss_offdiag",
        "orth_body_loss",
        "orth_body_loss_offdiag",
        "Q_fb",
        "Q_fb_body",
    ]

    # Filter to existing columns
    existing = _valid_metric_cols(train_df, "timestep", target_metrics)

    for metric in existing:
        out_file = out_dir / f"{metric}_tail75.png"
        _plot_tail_single(
            train_df,
            x_col="timestep",
            metric=metric,
            out_file=out_file,
            smooth_window=args.smooth_window,
            tail_start_ratio=0.25,   # keep last 75%
        )
        print(f"Saved {out_file}")

    if not existing:
        print("None of the requested metrics found in train_log.")
    else:
        print(f"All tail plots saved to {out_dir}")


if __name__ == "__main__":
    main()