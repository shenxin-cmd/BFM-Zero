"""
Plot training curves for the BFM-Zero split-z experiment.
Modified: only plot 5 specified FB loss terms.
For each: one full-range plot; if early values are huge (max >= 1e4),
an additional tail 75% plot is generated.

Metrics:
    fb_loss_offdiag, fb_hand_loss_offdiag, fb_body_loss_offdiag,
    fb_hand_loss_diag, fb_body_loss_diag
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared data utilities
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


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _plot_full_single(
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    out_file: Path,
    smooth_window: int,
) -> None:
    """Plot a single metric over the full training range."""
    x = df[x_col]
    y = df[metric].astype(float)
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    if mask.sum() < 2:
        return

    x = x[mask]
    y = y[mask].interpolate(limit_direction="both")
    y_smooth = _smooth(y, smooth_window)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(x, y, alpha=0.25, linewidth=1.0, label="raw")
    ax.plot(x, y_smooth, linewidth=1.8, label=f"smooth(w={smooth_window})")
    ax.set_title(f"{metric}  (full training)")
    ax.set_xlabel(x_col)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def _plot_tail_single(
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    out_file: Path,
    smooth_window: int,
    tail_start_ratio: float = 0.25,
) -> None:
    """Plot only the last (1 - tail_start_ratio) of the training steps."""
    x = df[x_col]
    y = df[metric].astype(float)
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    if mask.sum() < 2:
        return

    # Slice to tail
    start_idx = int(len(x) * tail_start_ratio)
    start_idx = min(start_idx, len(x) - 2)  # at least 2 points
    x = x[mask].iloc[start_idx:]
    y = y[mask].iloc[start_idx:]

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
        description="Plot selected FB loss curves (full range + optional tail)."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./workdir/bfmzero-split-z",
        help="Path to the training run directory containing train_log.txt",
    )
    parser.add_argument("--train_log", type=str, default="train_log.txt")
    parser.add_argument("--out_dir", type=str, default="plots_selected_fb")
    parser.add_argument("--smooth_window", type=int, default=7)
    parser.add_argument(
        "--tail_threshold",
        type=float,
        default=1e4,
        help="If the metric's maximum value exceeds this, an additional tail plot is created.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    train_log = result_dir / args.train_log
    out_dir = result_dir / args.out_dir

    train_df = _clean_train_df(train_log)

    # Exactly the 5 requested metrics
    target_metrics = [
        "fb_loss_offdiag",
        "fb_hand_loss_offdiag",
        "fb_body_loss_offdiag",
        "fb_hand_loss_diag",
        "fb_body_loss_diag",
    ]

    # Keep only those present in the logged data
    existing = [m for m in target_metrics if m in train_df.columns and train_df[m].notna().sum() >= 2]

    if not existing:
        print("None of the requested metrics were found in the train log.")
        return

    for metric in existing:
        # 1. Full range plot
        full_path = out_dir / f"{metric}.png"
        _plot_full_single(
            train_df,
            x_col="timestep",
            metric=metric,
            out_file=full_path,
            smooth_window=args.smooth_window,
        )
        print(f"Saved {full_path}")

        # 2. Check if tail plot is needed
        vals = train_df[metric].dropna()
        if len(vals) >= 2 and vals.max() >= args.tail_threshold:
            tail_path = out_dir / f"{metric}_tail75.png"
            _plot_tail_single(
                train_df,
                x_col="timestep",
                metric=metric,
                out_file=tail_path,
                smooth_window=args.smooth_window,
                tail_start_ratio=0.25,
            )
            print(f"  -> additional tail plot saved to {tail_path} (max value {vals.max():.2e} >= threshold)")

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()