"""
Plot training curves for the BFM-Zero split-z experiment.

Compared to plot_bfmzero_curves.py, this script adds dedicated panels for the
new split-z metrics introduced by train_bfm_zero_split_z():

  z-norms  : z_body_norm, z_hand_norm  (vs. original z_norm)
  B-norms  : B_body_norm, B_hand_norm  (vs. original B_norm)
  FB losses: fb_body_loss, fb_hand_loss, fb_total_loss (weighted sum)
  Orth     : orth_body/hand_loss, *_diag, *_offdiag
  Q (actor): q_body, q_hand, q_total  (from update_td3_actor)
  Q_fb     : Q_fb_body, Q_fb_hand, Q_fb_total  (from update_actor in CPR/Aux)

Usage:
    python plot_bfmzero_split_z_curves.py \\
        --result_dir workdir/bfmzero-split-z/<run-id> \\
        [--train_log train_log.txt] \\
        [--eval_log humanoidverse_tracking_eval.csv] \\
        [--out_dir plots_split_z] \\
        [--smooth_window 7] \\
        [--burn_in_ratio 0.08]
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
# Shared data utilities (identical to the original script)
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


def _clean_eval_df(eval_log: Path) -> pd.DataFrame:
    df = pd.read_csv(eval_log)
    if "timestep" not in df.columns:
        raise ValueError(f"{eval_log} does not contain 'timestep' column.")
    df = _as_numeric(df)
    df = df[np.isfinite(df["timestep"])].copy()
    df = df.sort_values("timestep")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if "timestep" not in num_cols:
        num_cols.append("timestep")
    df = df[num_cols].groupby("timestep", as_index=False).mean(numeric_only=True)
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
# Plotting helpers (identical to the original script)
# ---------------------------------------------------------------------------

def _plot_grid(
    df: pd.DataFrame,
    x_col: str,
    metrics: list[str],
    out_file: Path,
    title: str,
    smooth_window: int,
    burn_in_ratio: float,
) -> None:
    if not metrics:
        return

    n = len(metrics)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), dpi=160)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    x = df[x_col]
    if len(x) == 0:
        return
    burn_idx = int(len(x) * burn_in_ratio)
    burn_idx = max(1, min(len(x) - 1, burn_idx)) if len(x) > 2 else 0

    for i, metric in enumerate(metrics):
        r = i // ncols
        c = i % ncols
        ax = axes_arr[r, c]
        y = df[metric].astype(float)
        mask = np.isfinite(y.to_numpy()) & np.isfinite(x.to_numpy())
        if mask.sum() < 2:
            ax.set_title(f"{metric} (insufficient data)")
            ax.axis("off")
            continue

        xx = x[mask]
        yy = y[mask].interpolate(limit_direction="both")
        y_s = _smooth(yy, smooth_window)

        ax.plot(xx, yy, alpha=0.25, linewidth=1.0, label="raw")
        ax.plot(xx, y_s, linewidth=1.8, label=f"smooth(w={smooth_window})")
        ax.set_title(metric)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)

        if burn_idx > 0 and len(xx) > burn_idx + 2:
            xx2 = xx.iloc[burn_idx:]
            yy2 = y_s.iloc[burn_idx:]
            if yy2.max() > yy2.min():
                yy2n = (yy2 - yy2.min()) / (yy2.max() - yy2.min())
                ax2 = ax.twinx()
                ax2.plot(xx2, yy2n, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.8)
                ax2.set_yticks([0, 1])
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_ylabel("late-trend(norm)", fontsize=8)
                ax2.tick_params(axis="y", labelsize=7)

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes_arr[r, c].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def _plot_single_metric(
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    out_file: Path,
    smooth_window: int,
    burn_in_ratio: float,
) -> None:
    x = df[x_col]
    y = df[metric].astype(float)
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    if mask.sum() < 2:
        return

    x = x[mask]
    y = y[mask].interpolate(limit_direction="both")
    ys = _smooth(y, smooth_window)

    burn_idx = int(len(x) * burn_in_ratio)
    burn_idx = max(1, min(len(x) - 1, burn_idx)) if len(x) > 2 else 0

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=170, sharex=False)
    axes[0].plot(x, y, alpha=0.25, linewidth=1.0, label="raw")
    axes[0].plot(x, ys, linewidth=1.8, label=f"smooth(w={smooth_window})")
    axes[0].set_title(f"{metric} (full)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    if burn_idx > 0 and len(x) > burn_idx + 2:
        x2 = x.iloc[burn_idx:]
        y2 = y.iloc[burn_idx:]
        ys2 = ys.iloc[burn_idx:]
        axes[1].plot(x2, y2, alpha=0.25, linewidth=1.0, label="raw")
        axes[1].plot(x2, ys2, linewidth=1.8, label=f"smooth(w={smooth_window})")
        axes[1].set_title(f"{metric} (zoom: last {int((1 - burn_in_ratio) * 100)}%)")
    else:
        axes[1].plot(x, y, alpha=0.25, linewidth=1.0, label="raw")
        axes[1].plot(x, ys, linewidth=1.8, label=f"smooth(w={smooth_window})")
        axes[1].set_title(f"{metric} (zoom fallback)")

    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel(x_col)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Overlay helper: plot body vs hand curves on the same axes
# ---------------------------------------------------------------------------

def _plot_body_hand_overlay(
    df: pd.DataFrame,
    x_col: str,
    pairs: list[tuple[str, str]],   # [(body_col, hand_col), ...]
    out_file: Path,
    title: str,
    smooth_window: int,
) -> None:
    """
    For each (body_col, hand_col) pair draw both curves in the same subplot so
    the body / hand gap is immediately visible.
    """
    valid_pairs = [
        (b, h) for b, h in pairs
        if b in df.columns and h in df.columns
        and df[b].notna().sum() >= 2 and df[h].notna().sum() >= 2
    ]
    if not valid_pairs:
        return

    n = len(valid_pairs)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), dpi=160)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    x = df[x_col]

    for i, (b_col, h_col) in enumerate(valid_pairs):
        r, c = i // ncols, i % ncols
        ax = axes_arr[r, c]

        for col, color, label in [(b_col, "tab:blue", "body"), (h_col, "tab:orange", "hand")]:
            y = df[col].astype(float)
            mask = np.isfinite(y.to_numpy()) & np.isfinite(x.to_numpy())
            if mask.sum() < 2:
                continue
            xx = x[mask]
            yy = y[mask].interpolate(limit_direction="both")
            ys = _smooth(yy, smooth_window)
            ax.plot(xx, yy, alpha=0.20, linewidth=0.8, color=color)
            ax.plot(xx, ys, linewidth=1.6, color=color, label=label)

        # Use the shared prefix (strip the "_body" / "_hand" suffix) as the title
        shared = b_col.replace("_body", "").replace("body_", "")
        ax.set_title(shared)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    for j in range(n, nrows * ncols):
        axes_arr[j // ncols, j % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BFM-Zero split-z training and eval curves."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./workdir/bfmzero-split-z",
        help=(
            "Path to the training run directory (the one that contains train_log.txt). "
            "For auto-named runs this is e.g. workdir/bfmzero-split-z/2026-5-6-21-30-15-K7QP3/"
        ),
    )
    parser.add_argument("--train_log", type=str, default="train_log.txt")
    parser.add_argument("--eval_log", type=str, default="humanoidverse_tracking_eval.csv")
    parser.add_argument("--out_dir", type=str, default="plots_split_z")
    parser.add_argument("--smooth_window", type=int, default=7)
    parser.add_argument(
        "--burn_in_ratio",
        type=float,
        default=0.08,
        help="Fraction of early timesteps removed in zoom plots.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    train_log = result_dir / args.train_log
    eval_log = result_dir / args.eval_log
    out_dir = result_dir / args.out_dir

    train_df = _clean_train_df(train_log)
    train_metrics = _valid_metric_cols(train_df, "timestep")

    # ------------------------------------------------------------------
    # Panel 01 – Core metrics  (original + split-z norms side-by-side)
    # ------------------------------------------------------------------
    core_cols = [
        "mean_disc_reward",
        "mean_aux_reward",
        "mean_next_Q",
        "mean_next_auxQ",
        "FPS",
        # aggregated norms (kept for reference)
        "z_norm",
        "B_norm",
        # split-z norms
        "z_body_norm",
        "z_hand_norm",
        "B_body_norm",
        "B_hand_norm",
    ]

    # ------------------------------------------------------------------
    # Panel 02 – Loss metrics  (aggregated + split-z per-subspace)
    # ------------------------------------------------------------------
    loss_cols = [
        "actor_loss",
        "critic_loss",
        "aux_critic_loss",
        "q_loss",
        # FB losses: aggregated + split
        "fb_loss",
        "fb_body_loss",
        "fb_hand_loss",
        "fb_total_loss",
        # discriminator losses
        "disc_loss",
        "disc_train_loss",
        "disc_expert_loss",
        "disc_wgan_gp_loss",
        # orth losses: aggregated + split
        "orth_loss",
        "orth_loss_diag",
        "orth_loss_offdiag",
        "orth_body_loss",
        "orth_hand_loss",
        "orth_body_loss_diag",
        "orth_hand_loss_diag",
        "orth_body_loss_offdiag",
        "orth_hand_loss_offdiag",
    ]

    # ------------------------------------------------------------------
    # Panel 03 – Q metrics  (aggregated + split-z)
    # ------------------------------------------------------------------
    q_cols = [
        # critic / aux_critic (not split)
        "Q1",
        "Q_aux",
        "Q_discriminator",
        "target_Q",
        "target_auxQ",
        "target_M",
        # fb Q: aggregated
        "Q_fb",
        "q",
        # fb Q: split-z
        "Q_fb_body",
        "Q_fb_hand",
        "Q_fb_total",
        "q_body",
        "q_hand",
        "q_total",
    ]

    # ------------------------------------------------------------------
    # Panel 04 – Aux rewards breakdown
    # ------------------------------------------------------------------
    aux_rew_cols = sorted([c for c in train_metrics if c.startswith("aux_rew/")])

    # ------------------------------------------------------------------
    # Panel 05 – FB diagnostics
    # ------------------------------------------------------------------
    diag_cols = ["fb_diag", "fb_offdiag", "unc_Q", "unc_auxQ", "M1", "F1", "B"]

    # ------------------------------------------------------------------
    # Panel 06 – Split-z FB losses comparison (body vs hand overlay)
    # ------------------------------------------------------------------
    fb_loss_pairs = [
        ("fb_body_loss",        "fb_hand_loss"),
        ("orth_body_loss",      "orth_hand_loss"),
        ("orth_body_loss_diag", "orth_hand_loss_diag"),
        ("orth_body_loss_offdiag", "orth_hand_loss_offdiag"),
    ]

    # ------------------------------------------------------------------
    # Panel 07 – Split-z norms comparison (body vs hand overlay)
    # ------------------------------------------------------------------
    norm_pairs = [
        ("z_body_norm", "z_hand_norm"),
        ("B_body_norm", "B_hand_norm"),
    ]

    # ------------------------------------------------------------------
    # Panel 08 – Split-z Q comparison (body vs hand overlay)
    # ------------------------------------------------------------------
    q_pairs = [
        ("Q_fb_body", "Q_fb_hand"),
        ("q_body",    "q_hand"),
    ]

    # ------------------------------------------------------------------
    # Render all grid panels
    # ------------------------------------------------------------------
    grouped = {
        "01_core_independent_axes.png": (
            "Core Metrics – split-z run (independent axes)", core_cols
        ),
        "02_losses_independent_axes.png": (
            "Loss Metrics – split-z run (independent axes)", loss_cols
        ),
        "03_q_independent_axes.png": (
            "Q Metrics – split-z run (independent axes)", q_cols
        ),
        "04_aux_rewards_independent_axes.png": (
            "Aux Reward Terms (independent axes)", aux_rew_cols
        ),
        "05_diag_independent_axes.png": (
            "FB Diagnostics (independent axes)", diag_cols
        ),
    }

    for file_name, (title, cols) in grouped.items():
        valid = _valid_metric_cols(train_df, "timestep", cols)
        _plot_grid(
            train_df,
            x_col="timestep",
            metrics=valid,
            out_file=out_dir / file_name,
            title=title,
            smooth_window=args.smooth_window,
            burn_in_ratio=args.burn_in_ratio,
        )

    # ------------------------------------------------------------------
    # Render body-vs-hand overlay panels
    # ------------------------------------------------------------------
    _plot_body_hand_overlay(
        train_df, "timestep", fb_loss_pairs,
        out_file=out_dir / "06_split_z_fb_orth_losses_overlay.png",
        title="Split-Z: FB & Orth Losses — body (blue) vs hand (orange)",
        smooth_window=args.smooth_window,
    )
    _plot_body_hand_overlay(
        train_df, "timestep", norm_pairs,
        out_file=out_dir / "07_split_z_norms_overlay.png",
        title="Split-Z: z & B Norms — body (blue) vs hand (orange)",
        smooth_window=args.smooth_window,
    )
    _plot_body_hand_overlay(
        train_df, "timestep", q_pairs,
        out_file=out_dir / "08_split_z_q_overlay.png",
        title="Split-Z: Q Values — body (blue) vs hand (orange)",
        smooth_window=args.smooth_window,
    )

    # ------------------------------------------------------------------
    # One-per-metric individual plots (all columns in the CSV)
    # ------------------------------------------------------------------
    single_dir = out_dir / "single_metrics"
    for metric in train_metrics:
        _plot_single_metric(
            train_df,
            x_col="timestep",
            metric=metric,
            out_file=single_dir / f"{metric.replace('/', '_')}.png",
            smooth_window=args.smooth_window,
            burn_in_ratio=args.burn_in_ratio,
        )

    # ------------------------------------------------------------------
    # Optional: tracking eval curves
    # ------------------------------------------------------------------
    if eval_log.exists():
        eval_df = _clean_eval_df(eval_log)
        eval_cols = _valid_metric_cols(
            eval_df,
            "timestep",
            ["emd", "obs_state_emd", "mpjpe_l", "vel_dist", "accel_dist", "distance", "proximity"],
        )
        _plot_grid(
            eval_df,
            x_col="timestep",
            metrics=eval_cols,
            out_file=out_dir / "09_tracking_eval_independent_axes.png",
            title="Tracking Eval (independent axes)",
            smooth_window=max(1, args.smooth_window // 2),
            burn_in_ratio=args.burn_in_ratio,
        )
        for metric in eval_cols:
            _plot_single_metric(
                eval_df,
                x_col="timestep",
                metric=metric,
                out_file=out_dir / "single_eval_metrics" / f"{metric}.png",
                smooth_window=max(1, args.smooth_window // 2),
                burn_in_ratio=args.burn_in_ratio,
            )

    print(f"Saved split-z plots to: {out_dir}")


if __name__ == "__main__":
    main()
