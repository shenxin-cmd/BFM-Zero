"""
Plot training curves for the BFM-Zero split-z experiment.

Supports both metric schemas:
  - Legacy bilinear hand: fb_hand_loss, q_hand, Q_fb_hand, fb_hand_loss_diag/offdiag
  - Hand MSE mode:        fb_hand_mse, actor_hand_mse (body keeps bilinear Q / FB)

Panels:
  01 core metrics (rewards, FPS, z/B norms)
  02 losses (actor, critic, FB body/hand, orth, discriminator)
  03 Q metrics (body Q, legacy hand Q if present, actor_hand_mse)
  04 aux reward breakdown
  05 FB diagnostics (diag/offdiag — body-only in MSE mode)
  06 split-z hand MSE dedicated panel (when MSE metrics exist)
  07 body vs hand overlays (auto-resolved column names)
  08+ catch-all for any remaining numeric columns in train_log
  single_metrics/ — one PNG per column in train_log

Usage:
    python plot_bfmzero_split_z_curves.py \\
        --result_dir workdir/bfmzero-split-z/<run-id> \\
        [--train_log train_log.txt] \\
        [--eval_log humanoidverse_tracking_eval.csv] \\
        [--out_dir plots_split_z] \\
        [--smooth_window 7] \\
        [--burn_in_ratio 0.08] \\
        [--tail_threshold 1e4]
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
# Metric name resolution (MSE vs legacy bilinear)
# ---------------------------------------------------------------------------

def _first_existing(df: pd.DataFrame, candidates: list[str], min_points: int = 2) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].notna().sum() >= min_points:
            return col
    return None


def _resolve_split_z_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Pick the best column name for each split-z concept present in the log."""
    return {
        "hand_fb": _first_existing(df, ["fb_hand_mse", "fb_hand_loss"]),
        "hand_actor": _first_existing(df, ["actor_hand_mse"]),
        "hand_q_actor": _first_existing(df, ["q_hand"]),
        "hand_q_cpr": _first_existing(df, ["Q_fb_hand"]),
        "hand_fb_diag": _first_existing(df, ["fb_hand_loss_diag"]),
        "hand_fb_offdiag": _first_existing(df, ["fb_hand_loss_offdiag"]),
    }


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
# Plotting helpers
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
    tail_threshold: float | None = None,
    tail_start_ratio: float = 0.25,
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

    n_subplots = 2
    need_tail = (
        tail_threshold is not None
        and len(y) >= 2
        and float(y.max()) >= tail_threshold
    )
    if need_tail:
        n_subplots = 3

    fig, axes = plt.subplots(n_subplots, 1, figsize=(9, 3.5 * n_subplots), dpi=170, sharex=False)
    axes = np.atleast_1d(axes)

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

    if need_tail:
        start_idx = int(len(x) * tail_start_ratio)
        start_idx = min(start_idx, len(x) - 2)
        x3 = x.iloc[start_idx:]
        y3 = y.iloc[start_idx:]
        ys3 = ys.iloc[start_idx:]
        axes[2].plot(x3, y3, alpha=0.25, linewidth=1.0, label="raw")
        axes[2].plot(x3, ys3, linewidth=1.8, label=f"smooth(w={smooth_window})")
        axes[2].set_title(
            f"{metric} (tail {int((1 - tail_start_ratio) * 100)}%, max={y.max():.2e} >= {tail_threshold:.0e})"
        )
        axes[2].grid(alpha=0.25)
        axes[2].legend(fontsize=8)
        axes[2].set_xlabel(x_col)
    else:
        axes[1].set_xlabel(x_col)

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def _plot_body_hand_overlay(
    df: pd.DataFrame,
    x_col: str,
    pairs: list[tuple[str, str, str]],  # (body_col, hand_col, title)
    out_file: Path,
    title: str,
    smooth_window: int,
) -> None:
    valid_pairs = [
        (b, h, t) for b, h, t in pairs
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

    for i, (b_col, h_col, subplot_title) in enumerate(valid_pairs):
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

        ax.set_title(subplot_title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    for j in range(n, nrows * ncols):
        axes_arr[j // ncols, j % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def _build_metric_groups(df: pd.DataFrame, resolved: dict[str, str | None]) -> dict[str, tuple[str, list[str]]]:
    """Return {filename: (title, ordered metric list)} for grouped panels."""
    hand_fb = resolved["hand_fb"]
    hand_actor = resolved["hand_actor"]

    core_cols = [
        "mean_disc_reward",
        "mean_aux_reward",
        "mean_next_Q",
        "mean_next_auxQ",
        "FPS",
        "duration [minutes]",
        "z_norm",
        "B_norm",
        "z_body_norm",
        "z_hand_norm",
        "B_body_norm",
        "B_hand_norm",
    ]

    loss_cols = [
        "actor_loss",
        "critic_loss",
        "aux_critic_loss",
        "q_loss",
        "fb_loss",
        "fb_body_loss",
        hand_fb,
        "fb_total_loss",
        "disc_loss",
        "disc_train_loss",
        "disc_expert_loss",
        "disc_wgan_gp_loss",
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

    q_cols = [
        "Q1",
        "Q_aux",
        "Q_discriminator",
        "target_Q",
        "target_auxQ",
        "target_M",
        "Q_fb",
        "q",
        "Q_fb_body",
        resolved["hand_q_cpr"],
        "Q_fb_total",
        "q_body",
        resolved["hand_q_actor"],
        "q_total",
        hand_actor,
    ]

    diag_cols = [
        "fb_diag",
        "fb_offdiag",
        "fb_loss_offdiag",
        "fb_body_loss_diag",
        "fb_body_loss_offdiag",
        resolved["hand_fb_diag"],
        resolved["hand_fb_offdiag"],
        "unc_Q",
        "unc_auxQ",
        "M1",
        "F1",
        "B",
    ]

    hand_mse_cols = [
        "fb_body_loss",
        hand_fb,
        "fb_total_loss",
        hand_actor,
        "q_body",
        "Q_fb_body",
        "q",
        "Q_fb",
        "actor_loss",
        "B_hand_norm",
        "z_hand_norm",
    ]

    groups: dict[str, tuple[str, list[str]]] = {
        "01_core_independent_axes.png": (
            "Core Metrics – split-z run (independent axes)", core_cols
        ),
        "02_losses_independent_axes.png": (
            "Loss Metrics – split-z run (independent axes)", loss_cols
        ),
        "03_q_independent_axes.png": (
            "Q / Hand-MSE Metrics – split-z run (independent axes)", q_cols
        ),
        "04_aux_rewards_independent_axes.png": (
            "Aux Reward Terms (independent axes)",
            sorted([c for c in df.columns if c.startswith("aux_rew/")]),
        ),
        "05_diag_independent_axes.png": (
            "FB Diagnostics (independent axes)", diag_cols
        ),
    }

    if hand_fb == "fb_hand_mse" or hand_actor == "actor_hand_mse":
        groups["06_hand_mse_independent_axes.png"] = (
            "Hand MSE Mode – dedicated panel", hand_mse_cols
        )

    return groups


def _build_overlay_pairs(resolved: dict[str, str | None]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []

    if resolved["hand_fb"]:
        pairs.append(("fb_body_loss", resolved["hand_fb"], "FB loss: body vs hand"))

    pairs.extend([
        ("orth_body_loss", "orth_hand_loss", "orth loss"),
        ("orth_body_loss_diag", "orth_hand_loss_diag", "orth diag"),
        ("orth_body_loss_offdiag", "orth_hand_loss_offdiag", "orth offdiag"),
        ("z_body_norm", "z_hand_norm", "z norm"),
        ("B_body_norm", "B_hand_norm", "B norm"),
    ])

    if resolved["hand_q_cpr"]:
        pairs.append(("Q_fb_body", resolved["hand_q_cpr"], "Q_fb: body vs hand (legacy)"))
    elif resolved["hand_actor"]:
        pairs.append(("Q_fb_body", resolved["hand_actor"], "Q_fb_body vs actor_hand_mse"))

    if resolved["hand_q_actor"]:
        pairs.append(("q_body", resolved["hand_q_actor"], "q: body vs hand (legacy)"))
    elif resolved["hand_actor"]:
        pairs.append(("q_body", resolved["hand_actor"], "q_body vs actor_hand_mse"))

    if resolved["hand_fb_diag"]:
        pairs.append(("fb_body_loss_diag", resolved["hand_fb_diag"], "FB diag (legacy hand)"))
    if resolved["hand_fb_offdiag"]:
        pairs.append(("fb_body_loss_offdiag", resolved["hand_fb_offdiag"], "FB offdiag (legacy hand)"))

    return pairs


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
    parser.add_argument(
        "--tail_threshold",
        type=float,
        default=1e4,
        help=(
            "If a metric's max exceeds this value, single-metric plots also include "
            "a tail-75%% panel (useful when early training spikes to 1e9)."
        ),
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    train_log = result_dir / args.train_log
    eval_log = result_dir / args.eval_log
    out_dir = result_dir / args.out_dir

    if not train_log.exists():
        raise FileNotFoundError(f"train log not found: {train_log}")

    train_df = _clean_train_df(train_log)
    train_metrics = _valid_metric_cols(train_df, "timestep")
    resolved = _resolve_split_z_columns(train_df)

    schema = "hand-MSE" if resolved["hand_fb"] == "fb_hand_mse" or resolved["hand_actor"] else "legacy-bilinear"
    print(f"Detected schema: {schema}")
    print(f"  hand FB metric   : {resolved['hand_fb'] or '(not logged)'}")
    print(f"  hand actor metric: {resolved['hand_actor'] or resolved['hand_q_actor'] or resolved['hand_q_cpr'] or '(not logged)'}")
    print(f"  total train metrics in log: {len(train_metrics)}")

    grouped = _build_metric_groups(train_df, resolved)
    plotted_in_groups: set[str] = set()

    for file_name, (title, cols) in grouped.items():
        valid = _valid_metric_cols(train_df, "timestep", [c for c in cols if c])
        plotted_in_groups.update(valid)
        _plot_grid(
            train_df,
            x_col="timestep",
            metrics=valid,
            out_file=out_dir / file_name,
            title=title,
            smooth_window=args.smooth_window,
            burn_in_ratio=args.burn_in_ratio,
        )
        if valid:
            print(f"  grouped panel {file_name}: {len(valid)} metrics")

    overlay_pairs = _build_overlay_pairs(resolved)
    _plot_body_hand_overlay(
        train_df, "timestep", overlay_pairs,
        out_file=out_dir / "07_split_z_body_hand_overlay.png",
        title="Split-Z: body (blue) vs hand (orange)",
        smooth_window=args.smooth_window,
    )

    # Catch-all panel for any numeric column not yet in grouped panels
    remaining = [m for m in train_metrics if m not in plotted_in_groups]
    if remaining:
        _plot_grid(
            train_df,
            x_col="timestep",
            metrics=remaining,
            out_file=out_dir / "08_remaining_metrics_independent_axes.png",
            title="Remaining Logged Metrics (catch-all)",
            smooth_window=args.smooth_window,
            burn_in_ratio=args.burn_in_ratio,
        )
        print(f"  catch-all panel: {len(remaining)} metrics")

    # One PNG per metric in train_log
    single_dir = out_dir / "single_metrics"
    for metric in train_metrics:
        _plot_single_metric(
            train_df,
            x_col="timestep",
            metric=metric,
            out_file=single_dir / f"{metric.replace('/', '_')}.png",
            smooth_window=args.smooth_window,
            burn_in_ratio=args.burn_in_ratio,
            tail_threshold=args.tail_threshold,
        )

    # Optional: tracking eval curves
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
                tail_threshold=args.tail_threshold,
            )

    print(f"\nSaved split-z plots to: {out_dir}")
    print(f"  single_metrics/: {len(train_metrics)} files (one per logged column)")


if __name__ == "__main__":
    main()
