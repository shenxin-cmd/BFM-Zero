from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _as_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
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
    return df.reset_index(drop=True)


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
    return df[num_cols].groupby("timestep", as_index=False).mean(numeric_only=True)


def _valid_metric_cols(df: pd.DataFrame, x_col: str, metrics: Iterable[str] | None = None) -> list[str]:
    if metrics is None:
        candidates = [c for c in df.columns if c != x_col]
    else:
        candidates = [c for c in metrics if c in df.columns and c != x_col]

    valid = []
    for col in candidates:
        if df[col].notna().sum() >= 2:
            valid.append(col)
    return valid


def _metric_file_name(metric: str) -> str:
    return metric.replace("/", "_").replace("\\", "_").replace(" ", "_")


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
        ys = _smooth(yy, smooth_window)

        ax.plot(xx, yy, alpha=0.25, linewidth=1.0, label="raw")
        ax.plot(xx, ys, linewidth=1.8, label=f"smooth(w={smooth_window})")
        ax.set_title(metric)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)

        if burn_idx > 0 and len(xx) > burn_idx + 2:
            xx2 = xx.iloc[burn_idx:]
            ys2 = ys.iloc[burn_idx:]
            if ys2.max() > ys2.min():
                ys2n = (ys2 - ys2.min()) / (ys2.max() - ys2.min())
                ax2 = ax.twinx()
                ax2.plot(xx2, ys2n, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.8)
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


def _collect_prefixed_metrics(df: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    return sorted(
        c
        for c in df.columns
        if any(c.startswith(prefix) for prefix in prefixes) and c != "timestep" and df[c].notna().sum() >= 2
    )


def _collect_contains_metrics(df: pd.DataFrame, keywords: Iterable[str]) -> list[str]:
    lowered = [kw.lower() for kw in keywords]
    return sorted(
        c
        for c in df.columns
        if c != "timestep" and any(kw in c.lower() for kw in lowered) and df[c].notna().sum() >= 2
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot structured-z BFM-Zero training and eval curves.")
    parser.add_argument("--result_dir", type=str, default="./results/bfmzero-isaac-rhand")
    parser.add_argument("--train_log", type=str, default="train_log.txt")
    parser.add_argument("--eval_log", type=str, default="humanoidverse_tracking_eval.csv")
    parser.add_argument("--out_dir", type=str, default="plots_structured_z")
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

    core_cols = [
        "mean_disc_reward",
        "mean_aux_reward",
        "mean_next_Q",
        "mean_next_auxQ",
        "FPS",
        "z_norm",
        "B_norm",
    ]
    loss_cols = [
        "actor_loss",
        "critic_loss",
        "aux_critic_loss",
        "q_loss",
        "fb_loss",
        "disc_loss",
        "disc_train_loss",
        "disc_expert_loss",
        "disc_wgan_gp_loss",
        "orth_loss",
        "orth_loss_diag",
        "orth_loss_offdiag",
        "anchor_loss",
        "hand_body_ortho_loss",
    ]
    q_cols = [
        "Q1",
        "Q_aux",
        "Q_discriminator",
        "Q_fb",
        "target_Q",
        "target_auxQ",
        "target_M",
    ]
    diag_cols = [
        "fb_diag",
        "fb_offdiag",
        "unc_Q",
        "unc_auxQ",
        "M1",
        "F1",
        "B",
    ]

    aux_cols = _collect_prefixed_metrics(train_df, ["aux_rew/"])
    structured_cols = _valid_metric_cols(
        train_df,
        "timestep",
        [
            "anchor_loss",
            "hand_body_ortho_loss",
            "z_norm",
            "B_norm",
            "Q_discriminator",
            "Q_fb",
            "Q_aux",
            "mean_disc_reward",
            "mean_aux_reward",
        ],
    )
    latent_cols = _collect_contains_metrics(
        train_df,
        ["anchor", "hand_body", "z_hand", "z_body", "disc_reward", "aux_reward"],
    )

    grouped = {
        "01_core_independent_axes.png": ("Core Metrics (independent axes)", core_cols),
        "02_losses_independent_axes.png": ("Loss Metrics (independent axes)", loss_cols),
        "03_q_independent_axes.png": ("Q Metrics (independent axes)", q_cols),
        "04_aux_independent_axes.png": ("Aux Terms (independent axes)", aux_cols),
        "05_diag_independent_axes.png": ("FB Diagnostics (independent axes)", diag_cols),
        "06_structured_z_independent_axes.png": ("Structured-z Key Metrics", structured_cols),
        "07_latent_related_independent_axes.png": ("Latent/Decoupling Metrics", latent_cols),
    }

    for file_name, (title, cols) in grouped.items():
        valid_cols = _valid_metric_cols(train_df, "timestep", cols)
        _plot_grid(
            train_df,
            x_col="timestep",
            metrics=valid_cols,
            out_file=out_dir / file_name,
            title=title,
            smooth_window=args.smooth_window,
            burn_in_ratio=args.burn_in_ratio,
        )

    single_dir = out_dir / "single_metrics"
    for metric in train_metrics:
        _plot_single_metric(
            train_df,
            x_col="timestep",
            metric=metric,
            out_file=single_dir / f"{_metric_file_name(metric)}.png",
            smooth_window=args.smooth_window,
            burn_in_ratio=args.burn_in_ratio,
        )

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
            out_file=out_dir / "08_tracking_eval_independent_axes.png",
            title="Tracking Eval (independent axes)",
            smooth_window=max(1, args.smooth_window // 2),
            burn_in_ratio=args.burn_in_ratio,
        )
        for metric in eval_cols:
            _plot_single_metric(
                eval_df,
                x_col="timestep",
                metric=metric,
                out_file=out_dir / "single_eval_metrics" / f"{_metric_file_name(metric)}.png",
                smooth_window=max(1, args.smooth_window // 2),
                burn_in_ratio=args.burn_in_ratio,
            )

    summary_lines = [
        f"Train log: {train_log}",
        f"Eval log: {eval_log if eval_log.exists() else 'not found'}",
        f"Detected train metrics: {len(train_metrics)}",
        f"Structured-z metrics present: {', '.join(structured_cols) if structured_cols else 'none'}",
        f"Latent-related metrics present: {', '.join(latent_cols) if latent_cols else 'none'}",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plot_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved structured-z plots to: {out_dir}")


if __name__ == "__main__":
    main()
