"""Visualize z_body / z_hand from tracking ``analysis_*.pkl`` via PCA → 3D → GIF.

Each ``analysis_*.pkl`` (from ``tracking_inference_split.py``) contains:
  - ``z_expert``  [T, z_dim]  target z fed to actor
  - ``z_actual``  [T, z_dim]  robot obs re-encoded through backward_map
  - ``metrics``   dict with ``z_body_dim``, ``z_hand_dim``, etc.

For every file this script:
  1. splits z_expert / z_actual into z_body and z_hand
  2. runs independent PCA (D → 3) on each of the 4 sequences
  3. prints explained-variance ratios to stdout
  4. saves static PNG + animated GIF for each sequence

Usage (server):
    python scripts/plot_analysis_z_pca_gif.py \\
        --analysis-dir results/ablation-residual-split-f-actor-324-64/tracking_inference_split \\
        --glob "analysis_*.pkl"

    python scripts/plot_analysis_z_pca_gif.py \\
        --analysis-pkl results/.../analysis_g1_circle_7_obs.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def _pca_fit_transform(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X: (T, D) → mean (D,), V (D, k), proj (T, k), ev_ratio (k,)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected 2D array, got {X.shape}")
    mean = X.mean(axis=0)
    Xc = X - mean
    t, d = Xc.shape
    k = min(n_components, d)
    if t < 2 or k == 0:
        V = np.zeros((d, max(k, 1)), dtype=np.float64)
        for j in range(max(k, 1)):
            V[j % d, j] = 1.0
        proj = (X - mean) @ V[:, :k]
        ev = np.ones(k) / max(k, 1)
        return mean, V[:, :k], proj[:, :k], ev

    _, s, vh = np.linalg.svd(Xc, full_matrices=False)
    V = vh[:k].T
    proj = Xc @ V
    var = (s[:k] ** 2) / max(t - 1, 1)
    total = var.sum() if var.sum() > 0 else 1.0
    ev = var / total
    return mean, V, proj, ev


def _axis_limits(proj: np.ndarray, margin: float = 0.12) -> tuple[float, float, float, float, float, float]:
    bmin = proj.min(axis=0)
    bmax = proj.max(axis=0)
    ctr = (bmin + bmax) * 0.5
    half = float(max((bmax - bmin).max() * 0.5, 1e-3) * (1.0 + margin))
    return (
        ctr[0] - half, ctr[0] + half,
        ctr[1] - half, ctr[1] + half,
        ctr[2] - half, ctr[2] + half,
    )


def _save_pca_figures(
    proj: np.ndarray,
    *,
    out_png: Path,
    out_gif: Path,
    title: str,
    anim_fps: float,
    anim_stride: int,
    anim_dpi: int,
    save_gif: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation as mpl_animation

    limits = _axis_limits(proj)
    t_steps = np.arange(proj.shape[0])

    # ── static PNG (full trajectory) ──
    fig = plt.figure(figsize=(7.5, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    if proj.shape[0] >= 2:
        ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color="0.35", linewidth=1.1, alpha=0.75)
    sc = ax.scatter(
        proj[:, 0], proj[:, 1], proj[:, 2],
        c=t_steps, cmap="viridis", s=22, alpha=0.95,
    )
    ax.scatter(*proj[0], marker="*", s=160, c="lime", edgecolors="k", linewidths=0.4, zorder=10, label="start")
    ax.scatter(*proj[-1], marker="^", s=70, c="red", edgecolors="k", linewidths=0.4, zorder=10, label="end")
    ax.set_xlim(*limits[0:2])
    ax.set_ylim(*limits[2:4])
    ax.set_zlim(*limits[4:6])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title + "\n(full trajectory, color = step)")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.55, label="step")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"    PNG → {out_png}")

    if not save_gif or proj.shape[0] <= 1:
        return

    frames = list(range(1, proj.shape[0] + 1, max(anim_stride, 1)))
    if frames[-1] != proj.shape[0]:
        frames.append(proj.shape[0])

    fig_a = plt.figure(figsize=(7.2, 6.5))
    ax_a = fig_a.add_subplot(111, projection="3d")

    def redraw(upto: int):
        ax_a.clear()
        upto = int(np.clip(upto, 1, proj.shape[0]))
        sub = proj[:upto]
        ts = t_steps[:upto]
        if upto >= 2:
            ax_a.plot(sub[:, 0], sub[:, 1], sub[:, 2], color="0.35", linewidth=1.05, alpha=0.72)
        ax_a.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=ts, cmap="viridis", s=22, alpha=0.95)
        ax_a.scatter(*proj[0], marker="*", s=140, c="lime", edgecolors="k", linewidths=0.4, zorder=10)
        if upto > 1:
            ax_a.scatter(*sub[-1], marker="o", s=55, c="orange", edgecolors="k", linewidths=0.4, zorder=11)
        ax_a.set_xlim(*limits[0:2])
        ax_a.set_ylim(*limits[2:4])
        ax_a.set_zlim(*limits[4:6])
        ax_a.set_xlabel("PC1")
        ax_a.set_ylabel("PC2")
        ax_a.set_zlabel("PC3")
        ax_a.set_title(title + f"\ncumulative steps 0…{upto - 1}")

    ani = mpl_animation.FuncAnimation(
        fig_a,
        redraw,
        frames=frames,
        interval=1000.0 / max(float(anim_fps), 1e-3),
        blit=False,
        repeat=True,
    )
    try:
        ani.save(str(out_gif), writer="pillow", dpi=anim_dpi)
        print(f"    GIF → {out_gif}  ({len(frames)} frames @ {anim_fps:g} fps)")
    except Exception as exc:
        print(f"    GIF failed ({exc}); install pillow: pip install pillow")
    plt.close(fig_a)


def process_analysis_pkl(
    pkl_path: Path,
    out_dir: Path,
    *,
    z_body_dim: int | None,
    anim_fps: float,
    anim_stride: int,
    anim_dpi: int,
    save_gif: bool,
) -> None:
    data = joblib.load(pkl_path)
    if "z_expert" not in data or "z_actual" not in data:
        raise KeyError(f"{pkl_path} missing z_expert/z_actual; keys={list(data.keys())}")

    z_expert = np.asarray(data["z_expert"], dtype=np.float64)
    z_actual = np.asarray(data["z_actual"], dtype=np.float64)
    metrics = data.get("metrics", {})
    if z_body_dim is None:
        z_body_dim = int(metrics.get("z_body_dim", 324))
    z_hand_dim = int(metrics.get("z_hand_dim", z_expert.shape[1] - z_body_dim))

    stem = pkl_path.stem.replace("analysis_", "", 1) if pkl_path.stem.startswith("analysis_") else pkl_path.stem
    od = out_dir / stem
    od.mkdir(parents=True, exist_ok=True)

    sequences: dict[str, np.ndarray] = {
        "z_expert_zbody": z_expert[:, :z_body_dim],
        "z_expert_zhand": z_expert[:, z_body_dim:],
        "z_actual_zbody": z_actual[:, :z_body_dim],
        "z_actual_zhand": z_actual[:, z_body_dim:],
    }

    print(f"\n{'=' * 72}")
    print(f"File: {pkl_path.name}")
    print(f"  T={z_expert.shape[0]}  z_dim={z_expert.shape[1]}  z_body_dim={z_body_dim}  z_hand_dim={z_hand_dim}")
    print(f"  Output: {od}")

    summary: dict[str, object] = {"source": str(pkl_path), "z_body_dim": z_body_dim, "z_hand_dim": z_hand_dim}
    pca_export: dict[str, np.ndarray] = {}

    for name, Z in sequences.items():
        mean, V, proj, ev = _pca_fit_transform(Z, n_components=3)
        print(
            f"  [{name}]  shape={Z.shape}  "
            f"PCA var: PC1={ev[0]:.3f} PC2={ev[1]:.3f} PC3={ev[2]:.3f}  cum={ev.sum():.3f}"
        )
        title = f"{stem} · {name} · PCA 3D"
        _save_pca_figures(
            proj,
            out_png=od / f"{name}_pca3d.png",
            out_gif=od / f"{name}_pca3d.gif",
            title=title,
            anim_fps=anim_fps,
            anim_stride=anim_stride,
            anim_dpi=anim_dpi,
            save_gif=save_gif,
        )
        summary[name] = {
            "explained_variance_ratio": ev.tolist(),
            "pca_mean_shape": list(mean.shape),
            "proj_shape": list(proj.shape),
        }
        pca_export[f"{name}_pca3d"] = proj.astype(np.float32)
        pca_export[f"{name}_pca_components"] = V.astype(np.float32)
        pca_export[f"{name}_pca_mean"] = mean.astype(np.float32)

    with open(od / "pca_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez(od / "pca_coords.npz", **pca_export)
    print(f"  summary → {od / 'pca_summary.json'}")
    print(f"  coords  → {od / 'pca_coords.npz'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--analysis-dir", type=Path, help="Directory containing analysis_*.pkl")
    src.add_argument("--analysis-pkl", type=Path, help="Single analysis pkl file")
    ap.add_argument("--glob", type=str, default="analysis_*.pkl", help="Glob under --analysis-dir")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output root (default: sibling z_pca_gifs/)")
    ap.add_argument("--z-body-dim", type=int, default=None, help="Override z_body_dim (default: read from metrics)")
    ap.add_argument("--anim-fps", type=float, default=20.0)
    ap.add_argument("--anim-stride", type=int, default=2, help="Use every Nth frame in GIF")
    ap.add_argument("--anim-dpi", type=int, default=100)
    ap.add_argument("--no-gif", action="store_true", help="Only save PNG, skip GIF")
    args = ap.parse_args()

    if args.analysis_pkl:
        files = [Path(args.analysis_pkl)]
        out_root = args.out_dir or (files[0].parent / "z_pca_gifs")
    else:
        ad = Path(args.analysis_dir)
        files = sorted(ad.glob(args.glob))
        if not files:
            raise FileNotFoundError(f"No files matching {args.glob!r} in {ad}")
        out_root = args.out_dir or (ad / "z_pca_gifs")

    print(f"Processing {len(files)} analysis file(s) → {out_root}")
    for p in files:
        process_analysis_pkl(
            p,
            out_root,
            z_body_dim=args.z_body_dim,
            anim_fps=args.anim_fps,
            anim_stride=args.anim_stride,
            anim_dpi=args.anim_dpi,
            save_gif=not args.no_gif,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
