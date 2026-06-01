"""
visualize_z_sphere_multi.py
============================
将多条 z pkl 轨迹投影到同一个球面上进行比较。

流程
----
  1. 加载指定目录（或显式列表）下的所有 .pkl 文件
  2. 按 z_body_dim 拆分为 z_body / z_hand
  3. 将所有轨迹合并，拟合共享 PCA（3D）
     → 共享坐标系使各轨迹在同一球上可直接比较
  4. 每条轨迹独立投影到球面（z_body R=15，z_hand R=6）
  5. 用高对比度颜色绘制双子图，支持保存 PNG / GIF / MP4

用法
----
  # 批量加载目录下所有 pkl
  python visualize_z_sphere_multi.py  --pkl-dir  /path/to/pkls/

  # 指定具体文件（最多支持任意多条）
  python visualize_z_sphere_multi.py  --pkl-files a.pkl b.pkl c.pkl d.pkl

  # 保存动图
  python visualize_z_sphere_multi.py  --pkl-dir /path/pkls --save-gif --save-mp4

  # 给每条轨迹自定义标签
  python visualize_z_sphere_multi.py  --pkl-dir /path/pkls  --labels trot walk jump run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

# ─── 4 条轨迹用的高对比度颜色（可扩展至更多） ─────────────────────────────────
_PALETTE = [
    "#E63946",  # 鲜红
    "#1D7FD4",  # 钴蓝
    "#2DC653",  # 翠绿
    "#FF9F1C",  # 橙黄
    "#9B5DE5",  # 紫
    "#F15BB5",  # 粉
    "#00BBF9",  # 青
    "#00F5D4",  # 薄荷绿
]

# ─── 工具 ─────────────────────────────────────────────────────────────────────

def _load_pkls(paths: list[Path]) -> list[np.ndarray]:
    zs = []
    for p in paths:
        z = joblib.load(p)
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 3:
            z = z.squeeze(1)
        if z.ndim != 2:
            raise ValueError(f"{p.name}: 形状应为 (T, D)，当前为 {z.shape}")
        zs.append(z)
    return zs


def _split_and_check(zs: list[np.ndarray], z_body_dim: int):
    """拆分并检查每条轨迹的维度一致性，返回 (bodies, hands) 列表。"""
    bodies, hands = [], []
    for z in zs:
        D = z.shape[1]
        if z_body_dim >= D:
            raise ValueError(
                f"z_body_dim={z_body_dim} >= z.shape[1]={D}，z_hand_dim 会是 0 或负数。"
            )
        bodies.append(z[:, :z_body_dim].astype(np.float64))
        hands.append(z[:, z_body_dim:].astype(np.float64))
    return bodies, hands


def _shared_pca(parts: list[np.ndarray]) -> tuple[PCA, list[np.ndarray]]:
    """
    在所有轨迹合并后的数据上拟合共享 PCA(3D)，
    再将每条轨迹分别投影到同一坐标系。
    返回 (pca, projected_list)。
    """
    combined = np.concatenate(parts, axis=0)
    pca = PCA(n_components=3, whiten=False)
    pca.fit(combined)
    ev = pca.explained_variance_ratio_
    print(f"  共享 PCA 解释方差: "
          f"PC1={ev[0]:.3f} PC2={ev[1]:.3f} PC3={ev[2]:.3f}  Σ={sum(ev):.3f}")
    projected = [pca.transform(p) for p in parts]
    return pca, projected


def _project_sphere(coords: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    return coords / safe * radius


def _draw_wireframe(ax: "Axes3D", r: float, alpha: float = 0.10, n: int = 28):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones(n), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="lightgray", alpha=alpha,
                    linewidth=0, antialiased=True)


# ─── 静态绘图 ──────────────────────────────────────────────────────────────────

def _plot_static(
    ax: "Axes3D",
    sphere_list: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    radius: float,
    title: str,
    skip: int,
) -> None:
    _draw_wireframe(ax, radius)

    for sphere, label, color in zip(sphere_list, labels, colors):
        T = sphere.shape[0]
        # 轨迹线
        ax.plot(sphere[:, 0], sphere[:, 1], sphere[:, 2],
                color=color, linewidth=1.5, alpha=0.85, zorder=4)
        # 稀疏散点
        idx = np.arange(0, T, max(1, skip))
        ax.scatter(sphere[idx, 0], sphere[idx, 1], sphere[idx, 2],
                   color=color, s=14, alpha=0.8, zorder=5)
        # 起点 ★ / 终点 ▲
        ax.scatter(*sphere[0],  marker="*", s=160, color=color,
                   edgecolors="white", linewidths=0.8, zorder=10)
        ax.scatter(*sphere[-1], marker="^", s=80,  color=color,
                   edgecolors="white", linewidths=0.6, zorder=10)

    lim = radius * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_xlabel("PC1", labelpad=2, fontsize=8)
    ax.set_ylabel("PC2", labelpad=2, fontsize=8)
    ax.set_zlabel("PC3", labelpad=2, fontsize=8)
    ax.tick_params(labelsize=6)


def save_static(
    body_list: list[np.ndarray],
    hand_list: list[np.ndarray],
    body_pca: PCA,
    hand_pca: PCA,
    labels: list[str],
    colors: list[str],
    r_body: float,
    r_hand: float,
    z_body_dim: int,
    z_hand_dim: int,
    out_png: Path,
    title_prefix: str = "",
) -> None:
    T_max = max(s.shape[0] for s in body_list)
    skip = max(1, T_max // 300)

    body_ev = body_pca.explained_variance_ratio_
    hand_ev = hand_pca.explained_variance_ratio_

    fig = plt.figure(figsize=(16, 7.5))
    suptitle = f"Multi-Trajectory z-space Sphere  |  {len(labels)} trajectories"
    if title_prefix:
        suptitle = f"{title_prefix}  |  " + suptitle
    fig.suptitle(suptitle, fontsize=12, y=0.98)

    ax_body: Axes3D = fig.add_subplot(121, projection="3d")
    ax_hand: Axes3D = fig.add_subplot(122, projection="3d")

    _plot_static(
        ax_body, body_list, labels, colors, r_body,
        title=(
            f"z_body  (dim={z_body_dim}, R={r_body})\n"
            f"共享 PCA var: {body_ev[0]:.2f}/{body_ev[1]:.2f}/{body_ev[2]:.2f}"
            f"  Σ={sum(body_ev):.2f}"
        ),
        skip=skip,
    )
    _plot_static(
        ax_hand, hand_list, labels, colors, r_hand,
        title=(
            f"z_hand  (dim={z_hand_dim}, R={r_hand})\n"
            f"共享 PCA var: {hand_ev[0]:.2f}/{hand_ev[1]:.2f}/{hand_ev[2]:.2f}"
            f"  Σ={sum(hand_ev):.2f}"
        ),
        skip=skip,
    )

    # 图例（用 patch 标注）
    patches = [
        mpatches.Patch(color=c, label=f"{lb}  (T={s.shape[0]})")
        for c, lb, s in zip(colors, labels, body_list)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(labels),
               fontsize=9, framealpha=0.85,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"静态图已保存: {out_png}")
    plt.close(fig)


# ─── 动画 ──────────────────────────────────────────────────────────────────────

def _init_anim_ax(ax: "Axes3D", r: float, title: str, elev: float = 25, azim: float = 45):
    _draw_wireframe(ax, r)
    lim = r * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel("PC1", labelpad=2, fontsize=8)
    ax.set_ylabel("PC2", labelpad=2, fontsize=8)
    ax.set_zlabel("PC3", labelpad=2, fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6)


def save_animation(
    body_list: list[np.ndarray],
    hand_list: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    r_body: float,
    r_hand: float,
    z_body_dim: int,
    z_hand_dim: int,
    out_gif: Path | None = None,
    out_mp4: Path | None = None,
    fps: int = 30,
    frame_step: int = 1,
    rotate_view: bool = False,
    rotate_speed: float = 0.3,
    elev: float = 25.0,
    azim: float = 45.0,
) -> None:
    T_max = max(s.shape[0] for s in body_list)
    frame_indices = np.arange(0, T_max, max(1, frame_step), dtype=int)
    if frame_indices[-1] != T_max - 1:
        frame_indices = np.append(frame_indices, T_max - 1)

    fig = plt.figure(figsize=(14, 6.5))
    fig.suptitle(
        f"Multi-Trajectory Sphere Animation  |  {len(labels)} traj",
        fontsize=11, y=0.98,
    )

    ax_body: Axes3D = fig.add_subplot(121, projection="3d")
    ax_hand: Axes3D = fig.add_subplot(122, projection="3d")

    _init_anim_ax(ax_body, r_body, f"z_body  R={r_body}", elev, azim)
    _init_anim_ax(ax_hand,  r_hand,  f"z_hand  R={r_hand}", elev, azim + 15)

    # 为每条轨迹在两个子图里各建一组 artists
    body_lines, hand_lines = [], []
    body_dots, hand_dots = [], []
    for color, label in zip(colors, labels):
        bl, = ax_body.plot([], [], [], color=color, linewidth=1.8, alpha=0.88,
                            label=label)
        hl, = ax_hand.plot([], [], [], color=color, linewidth=1.8, alpha=0.88,
                            label=label)
        bd, = ax_body.plot([], [], [], "o", color=color, markersize=9, zorder=10)
        hd, = ax_hand.plot([], [], [], "o", color=color, markersize=9, zorder=10)
        body_lines.append(bl)
        hand_lines.append(hl)
        body_dots.append(bd)
        hand_dots.append(hd)

    ax_body.legend(fontsize=7, loc="upper left")
    ax_hand.legend(fontsize=7, loc="upper left")
    time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=10)

    def _upd_line(line, dot, sphere, t):
        if t < 0 or len(sphere) == 0:
            return
        t = min(t, len(sphere) - 1)
        seg = sphere[: t + 1]
        line.set_data(seg[:, 0], seg[:, 1])
        line.set_3d_properties(seg[:, 2])
        dot.set_data([sphere[t, 0]], [sphere[t, 1]])
        dot.set_3d_properties([sphere[t, 2]])

    def _update(fi: int) -> Sequence:
        t = int(frame_indices[fi])
        for bline, hdot, bsphere in zip(body_lines, body_dots, body_list):
            _upd_line(bline, hdot, bsphere, t)
        for hline, hd, hsphere in zip(hand_lines, hand_dots, hand_list):
            _upd_line(hline, hd, hsphere, t)
        if rotate_view:
            az = azim + fi * rotate_speed
            ax_body.view_init(elev=elev, azim=az)
            ax_hand.view_init(elev=elev, azim=az + 15)
        time_text.set_text(f"frame {t + 1} / {T_max}")
        return (*body_lines, *hand_lines, *body_dots, *hand_dots, time_text)

    interval_ms = max(1, int(1000 / fps))
    anim = FuncAnimation(fig, _update, frames=len(frame_indices),
                         interval=interval_ms, blit=False, repeat=True)

    if out_gif is not None:
        out_gif = Path(out_gif)
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        print(f"正在保存 GIF ({len(frame_indices)} 帧, {fps} fps) → {out_gif}")
        anim.save(str(out_gif), writer=PillowWriter(fps=fps))
        print(f"GIF 已保存: {out_gif}")

    if out_mp4 is not None:
        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        print(f"正在保存 MP4 ({len(frame_indices)} 帧, {fps} fps) → {out_mp4}")
        saved = False
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(str(out_mp4), writer=FFMpegWriter(fps=fps, bitrate=4000))
            saved = True
        except Exception as e1:
            try:
                import imageio.v3 as iio
                frames = []
                for i in range(len(frame_indices)):
                    _update(i)
                    fig.canvas.draw()
                    rgba = np.asarray(fig.canvas.buffer_rgba())
                    frames.append(rgba[..., :3])
                iio.imwrite(out_mp4, frames, fps=fps)
                saved = True
            except Exception as e2:
                raise RuntimeError(
                    "MP4 保存失败，请安装 ffmpeg 或 imageio+ffmpeg。\n"
                    f"  FFMpegWriter: {e1}\n  imageio: {e2}"
                ) from e2
        if saved:
            print(f"MP4 已保存: {out_mp4}")

    plt.close(fig)


# ─── 主函数 ────────────────────────────────────────────────────────────────────

def main(
    pkl_dir: Path | None = None,
    pkl_files: list[Path] | None = None,
    z_body_dim: int = 225,
    r_body: float = 15.0,
    r_hand: float = 6.0,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    out_dir: Path | None = None,
    prefix: str = "multi",
    save_gif: bool = False,
    save_mp4: bool = False,
    anim_fps: int = 30,
    frame_step: int = 1,
    rotate_view: bool = False,
    no_png: bool = False,
) -> None:
    # ── 收集 pkl 路径 ────────────────────────────────────────────────────────
    paths: list[Path] = []
    if pkl_files:
        paths = [Path(p) for p in pkl_files]
    elif pkl_dir is not None:
        paths = sorted(Path(pkl_dir).glob("*.pkl"))
    if not paths:
        print("错误：未找到任何 pkl 文件。请通过 --pkl-dir 或 --pkl-files 指定。",
              file=sys.stderr)
        sys.exit(1)
    print(f"共找到 {len(paths)} 条轨迹:")
    for i, p in enumerate(paths):
        print(f"  [{i}] {p.name}")

    # ── 自动分配颜色与标签 ───────────────────────────────────────────────────
    n = len(paths)
    if colors is None:
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(n)]
    else:
        if len(colors) < n:
            colors = colors + [_PALETTE[i % len(_PALETTE)] for i in range(len(colors), n)]
    if labels is None:
        labels = [p.stem for p in paths]
    else:
        if len(labels) < n:
            labels = labels + [p.stem for p in paths[len(labels):]]

    # ── 加载与拆分 ───────────────────────────────────────────────────────────
    zs = _load_pkls(paths)
    z_hand_dim = zs[0].shape[1] - z_body_dim
    print(f"\nz 维度: total={zs[0].shape[1]}, z_body={z_body_dim}, z_hand={z_hand_dim}")
    bodies_raw, hands_raw = _split_and_check(zs, z_body_dim)

    # ── 共享 PCA ─────────────────────────────────────────────────────────────
    print("\n[z_body] 共享 PCA:")
    body_pca, body_pca3d_list = _shared_pca(bodies_raw)
    print("[z_hand] 共享 PCA:")
    hand_pca, hand_pca3d_list = _shared_pca(hands_raw)

    # ── 投影到球面 ───────────────────────────────────────────────────────────
    body_sph = [_project_sphere(p, r_body) for p in body_pca3d_list]
    hand_sph = [_project_sphere(p, r_hand) for p in hand_pca3d_list]

    for i, (label, bs, hs) in enumerate(zip(labels, body_sph, hand_sph)):
        print(f"  [{label}] body 球面范数≈{np.linalg.norm(bs, axis=1).mean():.2f}  "
              f"hand 球面范数≈{np.linalg.norm(hs, axis=1).mean():.2f}")

    # ── 输出路径 ─────────────────────────────────────────────────────────────
    base_dir = Path(out_dir) if out_dir else (
        paths[0].parent if pkl_dir is None else Path(pkl_dir)
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    png_path  = base_dir / f"{prefix}_sphere.png"
    gif_path  = base_dir / f"{prefix}_sphere.gif"
    mp4_path  = base_dir / f"{prefix}_sphere.mp4"
    npz_path  = base_dir / f"{prefix}_sphere_pca.npz"

    # ── 静态 PNG ──────────────────────────────────────────────────────────────
    if not no_png:
        save_static(
            body_sph, hand_sph,
            body_pca, hand_pca,
            labels, colors,
            r_body, r_hand,
            z_body_dim, z_hand_dim,
            png_path,
        )

    # ── 动画 ──────────────────────────────────────────────────────────────────
    if save_gif or save_mp4:
        save_animation(
            body_sph, hand_sph,
            labels, colors,
            r_body, r_hand,
            z_body_dim, z_hand_dim,
            out_gif=gif_path if save_gif else None,
            out_mp4=mp4_path if save_mp4 else None,
            fps=anim_fps,
            frame_step=frame_step,
            rotate_view=rotate_view,
        )

    # ── 导出 PCA 坐标 npz ─────────────────────────────────────────────────────
    save_dict: dict[str, np.ndarray] = {}
    for i, label in enumerate(labels):
        safe = label.replace("/", "_").replace(" ", "_")
        save_dict[f"{safe}_body_pca3d"]   = body_pca3d_list[i]
        save_dict[f"{safe}_body_sphere"]  = body_sph[i]
        save_dict[f"{safe}_hand_pca3d"]   = hand_pca3d_list[i]
        save_dict[f"{safe}_hand_sphere"]  = hand_sph[i]
    save_dict["shared_body_pca_components"] = body_pca.components_
    save_dict["shared_hand_pca_components"] = hand_pca.components_
    save_dict["shared_body_ev_ratio"]        = body_pca.explained_variance_ratio_
    save_dict["shared_hand_ev_ratio"]        = hand_pca.explained_variance_ratio_
    np.savez(npz_path, **save_dict)
    print(f"PCA 坐标已保存: {npz_path}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="多条 z 轨迹在同一球面上的 PCA 可视化"
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pkl-dir",   type=Path, help="包含 *.pkl 的目录")
    grp.add_argument("--pkl-files", type=Path, nargs="+", help="显式指定 pkl 文件列表")

    ap.add_argument("--z-body-dim", type=int,   default=225,
                    help="z_body 维度（默认 225）")
    ap.add_argument("--r-body",     type=float, default=15.0, help="z_body 球半径（默认 15）")
    ap.add_argument("--r-hand",     type=float, default=6.0,  help="z_hand 球半径（默认 6）")
    ap.add_argument("--labels",     type=str,   nargs="+",    help="每条轨迹的标签")
    ap.add_argument("--colors",     type=str,   nargs="+",    help="每条轨迹的颜色（hex/名称）")
    ap.add_argument("--out-dir",    type=Path,  default=None,
                    help="输出目录（默认与 pkl 同目录）")
    ap.add_argument("--prefix",     type=str,   default="multi",
                    help="输出文件名前缀（默认 multi）")
    ap.add_argument("--save-gif",   action="store_true", help="保存 GIF 动图")
    ap.add_argument("--save-mp4",   action="store_true", help="保存 MP4 视频")
    ap.add_argument("--anim-fps",   type=int,   default=30,  help="动画帧率（默认 30）")
    ap.add_argument("--frame-step", type=int,   default=1,
                    help="动画每 N 帧取一帧（轨迹很长时设 2~5 加速导出）")
    ap.add_argument("--rotate-view",action="store_true", help="动画缓慢旋转视角")
    ap.add_argument("--no-png",     action="store_true", help="不保存静态 PNG")

    args = ap.parse_args()
    main(
        pkl_dir=args.pkl_dir,
        pkl_files=args.pkl_files,
        z_body_dim=args.z_body_dim,
        r_body=args.r_body,
        r_hand=args.r_hand,
        labels=args.labels,
        colors=args.colors,
        out_dir=args.out_dir,
        prefix=args.prefix,
        save_gif=args.save_gif,
        save_mp4=args.save_mp4,
        anim_fps=args.anim_fps,
        frame_step=args.frame_step,
        rotate_view=args.rotate_view,
        no_png=args.no_png,
    )
