"""
visualize_z_sphere.py
=====================
对 tracking_inference_npz.py 输出的 z 序列做以下处理：

  1. 按 z_body_dim 拆分为 z_body（前 z_body_dim 维）和 z_hand（后 z_hand_dim 维）
  2. 分别做 PCA 降至 3D
  3. 将每条轨迹投影到球面（z_body → R=15，z_hand → R=6）
  4. 绘制双子图：左球(body) / 右球(hand)，时间着色，保存 PNG
  5. 可选：导出 GIF 动图或 MP4 视频（轨迹随时间生长）

用法
----
    python visualize_z_sphere.py  path/to/zs_xxx.pkl  [--z-body-dim 225]
    python visualize_z_sphere.py  path/to/zs_xxx.pkl  --save-gif
    python visualize_z_sphere.py  path/to/zs_xxx.pkl  --save-mp4 --anim-fps 30
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def project_to_sphere(coords: np.ndarray, radius: float) -> np.ndarray:
    """
    将 (T, 3) 的坐标投影到半径为 radius 的球面上。
    即对每个点做归一化后再乘以 radius。
    若某点是零向量（极少数情况），保持原样。
    """
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    return coords / safe * radius


def pca_3d(data: np.ndarray) -> tuple[np.ndarray, PCA]:
    """
    对 (T, D) 数据做 PCA，返回 (T, 3) 投影和拟合好的 PCA 对象。
    同时打印各主成分解释方差比例。
    """
    pca = PCA(n_components=3, whiten=False)
    proj = pca.fit_transform(data)
    explained = pca.explained_variance_ratio_
    print(f"    解释方差比: PC1={explained[0]:.3f}  PC2={explained[1]:.3f}  "
          f"PC3={explained[2]:.3f}  累计={sum(explained):.3f}")
    return proj.astype(np.float64), pca


def draw_sphere_wireframe(ax: "Axes3D", radius: float,
                          color: str = "lightgray",
                          alpha: float = 0.12,
                          n: int = 30) -> None:
    """在 ax 上绘制透明球面网格作为参考背景。"""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(n), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha,
                    linewidth=0, antialiased=True)


def plot_sphere_trajectory(
    ax: "Axes3D",
    traj_sphere: np.ndarray,
    traj_pca: np.ndarray,
    radius: float,
    title: str,
    cmap_name: str = "plasma",
) -> None:
    """
    在 3D 轴上绘制：
      - 半透明球面网格（参考背景）
      - 球面上的轨迹（时间着色渐变）
      - 起始点（绿色星形）和终止点（红色三角）
    """
    T = traj_sphere.shape[0]
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.05, 0.95, T))

    # 参考球面
    draw_sphere_wireframe(ax, radius)

    # 连续轨迹线段（分段着色）
    for i in range(T - 1):
        seg = traj_sphere[i: i + 2]
        ax.plot(
            seg[:, 0], seg[:, 1], seg[:, 2],
            color=colors[i], linewidth=1.2, alpha=0.8,
        )

    # 各点散点（增强可见性）
    skip = max(1, T // 200)   # 最多显示 ~200 个点
    idx = np.arange(0, T, skip)
    sc = ax.scatter(
        traj_sphere[idx, 0],
        traj_sphere[idx, 1],
        traj_sphere[idx, 2],
        c=idx / T, cmap=cmap_name,
        s=12, alpha=0.85, zorder=5,
    )

    # 起终点
    ax.scatter(*traj_sphere[0],  marker="*", s=150, c="lime",
               edgecolors="black", linewidths=0.5, zorder=10, label="start")
    ax.scatter(*traj_sphere[-1], marker="^", s=80,  c="red",
               edgecolors="black", linewidths=0.5, zorder=10, label="end")

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("PC1", labelpad=2, fontsize=8)
    ax.set_ylabel("PC2", labelpad=2, fontsize=8)
    ax.set_zlabel("PC3", labelpad=2, fontsize=8)
    ax.legend(fontsize=7, loc="upper left")

    lim = radius * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.tick_params(labelsize=6)

    return sc


def _setup_anim_axis(
    ax: "Axes3D",
    radius: float,
    title: str,
    elev: float,
    azim: float,
) -> dict:
    """初始化动画用 3D 轴：球面网格 + 空轨迹线 + 当前点 + 起点。"""
    draw_sphere_wireframe(ax, radius)
    (trail_line,) = ax.plot([], [], [], color="steelblue", linewidth=1.8, alpha=0.85)
    (cur_point,) = ax.plot([], [], [], "o", color="orangered", markersize=9, zorder=10)
    (start_point,) = ax.plot([], [], [], "*", color="lime", markersize=14, zorder=11)

    lim = radius * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("PC1", labelpad=2, fontsize=8)
    ax.set_ylabel("PC2", labelpad=2, fontsize=8)
    ax.set_zlabel("PC3", labelpad=2, fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6)
    return {
        "trail_line": trail_line,
        "cur_point": cur_point,
        "start_point": start_point,
        "elev": elev,
        "azim": azim,
    }


def _update_trail(artists: dict, traj: np.ndarray, t: int) -> None:
    """更新单个子图的轨迹线、当前点和起点。"""
    if t < 0:
        return
    seg = traj[: t + 1]
    artists["trail_line"].set_data(seg[:, 0], seg[:, 1])
    artists["trail_line"].set_3d_properties(seg[:, 2])
    artists["cur_point"].set_data([traj[t, 0]], [traj[t, 1]])
    artists["cur_point"].set_3d_properties([traj[t, 2]])
    artists["start_point"].set_data([traj[0, 0]], [traj[0, 1]])
    artists["start_point"].set_3d_properties([traj[0, 2]])


def save_sphere_animation(
    body_sphere: np.ndarray,
    hand_sphere: np.ndarray,
    r_body: float,
    r_hand: float,
    z_body_dim: int,
    z_hand_dim: int,
    pkl_name: str,
    out_gif: Path | None = None,
    out_mp4: Path | None = None,
    fps: int = 30,
    frame_step: int = 1,
    elev: float = 25.0,
    azim: float = 45.0,
    rotate_view: bool = False,
    rotate_speed: float = 0.4,
) -> None:
    """
    生成双子图动画：轨迹随时间逐帧生长。
    支持保存 GIF（Pillow）和 MP4（ffmpeg / imageio）。
    """
    T = body_sphere.shape[0]
    frame_indices = np.arange(0, T, max(1, frame_step), dtype=int)
    if frame_indices[-1] != T - 1:
        frame_indices = np.append(frame_indices, T - 1)

    fig = plt.figure(figsize=(14, 6.5))
    fig.suptitle(
        f"z-space Sphere Animation  |  {pkl_name}  |  T={T}",
        fontsize=12,
        y=0.98,
    )
    ax_body: Axes3D = fig.add_subplot(121, projection="3d")
    ax_hand: Axes3D = fig.add_subplot(122, projection="3d")

    body_art = _setup_anim_axis(
        ax_body, r_body,
        f"z_body (dim={z_body_dim}, R={r_body})",
        elev, azim,
    )
    hand_art = _setup_anim_axis(
        ax_hand, r_hand,
        f"z_hand (dim={z_hand_dim}, R={r_hand})",
        elev, azim + 15,
    )

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10)

    def _update(i: int) -> Sequence:
        t = int(frame_indices[i])
        _update_trail(body_art, body_sphere, t)
        _update_trail(hand_art, hand_sphere, t)
        if rotate_view:
            az = azim + i * rotate_speed
            ax_body.view_init(elev=elev, azim=az)
            ax_hand.view_init(elev=elev, azim=az + 15)
        time_text.set_text(f"frame {t + 1} / {T}")
        return (
            body_art["trail_line"], body_art["cur_point"], body_art["start_point"],
            hand_art["trail_line"], hand_art["cur_point"], hand_art["start_point"],
            time_text,
        )

    interval_ms = max(1, int(1000 / fps))
    anim = FuncAnimation(
        fig, _update, frames=len(frame_indices),
        interval=interval_ms, blit=False, repeat=True,
    )

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
                    "MP4 保存失败。请安装 ffmpeg（matplotlib FFMpegWriter）"
                    f"或 imageio+ffmpeg。\n  FFMpegWriter: {e1}\n  imageio: {e2}"
                ) from e2
        if saved:
            print(f"MP4 已保存: {out_mp4}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main(
    pkl_path: Path,
    z_body_dim: int = 225,
    r_body: float = 15.0,
    r_hand: float = 6.0,
    out: Path | None = None,
    save_gif: bool = False,
    save_mp4: bool = False,
    gif_path: Path | None = None,
    mp4_path: Path | None = None,
    anim_fps: int = 30,
    frame_step: int = 1,
    rotate_view: bool = False,
    no_png: bool = False,
) -> None:
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"找不到 pkl 文件: {pkl_path}")

    # ── 加载 z ─────────────────────────────────────────────────────────────
    z = joblib.load(pkl_path)
    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 3:
        z = z.squeeze(1)
    if z.ndim != 2:
        raise ValueError(f"z 形状应为 (T, D)，当前为 {z.shape}")

    T, D = z.shape
    z_hand_dim = D - z_body_dim

    if z_body_dim <= 0 or z_hand_dim <= 0:
        raise ValueError(
            f"z 维度 {D}，z_body_dim={z_body_dim} 导致 z_hand_dim={z_hand_dim}≤0，"
            "请用 --z-body-dim 指定正确值。"
        )

    print(f"z shape  : {z.shape}")
    print(f"z_body   : dims [0:{z_body_dim}]  → {z_body_dim}-d")
    print(f"z_hand   : dims [{z_body_dim}:{D}]  → {z_hand_dim}-d")

    z_body = z[:, :z_body_dim].astype(np.float64)
    z_hand = z[:, z_body_dim:].astype(np.float64)

    # ── PCA ────────────────────────────────────────────────────────────────
    print("\n[z_body] PCA 3D:")
    pca_body_3d, pca_body = pca_3d(z_body)

    print("[z_hand] PCA 3D:")
    pca_hand_3d, pca_hand = pca_3d(z_hand)

    # ── 投影到球面 ───────────────────────────────────────────────────────────
    body_sphere = project_to_sphere(pca_body_3d, r_body)
    hand_sphere = project_to_sphere(pca_hand_3d, r_hand)

    print(f"\nbody 球面轨迹范数 (应≈{r_body}): "
          f"min={np.linalg.norm(body_sphere, axis=1).min():.2f}  "
          f"max={np.linalg.norm(body_sphere, axis=1).max():.2f}")
    print(f"hand 球面轨迹范数 (应≈{r_hand}): "
          f"min={np.linalg.norm(hand_sphere, axis=1).min():.2f}  "
          f"max={np.linalg.norm(hand_sphere, axis=1).max():.2f}")

    body_ev = pca_body.explained_variance_ratio_
    hand_ev = pca_hand.explained_variance_ratio_

    # ── 静态 PNG ─────────────────────────────────────────────────────────────
    if not no_png:
        fig = plt.figure(figsize=(16, 7.5))
        fig.suptitle(
            f"z-space Sphere Projection  |  {pkl_path.name}  |  T={T}",
            fontsize=13, y=0.98,
        )

        ax_body: Axes3D = fig.add_subplot(121, projection="3d")
        ax_hand: Axes3D = fig.add_subplot(122, projection="3d")

        plot_sphere_trajectory(
            ax_body, body_sphere, pca_body_3d, r_body,
            title=(
                f"z_body (dim={z_body_dim})  →  R={r_body}\n"
                f"PCA var: {body_ev[0]:.2f} / {body_ev[1]:.2f} / {body_ev[2]:.2f}"
                f"  Σ={sum(body_ev):.2f}"
            ),
            cmap_name="plasma",
        )
        plot_sphere_trajectory(
            ax_hand, hand_sphere, pca_hand_3d, r_hand,
            title=(
                f"z_hand (dim={z_hand_dim})  →  R={r_hand}\n"
                f"PCA var: {hand_ev[0]:.2f} / {hand_ev[1]:.2f} / {hand_ev[2]:.2f}"
                f"  Σ={sum(hand_ev):.2f}"
            ),
            cmap_name="viridis",
        )

        cax = fig.add_axes([0.45, 0.08, 0.1, 0.03])
        sm = plt.cm.ScalarMappable(
            cmap="gray",
            norm=mcolors.Normalize(vmin=0, vmax=T - 1),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("frame index  (time →)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        plt.tight_layout(rect=[0, 0.10, 1, 0.97])

        if out is None:
            out = pkl_path.with_name(pkl_path.stem + "_sphere.png")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=160, bbox_inches="tight")
        print(f"\n图像已保存: {out}")
        plt.close(fig)
    else:
        out = Path(out) if out is not None else pkl_path.with_name(pkl_path.stem + "_sphere.png")

    # ── GIF / MP4 动画 ───────────────────────────────────────────────────────
    out_gif = None
    out_mp4 = None
    if save_gif or gif_path is not None:
        out_gif = Path(gif_path) if gif_path is not None else out.with_suffix(".gif")
    if save_mp4 or mp4_path is not None:
        out_mp4 = Path(mp4_path) if mp4_path is not None else out.with_suffix(".mp4")

    if out_gif is not None or out_mp4 is not None:
        save_sphere_animation(
            body_sphere=body_sphere,
            hand_sphere=hand_sphere,
            r_body=r_body,
            r_hand=r_hand,
            z_body_dim=z_body_dim,
            z_hand_dim=z_hand_dim,
            pkl_name=pkl_path.name,
            out_gif=out_gif,
            out_mp4=out_mp4,
            fps=anim_fps,
            frame_step=frame_step,
            rotate_view=rotate_view,
        )

    # ── 额外：z_body 和 z_hand PCA 坐标也导出为 npz ─────────────────────────
    npz_out = out.with_suffix(".npz")
    np.savez(
        npz_out,
        z_body_pca3d=pca_body_3d,
        z_body_sphere=body_sphere,
        z_hand_pca3d=pca_hand_3d,
        z_hand_sphere=hand_sphere,
        pca_body_components=pca_body.components_,
        pca_hand_components=pca_hand.components_,
        pca_body_explained_variance_ratio=pca_body.explained_variance_ratio_,
        pca_hand_explained_variance_ratio=pca_hand.explained_variance_ratio_,
    )
    print(f"PCA 坐标已保存: {npz_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="z 序列拆分 PCA 后投影到球面可视化"
    )
    parser.add_argument(
        "pkl_path",
        type=Path,
        help="tracking_inference_npz.py 输出的 z pkl 文件路径",
    )
    parser.add_argument(
        "--z-body-dim",
        type=int,
        default=225,
        help="z_body 的维度数（默认 225）",
    )
    parser.add_argument(
        "--r-body",
        type=float,
        default=15.0,
        help="z_body 球半径（默认 15）",
    )
    parser.add_argument(
        "--r-hand",
        type=float,
        default=6.0,
        help="z_hand 球半径（默认 6）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 PNG 路径（默认与 pkl 同目录，文件名加 _sphere.png 后缀）",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="保存 GIF 动图（默认路径：与 PNG 同名 .gif）",
    )
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="保存 MP4 视频（默认路径：与 PNG 同名 .mp4，需 ffmpeg 或 imageio）",
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=None,
        help="自定义 GIF 输出路径（等价于指定 --save-gif）",
    )
    parser.add_argument(
        "--mp4-path",
        type=Path,
        default=None,
        help="自定义 MP4 输出路径（等价于指定 --save-mp4）",
    )
    parser.add_argument(
        "--anim-fps",
        type=int,
        default=30,
        help="GIF/MP4 帧率（默认 30）",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="动画每隔 N 帧取一帧（轨迹很长时可设为 2/5 以加快导出）",
    )
    parser.add_argument(
        "--rotate-view",
        action="store_true",
        help="动画中缓慢旋转视角",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="不保存静态 PNG，仅导出动画",
    )
    args = parser.parse_args()
    main(
        pkl_path=args.pkl_path,
        z_body_dim=args.z_body_dim,
        r_body=args.r_body,
        r_hand=args.r_hand,
        out=args.out,
        save_gif=args.save_gif,
        save_mp4=args.save_mp4,
        gif_path=args.gif_path,
        mp4_path=args.mp4_path,
        anim_fps=args.anim_fps,
        frame_step=args.frame_step,
        rotate_view=args.rotate_view,
        no_png=args.no_png,
    )
