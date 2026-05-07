"""
从 ``split_z_goal_inference_isaac`` 保存的 ``.npz`` 读取 rollout 上的推断 z，
对 ``z_body`` / ``z_hand`` 分别做 PCA（NumPy SVD，前 3 维），绘制 3D 轨迹并标注 policy 目标 z。

用法::

  python -m humanoidverse.plot_split_z_rollout_pca --npz path/to/trace.npz

静止图： ``*_zbody_pca.png`` / ``*_zhand_pca.png`` （``--no-save-animation`` 时仅这两项）。

GIF：默认另存 ``*_zbody_pca.gif`` / ``*_zhand_pca.gif``，按 rollout 步数累积显示轨迹；可用 ``--anim-stride`` 跳帧、``--anim-max-frames`` 封顶。

参考球：**线框网格**（无材质/填充），GIF 导出更快；``--sphere-wireframe-rstride`` / ``--sphere-wireframe-cstride`` 可再稀疏取样。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# matplotlib 延迟导入


def _pca_fit_transform(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (T, D)。返回 mean (D,), V (D, k), proj (T, k)。"""
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.shape}")
    mean = X.mean(axis=0)
    Xc = X - mean
    t = Xc.shape[0]
    if t < 2:
        d = X.shape[1]
        k = min(n_components, d)
        V = np.zeros((d, k), dtype=np.float64)
        for j in range(k):
            V[j % d, j] = 1.0
        proj = (X - mean) @ V
        return mean, V, proj

    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, vh.shape[0])
    V = vh[:k].T
    proj = Xc @ V
    return mean, V, proj


def _transform_new(X_new: np.ndarray, mean: np.ndarray, V: np.ndarray) -> np.ndarray:
    return (X_new - mean) @ V


def _sphere_mesh(radius: float, n_u: int = 24, n_v: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    x = radius * np.cos(uu) * np.sin(vv)
    y = radius * np.sin(uu) * np.sin(vv)
    z = radius * np.cos(vv)
    return x, y, z


def _uniform_axis_limits(proj: np.ndarray, radius_ref: float, *, margin_frac: float = 0.12):
    bmin = proj.min(axis=0)
    bmax = proj.max(axis=0)
    ctr = (bmin + bmax) * 0.5
    half = float(np.maximum((bmax - bmin).max() * 0.5, radius_ref) * (1.0 + margin_frac))
    return (
        ctr[0] - half,
        ctr[0] + half,
        ctr[1] - half,
        ctr[1] + half,
        ctr[2] - half,
        ctr[2] + half,
    )


def main(
    npz_path: Path,
    out_dir: Path | None = None,
    reference_radius_body: float = 15.0,
    reference_radius_hand: float = 6.0,
    save_animation: bool = True,
    anim_fps: float = 22.0,
    anim_stride: int = 2,
    anim_max_frames: int | None = None,
    anim_dpi: int = 100,
    perspective: bool = True,
    sphere_wireframe_rstride: int = 2,
    sphere_wireframe_cstride: int = 2,
) -> None:
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    od = Path(out_dir) if out_dir is not None else npz_path.parent
    od.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    zb_trace = np.asarray(data["inferred_z_body"], dtype=np.float64)
    zh_trace = np.asarray(data["inferred_z_hand"], dtype=np.float64)

    meta_raw = data.get("meta_json")
    if meta_raw is None:
        meta = {}
    else:
        raw = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw
        meta = json.loads(str(raw))

    policy_body_ref = np.asarray(data["policy_goal_z_body"]).reshape(-1) if "policy_goal_z_body" in data else None
    policy_hand_ref = np.asarray(data["policy_goal_z_hand"]).reshape(-1) if "policy_goal_z_hand" in data else None
    policy_body_used = np.asarray(data["policy_actor_z_body"]).reshape(-1) if "policy_actor_z_body" in data else None
    policy_hand_used = np.asarray(data["policy_actor_z_hand"]).reshape(-1) if "policy_actor_z_hand" in data else None

    def fig_block(
        name: str,
        Z: np.ndarray,
        policy_ref: np.ndarray | None,
        policy_used: np.ndarray | None,
        radius_ref: float,
        title_suffix: str,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        mean, V, proj = _pca_fit_transform(Z, n_components=3)

        xmin, xmax, ymin, ymax, zmin, zmax = _uniform_axis_limits(proj, radius_ref)

        sx, sy, sz = _sphere_mesh(radius_ref)

        rs = max(1, int(sphere_wireframe_rstride))
        cs = max(1, int(sphere_wireframe_cstride))

        def _draw_sphere(ax_):
            """仅网格线示意参考球（无曲面、无银色/高光），便于快速画图与导出 GIF。"""
            ax_.plot_wireframe(sx, sy, sz, rstride=rs, cstride=cs, color="0.72", linewidth=0.65, alpha=0.48)

        p_ref = (
            _transform_new(policy_ref.reshape(1, -1), mean, V)[0]
            if policy_ref is not None and policy_ref.size > 0
            else None
        )
        p_use = (
            _transform_new(policy_used.reshape(1, -1), mean, V)[0]
            if policy_used is not None and policy_used.size > 0
            else None
        )

        t_steps = np.arange(Z.shape[0])
        vmin_s, vmax_s = 0.0, float(max(Z.shape[0] - 1, 0))

        title_lines = (
            f"{title_suffix} · PCA rollout inferred z · ref sphere r≈{radius_ref:g}",
            f"motion={meta.get('motion_id')} · frame={meta.get('goal_frame')} · "
            f"hnoise_std={meta.get('hand_noise_std')}",
        )
        title_base = "\n".join(title_lines)

        def _style_axes(ax_):
            ax_.set_xlim(xmin, xmax)
            ax_.set_ylim(ymin, ymax)
            ax_.set_zlim(zmin, zmax)
            ax_.set_xlabel("PC1")
            ax_.set_ylabel("PC2")
            ax_.set_zlabel("PC3")
            ax_.set_box_aspect([1.0, 1.0, 1.0])
            ax_.tick_params(axis="both", labelsize=8)
            if perspective:
                try:
                    ax_.set_proj_type("persp")
                    ax_.dist = 10.15
                except Exception:
                    pass

        def _draw_trajectory(ax_, end_exclusive: int):
            """画出 0:end_exclusive；返回 scatter 对应的 PathCollection 供静止图挂 colorbar；无点时返回 None。"""
            end_exclusive = int(np.clip(end_exclusive, 0, proj.shape[0]))
            subs = proj[:end_exclusive]
            ts = t_steps[:end_exclusive]
            coll = None
            if end_exclusive > 0:
                coll = ax_.scatter(
                    subs[:, 0],
                    subs[:, 1],
                    subs[:, 2],
                    c=ts,
                    cmap="viridis",
                    s=26,
                    vmin=vmin_s,
                    vmax=max(vmax_s, vmin_s + 1e-6),
                    alpha=0.96,
                    depthshade=True,
                )
            if end_exclusive >= 2:
                ax_.plot(subs[:, 0], subs[:, 1], subs[:, 2], color="0.38", linewidth=1.08, alpha=0.72)
            if p_ref is not None:
                ax_.scatter([p_ref[0]], [p_ref[1]], [p_ref[2]], c="tab:blue", marker="*", s=200, label="goal policy z (pre-noise)", zorder=20)
            if p_use is not None:
                ax_.scatter([p_use[0]], [p_use[1]], [p_use[2]], c="tab:red", marker="X", s=130, label="policy z (actor input)", zorder=21)
            return coll

        # --------- 静止 PNG ---------
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        _draw_sphere(ax)
        _style_axes(ax)
        traj_pc = _draw_trajectory(ax, proj.shape[0])
        ax.set_title(title_base + "\ninferred z(t) cumulative (full rollout)")
        if traj_pc is not None:
            fig.colorbar(traj_pc, ax=ax, shrink=0.55, label="step index")
        ax.legend(loc="upper left", fontsize=8)
        out_png = od / f"{npz_path.stem}_{name}_pca.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"Saved {out_png}")

        if not save_animation or proj.shape[0] <= 1:
            if not save_animation:
                pass
            return

        frames = list(range(1, proj.shape[0] + 1, max(anim_stride, 1)))
        if anim_max_frames is not None and len(frames) > anim_max_frames:
            pick_every = max(1, len(frames) // anim_max_frames)
            frames = frames[::pick_every]
        if frames[-1] != proj.shape[0]:
            frames.append(proj.shape[0])

        fig_a = plt.figure(figsize=(7.6, 6.9))
        ax_a = fig_a.add_subplot(111, projection="3d")

        def redraw(upto: int):
            ax_a.clear()
            _draw_sphere(ax_a)
            _style_axes(ax_a)
            _draw_trajectory(ax_a, upto)
            ax_a.set_title(title_base + f"\nanimate · cumulative steps 0…{upto - 1}   (GIF: viridis = step)")
            ax_a.legend(loc="upper left", fontsize=7)

        ani = mpl_animation.FuncAnimation(
            fig_a,
            redraw,
            frames=frames,
            interval=1000.0 / max(float(anim_fps), 1e-3),
            blit=False,
            repeat=False,
        )
        gif_path = od / f"{npz_path.stem}_{name}_pca.gif"
        try:
            ani.save(str(gif_path), writer="pillow", dpi=anim_dpi)
        except Exception as e:
            print(f"GIF 保存失败 ({e})；确认已安装 pillow，或减小 --anim-max-frames")
            plt.close(fig_a)
            return

        plt.close(fig_a)
        print(f"Saved {gif_path}  (~{len(frames)} 关键帧 · {anim_fps:g} fps)")

    fig_block(
        "zbody",
        zb_trace,
        policy_body_ref,
        policy_body_used,
        reference_radius_body,
        "z_body",
    )
    fig_block(
        "zhand",
        zh_trace,
        policy_hand_ref,
        policy_hand_used,
        reference_radius_hand,
        "z_hand",
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
