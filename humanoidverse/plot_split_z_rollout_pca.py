"""
从 ``split_z_goal_inference_isaac`` 保存的 ``.npz`` 读取 rollout 上的推断 z，
对 ``z_body`` / ``z_hand`` 分别做 PCA（NumPy SVD，前 3 维），绘制 3D 轨迹并标注 policy 目标 z。

用法::

  python -m humanoidverse.plot_split_z_rollout_pca --npz path/to/trace.npz

输出：同目录下 ``*_zbody_pca.png``、``*_zhand_pca.png``（或可指定 ``--out-dir``）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# matplotlib 延迟导入（无 headed 服务器时仍可保存）


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


def _sphere_mesh(radius: float, n_u: int = 28, n_v: int = 18) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    x = radius * np.cos(uu) * np.sin(vv)
    y = radius * np.sin(uu) * np.sin(vv)
    z = radius * np.cos(vv)
    return x, y, z


def main(
    npz_path: Path,
    out_dir: Path | None = None,
    reference_radius_body: float = 15.0,
    reference_radius_hand: float = 6.0,
    sphere_alpha: float = 0.12,
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

    def fig_block(
        name: str,
        Z: np.ndarray,
        policy_ref: np.ndarray | None,
        policy_used: np.ndarray | None,
        radius_ref: float,
        title_suffix: str,
    ) -> None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – 注册 3D

        mean, V, proj, _ = _pca_fit_transform(Z, n_components=3)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        # 轨迹（时间上色）
        t_steps = np.arange(Z.shape[0])
        sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=t_steps, cmap="viridis", s=16, label="inferred z(t)")
        ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color="gray", alpha=0.45, linewidth=0.8)

        if policy_ref is not None and policy_ref.size > 0:
            p0 = _transform_new(policy_ref.reshape(1, -1), mean, V)[0]
            ax.scatter([p0[0]], [p0[1]], [p0[2]], c="tab:blue", marker="*", s=220, label="goal policy z (pre-noise)")
        if policy_used is not None and policy_used.size > 0:
            p1 = _transform_new(policy_used.reshape(1, -1), mean, V)[0]
            ax.scatter([p1[0]], [p1[1]], [p1[2]], c="tab:red", marker="X", s=140, label="policy z (actor input)")
        # 参照球（PCA 空间中的半径取 reference_radius 作为视觉尺度；与高维球半径语义近似）
        xs, ys, zs = _sphere_mesh(radius_ref, n_u=32, n_v=20)
        ax.plot_surface(xs, ys, zs, color="lightgray", alpha=sphere_alpha, linewidth=0)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(
            f"{title_suffix}  (PCA on rollout inferred z; ref sphere r≈{radius_ref:g} in z-space)\n"
            f"motion={meta.get('motion_id')} frame={meta.get('goal_frame')} hnoise_std={meta.get('hand_noise_std')}"
        )
        fig.colorbar(sc, ax=ax, shrink=0.55, label="step index")
        ax.legend(loc="upper left", fontsize=8)
        out_png = od / f"{npz_path.stem}_{name}_pca.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"Saved {out_png}")

    policy_body_ref = np.asarray(data["policy_goal_z_body"]).reshape(-1) if "policy_goal_z_body" in data else None
    policy_hand_ref = np.asarray(data["policy_goal_z_hand"]).reshape(-1) if "policy_goal_z_hand" in data else None
    policy_body_used = np.asarray(data["policy_actor_z_body"]).reshape(-1) if "policy_actor_z_body" in data else None
    policy_hand_used = np.asarray(data["policy_actor_z_hand"]).reshape(-1) if "policy_actor_z_hand" in data else None

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
