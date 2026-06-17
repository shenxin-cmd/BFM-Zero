"""
从外部轨迹目录加载 backward 观测（``state`` / ``last_action`` / ``privileged_state``），
做与 ``tracking_inference.py`` 相同的 ``backward_map → cumulative mean → project_z``，
再在 Isaac 中 rollout 并可保存并排视频。

不依赖 ``lafan_29dof.pkl`` 的运动库回放；环境与 actor 仍需正常 ``HumanoidVerseIsaacConfig``，
``config.json`` 里仍可保留合法的 ``lafan_tail_path`` 仅用于环境初始化。

轨迹文件：
  - ``.pkl``：``joblib`` 或 pickle，内容为 dict；或长度为 T 的 list，每项为帧级 dict。
  - ``.npz``：每组一个数组，维度为 ``(T, dim)``.

默认按 ``*.npz`` 扫描；若 ``*.npz`` 无文件且模式为 ``*.npz`` 或 ``*.pkl``，会自动尝试另一种扩展名。

必备键（不区分大小写；``last action`` → ``last_action``；兼容 ``priviledged_state``）：
  - ``state``          形状 ``(T, 64)``
  - ``last_action``   形状 ``(T, 29)``
  - ``privileged_state`` 通常为 ``compute_humanoid_observations_max`` 拼出的维度：
      训练若 ``root_height_obs=True`` 常为 **463**（30 体 + 扩展 ``head_link``）或 **448**（无扩展体）；
      若 ``root_height_obs=False`` 则对应 **462** / **447**。
    脚本会按 checkpoint 的 ``obs_space`` **自动对齐**：若轨迹比模型多一整块末端刚体观测（常见于多录了虚拟头），会自动裁掉等价于 Isaac **``nums_extend_bodies=1``** 的那一段。

可选键（若提供则用于更准确的 MuJoCo 专家侧）：
  - ``mujoco_qpos`` 形状 ``(T, 36)``：7 自由根 + 29 关节，与 ``IsaacRendererWithMuJoco`` 一致。

若无 ``mujoco_qpos`` / ``qpos``：用每帧 ``state[:, :29] + default_dof_pos`` 作为关节绝对角，
根姿态使用环境 ``reset`` 后的默认根（仅用于专家侧渲染，可能与真实采集略有偏差）。

录制轨迹（``data/recordings/*_obs.npz``）的 ``privileged_state`` 常含错误 body 速度
（有限差分 ~30× 偏大）。默认 ``--repair-privileged`` 会从 ``state`` 重建 qpos/qvel 并用
MuJoCo FK 重算 ``privileged_state``，同时对每条轨迹分别 rollout 并保存指标。

Rollout 输出（``<model>/tracking_inference_split/``）按**来源**与**类型**分目录：

  ``summary/``
    汇总 ``summary_by_plane_shape.json`` / ``.csv``、``summary_overall.json``

  ``clips/{plane}/{shape}/{clip_id}/``
    ``metrics.json``、``body_input.json``、``analysis.pkl``、``zs_expert.pkl``
    ``z_expert.npz``、``z_actual.npz``、``z_actual_smoothed.npz``、``z_compare.npz``
    ``ee_traj_3d.png``、``ee_traj_plane_{xy,xz,yz}.png``、``tracking.mp4``（可选）
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import json
import mediapy as media
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch.utils._pytree import tree_map

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.tracking_inference_npz import (
    _EE_BODY,
    _fk_ee_positions_base,
    _fk_one_step_base,
)
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.utils.recording_obs_repair import repair_recording_traj

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

_BACKWARD_KEYS = ("state", "last_action", "privileged_state")

# ---------- inference metric helpers ----------

# DOF slices (G1 29-DOF)
_BODY_DOF_SLICE = slice(0, 22)   # 22 non-right-arm joints
_HAND_DOF_SLICE = slice(22, 29)  # 7 right-arm joints


def _wrist_local_pos_slice(config: dict) -> slice:
    """Right wrist (right_wrist_yaw_link) local position slice in privileged_state.

    G1 29-DOF body ordering: body index 29 = right_wrist_yaw_link.
    After removing root from local_body_pos, body 29 is at offset (29-1)*3 = 84.
    If root_height_obs=True, privstate starts with root_height (1 dim), shifting by +1.
    """
    rho = bool(config["env"].get("root_height_obs", False))
    start = 85 if rho else 84
    return slice(start, start + 3)


def _backward_obs_from_env(observation: dict, last_action_buf: "torch.Tensor") -> dict:
    return {
        "state": observation["state"],
        "privileged_state": observation["privileged_state"],
        "last_action": observation.get("last_action", last_action_buf),
    }


def _compute_B_raw(model, bmap_obs: dict) -> np.ndarray:
    """Raw backward-map output B (before project_z). Shape (z_dim,)."""
    return model.backward_map(bmap_obs)[0].detach().cpu().numpy()


def _compute_z_from_B(model, B: np.ndarray) -> np.ndarray:
    B_t = torch.from_numpy(B).to(device=next(model.parameters()).device, dtype=torch.float32)
    if B_t.ndim == 1:
        B_t = B_t.unsqueeze(0)
    return model.project_z(B_t)[0].detach().cpu().numpy()


def _smooth_and_project_B(
    model,
    B_seq: np.ndarray,
    *,
    z_window: int,
    z_ema_alpha: float | None,
) -> np.ndarray:
    """Same window mean + EMA + project_z as expert ``tracking_inference``."""
    dev = next(model.parameters()).device
    z = torch.from_numpy(np.asarray(B_seq, dtype=np.float32)).to(device=dev)
    for step in range(z.shape[0]):
        end_idx = min(step + max(1, z_window), z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    if z_ema_alpha is not None and z_ema_alpha < 1.0:
        for step in range(1, z.shape[0]):
            z[step] = z_ema_alpha * z[step] + (1.0 - z_ema_alpha) * z[step - 1]
    return model.project_z(z).detach().cpu().numpy()


def _flat_backward_numpy(state: np.ndarray, priv: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.asarray(state, dtype=np.float64).reshape(-1), np.asarray(priv, dtype=np.float64).reshape(-1)],
        axis=-1,
    )


def _body_idx_from_model(model) -> np.ndarray:
    bmap = model._backward_map
    if hasattr(bmap, "body_idx"):
        return bmap.body_idx.detach().cpu().numpy()
    total = int(model.cfg.archi.total_z_dim)
    return np.arange(int(model.cfg.archi.z_body_dim), dtype=np.int64)


def _compute_body_input_diagnostics(
    expert_flat: np.ndarray,
    actual_flat: np.ndarray,
    body_idx: np.ndarray,
) -> dict:
    """L2 gaps in B_network body observation slice (expert NPZ vs Isaac)."""
    n = min(len(expert_flat), len(actual_flat))
    if n == 0:
        return {}
    exp_b = expert_flat[:n, body_idx]
    act_b = actual_flat[:n, body_idx]
    diff = act_b - exp_b
    per_frame = np.linalg.norm(diff, axis=1)

    flat_idx = body_idx.astype(np.int64)
    root_mask = np.isin(flat_idx, [58, 59, 60, 61, 62, 63])
    body_dof_mask = np.isin(flat_idx, list(range(22)) + list(range(29, 51)))
    priv_mask = flat_idx >= 64

    def _masked_mean_l2(mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        return float(np.linalg.norm(diff[:, mask], axis=1).mean())

    return {
        "n_frames": int(n),
        "body_input_l2_mean": float(per_frame.mean()),
        "body_input_l2_p99": float(np.percentile(per_frame, 99)),
        "body_dof_input_l2_mean": _masked_mean_l2(body_dof_mask),
        "root_grav_angvel_input_l2_mean": _masked_mean_l2(root_mask),
        "privileged_body_input_l2_mean": _masked_mean_l2(priv_mask),
    }


def _z_subspace_cosine_metrics(
    z_expert: np.ndarray,
    z_actual: np.ndarray,
    z_actual_smoothed: np.ndarray | None,
    z_body_dim: int,
) -> dict:
    """Cosine similarity between expert and actual z (full / body / hand)."""

    def _mean_cos(a: np.ndarray, b: np.ndarray) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        ab = (a[:n] * b[:n]).sum(axis=-1)
        na = np.linalg.norm(a[:n], axis=-1)
        nb = np.linalg.norm(b[:n], axis=-1)
        return float(np.mean(ab / (na * nb + 1e-12)))

    zb_e = z_expert[:, :z_body_dim]
    zh_e = z_expert[:, z_body_dim:]
    out = {
        "z_cos_expert_vs_actual": _mean_cos(z_expert, z_actual),
        "z_body_cos_expert_vs_actual": _mean_cos(zb_e, z_actual[:, :z_body_dim]),
        "z_hand_cos_expert_vs_actual": _mean_cos(zh_e, z_actual[:, z_body_dim:]),
    }
    if z_actual_smoothed is not None:
        out["z_cos_expert_vs_actual_smoothed"] = _mean_cos(z_expert, z_actual_smoothed)
        out["z_body_cos_expert_vs_actual_smoothed"] = _mean_cos(
            zb_e, z_actual_smoothed[:, :z_body_dim]
        )
        out["z_hand_cos_expert_vs_actual_smoothed"] = _mean_cos(
            zh_e, z_actual_smoothed[:, z_body_dim:]
        )
    return out


def _compute_z_actual(model, observation: dict, last_action_buf: "torch.Tensor") -> np.ndarray:
    """Re-encode current robot observation: single-frame B → project_z (no smoothing)."""
    return _compute_z_from_B(model, _compute_B_raw(model, _backward_obs_from_env(observation, last_action_buf)))


@dataclass
class InferenceOutputLayout:
    """Rollout artifacts grouped by source (plane/shape) and clip id."""

    root: Path

    @classmethod
    def create(cls, model_folder: Path) -> "InferenceOutputLayout":
        root = model_folder / "tracking_inference_split"
        root.mkdir(parents=True, exist_ok=True)
        summary = root / "summary"
        summary.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    @property
    def summary_dir(self) -> Path:
        return self.root / "summary"

    def clip_dir(self, clip_meta: dict | None, clip_id: str) -> Path:
        meta = clip_meta or {}
        plane = str(meta.get("plane") or "unknown")
        shape = str(meta.get("shape") or "unknown")
        d = self.root / "clips" / plane / shape / clip_id
        d.mkdir(parents=True, exist_ok=True)
        return d


def _clip_id_from_path(traj_path: Path) -> str:
    clip_id = traj_path.stem.removesuffix("_obs")
    return re.sub(r"[^\w\-.]+", "_", clip_id)


def _discover_traj_paths(traj_obs_dir: Path, traj_glob: str) -> list[Path]:
    paths = sorted(traj_obs_dir.glob(traj_glob))
    if not paths and traj_glob in ("*.npz", "*.pkl", "**/*_obs.npz"):
        for alt in ("**/*_obs.npz", "clips/**/*_obs.npz", "*.npz", "*.pkl"):
            if alt == traj_glob:
                continue
            paths = sorted(traj_obs_dir.glob(alt))
            if paths:
                print(f"提示: {traj_glob!r} 无匹配，已改用 {alt!r}，共 {len(paths)} 个文件。")
                break
    return paths


def _save_inference_metrics(
    clip_dir: Path,
    clip_id: str,
    n_steps: int,
    joint_pos_arr: np.ndarray,
    z_actual_arr: np.ndarray,
    ee_pred_arr: np.ndarray,
    ref_dof_arr: np.ndarray,
    ref_ee_arr: np.ndarray,
    z_expert_arr: np.ndarray,
    z_body_dim: int,
    clip_meta: dict | None = None,
    z_actual_smoothed_arr: np.ndarray | None = None,
    body_input_diag: dict | None = None,
) -> dict:
    """Compute inference metrics, save pkl/json/npz, return metrics dict."""
    n_cmp = min(n_steps, ref_dof_arr.shape[0] - 1, len(z_actual_arr))

    # joint_pos: [n_steps+1, 29]; compare after-step states [1:n_cmp+1] vs ref [1:n_cmp+1]
    jp_pred = joint_pos_arr[1 : n_cmp + 1]
    jp_ref   = ref_dof_arr[1 : n_cmp + 1]
    # z and EE: both collected before-acting [0:n_cmp]
    z_act  = z_actual_arr[:n_cmp]
    z_exp  = z_expert_arr[:n_cmp]
    ee_pred = ee_pred_arr[:n_cmp]
    ee_ref  = ref_ee_arr[:n_cmp]

    # ---- DOF errors ----
    all_dof_err  = np.linalg.norm(jp_pred - jp_ref,                                axis=-1)
    body_dof_err = np.linalg.norm(jp_pred[:, _BODY_DOF_SLICE] - jp_ref[:, _BODY_DOF_SLICE], axis=-1)
    hand_dof_err = np.linalg.norm(jp_pred[:, _HAND_DOF_SLICE] - jp_ref[:, _HAND_DOF_SLICE], axis=-1)

    # ---- EE local (heading-frame) error ----
    hand_ee_err = np.linalg.norm(ee_pred - ee_ref, axis=-1)   # metres

    # ---- z_hand jump rate (in z_actual) ----
    z_hand_act = z_act[:, z_body_dim:]
    if z_hand_act.shape[1] > 0 and len(z_hand_act) > 1:
        dz_h       = np.diff(z_hand_act, axis=0)
        hand_scale = np.sqrt(max(1, z_hand_act.shape[1]))
        spike_rate = float(np.mean(np.linalg.norm(dz_h, axis=1) / hand_scale > 0.5))
    else:
        spike_rate = 0.0

    metrics = {
        "n_frames":               int(n_cmp),
        "all_dof_error_norm":     float(np.mean(all_dof_err)),
        "body_dof_error_norm":    float(np.mean(body_dof_err)),
        "hand_dof_error_norm":    float(np.mean(hand_dof_err)),
        "hand_ee_local_err_m":    float(np.mean(hand_ee_err)),
        "hand_ee_local_err_mm":   float(np.mean(hand_ee_err) * 1000),
        "z_hand_spike_rate":      spike_rate,
        "z_body_dim":             int(z_body_dim),
        "z_hand_dim":             int(z_act.shape[1] - z_body_dim),
    }
    if clip_meta:
        metrics.update({k: v for k, v in clip_meta.items() if k not in metrics})

    z_cos = _z_subspace_cosine_metrics(
        z_exp, z_act, z_actual_smoothed_arr[:n_cmp] if z_actual_smoothed_arr is not None else None, z_body_dim
    )
    metrics.update(z_cos)
    if body_input_diag:
        metrics.update({k: v for k, v in body_input_diag.items() if v is not None and k not in metrics})

    print(f"\n=== Inference Metrics ({clip_id}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    z_hand_dim = int(z_act.shape[1] - z_body_dim)
    metrics["output_relpath"] = str(clip_dir.relative_to(clip_dir.parents[3]))
    _save_z_sequences(
        clip_dir=clip_dir,
        z_expert=z_exp,
        z_actual=z_act,
        z_body_dim=z_body_dim,
        z_hand_dim=z_hand_dim,
        clip_meta=clip_meta,
        z_actual_smoothed=z_actual_smoothed_arr[:n_cmp] if z_actual_smoothed_arr is not None else None,
    )

    if body_input_diag:
        body_path = clip_dir / "body_input.json"
        with open(body_path, "w", encoding="utf-8") as f:
            json.dump(body_input_diag, f, indent=2)
        print(f"Saved body input diag → {body_path}")

    analysis = {
        "clip_id":        clip_id,
        "metrics":        metrics,
        "z_expert":       z_exp,
        "z_actual":       z_act,
        "z_actual_smoothed": (
            z_actual_smoothed_arr[:n_cmp] if z_actual_smoothed_arr is not None else None
        ),
        "body_input_diag": body_input_diag,
        "joint_pos_pred": jp_pred,
        "joint_pos_ref":  jp_ref,
        "ee_local_pred":  ee_pred,
        "ee_local_ref":   ee_ref,
    }
    analysis_path = clip_dir / "analysis.pkl"
    joblib.dump(analysis, analysis_path)
    print(f"Saved analysis data → {analysis_path}")

    metrics_path = clip_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics      → {metrics_path}")
    return metrics


def _save_z_sequences(
    *,
    clip_dir: Path,
    z_expert: np.ndarray,
    z_actual: np.ndarray,
    z_body_dim: int,
    z_hand_dim: int,
    clip_meta: dict | None,
    z_actual_smoothed: np.ndarray | None = None,
) -> None:
    """Save expert / actual z as NPZ for sphere visualization."""
    meta = clip_meta or {}
    common = {
        "z_body_dim": np.int32(z_body_dim),
        "z_hand_dim": np.int32(z_hand_dim),
    }
    np.savez(
        clip_dir / "z_expert.npz",
        z=z_expert.astype(np.float32),
        **common,
    )
    np.savez(
        clip_dir / "z_actual.npz",
        z=z_actual.astype(np.float32),
        **common,
    )
    if z_actual_smoothed is not None:
        np.savez(
            clip_dir / "z_actual_smoothed.npz",
            z=z_actual_smoothed.astype(np.float32),
            **common,
        )
    compare_kw = {
        "z_expert": z_expert.astype(np.float32),
        "z_actual": z_actual.astype(np.float32),
        "plane": np.array(str(meta.get("plane", ""))),
        "shape": np.array(str(meta.get("shape", ""))),
        **common,
    }
    if z_actual_smoothed is not None:
        compare_kw["z_actual_smoothed"] = z_actual_smoothed.astype(np.float32)
    np.savez(clip_dir / "z_compare.npz", **compare_kw)
    saved = "z_expert.npz, z_actual.npz"
    if z_actual_smoothed is not None:
        saved += ", z_actual_smoothed.npz"
    saved += ", z_compare.npz"
    print(f"Saved z sequences  → {clip_dir.name}/{saved}")


def _npz_scalar_str(path: Path, key: str) -> str | None:
    with np.load(path, allow_pickle=True) as z:
        if key not in z.files:
            return None
        arr = np.asarray(z[key]).reshape(-1)
        if arr.size == 0:
            return None
        return str(arr[0])


def parse_clip_meta(path: Path) -> dict[str, str]:
    """Parse shape / plane from NPZ metadata or filename."""
    shape = _npz_scalar_str(path, "shape_name")
    plane = _npz_scalar_str(path, "shape_plane")
    stem = path.stem.removesuffix("_obs")
    if not shape:
        m = re.match(r"^([a-z_]+)_P(xy|xz|yz)_", stem, re.I)
        if m:
            shape = m.group(1).lower()
    if not plane:
        m = re.search(r"_P(xy|xz|yz)_", stem, re.I)
        if m:
            plane = m.group(1).lower()
        elif "_xy_" in stem.lower() or stem.lower().endswith("_xy"):
            plane = "xy"
        elif "_xz_" in stem.lower():
            plane = "xz"
        elif "_yz_" in stem.lower():
            plane = "yz"
    return {
        "shape": (shape or "unknown").lower(),
        "plane": (plane or "unknown").lower(),
        "traj_file": path.name,
    }


def _load_prepare_quality_map(traj_obs_dir: Path) -> dict[str, dict]:
    for candidate in (
        traj_obs_dir / "reports" / "prepare_report.json",
        traj_obs_dir / "prepare_report.json",
    ):
        if candidate.is_file():
            report_path = candidate
            break
    else:
        return {}
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    out: dict[str, dict] = {}
    for clip in report.get("clips", []):
        out_npz = clip.get("output_npz") or ""
        name = Path(out_npz).name
        if not name:
            continue
        cont = clip.get("continuity") or {}
        out[name] = {
            "jump_flagged": bool(clip.get("jump_flagged", False)),
            "max_arm_delta_rad": float(cont.get("max_arm_delta_rad", 999.0)),
        }
    return out


def select_one_per_shape_plane(
    paths: list[Path],
    quality_map: dict[str, dict],
) -> list[Path]:
    """Pick one clip per (plane, shape); prefer smooth (non-jump) clips."""
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in paths:
        meta = parse_clip_meta(p)
        groups[(meta["plane"], meta["shape"])].append(p)

    selected: list[Path] = []
    for key in sorted(groups.keys()):
        plist = groups[key]
        plist.sort(
            key=lambda p: (
                quality_map.get(p.name, {}).get("jump_flagged", True),
                quality_map.get(p.name, {}).get("max_arm_delta_rad", 999.0),
                p.name,
            )
        )
        selected.append(plist[0])
        print(f"  select {key[0]}/{key[1]}: {plist[0].name} (from {len(plist)} candidates)")
    return selected


def _save_ee_traj_plots(
    expert_ee_base: np.ndarray,
    policy_ee_base: np.ndarray,
    clip_dir: Path,
    title_suffix: str = "",
) -> None:
    """Save 3D and XY/XZ/YZ pelvis-frame EE comparison plots (English labels only)."""
    n_cmp = min(len(expert_ee_base), len(policy_ee_base))
    if n_cmp < 2:
        print(f"WARNING: skip EE plots for {clip_dir.name}: too few frames ({n_cmp})")
        return
    exp = expert_ee_base[:n_cmp]
    pol = policy_ee_base[:n_cmp]
    dev = np.linalg.norm(exp - pol, axis=1)
    mean_mm = float(dev.mean() * 1000)
    max_mm = float(dev.max() * 1000)
    suffix = f" | {title_suffix}" if title_suffix else ""
    subtitle = f"mean deviation={mean_mm:.2f} mm  max={max_mm:.2f} mm"

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        exp[:, 0], exp[:, 1], exp[:, 2],
        color="#1D7FD4", lw=1.8, linestyle="--", alpha=0.9,
        label=f"Expert EE (T={n_cmp})",
    )
    ax.plot(
        pol[:, 0], pol[:, 1], pol[:, 2],
        color="#E63946", lw=2.0, linestyle="-", alpha=0.9,
        label=f"Policy EE (T={n_cmp})",
    )
    ax.scatter(*exp[0], marker="*", s=120, color="#1D7FD4", zorder=8)
    ax.scatter(*pol[0], marker="*", s=120, color="#E63946", zorder=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Right-hand EE trajectory (pelvis frame){suffix}\n{subtitle}", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(clip_dir / "ee_traj_3d.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    plane_cfg = [
        ("xy", 0, 1, "X (m)", "Y (m)"),
        ("xz", 0, 2, "X (m)", "Z (m)"),
        ("yz", 1, 2, "Y (m)", "Z (m)"),
    ]
    for pname, i, j, xlabel, ylabel in plane_cfg:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(exp[:, i], exp[:, j], color="#1D7FD4", lw=1.8, linestyle="--", alpha=0.9, label="Expert EE")
        ax.plot(pol[:, i], pol[:, j], color="#E63946", lw=2.0, linestyle="-", alpha=0.9, label="Policy EE")
        ax.scatter(exp[0, i], exp[0, j], marker="*", s=200, color="#1D7FD4", zorder=8)
        ax.scatter(pol[0, i], pol[0, j], marker="*", s=200, color="#E63946", zorder=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)
        ax.set_title(
            f"Right-hand EE — {pname.upper()} projection{suffix}\n{subtitle}",
            fontsize=10,
        )
        ax.legend(fontsize=9, loc="best")
        fig.savefig(clip_dir / f"ee_traj_plane_{pname}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved EE traj plots  → {clip_dir.name}/ee_traj_3d.png + plane XY/XZ/YZ")


def _save_aggregate_summaries(
    summary_dir: Path,
    per_clip_metrics: list[dict],
) -> None:
    """Aggregate per-clip metrics by (plane, shape) and overall."""
    if not per_clip_metrics:
        return

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in per_clip_metrics:
        groups[(str(m.get("plane", "unknown")), str(m.get("shape", "unknown")))].append(m)

    agg_rows: list[dict] = []
    metric_keys = [
        "hand_ee_local_err_mm",
        "hand_dof_error_norm",
        "body_dof_error_norm",
        "all_dof_error_norm",
        "z_hand_spike_rate",
    ]
    for (plane, shape), items in sorted(groups.items()):
        row: dict = {"plane": plane, "shape": shape, "n_clips": len(items)}
        for k in metric_keys:
            vals = [float(x[k]) for x in items if k in x]
            row[f"mean_{k}"] = float(np.mean(vals)) if vals else None
        row["clip_files"] = [x.get("traj_file", "") for x in items]
        row["output_relpaths"] = [x.get("output_relpath", "") for x in items]
        agg_rows.append(row)

    json_path = summary_dir / "summary_by_plane_shape.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg_rows, f, indent=2)
    print(f"Saved aggregate     → {json_path}")

    csv_path = summary_dir / "summary_by_plane_shape.csv"
    if agg_rows:
        fieldnames = list(agg_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in agg_rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"Saved aggregate CSV → {csv_path}")

    overall: dict = {"n_clips": len(per_clip_metrics)}
    for k in metric_keys:
        vals = [float(x[k]) for x in per_clip_metrics if k in x]
        overall[f"mean_{k}"] = float(np.mean(vals)) if vals else None
    overall_path = summary_dir / "summary_overall.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(f"Saved overall       → {overall_path}")


def _build_fk_context():
    """Lightweight MuJoCo FK for pelvis-frame right-wrist positions."""
    from humanoidverse.utils.g1_env_config import G1EnvConfig

    fk_env, _ = G1EnvConfig(render_height=16, render_width=16).build(num_envs=1)
    fk_g1 = IsaacRendererWithMuJoco._inner_g1_env(fk_env)
    fk_model = fk_g1._mj_model
    fk_data = mujoco.MjData(fk_model)
    ee_id = int(mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_BODY, _EE_BODY))
    return fk_model, fk_data, ee_id


# ---------- end metric helpers ----------


def _normalize_traj_key(name: str) -> str:
    k = name.strip().lower().replace(" ", "_")
    if k == "priviledged_state":
        k = "privileged_state"
    return k


def _as_traj_array(v: object) -> np.ndarray | None:
    """Convert to (T, D) float32 trajectory array; skip metadata / non-numeric."""
    arr = np.asarray(v)
    if arr.dtype.kind in ("U", "S", "O", "M", "m") or not np.issubdtype(arr.dtype, np.number):
        return None
    try:
        arr = np.asarray(arr, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if arr.ndim != 2:
        return None
    return arr


def _stack_traj_obs(raw: object) -> dict[str, np.ndarray]:
    """统一为 {key: (T, D) float32 numpy}；NPZ 内 timestamps/fps/字符串元数据会被忽略。"""
    if isinstance(raw, list):
        if len(raw) == 0:
            raise ValueError("轨迹 list 为空")
        keys = set()
        for row in raw:
            if not isinstance(row, dict):
                raise TypeError("轨迹 list 的每个元素应为 dict")
            keys.update(row.keys())
        out: dict[str, list[np.ndarray]] = { _normalize_traj_key(k): [] for k in keys }
        for row in raw:
            for k, v in row.items():
                nk = _normalize_traj_key(k)
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                out[nk].append(arr)
        return {k: np.stack(vs, axis=0) for k, vs in out.items()}

    if not isinstance(raw, dict):
        raise TypeError(f"不支持的轨迹类型: {type(raw)}")

    fixed: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        nk = _normalize_traj_key(k)
        arr = _as_traj_array(v)
        if arr is None:
            continue
        fixed[nk] = arr
    return fixed


def load_traj_obs_file(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        raw = joblib.load(path)
    elif path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        raw = {k: z[k] for k in z.files}
        z.close()
    else:
        raise ValueError(f"仅支持 .pkl / .npz: {path}")
    return _stack_traj_obs(raw)


def traj_to_backward_batch(traj: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    for k in _BACKWARD_KEYS:
        if k not in traj:
            raise KeyError(f"轨迹缺键 {k!r}（或拼写 priviledged_state）；现有: {sorted(traj.keys())}")
        if traj[k].ndim != 2:
            raise ValueError(f"{k} 应为 (T, D)，got {traj[k].shape}")
    return {
        k: torch.from_numpy(traj[k]).to(device=device, dtype=torch.float32)
        for k in _BACKWARD_KEYS
    }


def _privileged_dim_from_model(model) -> int:
    sp = model.obs_space
    if getattr(sp, "spaces", None) is None:
        raise TypeError(f"模型 obs_space 非 Dict（{type(sp)}）；无法获知 privileged_state 期望维数")
    priv = sp.spaces.get("privileged_state")
    if priv is None:
        keys = getattr(sp, "spaces", {}).keys()
        raise KeyError(f"obs_space 中没有 privileged_state，现有键: {list(keys)}")
    return int(priv.shape[0])


def _drop_last_body_max_local_slices(priv: np.ndarray, *, root_height_obs: bool) -> np.ndarray:
    """移除 ``max_local_self`` 中末尾刚体的一块（通常为 ``head_link`` 扩展），与 ``legged_robot_motions.compute_humanoid_observations_max`` 拼接顺序一致。"""
    assert priv.ndim == 2
    idx = int(root_height_obs)
    slices: list[np.ndarray] = []
    seg_dims = [(90, 3), (186, 6), (93, 3), (93, 3)]  # (length, chop from end)

    offset = idx
    for length, chop in seg_dims:
        seg = priv[:, offset : offset + length]
        slices.append(seg[:, :-chop])
        offset += length
    tail = idx + sum(d for d, _ in seg_dims)
    assert offset == tail and tail == priv.shape[-1]

    heads = [priv[:, :1]] if root_height_obs else []
    merged = np.concatenate(heads + slices, axis=-1)
    assert merged.ndim == 2
    return np.ascontiguousarray(merged.astype(np.float32))


def align_traj_privileged_for_model(traj: dict[str, np.ndarray], model) -> dict[str, np.ndarray]:
    """
    将 ``privileged_state`` 维数对齐到 checkpoint 的 BatchNorm / B 网络。

    支持：463→448、462→447（即多 1 个刚体 × 局部 pos/rot/vel/ang_vel，与 yaml 里 ``nums_extend_bodies: 1`` 一致）。
    """
    exp = _privileged_dim_from_model(model)
    p = traj["privileged_state"]
    d = int(p.shape[-1])
    if d == exp:
        return traj

    trim_from: dict[tuple[int, int], bool] = {
        (463, 448): True,
        (462, 447): False,
    }
    key = (d, exp)
    if key not in trim_from:
        raise ValueError(
            f"privileged_state 维数 {d} 与 checkpoint 期望 {exp} 不符，且无法自动转换。"
            f" 若为刚体数不一致，请用与训练相同 body 配置的轨迹，或另行重算 max_local_self。"
        )
    new_p = _drop_last_body_max_local_slices(p, root_height_obs=trim_from[key])
    if new_p.shape[-1] != exp:
        raise RuntimeError(f"对齐后 privileged 维数为 {new_p.shape[-1]}，仍不等于模型期望 {exp}")

    out = dict(traj)
    out["privileged_state"] = new_p
    print(
        f"提示: privileged_state {d} → {exp}（按末端刚体块裁剪，root_height_obs={trim_from[key]}），"
        "与仅 30 个实体刚体的 checkpoint 对齐。"
    )
    return out


def _base_init_root_pose_7(env) -> np.ndarray:
    """``LeggedRobotBase.base_init_state`` 多为 ``(13,)``，部分后端可能为 ``(N,13)``；取 pos+quat (7,)。"""
    bis = env.base_init_state.detach().float()
    if bis.ndim == 1:
        if bis.shape[0] < 7:
            raise ValueError(f"base_init_state 维数 {bis.shape[0]} < 7")
        root7 = bis[:7]
    elif bis.ndim >= 2:
        if bis.shape[-1] < 7:
            raise ValueError(f"base_init_state 最后一维 {bis.shape[-1]} < 7")
        root7 = bis[0, :7]
    else:
        raise ValueError(f"无法解析 base_init_state，ndim={bis.ndim}")
    return root7.cpu().numpy().astype(np.float32)


def _build_expert_qpos(
    traj: dict[str, np.ndarray],
    *,
    env,
    wrapped_env,
    device: torch.device,
    num_envs: int,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    返回 expert_qpos (T,36)、ref_root (13,)、dof_init for reset。

    ``mujoco_qpos`` 若存在：每行 36 = 根位置(3)+根四元数 wxyz(4)+29 关节；与 ``IsaacRendererWithMuJoco`` / ``robot_root_states`` 拼 qpos 的约定一致。

    否则：各帧关节 = ``state[:, :29] + default_dof_pos``，根固定在 ``base_init_state`` 的首帧姿态（仅便于渲染对齐，不等价于采集真值根轨迹）。
    """
    T = traj["state"].shape[0]
    st = traj["state"]
    if st.shape[1] < 58:
        raise ValueError(
            f"state 第二维至少 58（29 相对关节 + 29 关节速度 + …），当前 {st.shape[1]}"
        )
    ddp = env.default_dof_pos.detach().float()
    if ddp.ndim == 1:
        default_dof = ddp.cpu().numpy()
    else:
        default_dof = ddp[0].cpu().numpy()

    qpos_key = "mujoco_qpos" if "mujoco_qpos" in traj else ("qpos" if "qpos" in traj else None)
    if qpos_key is not None:
        q = np.asarray(traj[qpos_key], dtype=np.float64)
        if q.shape != (T, 36):
            raise ValueError(f"{qpos_key} 期望 (T,36) T={T}, got {q.shape}")
        expert_qpos = q.astype(np.float32)
        root_pos = torch.from_numpy(expert_qpos[0, :3]).float().to(device)
        quat_wxyz = torch.from_numpy(expert_qpos[0, 3:7]).float().to(device)
        lin = torch.zeros(3, device=device, dtype=torch.float32)
        ang = torch.zeros(3, device=device, dtype=torch.float32)
        ref_root = torch.cat([root_pos, quat_wxyz, lin, ang], dim=0)
    else:
        dof_rel = st[:, :29].astype(np.float64)
        dof_abs = dof_rel + default_dof[None, :]
        root_template = _base_init_root_pose_7(env)
        root_pos = root_template[:3]
        root_quat = root_template[3:7]
        expert_qpos = np.zeros((T, 36), dtype=np.float32)
        expert_qpos[:, :3] = root_pos
        expert_qpos[:, 3:7] = root_quat
        expert_qpos[:, 7:] = dof_abs.astype(np.float32)
        ref_root = torch.cat(
            [
                torch.from_numpy(root_pos).float().to(device),
                torch.from_numpy(root_quat).float().to(device),
                torch.zeros(3, device=device),
                torch.zeros(3, device=device),
            ],
            dim=0,
        )

    dof_vel0 = st[0, 29:58].astype(np.float32)
    dof_template = wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0]
    dof_init = torch.zeros_like(dof_template)
    dof_init[..., 0] = torch.from_numpy(expert_qpos[0, 7:]).float().to(dof_template.device)
    dof_init[..., 1] = torch.from_numpy(dof_vel0).float().to(dof_template.device)

    return expert_qpos, ref_root, dof_init


def _run_single_traj_rollout(
    *,
    traj_np: dict[str, np.ndarray],
    z: torch.Tensor,
    clip_id: str,
    clip_dir: Path,
    model,
    wrapped_env,
    env,
    dev: torch.device,
    num_envs: int,
    _wrist_slice: slice,
    _z_body_dim: int,
    episode_len: int | None,
    save_mp4: bool,
    fk_model,
    fk_data,
    ee_id: int,
    clip_meta: dict | None = None,
    z_window: int = 1,
    z_ema_alpha: float | None = None,
) -> dict:
    """Reset env to traj init, rollout tracking, save metrics for one trajectory."""
    expert_qpos, ref_root, dof_init_state = _build_expert_qpos(
        traj_np, env=env, wrapped_env=wrapped_env, device=dev, num_envs=num_envs
    )

    sim_dev = wrapped_env._env.device
    env.set_is_evaluating(0)
    wrapped_env.reset(to_numpy=False)

    env_ids = torch.arange(num_envs, dtype=torch.long, device=sim_dev)
    target_states = {
        "dof_states": dof_init_state,
        "root_states": torch.stack([ref_root.clone().to(sim_dev) for _ in range(num_envs)]),
    }
    wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
    wrapped_env.step(
        torch.zeros((num_envs, wrapped_env.action_space.shape[-1]), dtype=torch.float32, device=sim_dev),
        to_numpy=False,
    )
    observation = wrapped_env._get_g1env_observation(to_numpy=False)

    Tz = z.shape[0]
    traj_T = traj_np["state"].shape[0]
    n_steps = min(Tz, traj_T - 1, expert_qpos.shape[0] - 1)
    if episode_len is not None:
        n_steps = min(n_steps, episode_len)
    print(f"Rollout [{clip_id}]: {n_steps} steps (z={Tz}, traj_T={traj_T})")

    joint_pos = [wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy()]
    _z_actual_list: list[np.ndarray] = []
    _B_actual_list: list[np.ndarray] = []
    _actual_flat_list: list[np.ndarray] = []
    _ee_local_pred_list: list[np.ndarray] = []
    _policy_ee_base_list: list[np.ndarray] = []
    _action_dim = wrapped_env.action_space.shape[-1]
    _last_act_buf = torch.zeros((num_envs, _action_dim), dtype=torch.float32, device=sim_dev)
    body_idx = _body_idx_from_model(model)

    expert_video = None
    frames: list[np.ndarray] = []
    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
        expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + n_steps])
        frames = [rgb_renderer.render(wrapped_env._env, 0)[0]]

    expert_ee_base = _fk_ee_positions_base(
        fk_model,
        fk_data,
        ee_id,
        expert_qpos[1 : 1 + n_steps, :7],
        expert_qpos[1 : 1 + n_steps, 7:].astype(np.float64),
    )

    for i in range(n_steps):
        print(f"  [{clip_id}] step {i + 1}/{n_steps}")
        bmap_obs = _backward_obs_from_env(observation, _last_act_buf)
        with torch.no_grad():
            B_raw = _compute_B_raw(model, bmap_obs)
            _B_actual_list.append(B_raw)
            _z_actual_list.append(_compute_z_from_B(model, B_raw))
        st_np = observation["state"][0].detach().cpu().numpy()
        priv_np = observation["privileged_state"][0].detach().cpu().numpy()
        _actual_flat_list.append(_flat_backward_numpy(st_np, priv_np))
        _ee_local_pred_list.append(
            observation["privileged_state"][0, _wrist_slice].detach().cpu().numpy()
        )
        _rs = wrapped_env._env.simulator.robot_root_states[0].float().detach().cpu().numpy()
        _d = wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0, :, 0].float().detach().cpu().numpy()
        _policy_ee_base_list.append(_fk_one_step_base(fk_model, fk_data, ee_id, _rs, _d))
        action = model.act(observation, z[i % len(z)].unsqueeze(0).expand(num_envs, -1), mean=True)
        _last_act_buf = action.detach()
        observation, _r, _t, _trunc, _info = wrapped_env.step(action, to_numpy=False)
        joint_pos.append(wrapped_env._env.simulator.dof_state[..., 0].clone().cpu().numpy())
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])

    _default_dof = wrapped_env._env.default_dof_pos[0].cpu().numpy()
    _ref_dof_arr = traj_np["state"][:, :29] + _default_dof
    _ref_ee_arr = traj_np["privileged_state"][:, _wrist_slice].astype(np.float32)
    z_expert_np = z.detach().cpu().numpy()

    B_actual_arr = np.stack(_B_actual_list, axis=0)
    z_actual_smoothed = _smooth_and_project_B(
        model, B_actual_arr, z_window=z_window, z_ema_alpha=z_ema_alpha
    )

    expert_flat = np.stack(
        [
            _flat_backward_numpy(traj_np["state"][t], traj_np["privileged_state"][t])
            for t in range(1, 1 + n_steps)
        ],
        axis=0,
    )
    actual_flat = np.stack(_actual_flat_list, axis=0)
    body_input_diag = _compute_body_input_diagnostics(expert_flat, actual_flat, body_idx)

    metrics = _save_inference_metrics(
        clip_dir=clip_dir,
        clip_id=clip_id,
        n_steps=n_steps,
        joint_pos_arr=np.stack(joint_pos, axis=0).squeeze(1),
        z_actual_arr=np.stack(_z_actual_list),
        ee_pred_arr=np.stack(_ee_local_pred_list),
        ref_dof_arr=_ref_dof_arr,
        ref_ee_arr=_ref_ee_arr,
        z_expert_arr=z_expert_np,
        z_body_dim=_z_body_dim,
        clip_meta=clip_meta,
        z_actual_smoothed_arr=z_actual_smoothed,
        body_input_diag=body_input_diag,
    )

    policy_ee_base = np.asarray(_policy_ee_base_list, dtype=np.float64)
    title_suffix = ""
    if clip_meta:
        title_suffix = f"{clip_meta.get('shape', '')} / {clip_meta.get('plane', '')}"
    _save_ee_traj_plots(
        expert_ee_base[: len(policy_ee_base)],
        policy_ee_base,
        clip_dir,
        title_suffix=title_suffix,
    )

    if save_mp4 and expert_video is not None:
        new_frames = [np.concatenate([a, b], axis=1) for a, b in zip(expert_video, frames)]
        video_path = clip_dir / "tracking.mp4"
        media.write_video(str(video_path), new_frames, fps=50)
        print(f"Saved video: {video_path}")
    return metrics


def main(
    model_folder: Path,
    traj_obs_dir: Path,
    traj_glob: str = "*.npz",
    data_path: Path | None = None,
    headless: bool = True,
    device: str = "cuda",
    simulator: str = "isaacsim",
    save_mp4: bool = False,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    episode_len: int | None = None,
    z_window: int = 1,
    z_ema_alpha: float | None = None,
    repair_privileged: bool = True,
    recording_dt: float = 1.0 / 30.0,
    one_per_shape_plane: bool = False,
    max_clips: int | None = None,
) -> None:
    """z smoothing options (anti-jump for OOD target trajectories):

    z_window    : forward-looking window mean over B projections (1 = off, the
                  historical behaviour of this script; training-time tracking
                  uses seq_length=8).
    z_ema_alpha : causal low-pass z[t] <- a*z[t] + (1-a)*z[t-1]; None/1.0 = off.
                  Start with z_window=8, z_ema_alpha=0.6 when the right hand jumps.
                  The same window/EMA is applied to Isaac B projections to produce
                  ``z_actual_smoothed_*`` for fair comparison with ``z_expert``.
    repair_privileged : recompute privileged_state from state via MuJoCo FK (fixes
                  ~30× velocity bug in recording pipeline). Default True.
    recording_dt: frame interval used when reconstructing root linear velocity from state.
    one_per_shape_plane : keep one clip per (plane, shape) group (~30 for V3 benchmark).
    max_clips   : debug cap on number of trajectories to process.
    """
    model_folder = Path(model_folder)
    traj_obs_dir = Path(traj_obs_dir)

    model = load_model_from_checkpoint_dir(str(model_folder / "checkpoint"), device=device)
    model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    dev = next(model.parameters()).device

    with open(model_folder / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    _wrist_slice = _wrist_local_pos_slice(config)
    try:
        _z_body_dim: int = model.cfg.archi.z_body_dim
    except AttributeError:
        _z_body_dim = model.cfg.archi.total_z_dim  # no hand split

    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"

    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    export_root = model_folder / "exported"
    export_root.mkdir(parents=True, exist_ok=True)
    z_export_dim = model.cfg.archi.total_z_dim
    export_meta_policy_as_onnx(
        model,
        export_root,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + z_export_dim)},
        z_dim=z_export_dim,
        history=("history_actor" in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"Exported model to {export_root}/{model_name}.onnx")

    def tracking_inference(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + max(1, z_window), z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        if z_ema_alpha is not None and z_ema_alpha < 1.0:
            for step in range(1, z.shape[0]):
                z[step] = z_ema_alpha * z[step] + (1.0 - z_ema_alpha) * z[step - 1]
        return model.project_z(z)

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    print("=" * 80)
    print(env.config.simulator)
    print("-" * 80)

    output_layout = InferenceOutputLayout.create(model_folder)
    print(f"Output root: {output_layout.root}")

    paths = _discover_traj_paths(traj_obs_dir, traj_glob)
    if not paths:
        raise FileNotFoundError(
            f"在 {traj_obs_dir} 下未找到匹配 {traj_glob!r} 的文件"
            f"（若轨迹为另一种格式，请显式设置 --traj-glob，例如 '**/*_obs.npz' 或 '*.pkl'）"
        )

    quality_map = _load_prepare_quality_map(traj_obs_dir)
    if one_per_shape_plane:
        print(f"Selecting one clip per (plane, shape) from {len(paths)} files …")
        paths = select_one_per_shape_plane(paths, quality_map)
    if max_clips is not None:
        paths = paths[:max_clips]
    print(f"Will process {len(paths)} trajectory file(s).")

    fk_model, fk_data, ee_id = _build_fk_context()
    print(f"FK EE body '{_EE_BODY}' id = {ee_id}")

    traj_items: list[tuple[str, Path, dict[str, np.ndarray], torch.Tensor, dict]] = []

    for traj_path in paths:
        print(f"\nLoad trajectory: {traj_path}")
        clip_meta = parse_clip_meta(traj_path)
        clip_id = _clip_id_from_path(traj_path)
        clip_dir = output_layout.clip_dir(clip_meta, clip_id)
        traj_np = load_traj_obs_file(traj_path)
        if repair_privileged:
            print("  Repairing privileged_state from state (MuJoCo FK) …")
            traj_np = repair_recording_traj(traj_np, dt=recording_dt, verbose=True)
        traj_np = align_traj_privileged_for_model(traj_np, model)
        obs_full = traj_to_backward_batch(traj_np, dev)
        z = tracking_inference(tree_map(lambda x: x[1:], obs_full))
        zs_path = clip_dir / "zs_expert.pkl"
        joblib.dump(z.detach().cpu().numpy(), zs_path)
        print(f"Saved {zs_path}  (z steps={z.shape[0]})")
        traj_items.append((clip_id, clip_dir, traj_np, z, clip_meta))

    if not traj_items:
        raise RuntimeError("No trajectories processed")

    per_clip_metrics: list[dict] = []
    for clip_id, clip_dir, traj_np, z, clip_meta in traj_items:
        metrics = _run_single_traj_rollout(
            traj_np=traj_np,
            z=z,
            clip_id=clip_id,
            clip_dir=clip_dir,
            model=model,
            wrapped_env=wrapped_env,
            env=env,
            dev=dev,
            num_envs=num_envs,
            _wrist_slice=_wrist_slice,
            _z_body_dim=_z_body_dim,
            episode_len=episode_len,
            save_mp4=save_mp4,
            fk_model=fk_model,
            fk_data=fk_data,
            ee_id=ee_id,
            clip_meta=clip_meta,
            z_window=z_window,
            z_ema_alpha=z_ema_alpha,
        )
        per_clip_metrics.append(metrics)

    _save_aggregate_summaries(output_layout.summary_dir, per_clip_metrics)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
