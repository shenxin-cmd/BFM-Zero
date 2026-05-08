"""
Isaac 上 Split-z：**单一 MuJoCo 关节目标** → ``goal_inference`` 得 z → 策略与环境闭环 rollout，
录 mp4 并保存 rollout 过程的 z trace（``.npz``）。

Goal 观测的构造对齐仓库根目录 ``env.py`` 中 ``MuJoCoBFMZeroEnv._create_observation_backward`` /
``get_privileged_state``（单位重力投影 ``[0,0,-1]``、静态姿态下速度与 ``last_action`` 置零）。

关节目标：在本文件顶部 ``GOAL_POSE_JOINT_ABS_RAD`` 中为 **铰链角绝对值** [rad]，未列出者使用本文件内的
``DEFAULT_ABS_DOF_FOR_MISSING_GOAL_KEYS_RAD``（可自行修改以匹配所用 checkpoint / 机器人配置）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mediapy as media
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation
from tqdm import tqdm

import humanoidverse
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.g1_env_config import get_g1_robot_xml_root
from humanoidverse.utils.torch_utils import calc_heading_quat_inv, quat_mul, quat_rotate, quat_to_tan_norm

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 本脚本内联 MJCF 默认文件名、关节顺序与缺省绝对角。
# -----------------------------------------------------------------------------
DEFAULT_MUJOCO_SCENE = "scene_29dof_freebase_mujoco.xml"

# 与 g1_29dof.yaml 中 robot.dof_names 顺序一致。
G1_29DOF_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# backward 观测里 dof_pos_rel 的减数、以及 GOAL_POSE 未写关节时的绝对角占位；与训练中性站姿对齐。
DEFAULT_ABS_DOF_FOR_MISSING_GOAL_KEYS_RAD: dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": -0.09,
    "left_shoulder_roll_joint": 0.41,
    "left_shoulder_yaw_joint": 0.38,
    "left_elbow_joint": -0.05,
    "left_wrist_roll_joint": 0.03,
    "left_wrist_pitch_joint": 0.38,
    "left_wrist_yaw_joint": 0.38,
    "right_shoulder_pitch_joint": -0.09,
    "right_shoulder_roll_joint": -0.25,
    "right_shoulder_yaw_joint": 0.38,
    "right_elbow_joint": -0.05,
    "right_wrist_roll_joint": 0.03,
    "right_wrist_pitch_joint": 0.38,
    "right_wrist_yaw_joint": 0.38,
}


def _training_default_dof_vec() -> np.ndarray:
    return np.array([DEFAULT_ABS_DOF_FOR_MISSING_GOAL_KEYS_RAD[j] for j in G1_29DOF_JOINT_ORDER], dtype=np.float64)


# -----------------------------------------------------------------------------
# 目标关节绝对角 [rad]；键名为 g1 29dof 铰链 joint 名，未列出者用 DEFAULT_ABS_DOF_FOR_MISSING_GOAL_KEYS_RAD。
# -----------------------------------------------------------------------------
GOAL_POSE_JOINT_ABS_RAD: dict[str, float] = {}

# Isaac eval 装载的参考 motion 数据集起始 id（与 goal z 无关，仅初始化 tracking 环境）。
ROLLOUT_REF_MOTION_ID = 0

# -----------------------------------------------------------------------------
# Goal 观测：按 env.py 逻辑从 MuJoCo 状态拼装（与 inference_tutorial + env.py 一致思路）
# -----------------------------------------------------------------------------


class _PrivilegedPrevHolder:
    __slots__ = ("body_pos_prev", "body_quat_prev")

    def __init__(self) -> None:
        self.body_pos_prev: np.ndarray | None = None
        self.body_quat_prev: np.ndarray | None = None


def _calc_angular_velocity(quat_cur: np.ndarray, quat_prev: np.ndarray, dt: float) -> np.ndarray:
    """env.py:calc_angular_velocity；四元数为 [w,x,y,z]."""
    from scipy.spatial.transform import Rotation as R

    quat_cur = np.asarray(quat_cur)
    quat_prev = np.asarray(quat_prev)
    orig_ndim = quat_cur.ndim
    if quat_cur.ndim == 1:
        quat_cur = quat_cur[None, :]
        quat_prev = quat_prev[None, :]

    qc_xyzw = np.stack([quat_cur[:, 1], quat_cur[:, 2], quat_cur[:, 3], quat_cur[:, 0]], axis=-1)
    qp_xyzw = np.stack([quat_prev[:, 1], quat_prev[:, 2], quat_prev[:, 3], quat_prev[:, 0]], axis=-1)
    rot_cur = R.from_quat(qc_xyzw)
    rot_prev = R.from_quat(qp_xyzw)
    delta_rot = rot_prev.inv() * rot_cur
    angular_velocity = delta_rot.as_rotvec() / max(dt, 1e-9)
    out = angular_velocity[0] if orig_ndim == 1 else angular_velocity
    return out.astype(np.float64)


def _absolute_dof_array(joint_abs: dict[str, float]) -> np.ndarray:
    """绝对铰链角 [rad]；``joint_abs`` 未列出项用 ``DEFAULT_ABS_DOF_FOR_MISSING_GOAL_KEYS_RAD``。"""
    ref = DEFAULT_ABS_DOF_FOR_MISSING_GOAL_KEYS_RAD
    return np.array([float(joint_abs[n]) if n in joint_abs else float(ref[n]) for n in G1_29DOF_JOINT_ORDER], dtype=np.float64)


def _set_mujoco_pose_from_abs(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    dof_absolute: np.ndarray,
) -> None:
    kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid < 0:
        raise RuntimeError("MJCF 中需要名为 'stand' 的 keyframe（与 inference_tutorial 一致）。")
    mj_data.qpos[:] = mj_model.key_qpos[kid]
    mj_data.qvel[:] = 0.0
    mj_type_hinge = int(mujoco.mjtJoint.mjJNT_HINGE)

    assert dof_absolute.shape[0] == 29
    for i, jn in enumerate(G1_29DOF_JOINT_ORDER):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise KeyError(f"Joint {jn!r} not in MJCF.")
        if mj_model.jnt_type[jid] != mj_type_hinge:
            raise TypeError(f"Joint {jn!r}: 需要 mjJNT_HINGE。")
        qadr = int(mj_model.jnt_qposadr[jid])
        mj_data.qpos[qadr] = float(dof_absolute[i])

    mujoco.mj_forward(mj_model, mj_data)


def _get_privileged_state_env_style(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, holder: _PrivilegedPrevHolder, dt_stub: float) -> np.ndarray:
    """等价 env.py ``get_privileged_state``（首次调用速度与角速度视为 0）。"""
    total_bodies = mj_model.nbody
    valid_body_indices: list[int] = []
    head_link_idx: int | None = None

    for i in range(total_bodies):
        try:
            body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        except Exception:
            body_name = None
        if body_name:
            if body_name.startswith("dummy") or body_name.endswith("hand") or body_name.startswith("world"):
                continue
            if body_name == "head_link":
                head_link_idx = i
            else:
                valid_body_indices.append(i)
        else:
            valid_body_indices.append(i)

    if head_link_idx is not None:
        valid_body_indices.append(head_link_idx)

    num_bodies = len(valid_body_indices)
    valid_body_indices_np = np.array(valid_body_indices, dtype=np.int64)

    body_pos = mj_data.xpos[valid_body_indices_np].copy().astype(np.float32)
    body_quat_wxyz = mj_data.xquat[valid_body_indices_np].copy().astype(np.float32)

    if holder.body_pos_prev is None:
        body_vel = np.zeros((num_bodies, 3), dtype=np.float32)
        body_ang_vel = np.zeros((num_bodies, 3), dtype=np.float32)
        holder.body_pos_prev = body_pos.copy()
        holder.body_quat_prev = body_quat_wxyz.copy()
    else:
        body_vel = (body_pos - holder.body_pos_prev) / dt_stub
        body_ang_vel = _calc_angular_velocity(body_quat_wxyz, holder.body_quat_prev, dt_stub).astype(np.float32)
        holder.body_pos_prev = body_pos.copy()
        holder.body_quat_prev = body_quat_wxyz.copy()

    body_pos_t = torch.from_numpy(body_pos).unsqueeze(0)
    body_rot_t = torch.from_numpy(body_quat_wxyz[:, [1, 2, 3, 0]].copy()).unsqueeze(0).float()
    body_vel_t = torch.from_numpy(body_vel).unsqueeze(0).float()
    body_ang_vel_t = torch.from_numpy(body_ang_vel).unsqueeze(0).float()

    root_pos = body_pos_t[:, 0:1, :]
    root_rot = body_rot_t[:, 0:1, :]

    heading_rot_inv = calc_heading_quat_inv(root_rot, w_last=True)
    heading_rot_inv_expand = heading_rot_inv.repeat(1, num_bodies, 1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(-1, 4)

    root_pos_expand = root_pos.repeat(1, num_bodies, 1)
    local_body_pos = body_pos_t - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(-1, 3)
    flat_local_body_pos = quat_rotate(flat_heading_rot_inv, flat_local_body_pos, w_last=True)
    local_body_pos_obs = flat_local_body_pos.reshape(1, -1)
    local_body_pos_obs = local_body_pos_obs[..., 3:]

    flat_body_rot = body_rot_t.reshape(-1, 4)
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot, w_last=True)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot, w_last=True)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(1, -1)

    flat_body_vel = body_vel_t.reshape(-1, 3)
    flat_local_body_vel = quat_rotate(flat_heading_rot_inv, flat_body_vel, w_last=True)
    local_body_vel_obs = flat_local_body_vel.reshape(1, -1)

    flat_body_ang_vel = body_ang_vel_t.reshape(-1, 3)
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot_inv, flat_body_ang_vel, w_last=True)
    local_body_ang_vel_obs = flat_local_body_ang_vel.reshape(1, -1)

    root_h = root_pos[:, :, 2:3].squeeze(0)

    privileged = torch.cat(
        [root_h, local_body_pos_obs, local_body_rot_obs, local_body_vel_obs, local_body_ang_vel_obs], dim=-1
    ).squeeze(0)
    return privileged.detach().cpu().numpy().astype(np.float32)


def _create_observation_backward_env_style(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> tuple[dict[str, torch.Tensor], int]:
    """
    env.py:_create_observation_backward；返回 tensors (1,batch) CPU -> 外层再拷贝到模型 device。
    第二返回值：privileged_dim（用于断言与 config）。
    """
    dof_pos_rel = mj_data.qpos[7 : 7 + 29].copy().astype(np.float64) - _training_default_dof_vec()
    dof_vel = mj_data.qvel[6 : 6 + 29].copy().astype(np.float64)

    root_quat_wxyz = mj_data.qpos[3:7].astype(np.float64)
    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    rot = ScipyRotation.from_quat([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]])
    projected_gravity = rot.inv().apply(gravity_vec).astype(np.float64)
    ang_vel = rot.apply(mj_data.qvel[3:6].copy()).astype(np.float64)

    holder = _PrivilegedPrevHolder()
    privileged_np = _get_privileged_state_env_style(mj_model, mj_data, holder, dt_stub=1.0 / 300.0)
    privileged_dim = int(privileged_np.shape[-1])

    state = torch.tensor(np.concatenate([dof_pos_rel, dof_vel, projected_gravity, ang_vel]).astype(np.float32)).unsqueeze(0)
    new_obs = {
        "state": state,
        "last_action": torch.zeros((1, 29), dtype=torch.float32),
        "privileged_state": torch.from_numpy(privileged_np.copy()).unsqueeze(0),
    }
    return new_obs, privileged_dim


def _build_goal_observation(
    mujoco_xml: Path,
    device: torch.device,
    *,
    use_root_height_obs: bool,
) -> dict[str, torch.Tensor]:
    xml_path = Path(mujoco_xml).resolve()
    if not xml_path.is_file():
        xml_path = (get_g1_robot_xml_root() / DEFAULT_MUJOCO_SCENE).resolve()

    dof_abs = _absolute_dof_array(GOAL_POSE_JOINT_ABS_RAD)

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    _set_mujoco_pose_from_abs(mj_model, mj_data, dof_abs)

    raw, privileged_dim = _create_observation_backward_env_style(mj_model, mj_data)
    expect = 463 if use_root_height_obs else 462
    if privileged_dim != expect:
        raise RuntimeError(
            f"privileged_state 维度 {privileged_dim} 与 checkpoint 预期的 {expect} "
            f"(root_height_obs={use_root_height_obs}) 不符；请核对 MJCF 与 env.root_height_obs。"
        )
    return {k: v.to(device=device, dtype=torch.float32) for k, v in raw.items()}


def main(
    model_folder: Path,
    data_path: Path | None = None,
    rollout_motion_id: int = ROLLOUT_REF_MOTION_ID,
    mujoco_goal_xml: Path | None = None,
    headless: bool = True,
    device: str = "cuda",
    simulator: str = "isaacsim",
    hand_noise_std: float = 0.0,
    seed: int = 0,
    episode_len: int = 500,
    save_mp4: bool = True,
    video_folder: Path | None = None,
    video_suffix: str = "",
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    video_camera: str = "face_torso",
    face_torso_camera_distance: float = 2.75,
    face_torso_camera_elevation_deg: float = 14.0,
    policy_sample: bool = False,
    save_z_trace: bool = True,
    z_trace_out: Path | None = None,
) -> None:
    """
    从模块级 ``GOAL_POSE_JOINT_ABS_RAD`` 定义目标铰链角，经 ``env.py`` 风格 backward 观测做 ``goal_inference``，
    再 ``project_z(act on z_hand 噪声同上)`` 与 Isaac env 闭环 ``episode_len`` 步；
    默认写 mp4 与 ``--z-trace-out`` / 默认路径下的 rollout z ``.npz``。
    """
    model_folder = Path(model_folder)
    vid_dir = Path(video_folder) if video_folder is not None else model_folder / "split_z_goal_isaac" / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()

    if not model.cfg.archi.is_split_mode:
        raise RuntimeError("需要 split-z checkpoint（z_body_dim>0 且 z_hand_dim>0）。")

    with open(model_folder / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    use_root_height_obs = bool(config["env"].get("root_height_obs", False))

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

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    rollout_mid = int(rollout_motion_id)

    xml_goal = mujoco_goal_xml if mujoco_goal_xml is not None else get_g1_robot_xml_root() / DEFAULT_MUJOCO_SCENE

    backward_keys = ["state", "last_action", "privileged_state"]
    goal_observation = _build_goal_observation(xml_goal, model.device, use_root_height_obs=use_root_height_obs)

    print(
        f"goal=Mujoco(env.py-style) xml={xml_goal} · rollout_motion_id={rollout_mid} · "
        f"use_root_height_obs={use_root_height_obs}"
    )
    vc = video_camera.lower().strip()
    if vc not in ("face_torso", "track"):
        raise ValueError("video_camera must be 'face_torso' or 'track'.")
    print(f"hand_noise_std={hand_noise_std} · video_camera={vc} · policy_sample={policy_sample}")

    env.set_is_evaluating(rollout_mid)

    d_body = int(model.cfg.archi.z_body_dim)
    with torch.no_grad():
        z_ref = model.goal_inference(goal_observation)

    zb = z_ref[:, :d_body]
    zh = z_ref[:, d_body:]
    policy_goal_z_body_np = zb.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    policy_goal_z_hand_np = zh.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    rng = torch.Generator(device=z_ref.device)
    rng.manual_seed(seed)
    if hand_noise_std > 0:
        zh = zh + float(hand_noise_std) * torch.randn(zh.shape, device=zh.device, dtype=zh.dtype, generator=rng)
    z = model.project_z(torch.cat([zb, zh], dim=-1))
    zb2 = z[:, :d_body]
    zh2 = z[:, d_body:]

    def _backward_obs_from_wrapped(obs_dict: dict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k in backward_keys:
            if k not in obs_dict:
                raise KeyError(f"backward 缺键 {k!r}; 现有 {list(obs_dict.keys())}")
            t = obs_dict[k]
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t, device=model.device, dtype=torch.float32)
            else:
                t = t.to(device=model.device, dtype=torch.float32)
            out[k] = t
        return out

    def _record_inferred_z(obs_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ob = _backward_obs_from_wrapped(obs_dict)
        zt = model.project_z(model.backward_map(ob))
        zb_np = zt[:, :d_body].detach().float().cpu().numpy().squeeze(0).astype(np.float32)
        zh_np = zt[:, d_body:].detach().float().cpu().numpy().squeeze(0).astype(np.float32)
        zf_np = zt.detach().float().cpu().numpy().squeeze(0).astype(np.float32)
        return zb_np, zh_np, zf_np

    inferred_b: list[np.ndarray] = []
    inferred_h: list[np.ndarray] = []
    inferred_f: list[np.ndarray] = []

    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(
            render_size=256,
            video_camera=vc,
            face_torso_distance=face_torso_camera_distance,
            face_torso_elevation_deg=face_torso_camera_elevation_deg,
        )

    observation, _ = wrapped_env.reset(to_numpy=False)
    observation, _ = wrapped_env.reset(to_numpy=False)
    observation, _ = wrapped_env.reset(to_numpy=False)
    observation, _ = wrapped_env.reset(to_numpy=False)

    frames: list = []
    zn = z.repeat(env.num_envs, 1)
    act_mean = not policy_sample
    infer_n = tqdm(range(episode_len), desc="steps", leave=False)

    with torch.no_grad():
        b0, h0, f0 = _record_inferred_z(observation)
        inferred_b.append(b0)
        inferred_h.append(h0)
        inferred_f.append(f0)

    for _counter in infer_n:
        action = model.act(observation, zn, mean=act_mean)
        observation, _r, _t, _trunc, _i = wrapped_env.step(action, to_numpy=False)
        with torch.no_grad():
            b1, h1, f1 = _record_inferred_z(observation)
            inferred_b.append(b1)
            inferred_h.append(h1)
            inferred_f.append(f1)
        if save_mp4:
            frames.append(rgb_renderer.render(env, 0)[0])

    tag = "mujoco_env_py_goal"
    if save_z_trace:
        zb_arr = np.stack(inferred_b, axis=0)
        zh_arr = np.stack(inferred_h, axis=0)
        zf_arr = np.stack(inferred_f, axis=0)
        trace_path = Path(z_trace_out) if z_trace_out is not None else vid_dir / (
            f"ztrace_{tag}_hnoise{hand_noise_std:.4f}".replace(".", "p") + ".npz"
        )
        meta = {
            "goal_source": "env_py_style_mujoco",
            "mujoco_xml": str(xml_goal),
            "rollout_motion_id": rollout_mid,
            "motion_id": -1,
            "goal_frame": 0,
            "hand_noise_std": hand_noise_std,
            "episode_len": episode_len,
            "z_body_dim": d_body,
            "z_hand_dim": int(model.cfg.archi.z_hand_dim),
            "seed": int(seed),
            "policy_sample": bool(policy_sample),
            "backward_keys": backward_keys,
        }
        np.savez(
            trace_path,
            inferred_z_body=zb_arr,
            inferred_z_hand=zh_arr,
            inferred_z_full=zf_arr,
            policy_goal_z_body=policy_goal_z_body_np,
            policy_goal_z_hand=policy_goal_z_hand_np,
            policy_goal_z_full=z_ref.detach().cpu().numpy().reshape(-1).astype(np.float32),
            policy_actor_z_body=zb2.detach().cpu().numpy().reshape(-1).astype(np.float32),
            policy_actor_z_hand=zh2.detach().cpu().numpy().reshape(-1).astype(np.float32),
            policy_actor_z_full=z.detach().cpu().numpy().reshape(-1).astype(np.float32),
            meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
        )
        print(f"Saved z trace: {trace_path} (samples={zb_arr.shape[0]})")

    if save_mp4 and frames:
        sfx = f"_{video_suffix}" if video_suffix else ""
        noise_tag = f"hnoise{hand_noise_std:.4f}".replace(".", "p")
        out_mp4 = vid_dir / f"goal_{tag}_{noise_tag}_{vc}{'_psample' if policy_sample else ''}{sfx}.mp4"
        media.write_video(str(out_mp4), frames, fps=50)
        print(f"Saved video: {out_mp4}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
