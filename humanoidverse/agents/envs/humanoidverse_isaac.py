import os

os.environ["HYDRA_FULL_ERROR"] = "1"
# For Isaac Sim
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
import typing as tp
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import gymnasium
import humanoidverse
import hydra
import mujoco
import numpy as np
import torch
import pydantic
from gymnasium import Env
from gymnasium.vector import VectorEnv
from humanoidverse.envs.legged_robot_motions.legged_robot_motions import LeggedRobotMotions, compute_humanoid_observations_max
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.utils.torch_utils import calc_heading_quat_inv, quat_rotate, quat_rotate_inverse
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..buffers.trajectory import TrajectoryDictBuffer
from .utils.history_handler import HistoryHandler
from humanoidverse.envs.env_utils.history_handler import HistoryHandler as HVHistoryHandler

# Resolve humanoidverse root: use package __file__ when set,
# otherwise derive from this file (e.g. when run from release_version/humanoidverse where __file__ can be None).
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = os.path.dirname(humanoidverse.__file__)
else:
    HUMANOIDVERSE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# THIS_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
HYDRA_CONFIG_DIR = os.path.join(HUMANOIDVERSE_DIR, "config")
HYDRA_CONFIG_REL_PATH = os.path.join("exp", "bfm_zero", "bfm_zero")


@dataclass
class RightWristControlData:
    """Simulator tensors needed by the inference-time hand controller.

    Cartesian tensors and the Jacobian are expressed in ``control_frame``.
    Jacobian rows 0:3 are linear, as specified by the simulator tensor APIs.
    """

    current_pos_control: torch.Tensor
    target_pos_control: torch.Tensor
    linear_vel_control: torch.Tensor
    target_linear_vel_control: torch.Tensor
    jacobian_pos_control: torch.Tensor
    control_frame: str
    jacobian_pos_simulator: torch.Tensor
    jacobian_source_frame: str
    right_arm_q: torch.Tensor
    default_right_arm_q: torch.Tensor
    lower_limits: torch.Tensor
    upper_limits: torch.Tensor
    action_scale: torch.Tensor
    action_lower: torch.Tensor
    action_upper: torch.Tensor
    right_arm_dof_indices: torch.Tensor


def _expert_source_base_weights(file_names: list, source_weights: dict[str, float]) -> list[float]:
    """Per-trajectory base sampling weights from per-source weights.

    Each motion key is matched to the LONGEST matching prefix in source_weights
    ("" acts as catch-all, e.g. for original lafan keys). The weight of a source is
    its *total* sampling mass: per-trajectory weight = source_weight / n_in_source,
    so the source-level mix is exactly proportional to source_weights regardless of
    how many clips each source contributes.
    """
    prefixes = sorted(source_weights.keys(), key=len, reverse=True)

    def match(name: str) -> str:
        for p in prefixes:
            if str(name).startswith(p):
                return p
        raise ValueError(f"motion key {name!r} matches no prefix in {list(source_weights)}")

    sources = [match(n) for n in file_names]
    counts = {p: sources.count(p) for p in set(sources)}
    # rescale so weights stay O(1) (mean 1) for numerical niceness
    raw = [source_weights[s] / counts[s] for s in sources]
    mean_raw = sum(raw) / len(raw)
    weights = [w / mean_raw for w in raw]
    print("Expert source balance:", {p: f"n={counts[p]}, total_mass={source_weights[p]}" for p in counts})
    return weights


def load_expert_trajectories_from_motion_lib(
    env, agent_cfg, device="cpu", add_history_noaction: bool = False, source_weights: dict[str, float] | None = None
):
    """
    Load expert trajectories from motion library.

    source_weights: optional {key_prefix: total_sampling_mass} to balance sampling
    across data sources in the merged dataset (see _expert_source_base_weights).
    """
    env._motion_lib.load_motions_for_training()  # loading all motions from the motion lib
    episodes = []
    file_names = []
    history_handler = HVHistoryHandler(1, env.config.obs.obs_auxiliary, env.config.obs.obs_dims, device)
    history_config = env.config.obs.obs_auxiliary["history_actor"]
    for i in range(env._motion_lib._num_unique_motions):
        motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[i] / env.dt).cpu()))).to(env.device) * env.dt
        # import ipdb; ipdb.set_trace()
        motion_id = torch.tensor([i]).to(env.device).repeat(motion_times.shape[0])
        motion_res = env._motion_lib.get_motion_state(motion_id, motion_times)
        file_names.append(env._motion_lib._motion_data_keys[i])
        # import ipdb; ipdb.set_trace()
        ref_body_pos = motion_res["rg_pos_t"]
        ref_body_rots = motion_res["rg_rot_t"]
        ref_body_vels = motion_res["body_vel_t"]
        ref_body_angular_vels = motion_res["body_ang_vel_t"]

        # construct observation
        # TODO is this aligned with the environment observation logic?
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
        )
        max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

        # Aligned with the logic below to create proprio state
        base_quat = ref_body_rots[:, 0]
        ref_dof_pos = motion_res["dof_pos"] - env.default_dof_pos[0]
        ref_dof_vel = motion_res["dof_vel"]
        ref_ang_vel = ref_body_angular_vels[:, 0]
        projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True)
        # NOTE we multiply by zero to align with mujoco data
        bogus_actions = ref_dof_pos * 0  # bogus actions

        state = torch.cat(
            [
                ref_dof_pos,
                ref_dof_vel,
                projected_gravity,
                ref_ang_vel,
            ],
            dim=-1,
        )

        data = {
            "base_ang_vel": ref_ang_vel,
            "projected_gravity": projected_gravity,
            "dof_pos": ref_dof_pos,
            "dof_vel": ref_dof_vel,
        }

        ### compute history_actor
        # TODO: speed this up
        # We use the same history has the environment but we drop the action
        if add_history_noaction:
            history_handler.reset([0])
            history_actor = []
            for ii in range(state.shape[0]):
                history_tensors = []
                for key in sorted(history_config.keys()):
                    if key not in ["action", "actions"]:
                        history_length = history_config[key]
                        history_tensor = history_handler.query(key)[:, :history_length]
                        history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [1, history_length*obs_dim]
                        history_tensors.append(history_tensor)
                history_tensors = torch.cat(history_tensors, dim=1)  # this is history_actor at time i
                history_actor.append(history_tensors)

                # update history with the current state
                for key in history_handler.history.keys():
                    if key not in ["action", "actions"]:
                        history_handler.add(key, data[key][ii][None, ...])
            history_actor = torch.stack(history_actor, dim=0).squeeze(1)  # Shape: [T, history_length*obs_dim]
        ###

        curr_motion_len = state.shape[0]
        truncated = torch.zeros(curr_motion_len, dtype=bool).to(env.device)
        truncated[-1] = True

        assert state.shape[0] == curr_motion_len, f"{env._motion_lib._motion_data_keys[i]}: {state.shape[0]} vs {curr_motion_len}"
        assert max_local_self_obs.shape[0] == curr_motion_len, (
            f"{env._motion_lib._motion_data_keys[i]}: {max_local_self_obs.shape[0]} vs {curr_motion_len}"
        )
        assert bogus_actions.shape[0] == curr_motion_len, (
            f"{env._motion_lib._motion_data_keys[i]}: {bogus_actions.shape[0]} vs {curr_motion_len}"
        )
        assert truncated.shape[0] == curr_motion_len, f"{env._motion_lib._motion_data_keys[i]}: {truncated.shape[0]} vs {curr_motion_len}"
        if add_history_noaction:
            assert history_actor.shape[0] == curr_motion_len, (
                f"{env._motion_lib._motion_data_keys[i]}: {history_actor.shape[0]} vs {curr_motion_len}"
            )

        ep = {
            "observation": {
                "state": state,
                "last_action": bogus_actions,
                "privileged_state": max_local_self_obs,
            },
            "terminated": torch.zeros(curr_motion_len, dtype=bool).to(env.device),
            "truncated": truncated,
            "motion_id": torch.ones(curr_motion_len, dtype=torch.long) * i,
        }
        if add_history_noaction:
            ep["observation"]["history_noaction"] = history_actor
        episodes.append(ep)

    base_weights = None
    if source_weights is not None:
        base_weights = _expert_source_base_weights(file_names, source_weights)

    expert_buffer = TrajectoryDictBuffer(
        episodes=episodes,
        seq_length=agent_cfg.model.seq_length,
        device=device,
        base_weights=base_weights,
    )

    assert expert_buffer.storage["observation"]["state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["last_action"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["privileged_state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["terminated"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["motion_id"].shape[0] == expert_buffer.storage["truncated"].shape[0]

    expert_buffer.file_names = file_names
    return expert_buffer


def get_enabled_dr_dynamics_obs_names(env: LeggedRobotMotions) -> list[str]:
    """Get list of dr_* raw observations that are actually being varied in the code (e.g., if ctrl_delay is randomized, then include dr_ctrl_delay observation in the list)

    Useful for picking not including dr_* observations that are not actually varied in the environment."""
    isaac_config = env.config

    dr_dynamics = []
    if isaac_config.domain_rand.randomize_ctrl_delay:
        dr_dynamics.append("dr_ctrl_delay")
    if isaac_config.domain_rand.randomize_pd_gain:
        dr_dynamics.append("dr_pd_gains")
    if isaac_config.domain_rand.randomize_base_com:
        dr_dynamics.append("dr_coms")
    if isaac_config.domain_rand.randomize_link_mass:
        dr_dynamics.append("dr_masses")
    if isaac_config.domain_rand.randomize_friction:
        dr_dynamics.append("dr_frictions")
    if isaac_config.domain_rand.randomize_torque_rfi:
        dr_dynamics.append("dr_torque_rfi")
    if isaac_config.domain_rand.randomize_rfi_lim:
        dr_dynamics.append("dr_rfi_lim")

    # Special case for obs noise: check if any noise is actually applied
    if isaac_config.obs.noise_scales is not None:
        for key, noise_scale in isaac_config.obs.noise_scales.items():
            if noise_scale > 0.0:
                # At least one noise scale is applied, so lets include all obs_noise_scales
                dr_dynamics.append("dr_obs_noise_scales")
                break

    return dr_dynamics


def mj_camera_face_torso(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    distance: float = 2.75,
    elevation_deg: float = 14.0,
    lookat_height_above_torso_m: float = 0.42,
) -> mujoco.MjvCamera:
    """Look-at 躯干前方：摄像机置于骨盆–躯干大致朝向的反侧（胸口朝向屏幕），随 yaw 转动保持正面。"""
    torso_name = "torso_link"
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_name)
    if torso_id < 0:
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if torso_id < 0:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        return cam

    tpos = np.asarray(data.xpos[torso_id], dtype=np.float64).copy()
    R = np.asarray(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)
    fwd = R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    fwd[2] = 0.0
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        fwd = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        fwd /= n

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = tpos + np.array([0.0, 0.0, float(lookat_height_above_torso_m)], dtype=np.float64)
    cam.distance = distance
    cam.azimuth = float(np.degrees(np.arctan2(-fwd[1], -fwd[0])))
    cam.elevation = float(elevation_deg)

    return cam


class IsaacRendererWithMuJoco:
    """Renders Isaac state via MuJoCo. Only 29 DOF (36-D qpos: 7 free + 29 joints) is supported."""

    def __init__(
        self,
        render_size: int = 512,
        *,
        video_camera: str = "track",
        face_torso_distance: float = 2.75,
        face_torso_elevation_deg: float = 14.0,
        face_torso_lookat_height_above_torso_m: float = 0.42,
    ):
        from humanoidverse.utils.g1_env_config import G1EnvConfig

        self.mujoco_env, _ = G1EnvConfig(render_height=render_size, render_width=render_size).build(num_envs=1)
        self.video_camera = video_camera
        self.face_torso_distance = face_torso_distance
        self.face_torso_elevation_deg = face_torso_elevation_deg
        self.face_torso_lookat_height_above_torso_m = face_torso_lookat_height_above_torso_m

    @staticmethod
    def _inner_g1_env(wrapped):
        """``G1EnvConfig.build`` 外层有 Gymnasium Wrapper，必须用内层 ``G1Base.render(..., camera=)``。"""
        e = wrapped
        while hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        return e

    def render(self, hv_env: "HumanoidVerseVectorEnv", env_idxs: list[int] | None = None):
        base_pos = hv_env.simulator.robot_root_states[:, [0, 1, 2, 6, 3, 4, 5]].clone().detach().cpu().numpy()
        joint_pos = hv_env.simulator.dof_pos.clone().detach().cpu().numpy()
        if joint_pos.shape[1] != 29:
            raise ValueError(
                f"Isaac dof_pos must be 29-D (codebase is 29 DOF only), got {joint_pos.shape[1]}."
            )
        mujoco_qpos = np.concatenate([base_pos, joint_pos], axis=1)  # (n_envs, 36)

        all_images = []
        if env_idxs is None:
            env_idxs = list(range(hv_env.num_envs))
        elif not isinstance(env_idxs, (list, tuple)):
            env_idxs = [int(env_idxs)]  # e.g. render(env, 0) -> render env 0 only
        g1 = IsaacRendererWithMuJoco._inner_g1_env(self.mujoco_env)
        mj_inner = g1  # aliases for clearer reset/model access names used below

        for env_idx in env_idxs:
            qvel = mj_inner._mj_data.qvel.copy()
            self.mujoco_env.reset(options={"qpos": mujoco_qpos[env_idx], "qvel": qvel})
            mujoco.mj_forward(mj_inner.model, mj_inner.data)
            if self.video_camera == "face_torso":
                cam = mj_camera_face_torso(
                    mj_inner.model,
                    mj_inner.data,
                    distance=self.face_torso_distance,
                    elevation_deg=self.face_torso_elevation_deg,
                    lookat_height_above_torso_m=self.face_torso_lookat_height_above_torso_m,
                )
                all_images.append(g1.render(camera=cam))
            else:
                all_images.append(g1.render())

        return all_images

    @staticmethod
    def _estimate_foreground_mask(frame: np.ndarray) -> np.ndarray:
        corners = np.concatenate(
            [
                frame[:20, :20].reshape(-1, 3),
                frame[:20, -20:].reshape(-1, 3),
                frame[-20:, :20].reshape(-1, 3),
                frame[-20:, -20:].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(corners.astype(np.float32), axis=0)
        diff = np.linalg.norm(frame.astype(np.float32) - background[None, None, :], axis=-1)
        return diff > 28.0

    @staticmethod
    def _overlay_goal_frame(
        current_frame: np.ndarray, goal_frame: np.ndarray, *, overlay_alpha: float = 0.42
    ) -> np.ndarray:
        mask = IsaacRendererWithMuJoco._estimate_foreground_mask(goal_frame)
        tinted_goal = goal_frame.astype(np.float32).copy()
        blue_tint = np.array([90.0, 185.0, 255.0], dtype=np.float32)
        tinted_goal = 0.25 * tinted_goal + 0.75 * blue_tint
        out = current_frame.astype(np.float32).copy()
        alpha = float(overlay_alpha)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * tinted_goal[mask]
        return np.clip(out, 0, 255).astype(np.uint8)

    def render_with_goal_dof_overlay(
        self,
        hv_env: tp.Any,
        env_idx: int,
        goal_dof_pos: np.ndarray,
        *,
        overlay_alpha: float = 0.42,
    ) -> np.ndarray:
        """与当前仿真根姿态一致，将 29-D 关节设为目标铰链角，用半透明青色叠在 MuJoCo 画上（结构化 z probe 思路）。"""
        base_pos = hv_env.simulator.robot_root_states[:, [0, 1, 2, 6, 3, 4, 5]].clone().detach().cpu().numpy()
        joint_pos = hv_env.simulator.dof_pos.clone().detach().cpu().numpy()
        if joint_pos.shape[1] != 29:
            raise ValueError(f"Expected 29 DOF dof_pos; got shape {joint_pos.shape}.")
        q_cur = np.concatenate([base_pos[env_idx].astype(np.float32), joint_pos[env_idx].astype(np.float32)], axis=-1)

        gd = np.asarray(goal_dof_pos, dtype=np.float32).reshape(-1)
        if gd.size != 29:
            raise ValueError(f"goal_dof_pos must have 29 elements; got shape {gd.shape}.")
        q_goal = np.concatenate([q_cur[:7].astype(np.float32), gd], axis=-1)

        g1 = IsaacRendererWithMuJoco._inner_g1_env(self.mujoco_env)
        mj_inner = g1

        cam = None
        qvel = mj_inner._mj_data.qvel.copy()
        self.mujoco_env.reset(options={"qpos": q_cur, "qvel": qvel})
        mujoco.mj_forward(mj_inner.model, mj_inner.data)

        if self.video_camera == "face_torso":
            cam = mj_camera_face_torso(
                mj_inner.model,
                mj_inner.data,
                distance=self.face_torso_distance,
                elevation_deg=self.face_torso_elevation_deg,
                lookat_height_above_torso_m=self.face_torso_lookat_height_above_torso_m,
            )

        rgb_cur = g1.render(camera=cam)

        qvel_goal = mj_inner._mj_data.qvel.copy()
        self.mujoco_env.reset(options={"qpos": q_goal, "qvel": qvel_goal})
        mujoco.mj_forward(mj_inner.model, mj_inner.data)

        rgb_goal = g1.render(camera=cam)

        overlay = IsaacRendererWithMuJoco._overlay_goal_frame(rgb_cur, rgb_goal, overlay_alpha=overlay_alpha)
        # 恢复 MuJoCo 与当前仿真一致（便于下一帧继续使用同一 renderer）
        self.mujoco_env.reset(options={"qpos": q_cur, "qvel": qvel})
        mujoco.mj_forward(mj_inner.model, mj_inner.data)

        return overlay

    def from_qpos(self, qpos):
        """Render frames for each qpos. Only 36-D qpos (7 free + 29 joints) is supported."""
        frames = []
        n = len(qpos)
        for i, q in enumerate(qpos):
            if (i + 1) % 50 == 0 or i == 0 or i == n - 1:
                print(f"  Rendering expert frame {i + 1}/{n}")
            q = np.asarray(q).ravel()
            if q.size != 36:
                raise ValueError(
                    f"from_qpos expects 36-D qpos (7 free + 29 joints), got shape {q.shape}. "
                    "23-DOF / 30-D qpos is not supported."
                )
            self.mujoco_env.reset(options={"qpos": q})
            g1 = IsaacRendererWithMuJoco._inner_g1_env(self.mujoco_env)
            frames.append(g1.render())
        return frames



class HumanoidVerseVectorEnv(VectorEnv):
    """Wrapper to make the humanoidverse env compatible with gymnasium's VectorEnv interface."""

    def __init__(
        self,
        env: LeggedRobotMotions,
        add_time_aware_observation: bool = True,
        include_last_action: bool = True,
        context_length: int | None = None,
        include_dr_info: bool = False,
        included_dr_obs_names: list[str] | None = None,
        include_history_actor: bool = True,
        include_history_noaction: bool = False,
    ):
        super().__init__()
        self._env = env
        self.spec = self._env.spec
        self.num_envs = self._env.num_envs
        self.add_time_aware_observation = add_time_aware_observation
        self.include_last_action = include_last_action
        self.context_length = context_length
        self.history_handler = None
        self.include_dr_info = include_dr_info
        self.include_history_actor = include_history_actor
        self.include_history_noaction = include_history_noaction
        if included_dr_obs_names is not None:
            self.included_dr_obs_names = included_dr_obs_names
        else:
            self.included_dr_obs_names = get_enabled_dr_dynamics_obs_names(self._env)
        if self.include_dr_info and len(self.included_dr_obs_names) == 0:
            raise ValueError(
                "include_dr_info is True, but no dr_* observations are enabled in the environment. "
                "Check the domain randomization settings in the environment config."
            )

        # TODO does might not be ideal, but we need to reset things to get the observation space
        example_observation, _ = self.reset()
        observation_spaces = {}
        for key, value in example_observation.items():
            if key != "history":
                observation_spaces[key] = gymnasium.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=value.shape,
                    dtype=np.float32,
                )
        if self.include_history_noaction:
            history_length = 0
            for k, v in self._env.config.obs.obs_auxiliary["history_actor"].items():
                if k not in ["action", "actions"]:
                    history_length += v * self._env.config.obs.obs_dims[k]
            observation_spaces["history_noaction"] = gymnasium.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.num_envs, history_length),
                dtype=np.float32,
            )

        if add_time_aware_observation:
            # Update the observation spaces to include time-aware observation
            observation_spaces["time"] = gymnasium.spaces.Box(
                low=0.0,
                high=float("inf"),
                shape=(1,),
                dtype=np.int32,
            )
        self.observation_space = gymnasium.spaces.Dict(observation_spaces)
        self.single_action_space = self._env.get_wrapper_attr("single_action_space")
        # Repeat the action space to match the vectorized environment
        action_space_shape = (self.num_envs,) + self.single_action_space.shape
        self.action_space = gymnasium.spaces.Box(
            low=np.tile(self.single_action_space.low, (self.num_envs, 1)),
            high=np.tile(self.single_action_space.high, (self.num_envs, 1)),
            shape=action_space_shape,
            dtype=self.single_action_space.dtype,
        )
        if self.context_length:
            # keep track of the high level history of the observations and action
            dims = {"action": self.single_action_space.shape[0]}
            for k, v in self.single_observation_space.spaces.items():
                dims[k] = 1 if len(v.shape) == 0 else v.shape[0]
            self.history_handler = HistoryHandler(self.num_envs, context_length=self.context_length, keys_dims=dims, device=self.device)

        _target_dof_pos = self._env.default_dof_pos.clone().unsqueeze(0).repeat(self.num_envs, 1, 2)
        # Set velocities to zero
        _target_dof_pos[..., 1] = 0.0
        self._default_pose_target_reset = {
            "dof_states": _target_dof_pos,
            "root_states": self._env.base_init_state,
        }

    @property
    def single_observation_space(self):
        """Return the observation space for a single environment."""
        single_obs_spaces = {}
        for key, space in self.observation_space.spaces.items():
            single_obs_spaces[key] = gymnasium.spaces.Box(
                low=space.low[0],
                high=space.high[0],
                shape=space.shape[1:],
                dtype=space.dtype,
            )
        return gymnasium.spaces.Dict(single_obs_spaces)

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> Env:
        return self._env.unwrapped

    @property
    def unwrapped(self):
        return self.base_env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        to_numpy: bool = True,
        reset_to_default_pose: bool = False,
        target_states: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if reset_to_default_pose and target_states is not None:
            raise ValueError("Cannot specify both reset_to_default_pose and target_states.")
        target_states = target_states or (self._default_pose_target_reset if reset_to_default_pose else None)
        _, info = self.base_env.reset_all(target_states=target_states)
        observation = self._get_g1env_observation(to_numpy=to_numpy)
        qpos, qvel = self._get_qpos_qvel(to_numpy=to_numpy)
        # add only observation to the history
        # observation is now 1 step ahead
        if self.history_handler is not None:
            self._update_history(torch.arange(self.num_envs, device=self.device), observation, None)
            observation["history"] = {"observation": [], "action": []}
        self.env_to_reset = []
        info = {"qpos": qpos, "qvel": qvel}
        return observation, info

    def _get_qpos_qvel(self, to_numpy: bool = True):
        # base_pos = self._env.simulator._robot.data.root_state_w[:, :7].clone().detach()
        base_pos = self._env.simulator.robot_root_states[:, [0, 1, 2, 6, 3, 4, 5]].clone().detach()
        qpos = self._env.simulator.dof_pos.clone().detach()
        mujoco_qpos = torch.cat([base_pos, qpos], dim=-1)
        qvel = self._env.simulator.dof_vel.clone().detach()
        base_lin_vel = self._env.simulator.robot_root_states[:, 7:10].clone().detach()
        base_ang_vel = self._env._get_obs_base_ang_vel().clone().detach()
        mujoco_qvel = torch.cat([base_lin_vel, base_ang_vel, qvel], dim=-1)
        if to_numpy:
            mujoco_qpos = mujoco_qpos.cpu().numpy()
            mujoco_qvel = mujoco_qvel.cpu().numpy()
        return mujoco_qpos, mujoco_qvel

    def _get_g1env_observation(self, to_numpy: bool = True):
        """Turn current Isaac sim state into a G1Env-like observation."""

        # G1Env state: https://github.com/fairinternal/HumEnv/blob/main/g1env/robot.py#L545-L551
        raw_obs = self._env.obs_buf_dict_raw["actor_obs"]

        g1env_state = torch.cat(
            [
                raw_obs["dof_pos"],
                raw_obs["dof_vel"],
                raw_obs["projected_gravity"],
                raw_obs["base_ang_vel"],
            ],
            dim=-1,
        )
        last_action = raw_obs["actions"]

        # This is the function used to produce "max_local_self"
        # G1Env:
        #  https://github.com/fairinternal/HumEnv/blob/main/g1env/robot.py#L584-L587
        #  https://github.com/fairinternal/HumEnv/blob/main/g1env/robot.py#L713
        privileged_state = raw_obs["max_local_self"]
        observation = {
            "state": g1env_state,
            "privileged_state": privileged_state,
        }
        if self.include_last_action:
            observation["last_action"] = last_action
        if self.add_time_aware_observation:
            observation["time"] = self._env.episode_length_buf.unsqueeze(-1)
        if self.include_history_actor:
            observation["history_actor"] = raw_obs["history_actor"]
        if self.include_history_noaction:
            act_hlen = self._env.config.obs.obs_auxiliary["history_actor"]["actions"]
            observation["history_noaction"] = raw_obs["history_actor"][:, last_action.shape[-1] * act_hlen :]  # drop the action part

        if self.include_dr_info:
            dr_obs = torch.cat(
                [raw_obs[key] for key in self.included_dr_obs_names],
                dim=-1,
            )
            observation["dr_dynamics"] = dr_obs

        if to_numpy:
            observation = tree_map(lambda x: x.cpu().numpy(), observation)

        return observation

    def _initialize_right_wrist_control_metadata(self) -> None:
        """Validate ordering once and cache static action/Jacobian metadata."""

        simulator = self._env.simulator
        expected_names = list(self._env.config.robot.right_arm_dof_names)
        if len(expected_names) != 7:
            raise RuntimeError(f"Expected seven configured right-arm joints, got {expected_names}")
        name_to_index = {name: i for i, name in enumerate(simulator.dof_names)}
        missing = [name for name in expected_names if name not in name_to_index]
        if missing:
            raise RuntimeError(f"Configured right-arm joints absent from simulator order: {missing}")
        indices_list = [name_to_index[name] for name in expected_names]
        actual_names = [simulator.dof_names[i] for i in indices_list]
        if actual_names != expected_names:
            raise RuntimeError(f"Right-arm order mismatch: simulator={actual_names}, config={expected_names}")
        if indices_list != list(range(22, 29)):
            raise RuntimeError(f"This 29-DoF actor expects right arm at action indices 22:29, got {indices_list}")
        wrist_name = "right_wrist_yaw_link"
        if wrist_name not in simulator.body_names:
            raise RuntimeError(f"Body {wrist_name!r} is absent from simulator body order")

        dtype = simulator.dof_pos.dtype
        indices = torch.tensor(indices_list, device=self.device, dtype=torch.long)
        control_cfg = self._env.config.robot.control
        action_scale = torch.full(
            (self._env.num_dof,), float(control_cfg.action_scale), device=self.device, dtype=dtype
        )
        if control_cfg.get("action_rescale", False):
            effort = torch.as_tensor(self._env.config.robot.dof_effort_limit_list, device=self.device, dtype=dtype)
            action_scale *= effort / self._env.p_gains
        if control_cfg.get("normalize_action", False):
            normalize_to = float(control_cfg.get("normalize_action_to") or control_cfg.action_clip_value)
            action_scale *= normalize_to / float(control_cfg.normalize_action_from)

        self._right_arm_dof_indices_list = indices_list
        self._right_arm_dof_indices = indices
        self._right_wrist_body_index = simulator.body_names.index(wrist_name)
        self._right_arm_action_scale = action_scale.index_select(0, indices)
        self._right_arm_action_lower = torch.as_tensor(
            self.single_action_space.low, device=self.device, dtype=dtype
        ).index_select(0, indices)
        self._right_arm_action_upper = torch.as_tensor(
            self.single_action_space.high, device=self.device, dtype=dtype
        ).index_select(0, indices)
        if hasattr(simulator, "_robot") and hasattr(simulator._robot, "root_physx_view"):
            self._right_wrist_raw_body_index = int(simulator.body_ids[self._right_wrist_body_index])
            self._right_arm_raw_dof_indices = torch.tensor(
                [simulator.dof_ids[i] for i in indices_list], device=self.device, dtype=torch.long
            )
            self._right_wrist_raw_body_count = max(int(i) for i in simulator.body_ids) + 1
            self._right_arm_raw_dof_count = max(int(i) for i in simulator.dof_ids) + 1

    def _right_wrist_jacobian_simulator(
        self, wrist_body_index: int, right_arm_indices: torch.Tensor
    ) -> tuple[torch.Tensor, str]:
        """Slice the cached simulator Jacobian and report its documented frame.

        Isaac Gym includes six floating-base columns for a free-base asset;
        PhysX/Isaac Sim exposes articulation-joint columns in its internal joint
        order. Body and joint offsets are inferred from the returned tensor and
        the simulator's explicit order maps rather than hard-coded. PhysX view
        Jacobians are root-frame quantities (as used by Isaac Lab differential
        IK); legacy Isaac Gym tensors are world-frame quantities.
        """

        simulator = self._env.simulator
        if hasattr(simulator, "jacobian"):
            jacobian = simulator.jacobian
            source_frame = "world"
            raw_body_index = wrist_body_index
            raw_dof_indices = right_arm_indices
            raw_body_count = simulator.num_bodies
            raw_dof_count = simulator.num_dof
        elif hasattr(simulator, "_robot") and hasattr(simulator._robot, "root_physx_view"):
            jacobian = simulator._robot.root_physx_view.get_jacobians()
            source_frame = "root"
            raw_body_index = self._right_wrist_raw_body_index
            raw_dof_indices = self._right_arm_raw_dof_indices
            raw_body_count = self._right_wrist_raw_body_count
            raw_dof_count = self._right_arm_raw_dof_count
        else:
            raise RuntimeError(
                f"Simulator {type(simulator).__name__} does not expose a supported batched Jacobian API"
            )

        if jacobian.ndim != 4 or jacobian.shape[-2] != 6:
            raise RuntimeError(f"Expected Jacobian [B, bodies, 6, columns], got {tuple(jacobian.shape)}")
        if jacobian.shape[0] != self.num_envs:
            raise RuntimeError(f"Jacobian batch {jacobian.shape[0]} != num_envs {self.num_envs}")

        if jacobian.shape[1] == raw_body_count:
            jacobian_body_index = raw_body_index
        elif jacobian.shape[1] == raw_body_count - 1:
            # Fixed-base tensor APIs omit the root link from the Jacobian body axis.
            jacobian_body_index = raw_body_index - 1
        else:
            raise RuntimeError(
                f"Cannot map Jacobian body axis {jacobian.shape[1]} to articulation body count {raw_body_count}"
            )
        if not 0 <= jacobian_body_index < jacobian.shape[1]:
            raise RuntimeError(f"Mapped wrist Jacobian body index {jacobian_body_index} is invalid")

        if jacobian.shape[-1] == raw_dof_count:
            jacobian_dof_indices = raw_dof_indices
        elif jacobian.shape[-1] == raw_dof_count + 6:
            # Isaac Gym free-base actors prepend 6 unactuated root columns.
            jacobian_dof_indices = raw_dof_indices + 6
        else:
            raise RuntimeError(
                f"Cannot map Jacobian columns {jacobian.shape[-1]} to articulation DoFs {raw_dof_count}"
            )
        jacobian_dof_indices = jacobian_dof_indices.to(device=jacobian.device, dtype=torch.long)
        jacobian_pos_simulator = jacobian[:, jacobian_body_index, 0:3, :].index_select(
            -1, jacobian_dof_indices
        )
        if jacobian_pos_simulator.shape != (self.num_envs, 3, 7):
            raise RuntimeError(
                f"Right-wrist Jacobian slice has shape {tuple(jacobian_pos_simulator.shape)}, expected [B, 3, 7]"
            )
        return jacobian_pos_simulator, source_frame

    def get_right_wrist_control_data(
        self,
        target_wrist_pos_control: torch.Tensor,
        target_wrist_linear_vel_control: torch.Tensor | None = None,
        control_frame: str = "heading",
    ) -> RightWristControlData:
        """Collect and frame-align right-wrist control tensors on the simulator device."""

        if control_frame not in ("root", "heading"):
            raise ValueError(f"control_frame must be 'root' or 'heading', got {control_frame!r}")

        simulator = self._env.simulator
        if not hasattr(self, "_right_arm_dof_indices"):
            self._initialize_right_wrist_control_metadata()
        right_arm_indices = self._right_arm_dof_indices
        wrist_body_index = self._right_wrist_body_index
        root_state = simulator.robot_root_states
        root_pos_world = root_state[:, 0:3]
        root_quat_world = root_state[:, 3:7]  # xyzw
        root_linear_vel_world = root_state[:, 7:10]
        root_angular_vel_world = root_state[:, 10:13]
        wrist_pos_world = simulator._rigid_body_pos[:, wrist_body_index]
        wrist_linear_vel_world = simulator._rigid_body_vel[:, wrist_body_index]
        wrist_offset_world = wrist_pos_world - root_pos_world

        current_pos_root = quat_rotate_inverse(root_quat_world, wrist_offset_world, w_last=True)
        wrist_relative_vel_world = (
            wrist_linear_vel_world
            - root_linear_vel_world
            - torch.cross(root_angular_vel_world, wrist_offset_world, dim=-1)
        )
        linear_vel_root = quat_rotate_inverse(root_quat_world, wrist_relative_vel_world, w_last=True)

        jacobian_pos_simulator, jacobian_source_frame = self._right_wrist_jacobian_simulator(
            wrist_body_index, right_arm_indices
        )
        if jacobian_source_frame == "world":
            root_quat_per_column = root_quat_world[:, None, :].expand(-1, 7, -1).reshape(-1, 4)
            jacobian_columns_world = jacobian_pos_simulator.transpose(-1, -2).reshape(-1, 3)
            jacobian_pos_root = quat_rotate_inverse(
                root_quat_per_column, jacobian_columns_world, w_last=True
            ).reshape(self.num_envs, 7, 3).transpose(-1, -2)
        else:
            jacobian_pos_root = jacobian_pos_simulator

        if control_frame == "heading":
            heading_quat_inverse = calc_heading_quat_inv(root_quat_world, w_last=True)
            current_pos_control = quat_rotate(
                heading_quat_inverse, wrist_offset_world, w_last=True
            )
            # d(R_heading^-1 (p_wrist - p_root)) / dt subtracts only the
            # yaw-frame angular velocity. Roll and pitch remain observable in
            # the heading frame and therefore must not be removed here.
            root_x_local = torch.zeros_like(wrist_offset_world)
            root_x_local[:, 0] = 1.0
            root_x_world = quat_rotate(root_quat_world, root_x_local, w_last=True)
            root_x_world_dot = torch.cross(
                root_angular_vel_world, root_x_world, dim=-1
            )
            horizontal_norm_sq = root_x_world[:, 0].square() + root_x_world[:, 1].square()
            heading_rate = (
                root_x_world[:, 0] * root_x_world_dot[:, 1]
                - root_x_world[:, 1] * root_x_world_dot[:, 0]
            ) / horizontal_norm_sq.clamp_min(1e-8)
            heading_angular_vel_world = torch.zeros_like(root_angular_vel_world)
            heading_angular_vel_world[:, 2] = heading_rate
            wrist_heading_vel_world = (
                wrist_linear_vel_world
                - root_linear_vel_world
                - torch.cross(heading_angular_vel_world, wrist_offset_world, dim=-1)
            )
            linear_vel_control = quat_rotate(
                heading_quat_inverse, wrist_heading_vel_world, w_last=True
            )
            jacobian_columns_root = jacobian_pos_root.transpose(-1, -2).reshape(-1, 3)
            root_quat_per_column = root_quat_world[:, None, :].expand(-1, 7, -1).reshape(-1, 4)
            jacobian_columns_world = quat_rotate(
                root_quat_per_column, jacobian_columns_root, w_last=True
            )
            heading_inverse_per_column = (
                heading_quat_inverse[:, None, :].expand(-1, 7, -1).reshape(-1, 4)
            )
            jacobian_pos_control = quat_rotate(
                heading_inverse_per_column, jacobian_columns_world, w_last=True
            ).reshape(self.num_envs, 7, 3).transpose(-1, -2)
        else:
            current_pos_control = current_pos_root
            linear_vel_control = linear_vel_root
            jacobian_pos_control = jacobian_pos_root

        target = torch.as_tensor(
            target_wrist_pos_control, device=self.device, dtype=current_pos_control.dtype
        )
        if target.shape == (3,):
            target = target.unsqueeze(0).expand(self.num_envs, -1)
        if target.shape != (self.num_envs, 3):
            raise ValueError(
                f"target_wrist_pos_control must be [3] or [B, 3], got {tuple(target.shape)}"
            )
        if target_wrist_linear_vel_control is None:
            target_velocity = torch.zeros_like(target)
        else:
            target_velocity = torch.as_tensor(
                target_wrist_linear_vel_control,
                device=self.device,
                dtype=current_pos_control.dtype,
            )
            if target_velocity.shape == (3,):
                target_velocity = target_velocity.unsqueeze(0).expand(self.num_envs, -1)
            if target_velocity.shape != (self.num_envs, 3):
                raise ValueError(
                    "target_wrist_linear_vel_control must be [3] or [B, 3], "
                    f"got {tuple(target_velocity.shape)}"
                )

        return RightWristControlData(
            current_pos_control=current_pos_control,
            target_pos_control=target,
            linear_vel_control=linear_vel_control,
            target_linear_vel_control=target_velocity,
            jacobian_pos_control=jacobian_pos_control,
            control_frame=control_frame,
            jacobian_pos_simulator=jacobian_pos_simulator,
            jacobian_source_frame=jacobian_source_frame,
            right_arm_q=simulator.dof_pos.index_select(-1, right_arm_indices),
            default_right_arm_q=(
                self._env.default_dof_pos + self._env.default_dof_pos_offset
            ).index_select(-1, right_arm_indices),
            lower_limits=simulator.hard_dof_pos_limits[:, 0].index_select(0, right_arm_indices),
            upper_limits=simulator.hard_dof_pos_limits[:, 1].index_select(0, right_arm_indices),
            action_scale=self._right_arm_action_scale,
            action_lower=self._right_arm_action_lower,
            action_upper=self._right_arm_action_upper,
            right_arm_dof_indices=right_arm_indices,
        )

    def get_episodic_dr_info(self) -> Dict[str, np.ndarray]:
        """Get the episodic domain randomization information."""
        if not self.include_dr_info:
            raise ValueError("include_dr_info is False, cannot get episodic DR info.")

        dr_info = {}
        for key in self.included_dr_obs_names:
            dr_info[key] = self._env.obs_buf_dict_raw["actor_obs"][key].cpu().numpy()

    def step(
        self, actions: Union[torch.Tensor, Dict], to_numpy: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        # add actions to the history
        # observations are now aligned with the actions
        self._update_history(None, None, actions)
        if to_numpy:
            actions = torch.tensor(actions, device=self._env.device, dtype=torch.float32) if isinstance(actions, np.ndarray) else actions
        actions = {"actions": actions}
        _, reward, reset, new_info = self.base_env.step(actions)
        qpos, qvel = self._get_qpos_qvel(to_numpy=to_numpy)
        # handle the reset logic
        reset = reset.bool()
        env_to_reset = reset.nonzero(as_tuple=False).flatten()  # this is used only for history
        time_outs = new_info["time_outs"].bool()
        terminated = torch.logical_and(reset, ~time_outs)
        truncated = time_outs

        # update observation and history
        observation = self._get_g1env_observation(to_numpy=to_numpy)
        observation = self._add_history_to_observation(observation, to_numpy=to_numpy)
        self._update_history(env_to_reset, observation, None)
        if len(env_to_reset) > 0 and self.history_handler is not None:
            assert len(env_to_reset) == self.num_envs, "Partial resets not supported with history"
            observation["history"] = {"observation": [], "action": []}

        if to_numpy:
            reward = reward.cpu().numpy()
            terminated = terminated.cpu().numpy()
            truncated = truncated.cpu().numpy()
        new_info["qpos"] = qpos
        new_info["qvel"] = qvel
        return observation, reward, terminated, truncated, new_info

    def close(self):
        return self.base_env.close()

    # handle the history
    def _add_history_to_observation(self, observation: dict[str, torch.Tensor], to_numpy: bool = True) -> dict[str, torch.Tensor]:
        if self.context_length and self.history_handler is not None:
            # add history to the observation
            # this is used to create the context for the model
            _history = {"observation": {}, "action": self.history_handler.query("action", filter_by_length=True)}
            for k in observation.keys():
                _history["observation"][k] = self.history_handler.query(k, filter_by_length=True)

            if to_numpy:
                _history = tree_map(lambda x: x.cpu().numpy(), _history)
            observation["history"] = _history
        return observation

    def _update_history(self, reset_ids, observation, action):
        if self.history_handler is not None:
            if reset_ids is not None and len(reset_ids) > 0:
                self.history_handler.reset(reset_ids=reset_ids)
            if observation is not None:
                for key, value in observation.items():
                    if key != "history":
                        self.history_handler.add(key, value)
            if action is not None:
                self.history_handler.add("action", action)

    def render(self):
        return self.base_env.simulator.render()


_ISAAC_SIM_INITIALIZED = False


def instantiate_isaac_sim(num_envs: int, enable_cameras: bool = False, headless: bool = True):
    global _ISAAC_SIM_INITIALIZED
    if _ISAAC_SIM_INITIALIZED:
        return
    # Import things lazily
    import argparse
    import shutil
    from pathlib import Path

    import isaaclab
    from isaaclab.app import AppLauncher
    
    parser = argparse.ArgumentParser(description="")
    AppLauncher.add_app_launcher_args(parser)

    args_cli, _ = parser.parse_known_args()
    args_cli.num_envs = num_envs
    args_cli.enable_cameras = enable_cameras
    args_cli.headless = headless

    dest_path = Path(isaaclab.__file__) / "apps"
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    if args_cli.enable_cameras and args_cli.headless:
        source_file = current_file_dir_path + "/../apps/phc.isaaclab.python.headless.rendering.kit"
        shutil.copy(source_file, dest_path)
        args_cli.experience = dest_path + "/phc.isaaclab.python.headless.rendering.kit"
    elif args_cli.enable_cameras and not args_cli.headless:
        source_file = current_file_dir_path + "/../apps/phc.isaaclab.python.rendering.kit"
        shutil.copy(source_file, dest_path)
        args_cli.experience = dest_path + "/phc.isaaclab.python.rendering.kit"

    app_launcher = AppLauncher(args_cli)
    _ = app_launcher.app
    _ISAAC_SIM_INITIALIZED = True


_humanoidverse_env_singleton = None


class HumanoidVerseIsaacConfig(BaseConfig):
    name: tp.Literal["humanoidverse_isaac"] = "humanoidverse_isaac"

    device: str = "cuda:0"

    lafan_tail_path: str

    enable_cameras: bool = False
    camera_render_save_dir: str = "isaac_videos"

    # Max episode length in seconds
    max_episode_length_s: float | None = None

    # Set all obs noise_scales to 0.0 (overrides config)
    disable_obs_noise: bool = False
    # Disable all domain randomization (overrides config)
    disable_domain_randomization: bool = False

    # Relative path inside the humanoidverse/config directory
    relative_config_path: str = HYDRA_CONFIG_REL_PATH

    include_last_action: bool = True
    hydra_overrides: tp.List[str] = pydantic.Field(default_factory=list)

    context_length: int | None = None
    include_dr_info: bool = False
    included_dr_obs_names: tp.List[str] | None = None
    include_history_actor: bool = False
    include_history_noaction: bool = False

    make_config_g1env_compatible: bool = False

    root_height_obs: bool = False

    def build(self, num_envs: int = 1) -> tp.Tuple[HumanoidVerseVectorEnv, tp.Any]:
        global _humanoidverse_env_singleton
        assert num_envs >= 1

        if _humanoidverse_env_singleton is not None:
            # If the singleton exists, we need to check if it matches the requested num_envs and cfg
            if num_envs != _humanoidverse_env_singleton.num_envs:
                raise ValueError(
                    f"HumanoidVerse env was already created with num_envs={_humanoidverse_env_singleton.num_envs}, "
                    f"but requested num_envs={num_envs}. You can only spin up one HumanoidVerse per Python instance"
                )
            elif self != _humanoidverse_env_singleton._creation_config:
                raise ValueError(
                    f"HumanoidVerse env was already created with different config: "
                    f"{_humanoidverse_env_singleton._creation_config} vs {self}. You can only spin up one HumanoidVerse per Python instance"
                )
            return _humanoidverse_env_singleton, {}

        # Use config from humanoidverse to create the environment
        # however, we need to make sure we use isaacsim instead of isaacgym
        # --> create new file with that single line changed
        hydra_overrides = self.hydra_overrides.copy()

        with hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_DIR):
            cfg = hydra.compose(config_name=self.relative_config_path, overrides=hydra_overrides or [])
        unresolved_conf = OmegaConf.to_container(cfg, resolve=False)

        # Add custom resolvers used in the configs
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))

        # We need to manually fix some paths and values
        cfg.num_envs = num_envs
        cfg.exp_base = "__no_exp_base__"
        if not any(["env.config.headless" in k for k in hydra_overrides]):
            cfg.env.config.headless = True
        cfg.env.config.save_rendering_dir = self.camera_render_save_dir
        cfg.robot.asset.asset_root = cfg.robot.asset.asset_root.replace("humanoidverse", HUMANOIDVERSE_DIR)
        cfg.robot.motion.asset.assetRoot = cfg.robot.motion.asset.assetRoot.replace("humanoidverse", HUMANOIDVERSE_DIR)
        cfg.robot.motion.motion_file = self.lafan_tail_path

        # This sets obs/action dims etc
        pre_process_config(cfg)

        OmegaConf.set_struct(cfg, False)

        if self.make_config_g1env_compatible:
            cfg.robot.motion.num_extend_bodies = 0
            cfg.robot.motion.extend_config = []
            cfg.robot.motion.motion_tracking_link = []
            cfg.robot.motion.upper_body_link = [
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
            ]
            cfg.robot.motion.joint_matches = [
                ["pelvis", "Pelvis"],
                ["left_hip_pitch_link", "L_Hip"],
                ["left_knee_link", "L_Knee"],
                ["left_ankle_roll_link", "L_Ankle"],
                ["right_hip_pitch_link", "R_Hip"],
                ["right_knee_link", "R_Knee"],
                ["right_ankle_roll_link", "R_Ankle"],
                ["left_shoulder_roll_link", "L_Shoulder"],
                ["left_elbow_link", "L_Elbow"],
                ["right_shoulder_roll_link", "R_Shoulder"],
                ["right_elbow_link", "R_Elbow"],
            ]

        if self.enable_cameras:
            cfg.simulator.enable_cameras = True

        if self.disable_obs_noise:
            # Set all noise scales to 0.0
            for key in cfg.obs.noise_scales.keys():
                cfg.obs.noise_scales[key] = 0.0

        cfg.obs.root_height_obs = self.root_height_obs

        if self.disable_domain_randomization:
            # Disable all domain randomization (NOTE need to keep up-to-date with domain randomization configs)
            cfg.domain_rand.randomize_ctrl_delay = False
            cfg.domain_rand.randomize_pd_gain = False
            cfg.domain_rand.randomize_base_com = False
            cfg.domain_rand.randomize_link_mass = False
            cfg.domain_rand.randomize_friction = False
            cfg.domain_rand.randomize_torque_rfi = False
            cfg.domain_rand.randomize_rfi_lim = False
            cfg.domain_rand.randomize_push_robots = False
            cfg.domain_rand.randomize_default_dof_pos = False

        assert cfg.env.config.termination.terminate_when_close_to_dof_pos_limit is False
        assert cfg.env.config.termination.terminate_when_close_to_dof_vel_limit is False
        assert cfg.env.config.termination.terminate_when_close_to_torque_limit is False
        assert cfg.env.config.termination.terminate_by_contact is False
        assert cfg.env.config.termination.terminate_by_gravity is False
        assert cfg.env.config.termination.terminate_by_low_height is False
        assert cfg.env.config.termination.terminate_when_motion_end is False
        assert cfg.env.config.termination.terminate_when_motion_far is False
        assert cfg.env.config.robot.control.normalize_action_to == cfg.env.config.robot.control.action_clip_value, (
            "normalize_action_to and action_clip_value must be the same"
        )

        if self.max_episode_length_s is not None:
            cfg.env.config.max_episode_length_s = self.max_episode_length_s

        simulator_type = cfg.simulator["_target_"].split(".")[-1]
        if simulator_type == "IsaacSim":
            instantiate_isaac_sim(num_envs, enable_cameras=self.enable_cameras, headless=cfg.env.config.headless)
        isaac_env = LeggedRobotMotions(cfg.env.config, device=self.device)

        env = HumanoidVerseVectorEnv(
            isaac_env,
            include_last_action=self.include_last_action,
            include_dr_info=self.include_dr_info,
            included_dr_obs_names=self.included_dr_obs_names,
            context_length=self.context_length,
            include_history_actor=self.include_history_actor,
            include_history_noaction=self.include_history_noaction,
        )

        env._creation_config = self

        _humanoidverse_env_singleton = env

        return env, {"unresolved_conf": unresolved_conf}
