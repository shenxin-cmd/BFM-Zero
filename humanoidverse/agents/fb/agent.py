# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import json
import pickle
from pathlib import Path
from typing import Dict, Literal, Tuple

import safetensors
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..envs.utils.gym_spaces import json_to_space, space_to_json
from ..misc.zbuffer import ZBuffer
from ..nn_models import _soft_update_params, eval_mode, weight_init
from .model import FBModel, FBModelConfig


class FBAgentTrainConfig(BaseConfig):
    lr_f: float = 1e-4
    lr_b: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fb_target_tau: float = 0.01
    ortho_coef: float = 1.0
    train_goal_ratio: float = 0.5
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    batch_size: int = 1024
    discount: float = 0.99
    use_mix_rollout: bool = False
    update_z_every_step: int = 150
    z_buffer_size: int = 10000
    rollout_expert_trajectories: bool = False
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.25


class FBAgentConfig(BaseConfig):
    name: Literal["FBAgent"] = "FBAgent"
    model: FBModelConfig
    train: FBAgentTrainConfig
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim):
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBAgent


class FBAgent:
    config_class = FBAgentConfig

    def __init__(self, obs_space, action_dim, cfg: FBAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        self.fb_target_tau = float(min(max(self.cfg.train.fb_target_tau, 0), 1))
        self._model: FBModel = self.cfg.model.build(obs_space, action_dim)
        self.setup_training()
        self.setup_compile()
        # This is just to be sure? I think it should not change since build
        self._model.to(self.device)

        self.env_idx_with_expert_rollout = None

    @property
    def device(self):
        return self._model.device

    @property
    def optimizer_dict(self):
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
        }

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map.parameters(),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map.parameters(),
            lr=self.cfg.train.lr_f,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

        # prepare parameter list
        self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.total_z_dim, self._model.device)

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_fb = torch.compile(self.update_fb, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.update_actor = torch.compile(self.update_actor, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_fb = CudaGraphModule(self.update_fb, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor | dict[str, torch.Tensor] | None = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)

            if train_goal is not None:
                perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
                train_goal = tree_map(lambda x: x[perm], train_goal)
                goals = self._model._backward_map(train_goal)
                goals = self._model.project_z(goals)
                mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) < self.cfg.train.train_goal_ratio
                z = torch.where(mask, goals, z)
        return z

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        obs, action, next_obs, terminated = (
            batch["observation"],
            batch["action"],
            batch["next"]["observation"],
            batch["next"]["terminated"],
        )
        discount = self.cfg.train.discount * ~terminated

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)

        torch.compiler.cudagraph_mark_step_begin()
        z = self.sample_mixed_z(train_goal=next_obs).clone()
        self.z_buffer.add(z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        torch.compiler.cudagraph_mark_step_begin()
        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=next_obs,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        metrics.update(
            self.update_actor(
                obs=obs,
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.fb_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.fb_target_tau)

        return metrics

    def sample_action_from_norm_obs(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)
        return action

    def _fb_loss_single(
        self,
        Fs: torch.Tensor,
        B: torch.Tensor,
        discount: torch.Tensor,
        target_M: torch.Tensor,
    ):
        """Compute FB loss for one z sub-space (or the full z in non-split mode).

        Args:
            Fs: num_parallel × batch × z_sub_dim
            B:  batch × z_sub_dim
            discount: batch × 1
            target_M: batch × batch  (pre-computed pessimistic target)
        Returns:
            (fb_loss_scalar, Ms, fb_diag, fb_offdiag)
        """
        Ms = torch.matmul(Fs, B.T)  # num_parallel × batch × batch
        diff = Ms - discount * target_M
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        return fb_offdiag + fb_diag, Ms, fb_diag, fb_offdiag

    def _orth_loss_single(self, B: torch.Tensor):
        """Orthonormality loss for one B sub-space."""
        Cov = torch.matmul(B, B.T)
        diag = -Cov.diag().mean()
        offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        return offdiag + diag, diag, offdiag

    def update_fb(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        is_split = self.cfg.model.archi.is_split_mode
        z_body_dim = self.cfg.model.archi.z_body_dim if is_split else 0

        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                next_action = self.sample_action_from_norm_obs(next_obs, z)
                target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # num_parallel x batch x z_dim
                target_B = self._model._target_backward_map(goal)  # batch x z_dim

                if is_split:
                    target_Fs_body = target_Fs[..., :z_body_dim]
                    target_Fs_hand = target_Fs[..., z_body_dim:]
                    target_B_body = target_B[:, :z_body_dim]
                    target_B_hand = target_B[:, z_body_dim:]
                    target_Ms_body = torch.matmul(target_Fs_body, target_B_body.T)
                    target_Ms_hand = torch.matmul(target_Fs_hand, target_B_hand.T)
                    _, _, target_M_body = self.get_targets_uncertainty(target_Ms_body, self.cfg.train.fb_pessimism_penalty)
                    _, _, target_M_hand = self.get_targets_uncertainty(target_Ms_hand, self.cfg.train.fb_pessimism_penalty)
                    # keep a combined target_M for metrics only
                    target_Ms_all = torch.matmul(target_Fs, target_B.T)
                    _, _, target_M = self.get_targets_uncertainty(target_Ms_all, self.cfg.train.fb_pessimism_penalty)
                else:
                    target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
                    _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)

            # compute FB loss
            Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
            B = self._model._backward_map(goal)  # batch x z_dim

            if is_split:
                hand_weight = self.cfg.model.archi.z_body_dim / self.cfg.model.archi.z_hand_dim  # ≈6.25

                Fs_body = Fs[..., :z_body_dim]
                Fs_hand = Fs[..., z_body_dim:]
                B_body = B[:, :z_body_dim]
                B_hand = B[:, z_body_dim:]

                fb_loss_body, Ms_body, fb_diag_body, fb_offdiag_body = self._fb_loss_single(
                    Fs_body, B_body, discount, target_M_body
                )
                fb_loss_hand, Ms_hand, fb_diag_hand, fb_offdiag_hand = self._fb_loss_single(
                    Fs_hand, B_hand, discount, target_M_hand
                )
                fb_loss = fb_loss_body + hand_weight * fb_loss_hand

                # orthonormality losses (independent per sub-space)
                orth_loss_body, orth_diag_body, orth_offdiag_body = self._orth_loss_single(B_body)
                orth_loss_hand, orth_diag_hand, orth_offdiag_hand = self._orth_loss_single(B_hand)
                orth_loss = orth_loss_body + orth_loss_hand
                fb_loss += self.cfg.train.ortho_coef * orth_loss

                # reuse first parallel slice for metrics
                Ms = torch.cat([Ms_body, Ms_hand], dim=-1)
                fb_diag = fb_diag_body + hand_weight * fb_diag_hand
                fb_offdiag = fb_offdiag_body + hand_weight * fb_offdiag_hand
                orth_loss_diag = orth_diag_body + orth_diag_hand
                orth_loss_offdiag = orth_offdiag_body + orth_offdiag_hand
            else:
                Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch
                diff = Ms - discount * target_M
                fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
                fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
                fb_loss = fb_offdiag + fb_diag

                Cov = torch.matmul(B, B.T)
                orth_loss_diag = -Cov.diag().mean()
                orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
                orth_loss = orth_loss_offdiag + orth_loss_diag
                fb_loss += self.cfg.train.ortho_coef * orth_loss

            q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
            if q_loss_coef is not None:
                with torch.no_grad():
                    if is_split:
                        z_body = z[:, :z_body_dim]
                        z_hand = z[:, z_body_dim:]
                        next_Qs = (target_Fs_body * z_body).sum(dim=-1) + hand_weight * (target_Fs_hand * z_hand).sum(dim=-1)
                    else:
                        next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.fb_pessimism_penalty)
                    with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=False):
                        cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                    B_inv_conv = torch.linalg.solve(cov, B, left=False)
                    implicit_reward = (B_inv_conv * z).sum(dim=-1)  # batch
                    target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                    expanded_targets = target_Q.expand(Fs.shape[0], -1)
                if is_split:
                    Qs_body = (Fs_body * z_body).sum(dim=-1)
                    Qs_hand = (Fs_hand * z_hand).sum(dim=-1)
                    Qs = Qs_body + hand_weight * Qs_hand
                else:
                    Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
                q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
                fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }
            if is_split:
                output_metrics.update({
                    # z sub-space norms
                    "z_body_norm": torch.norm(z[:, :z_body_dim], dim=-1).mean(),
                    "z_hand_norm": torch.norm(z[:, z_body_dim:], dim=-1).mean(),
                    # B sub-space norms
                    "B_body_norm": torch.norm(B_body, dim=-1).mean(),
                    "B_hand_norm": torch.norm(B_hand, dim=-1).mean(),
                    # FB losses per sub-space
                    "fb_body_loss": fb_loss_body,
                    "fb_hand_loss": fb_loss_hand,
                    "fb_total_loss": fb_loss,
                    # orthonormality losses per sub-space
                    "orth_body_loss": orth_loss_body,
                    "orth_hand_loss": orth_loss_hand,
                    "orth_body_loss_diag": orth_diag_body,
                    "orth_hand_loss_diag": orth_diag_hand,
                    "orth_body_loss_offdiag": orth_offdiag_body,
                    "orth_hand_loss_offdiag": orth_offdiag_hand,
                })
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        return self.update_td3_actor(obs=obs, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(
        self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, clip_grad_norm: float | None
    ) -> Dict[str, torch.Tensor]:
        is_split = self.cfg.model.archi.is_split_mode
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)
            Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim

            if is_split:
                z_body_dim = self.cfg.model.archi.z_body_dim
                hand_weight = self.cfg.model.archi.z_body_dim / self.cfg.model.archi.z_hand_dim
                z_body = z[:, :z_body_dim]
                z_hand = z[:, z_body_dim:]
                Qs_body = (Fs[..., :z_body_dim] * z_body).sum(-1)  # num_parallel x batch
                Qs_hand = (Fs[..., z_body_dim:] * z_hand).sum(-1)
                _, _, Q_body = self.get_targets_uncertainty(Qs_body, self.cfg.train.actor_pessimism_penalty)
                _, _, Q_hand = self.get_targets_uncertainty(Qs_hand, self.cfg.train.actor_pessimism_penalty)
                Q = Q_body + hand_weight * Q_hand
            else:
                Qs = (Fs * z).sum(-1)  # num_parallel x batch
                _, _, Q = self.get_targets_uncertainty(Qs, self.cfg.train.actor_pessimism_penalty)

            actor_loss = -Q.mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        output_metrics = {"actor_loss": actor_loss.detach(), "q": Q.mean().detach()}
        if is_split:
            output_metrics.update({
                "q_body": Q_body.mean().detach(),
                "q_hand": Q_hand.mean().detach(),
                "q_total": Q.mean().detach(),
            })
        return output_metrics

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc
    
    def _sample_tracking_z(self, replay_buffer, batch_dim, traj_length):
        batch = replay_buffer["expert_slicer"].sample(batch_dim * traj_length, seq_length=traj_length)  # N*T x obs_dim
        z = self._model.backward_map(batch["next"]["observation"])  # NT x z_dim
        z = z.view(batch_dim, traj_length, z.shape[-1])  # N x T x z_dim
        for step in range(traj_length):
            end_idx = min(step + self.cfg.model.seq_length, traj_length)
            z[:, step] = z[:, step:end_idx].mean(dim=1)
        return self._model.project_z(z)  # N x T x z_dim

    def maybe_update_rollout_context(self, z: torch.Tensor | None, step_count: torch.Tensor, replay_buffer: None = None) -> torch.Tensor:
        # get mask for environmets where we need to change z
        if z is not None:
            mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=self._model.device)
            else:
                new_z = self._model.sample_z(z.shape[0], device=self._model.device)
            z = torch.where(mask_reset_z, new_z, z.to(self._model.device))
            if self.cfg.train.rollout_expert_trajectories:
                idxs = step_count % self.cfg.train.rollout_expert_trajectories_length
                if torch.any(idxs == 0):
                    n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage*step_count.shape[0])
                    self.env_idx_with_expert_rollout = torch.randint(0, step_count.shape[0], size=(n_elem,), device=self._model.device)
                    self.tracking_z = self._sample_tracking_z(replay_buffer, n_elem, self.cfg.train.rollout_expert_trajectories_length)  # N x T x z_dim
                mod_time = idxs[self.env_idx_with_expert_rollout].ravel()
                z[self.env_idx_with_expert_rollout] = self.tracking_z[torch.arange(len(self.env_idx_with_expert_rollout), device=self._model.device), mod_time]
        else:
            z = self._model.sample_z(step_count.shape[0], device=self._model.device)
            if self.cfg.train.rollout_expert_trajectories:
                n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage*step_count.shape[0])
                self.env_idx_with_expert_rollout = torch.randint(0, step_count.shape[0], size=(n_elem,), device=self._model.device)
                self.tracking_z = self._sample_tracking_z(replay_buffer, n_elem, self.cfg.train.rollout_expert_trajectories_length)  # N x T x z_dim
                z[self.env_idx_with_expert_rollout] = self.tracking_z[:, 0]
        return z

    @classmethod
    def load(cls, path: str, device: str | None = None):
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if device is not None:
            loaded_config["model"]["device"] = device

        if (path / "init_kwargs.pkl").exists():
            # Load arguments from a pickle file
            with (path / "init_kwargs.pkl").open("rb") as f:
                args = pickle.load(f)
            obs_space = args["obs_space"]
            action_dim = args["action_dim"]
        else:
            # load argeuments from a json file
            with (path / "init_kwargs.json").open("r") as f:
                args = json.load(f)
            obs_space = json_to_space(args["obs_space"])
            action_dim = args["action_dim"]

        config = cls.config_class(**loaded_config)
        agent = config.build(obs_space, action_dim)
        optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True, map_location=device)
        for k, v in optimizers.items():
            getattr(agent, k).load_state_dict(v)
        safetensors.torch.load_model(agent._model, path / "model/model.safetensors", device=device, strict=False)
        agent._model.train()
        agent._model.requires_grad_(True)
        return agent

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        json_dump = self.cfg.model_dump()
        with (output_folder / "config.json").open("w+") as f:
            json.dump(json_dump, f, indent=4)
        # save optimizer
        torch.save(
            self.optimizer_dict,
            output_folder / "optimizers.pth",
        )
        # save model
        model_folder = output_folder / "model"
        model_folder.mkdir(exist_ok=True)
        self._model.save(output_folder=str(model_folder))

        # Save the arguments required to create this agent (in addition to the config)
        init_kwargs = {
            "obs_space": space_to_json(self.obs_space),
            "action_dim": self.action_dim,
        }
        with (output_folder / "init_kwargs.json").open("w") as f:
            json.dump(init_kwargs, f, indent=4)
