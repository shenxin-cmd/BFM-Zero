# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import typing as tp

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..base_model import BaseModel, BaseModelConfig
from ..nn_filter_models import (
    ActorFilterArchiConfig,
    BackwardFilterArchiConfig,
    ForwardFilterArchiConfig,
    ResidualActorFilterArchiConfig,
    SimpleActorFilterArchiConfig,
)
from ..nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    ForwardArchiConfig,
    ResidualActorArchiConfig,
    SimpleActorArchiConfig,
    eval_mode,
)
from ..normalizers import ObsNormalizerConfig
from ..pytree_utils import tree_get_batch_size


class FBModelArchiConfig(BaseConfig):
    z_dim: int = 100
    norm_z: bool = True
    right_hand_z_start: int = 0
    right_hand_z_dim: int = 32
    z_norm_mode: tp.Literal["global", "per_part"] = "global"
    f: ForwardArchiConfig | ForwardFilterArchiConfig = pydantic.Field(ForwardArchiConfig(), discriminator="name")
    b: BackwardArchiConfig | BackwardFilterArchiConfig = pydantic.Field(BackwardArchiConfig(), discriminator="name")
    # Because of the "name" attribute, these two can be chosen between via strings easily
    actor: (
        ActorArchiConfig
        | ActorFilterArchiConfig
        | SimpleActorArchiConfig
        | ResidualActorArchiConfig
        | SimpleActorFilterArchiConfig
        | ResidualActorFilterArchiConfig
    ) = pydantic.Field(SimpleActorArchiConfig(), discriminator="name")


class FBModelConfig(BaseModelConfig):
    name: tp.Literal["FBModel"] = "FBModel"

    archi: FBModelArchiConfig = FBModelArchiConfig()
    obs_normalizer: ObsNormalizerConfig = ObsNormalizerConfig()
    inference_batch_size: int = 500_000
    seq_length: int = 1
    actor_std: float = 0.2
    amp: bool = False

    def build(self, obs_space, action_dim) -> "FBModel":
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBModel


class FBModel(BaseModel):
    config_class = FBModelConfig

    def __init__(self, obs_space, action_dim, cfg: FBModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg: FBModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device
        self.amp_dtype = torch.bfloat16

        # create networks
        self._backward_map = arch.b.build(obs_space, arch.z_dim)
        self._forward_map = arch.f.build(obs_space, arch.z_dim, action_dim)
        self._actor = arch.actor.build(obs_space, arch.z_dim, action_dim)
        self._right_hand_head = torch.nn.Sequential(
            torch.nn.Linear(arch.right_hand_z_dim, arch.right_hand_z_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(arch.right_hand_z_dim * 2, 7),
        )
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)

    def _normalize(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, std: float):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._actor(self._normalize(obs), z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            if self.cfg.archi.z_norm_mode == "per_part":
                right_start = self.cfg.archi.right_hand_z_start
                right_dim = self.cfg.archi.right_hand_z_dim
                right_end = right_start + right_dim
                if right_dim <= 0 or right_end > z.shape[-1]:
                    raise ValueError("Invalid right hand z partition.")

                right_part = z[..., right_start:right_end]
                rest_left = z[..., :right_start]
                rest_right = z[..., right_end:]
                z_right = math.sqrt(right_part.shape[-1]) * F.normalize(right_part, dim=-1)
                if rest_left.shape[-1] + rest_right.shape[-1] > 0:
                    rest = torch.cat([rest_left, rest_right], dim=-1)
                    z_rest = math.sqrt(rest.shape[-1]) * F.normalize(rest, dim=-1)
                    z = torch.cat([z_rest[..., :right_start], z_right, z_rest[..., right_start:]], dim=-1)
                else:
                    z = z_right
            else:
                z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def split_z(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        right_start = self.cfg.archi.right_hand_z_start
        right_dim = self.cfg.archi.right_hand_z_dim
        right_end = right_start + right_dim
        if right_dim <= 0 or right_end > z.shape[-1]:
            raise ValueError("Invalid right hand z partition.")
        return {
            "right_hand": z[..., right_start:right_end],
            "rest": torch.cat([z[..., :right_start], z[..., right_end:]], dim=-1),
        }

    def merge_z(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        right_start = self.cfg.archi.right_hand_z_start
        right_dim = self.cfg.archi.right_hand_z_dim
        right_end = right_start + right_dim
        z_dim = self.cfg.archi.z_dim

        z_right = parts["right_hand"]
        z_rest = parts["rest"]
        if z_right.shape[-1] != right_dim:
            raise ValueError(f"right_hand part dim mismatch: expected {right_dim}, got {z_right.shape[-1]}")
        if z_rest.shape[-1] != (z_dim - right_dim):
            raise ValueError(f"rest part dim mismatch: expected {z_dim - right_dim}, got {z_rest.shape[-1]}")
        if z_right.shape[:-1] != z_rest.shape[:-1]:
            raise ValueError("right_hand and rest parts must have matching batch dimensions.")

        return torch.cat([z_rest[..., :right_start], z_right, z_rest[..., right_start:]], dim=-1)

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()  # TODO we upcast to float32 to make sure the action can be converted to numpy later

    def reward_inference(
        self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            batch_size = tree_get_batch_size(next_obs)
            num_batches = int(np.ceil(batch_size / self.cfg.inference_batch_size))
            z = 0
            wr = reward if weight is None else reward * weight
            for i in range(num_batches):
                start_idx, end_idx = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size
                next_obs_slice = tree_map(lambda x: x[start_idx:end_idx].to(self.device), next_obs)
                B = self.backward_map(next_obs_slice)
                z += torch.matmul(wr[start_idx:end_idx].to(self.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)
