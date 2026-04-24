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
    StructuredBackwardMap,
    eval_mode,
)
from ..normalizers import ObsNormalizerConfig
from ..pytree_utils import tree_get_batch_size


class FBModelArchiConfig(BaseConfig):
    z_dim: int = 100
    norm_z: bool = True
    z_hand_low: list[float] = pydantic.Field(default_factory=lambda: [-1.0, -1.0, -1.0])
    z_hand_high: list[float] = pydantic.Field(default_factory=lambda: [1.0, 1.0, 1.0])
    z_hand_obs_key: str = "privileged_state"
    z_hand_obs_idxs: list[int] = pydantic.Field(default_factory=list)
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
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @property
    def z_hand_dim(self) -> int:
        return getattr(self.cfg.archi.b, "z_hand_dim", 0)

    @property
    def z_body_dim(self) -> int:
        return self.cfg.archi.z_dim - self.z_hand_dim

    @property
    def has_structured_z(self) -> bool:
        return self.z_hand_dim > 0

    @property
    def z_hand_scale(self) -> float:
        if not self.has_structured_z:
            return 1.0
        return getattr(self.cfg.archi.b, "z_hand_scale", 1.0)

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
        if self.has_structured_z:
            z_body = torch.randn((size, self.z_body_dim), dtype=torch.float32, device=device)
            z_body = self.project_z_body(z_body)
            hand_low = torch.as_tensor(self.cfg.archi.z_hand_low, dtype=torch.float32, device=device)
            hand_high = torch.as_tensor(self.cfg.archi.z_hand_high, dtype=torch.float32, device=device)
            if hand_low.numel() != self.z_hand_dim or hand_high.numel() != self.z_hand_dim:
                raise ValueError(
                    "z_hand_low and z_hand_high must match z_hand_dim. "
                    f"Got {hand_low.numel()} and {hand_high.numel()} for z_hand_dim={self.z_hand_dim}."
                )
            z_hand = hand_low + (hand_high - hand_low) * torch.rand((size, self.z_hand_dim), dtype=torch.float32, device=device)
            return self.merge_z(z_hand, z_body)
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def encode_z_hand(self, z_hand: torch.Tensor) -> torch.Tensor:
        if not self.has_structured_z:
            return z_hand
        return z_hand * self.z_hand_scale

    def decode_z_hand(self, z_hand: torch.Tensor) -> torch.Tensor:
        if not self.has_structured_z:
            return z_hand
        return z_hand / self.z_hand_scale

    def split_z(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_structured_z:
            return z[..., :0], z
        return z[..., : self.z_hand_dim], z[..., self.z_hand_dim :]

    def extract_z_hand(self, z: torch.Tensor, decode: bool = False) -> torch.Tensor:
        z_hand, _ = self.split_z(z)
        if decode:
            return self.decode_z_hand(z_hand)
        return z_hand

    def extract_z_body(self, z: torch.Tensor) -> torch.Tensor:
        _, z_body = self.split_z(z)
        return z_body

    def merge_z(self, z_hand: torch.Tensor, z_body: torch.Tensor, z_hand_encoded: bool = False) -> torch.Tensor:
        if not self.has_structured_z:
            return z_body
        if not z_hand_encoded:
            z_hand = self.encode_z_hand(z_hand)
        return torch.cat([z_hand, z_body], dim=-1)

    def project_z_body(self, z_body: torch.Tensor) -> torch.Tensor:
        if self.cfg.archi.norm_z:
            z_body = math.sqrt(z_body.shape[-1]) * F.normalize(z_body, dim=-1)
        return z_body

    def project_z(self, z):
        if not self.has_structured_z:
            return self.project_z_body(z)
        z_hand, z_body = self.split_z(z)
        z_body = self.project_z_body(z_body)
        return self.merge_z(z_hand, z_body, z_hand_encoded=True)

    def extract_hand_pos(self, obs: torch.Tensor | dict[str, torch.Tensor], encode: bool = False) -> torch.Tensor:
        if not self.has_structured_z:
            raise ValueError("extract_hand_pos is only available when z_hand_dim > 0.")
        if not isinstance(obs, dict):
            raise TypeError("Structured z hand extraction expects a dictionary observation.")
        hand_obs_key = self.cfg.archi.z_hand_obs_key
        if hand_obs_key not in obs:
            raise KeyError(f"Observation is missing key '{hand_obs_key}' required to extract z_hand targets.")
        if len(self.cfg.archi.z_hand_obs_idxs) != self.z_hand_dim:
            raise ValueError(
                f"z_hand_obs_idxs must have length {self.z_hand_dim}, got {len(self.cfg.archi.z_hand_obs_idxs)}."
            )
        z_hand = obs[hand_obs_key][..., self.cfg.archi.z_hand_obs_idxs]
        if encode:
            z_hand = self.encode_z_hand(z_hand)
        return z_hand

    def backward_components(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        target: bool = False,
        normalize_obs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_structured_z:
            raise ValueError("backward_components is only available when z_hand_dim > 0.")
        backward_map = self._target_backward_map if target else self._backward_map
        if not isinstance(backward_map, StructuredBackwardMap):
            raise TypeError("Structured z requires StructuredBackwardMap.")
        if normalize_obs:
            obs = self._normalize(obs)
        return backward_map.forward_components(obs)

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
        if self.has_structured_z:
            return z
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        if not self.has_structured_z:
            for step in range(z.shape[0]):
                end_idx = min(step + self.cfg.seq_length, z.shape[0])
                z[step] = z[step:end_idx].mean(dim=0)
            return self.project_z(z)

        z_hand = self.extract_z_hand(z)
        z_body = self.extract_z_body(z)
        z_body = z_body.clone()
        for step in range(z_body.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z_body.shape[0])
            z_body[step] = z_body[step:end_idx].mean(dim=0)
        z_body = self.project_z_body(z_body)
        return self.merge_z(z_hand, z_body, z_hand_encoded=True)
