import numpy as np
import torch
from gymnasium import spaces

from humanoidverse.agents.fb.agent import FBAgentConfig, FBAgentTrainConfig
from humanoidverse.agents.fb.model import FBModelArchiConfig, FBModelConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import ActorArchiConfig, BackwardArchiConfig, ForwardArchiConfig
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig


def _tiny_agent_config() -> FBAgentConfig:
    obs_filter = DictInputFilterConfig(name="DictInputFilterConfig", key="state")
    return FBAgentConfig(
        name="FBAgent",
        model=FBModelConfig(
            name="FBModel",
            device="cpu",
            archi=FBModelArchiConfig(
                name="FBModelArchiConfig",
                z_dim=4,
                norm_z=True,
                f=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=16,
                    model="simple",
                    hidden_layers=1,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=obs_filter,
                ),
                b=BackwardArchiConfig(
                    name="BackwardArchi",
                    hidden_dim=16,
                    hidden_layers=1,
                    norm=True,
                    input_filter=obs_filter,
                ),
                actor=ActorArchiConfig(
                    name="actor",
                    model="simple",
                    hidden_dim=16,
                    hidden_layers=1,
                    embedding_layers=2,
                    input_filter=obs_filter,
                ),
            ),
            obs_normalizer=ObsNormalizerConfig(
                name="ObsNormalizerConfig",
                normalizers={
                    "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                },
            ),
            seq_length=1,
            actor_std=0.05,
            amp=False,
        ),
        train=FBAgentTrainConfig(
            name="FBAgentTrainConfig",
            batch_size=8,
            z_buffer_size=16,
            lr_f=1e-3,
            lr_b=1e-3,
            lr_actor=1e-3,
            use_mix_rollout=False,
            rollout_expert_trajectories=False,
        ),
        cudagraphs=False,
        compile=False,
    )


def test_tiny_agent_backward_step_save_load_and_inference(tmp_path):
    torch.manual_seed(7)
    obs_space = spaces.Dict(
        {
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6,),
                dtype=np.float32,
            )
        }
    )
    action_dim = 5
    agent = _tiny_agent_config().build(obs_space=obs_space, action_dim=action_dim)

    obs = {"state": torch.randn(8, 6)}
    z = agent._model.sample_z(8, device="cpu")
    dist = agent._model._actor(obs, z, agent._model.cfg.actor_std)
    action = dist.sample(clip=agent.cfg.train.stddev_clip)
    f_pred = agent._model._forward_map(obs, z, action)
    b_pred = agent._model._backward_map(obs)
    loss = action.square().mean() + f_pred.square().mean() + b_pred.square().mean()

    agent.actor_optimizer.zero_grad(set_to_none=True)
    agent.forward_optimizer.zero_grad(set_to_none=True)
    agent.backward_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    agent.actor_optimizer.step()
    agent.forward_optimizer.step()
    agent.backward_optimizer.step()

    checkpoint_dir = tmp_path / "checkpoint"
    agent.save(str(checkpoint_dir))
    loaded = agent.__class__.load(str(checkpoint_dir), device="cpu")

    with torch.no_grad():
        loaded_action = loaded.act(obs=obs, z=z, mean=True)

    assert loaded_action.shape == (8, action_dim)
    assert torch.isfinite(loaded_action).all()
    assert (checkpoint_dir / "config.json").exists()
    assert (checkpoint_dir / "optimizers.pth").exists()
    assert (checkpoint_dir / "model" / "model.safetensors").exists()
