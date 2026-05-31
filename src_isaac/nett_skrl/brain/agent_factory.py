"""skrl agent construction for NETT brains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

from .experiment import apply_experiment_cfg, attach_tensorboard_tracking
from .models import build_models_for_algorithm
from .registry import algorithm_spec


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def encoder_cfg_for(brain) -> dict[str, Any]:
    return brain.encoder_cfg.as_kwargs()


def default_algorithm_cfg(brain):
    """Return a fresh skrl algorithm config from the algorithm registry."""
    return algorithm_spec(brain.algorithm).cfg_cls()


def memory_size_for(brain) -> int:
    """Use rollout-sized memory for on-policy skrl agents."""
    return brain.algorithm_cfg.agent_memory_size()


def build_agents(brain, env, device: torch.device, *, config=None) -> list:
    """Build one skrl agent per brain, each owning a contiguous env scope."""
    obs_space, act_space = env.observation_space, env.action_space
    encoder_kwargs = encoder_cfg_for(brain)
    spec = algorithm_spec(brain.algorithm)
    num_agents = int(
        getattr(config, "num_brains", None)
        or getattr(brain, "num_brains", None)
        or env.num_envs
    )
    if num_agents < 1:
        raise ValueError(f"num_brains must be >= 1; got {num_agents}.")
    if env.num_envs % num_agents != 0:
        raise ValueError(
            f"env.num_envs ({env.num_envs}) must be divisible by num_brains ({num_agents})."
        )
    scope = env.num_envs // num_agents
    agents = []
    for brain_id in range(num_agents):
        cfg = default_algorithm_cfg(brain)
        brain.algorithm_cfg.apply_to(cfg, spec)

        if hasattr(cfg, "value_preprocessor"):
            cfg.value_preprocessor = RunningStandardScaler
            cfg.value_preprocessor_kwargs = {"size": 1, "device": device}

        if config is not None:
            output_dir = Path(config.path)
            apply_experiment_cfg(
                cfg,
                wandb_cfg=brain.wandb_cfg,
                checkpoint_freq=brain.checkpoint_freq,
                condition=config.condition,
                brain_id=brain_id + 1,
                phase=config.current_mode,
                output_dir=output_dir,
                run_name=output_dir.parent.name,
            )

        memory = RandomMemory(
            memory_size=memory_size_for(brain),
            num_envs=scope,
            device=device,
        )
        models = build_models_for_algorithm(
            spec,
            encoder_cls=brain.encoder,
            encoder_kwargs=encoder_kwargs,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            cfg=brain.model_cfg,
        )
        if not brain.encoder_cfg.trainable:
            for model in models.values():
                if hasattr(model, "encoder"):
                    freeze(model.encoder)

        agent = brain.algorithm(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
        )
        # Per-agent method wrappers so NETT supplemental scalars are tracked
        # through skrl's TensorBoard writer. W&B picks them up from
        # sync_tensorboard=True when enabled.
        if config is not None:
            attach_tensorboard_tracking(agent)
        agents.append(agent)
    return agents
