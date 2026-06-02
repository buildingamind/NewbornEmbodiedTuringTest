"""skrl agent construction for NETT brains."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

from .experiment import apply_experiment_cfg
from .models import build_models_for_algorithm
from .registry import algorithm_spec
from ..recording.wandb import attach_wandb_init_hook


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def encoder_cfg_for(brain) -> dict[str, Any]:
    kwargs = brain.encoder_cfg.as_kwargs()
    kwargs.pop("trainable", None)
    return kwargs


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
    # Scale rollouts and memory_size so updates fire on `rollouts` transitions
    # regardless of num_envs. Without this, skrl's _rollout counter increments
    # by 1 per env.step() call, so the update fires after rollouts × scope
    # transitions instead of rollouts transitions.
    base_rollouts = memory_size_for(brain)
    scaled_rollouts = max(1, base_rollouts // scope)
    agents = []
    for brain_id in range(num_agents):
        cfg = default_algorithm_cfg(brain)
        brain.algorithm_cfg.apply_to(cfg, spec)
        if scope > 1 and hasattr(cfg, "rollouts"):
            cfg.rollouts = scaled_rollouts

        if hasattr(cfg, "value_preprocessor"):
            cfg.value_preprocessor = RunningStandardScaler
            cfg.value_preprocessor_kwargs = {"size": 1, "device": device}

        if config is not None:
            apply_experiment_cfg(
                cfg,
                brain=brain,
                config=config,
                brain_id=brain_id + 1,
            )

        memory = RandomMemory(
            memory_size=scaled_rollouts,
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
        if config is not None:
            attach_wandb_init_hook(agent)
        agents.append(agent)
    return agents
