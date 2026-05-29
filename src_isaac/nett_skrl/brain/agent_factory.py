"""skrl agent construction for NETT brains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

from .experiment import apply_experiment_cfg, attach_wandb_scalar_mirror
from .models import build_models_for_algorithm
from .registry import algorithm_spec


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def apply_cfg_overrides(cfg, overrides: dict[str, Any]) -> None:
    """In-place merge ``overrides`` into ``cfg`` (dataclass or dict)."""
    if isinstance(cfg, dict):
        cfg.update(overrides)
        return
    for k, v in overrides.items():
        setattr(cfg, k, v)


_DEFAULT_FEATURES_DIM = 512


def encoder_kwargs(brain, observation_space) -> dict[str, Any]:
    features_dim = int(brain.embedding_dim or _DEFAULT_FEATURES_DIM)
    return {"features_dim": features_dim, **brain.custom_encoder_args}


def default_algorithm_cfg(brain):
    """Return a fresh skrl algorithm config from the algorithm registry."""
    return algorithm_spec(brain.algorithm).cfg_cls()


def apply_algorithm_stability_defaults(cfg, spec) -> None:
    """Conservative defaults for image PPO on NETT's low-variance scenes."""
    if spec.cls.__name__ != "PPO":
        return
    if hasattr(cfg, "rollouts"):
        apply_cfg_overrides(cfg, {"rollouts": max(int(cfg.rollouts), 64)})
    if hasattr(cfg, "value_loss_scale"):
        apply_cfg_overrides(cfg, {"value_loss_scale": 0.25})
    if hasattr(cfg, "grad_norm_clip"):
        apply_cfg_overrides(cfg, {"grad_norm_clip": 0.25})


def memory_size_for(brain, cfg, spec) -> int:
    """Use rollout-sized memory for on-policy skrl agents."""
    if spec.family in {"on_policy", "cross_entropy"} and hasattr(cfg, "rollouts"):
        return int(cfg.rollouts)
    return brain.buffer_size


def build_agents(brain, env, device: torch.device, *, config=None) -> list:
    """Build one skrl agent per brain, each owning a contiguous env scope."""
    obs_space, act_space = env.observation_space, env.action_space
    enc_kwargs = encoder_kwargs(brain, obs_space)
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
        apply_algorithm_stability_defaults(cfg, spec)
        apply_cfg_overrides(cfg, brain.custom_algorithm_args)
        apply_cfg_overrides(cfg, {"learning_rate": brain.learning_rate})
        if hasattr(cfg, "mini_batches"):
            apply_cfg_overrides(cfg, {"mini_batches": max(1, brain.batch_size // 64)})
        if hasattr(cfg, "batch_size"):
            apply_cfg_overrides(cfg, {"batch_size": brain.batch_size})

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
            memory_size=memory_size_for(brain, cfg, spec),
            num_envs=scope,
            device=device,
        )
        models = build_models_for_algorithm(
            spec,
            encoder_cls=brain.encoder,
            encoder_kwargs=enc_kwargs,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            cfg=brain.model_cfg,
        )
        if not brain.train_encoder:
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
        # Per-agent method wrappers so each agent's tracked scalars
        # (reward, loss, etc.) land on its own wandb.run. Cheap when
        # wandb is disabled — the wrapper short-circuits if no run.
        if config is not None and brain.wandb_cfg.get("mode") != "disabled":
            attach_wandb_scalar_mirror(agent)
        agents.append(agent)
    return agents
