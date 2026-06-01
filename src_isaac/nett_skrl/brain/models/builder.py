"""Model factory helpers for NETT skrl agents."""

from __future__ import annotations

import torch.nn as nn

from .deterministic_actor import DeterministicActor
from .gaussian_actor import GaussianActor
from .model_cfg import ModelCfg
from .q_critic import QCritic
from .value_critic import ValueCritic


def build_models_for_algorithm(
    spec,
    *,
    encoder_cls,
    encoder_kwargs,
    observation_space,
    action_space,
    device,
    cfg: ModelCfg,
) -> dict[str, nn.Module]:
    """Build the skrl model dictionary required by an ``AlgorithmSpec``."""

    model_kwargs = {
        "encoder_cls": encoder_cls,
        "encoder_kwargs": encoder_kwargs,
        "observation_space": observation_space,
        "action_space": action_space,
        "device": device,
        "cfg": cfg,
    }

    def actor():
        cls = GaussianActor if spec.actor_type == "gaussian" else DeterministicActor
        return cls(**model_kwargs)

    def critic():
        cls = ValueCritic if spec.critic_type == "value" else QCritic
        return cls(**model_kwargs)

    models: dict[str, nn.Module] = {}
    for key in spec.model_keys:
        models[key] = actor() if "policy" in key else critic()
    return models
