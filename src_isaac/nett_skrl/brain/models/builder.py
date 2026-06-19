"""Model factory helpers for NETT skrl agents."""

from __future__ import annotations

import torch.nn as nn

from .deterministic_actor import DeterministicActor
from .gaussian_actor import GaussianActor
from .model_cfg import ModelCfg
from .multivariate_gaussian_actor import MultivariateGaussianActor
from .q_critic import QCritic
from .value_critic import ValueCritic
from .utils.init import orthogonal_init


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
    """Build the skrl model dictionary required by an ``AlgorithmSpec``.

    When ``cfg.shared_encoder`` is True, a single encoder instance is created
    and its reference is shared between the actor and all critic models — the
    same behaviour as SB3's ``share_features_extractor=True``. The shared
    encoder and its trunk are each initialised once; each model head gets its
    own independent MLP trunk so the critic can still learn a different value
    basis from the actor.
    """
    # Build one shared encoder if requested; otherwise each model builds its own.
    shared_enc = None
    if cfg.shared_encoder:
        shared_enc = encoder_cls(observation_space, **encoder_kwargs)
        shared_enc.to(device)
        if cfg.orthogonal_init:
            shared_enc.apply(lambda m: orthogonal_init(m, cfg.hidden_gain))

    model_kwargs = {
        "encoder_cls": encoder_cls,
        "encoder_kwargs": encoder_kwargs,
        "observation_space": observation_space,
        "action_space": action_space,
        "device": device,
        "cfg": cfg,
        "shared_encoder": shared_enc,
    }

    def actor():
        if spec.actor_type == "gaussian":
            # Opt-in distribution override (default keeps diagonal GaussianActor).
            if cfg.actor_distribution == "multivariate_gaussian":
                cls = MultivariateGaussianActor
            elif cfg.actor_distribution in (None, "gaussian"):
                cls = GaussianActor
            else:
                raise ValueError(
                    f"Unknown actor_distribution {cfg.actor_distribution!r}; "
                    "expected None, 'gaussian', or 'multivariate_gaussian'."
                )
        else:
            cls = DeterministicActor
        return cls(**model_kwargs)

    def critic():
        cls = ValueCritic if spec.critic_type == "value" else QCritic
        return cls(**model_kwargs)

    models: dict[str, nn.Module] = {}
    for key in spec.model_keys:
        models[key] = actor() if "policy" in key else critic()
    return models
