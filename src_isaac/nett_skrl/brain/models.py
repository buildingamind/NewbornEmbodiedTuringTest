"""Reusable skrl model classes for NETT continuous-control agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from ..observation import prepare_image_tensor, to_chw_space


@dataclass
class ModelCfg:
    """Configuration for NETT actor/critic MLP heads."""

    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "elu"
    initial_log_std: float = 0.0
    clip_actions: bool = True
    value_bound: float | None = 10.0
    orthogonal_init: bool = True
    hidden_gain: float = math.sqrt(2.0)
    output_gain: float = 0.01


def model_cfg_from(value: dict[str, Any] | None = None) -> ModelCfg:
    data = dict(value or {})
    return ModelCfg(**data)


def mlp_trunk(in_dim: int, hidden: list[int], activation: str) -> tuple[nn.Sequential, int]:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    if activation not in activations:
        raise ValueError(f"activation must be one of {sorted(activations)}; got {activation!r}")
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, int(h)), activations[activation]()]
        in_dim = int(h)
    return nn.Sequential(*layers), in_dim


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def _init_output(layer: nn.Linear, cfg: ModelCfg) -> None:
    if cfg.orthogonal_init:
        nn.init.orthogonal_(layer.weight, gain=cfg.output_gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


def features_forward(model, inputs):
    """Run NETT feature extractor + MLP trunk for skrl model inputs."""
    x = inputs.get("observations")
    if x is None:
        x = inputs.get("states")
    if x is None:
        raise KeyError("skrl model inputs must include 'observations' or 'states'.")
    target_device = next(model.encoder.parameters()).device
    if x.device != target_device:
        x = x.to(target_device, non_blocking=True)
    x = prepare_image_tensor(x, model.observation_space, device=target_device)
    return model.trunk(model.encoder(x))


class _FeatureBackbone:
    def _build_backbone(self, encoder_cls, encoder_kwargs, observation_space, cfg: ModelCfg) -> int:
        self.encoder = encoder_cls(to_chw_space(observation_space), **encoder_kwargs)
        self.trunk, last = mlp_trunk(
            int(self.encoder.features_dim),
            list(cfg.hidden_sizes),
            cfg.activation,
        )
        if cfg.orthogonal_init:
            self.encoder.apply(lambda module: _orthogonal_init(module, cfg.hidden_gain))
            self.trunk.apply(lambda module: _orthogonal_init(module, cfg.hidden_gain))
        return last


class GaussianActor(GaussianMixin, Model, _FeatureBackbone):
    """Continuous Gaussian actor for PPO/A2C/TRPO/RPO/SAC/CEM."""

    def __init__(self, *, encoder_cls, encoder_kwargs, observation_space, action_space, device, cfg: ModelCfg):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=cfg.clip_actions,
        )
        last = self._build_backbone(encoder_cls, encoder_kwargs, observation_space, cfg)
        self.mean_layer = nn.Linear(last, self.num_actions)
        _init_output(self.mean_layer, cfg)
        self.log_std = nn.Parameter(torch.full((self.num_actions,), float(cfg.initial_log_std)))

    def compute(self, inputs, role=""):
        mean = self.mean_layer(features_forward(self, inputs))
        return mean, {"log_std": self.log_std.expand_as(mean)}


class DeterministicActor(DeterministicMixin, Model, _FeatureBackbone):
    """Continuous deterministic actor for TD3/DDPG."""

    def __init__(self, *, encoder_cls, encoder_kwargs, observation_space, action_space, device, cfg: ModelCfg):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=cfg.clip_actions)
        last = self._build_backbone(encoder_cls, encoder_kwargs, observation_space, cfg)
        self.action_head = nn.Linear(last, self.num_actions)
        _init_output(self.action_head, cfg)

    def compute(self, inputs, role=""):
        return torch.tanh(self.action_head(features_forward(self, inputs))), {}


class ValueCritic(DeterministicMixin, Model, _FeatureBackbone):
    """V(s) critic for on-policy continuous agents."""

    def __init__(self, *, encoder_cls, encoder_kwargs, observation_space, action_space, device, cfg: ModelCfg):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        last = self._build_backbone(encoder_cls, encoder_kwargs, observation_space, cfg)
        self.value_head = nn.Linear(last, 1)
        self.value_bound = cfg.value_bound
        _init_output(self.value_head, cfg)

    def compute(self, inputs, role=""):
        value = self.value_head(features_forward(self, inputs))
        if self.value_bound is not None:
            value = torch.tanh(value) * float(self.value_bound)
        return value, {}


class QCritic(DeterministicMixin, Model, _FeatureBackbone):
    """Q(s, a) critic for SAC/TD3/DDPG."""

    def __init__(self, *, encoder_cls, encoder_kwargs, observation_space, action_space, device, cfg: ModelCfg):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        last = self._build_backbone(encoder_cls, encoder_kwargs, observation_space, cfg)
        self.q_head = nn.Linear(last + self.num_actions, 1)
        _init_output(self.q_head, cfg)

    def compute(self, inputs, role=""):
        features = features_forward(self, inputs)
        actions = inputs.get("taken_actions")
        if actions is None:
            actions = inputs.get("actions")
        if actions is None:
            raise KeyError("Q critic requires 'taken_actions' or 'actions' in skrl inputs.")
        actions = actions.to(features.device, non_blocking=True).float().view(features.shape[0], -1)
        return self.q_head(torch.cat([features, actions], dim=-1)), {}


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

    def actor():
        cls = GaussianActor if spec.actor_type == "gaussian" else DeterministicActor
        return cls(
            encoder_cls=encoder_cls,
            encoder_kwargs=encoder_kwargs,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )

    def critic():
        cls = ValueCritic if spec.critic_type == "value" else QCritic
        return cls(
            encoder_cls=encoder_cls,
            encoder_kwargs=encoder_kwargs,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )

    models: dict[str, nn.Module] = {}
    for key in spec.model_keys:
        models[key] = actor() if "policy" in key else critic()
    return models
