"""Shared encoder and MLP trunk wiring for NETT skrl models."""

from __future__ import annotations

from ..model_cfg import ModelCfg
from .init import orthogonal_init
from .mlp import mlp_trunk


class FeatureBackbone:
    def _build_backbone(self, encoder_cls, encoder_kwargs, observation_space, cfg: ModelCfg) -> int:
        # observation_space is already CHW — ChannelsFirst wrapper guarantees this.
        self.encoder = encoder_cls(observation_space, **encoder_kwargs)
        self.trunk, last = mlp_trunk(
            int(self.encoder.features_dim),
            list(cfg.hidden_sizes),
            cfg.activation,
        )
        if cfg.orthogonal_init:
            self.encoder.apply(lambda module: orthogonal_init(module, cfg.hidden_gain))
            self.trunk.apply(lambda module: orthogonal_init(module, cfg.hidden_gain))
        return last
