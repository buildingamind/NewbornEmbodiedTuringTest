"""Gaussian actor model for NETT continuous-control agents."""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, Model

from .model_cfg import ModelCfg
from .utils.backbone import FeatureBackbone
from .utils.features import features_forward
from .utils.init import init_output


class GaussianActor(GaussianMixin, Model, FeatureBackbone):
    """Continuous Gaussian actor for PPO/A2C/TRPO/RPO/SAC/CEM."""

    def __init__(self, *, encoder_cls, encoder_kwargs, observation_space, action_space, device, cfg: ModelCfg, shared_encoder=None):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=cfg.clip_actions,
            # Cap the exploration std when configured (default keeps skrl's max_log_std=2).
            **(
                {"clip_log_std": True, "max_log_std": float(cfg.max_log_std)}
                if cfg.max_log_std is not None
                else {}
            ),
        )
        last = self._build_backbone(encoder_cls, encoder_kwargs, observation_space, cfg, shared_encoder=shared_encoder)
        self.mean_layer = nn.Linear(last, self.num_actions)
        init_output(self.mean_layer, cfg)
        self.log_std = nn.Parameter(torch.full((self.num_actions,), float(cfg.initial_log_std)))

    def compute(self, inputs, role=""):
        mean = self.mean_layer(features_forward(self, inputs))
        return mean, {"log_std": self.log_std.expand_as(mean)}
