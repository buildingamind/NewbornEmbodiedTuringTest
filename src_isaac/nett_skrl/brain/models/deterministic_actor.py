"""Deterministic actor model for NETT continuous-control agents."""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model

from .model_cfg import ModelCfg
from .utils.backbone import FeatureBackbone
from .utils.features import features_forward
from .utils.init import init_output


class DeterministicActor(DeterministicMixin, Model, FeatureBackbone):
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
        init_output(self.action_head, cfg)

    def compute(self, inputs, role=""):
        return torch.tanh(self.action_head(features_forward(self, inputs))), {}
