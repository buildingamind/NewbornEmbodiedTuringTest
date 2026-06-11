"""Value critic model for NETT continuous-control agents."""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model

from .model_cfg import ModelCfg
from .utils.backbone import FeatureBackbone
from .utils.features import features_forward
from .utils.init import init_output


class ValueCritic(DeterministicMixin, Model, FeatureBackbone):
    """V(s) critic for on-policy continuous agents."""

    def __init__(self, *, encoder_cls, encoder_kwargs, observation_space, action_space, device, cfg: ModelCfg, shared_encoder=None):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        last = self._build_backbone(encoder_cls, encoder_kwargs, observation_space, cfg, shared_encoder=shared_encoder)
        self.value_head = nn.Linear(last, 1)
        self.value_bound = cfg.value_bound
        # SB3 inits the value head with gain 1.0 (not the 0.01 policy-head gain).
        init_output(self.value_head, cfg, gain=1.0)

    def compute(self, inputs, role=""):
        value = self.value_head(features_forward(self, inputs))
        if self.value_bound is not None:
            value = torch.tanh(value) * float(self.value_bound)
        return value, {}
