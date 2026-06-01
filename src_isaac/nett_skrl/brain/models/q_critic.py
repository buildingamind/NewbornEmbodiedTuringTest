"""Q critic model for NETT continuous-control agents."""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model

from .model_cfg import ModelCfg
from .utils.backbone import FeatureBackbone
from .utils.features import features_forward
from .utils.init import init_output


class QCritic(DeterministicMixin, Model, FeatureBackbone):
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
        init_output(self.q_head, cfg)

    def compute(self, inputs, role=""):
        features = features_forward(self, inputs)
        actions = inputs.get("taken_actions")
        if actions is None:
            actions = inputs.get("actions")
        if actions is None:
            raise KeyError("Q critic requires 'taken_actions' or 'actions' in skrl inputs.")
        actions = actions.to(features.device, non_blocking=True).float().view(features.shape[0], -1)
        return self.q_head(torch.cat([features, actions], dim=-1)), {}
