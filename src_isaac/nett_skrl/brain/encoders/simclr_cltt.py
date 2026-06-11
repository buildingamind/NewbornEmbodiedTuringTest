"""SimCLR encoder with CLTT (Contrastive Learning with Temporal Transformations).

The backbone follows the SimCLR CNN architecture from Chen et al. (2020)
"A Simple Framework for Contrastive Learning of Visual Representations".
The CLTT temporal contrastive signal comes from Wu et al. (2021) /
related works that treat temporally adjacent frames as positive pairs in a
contrastive objective.

In the NETT RL framework, the encoder provides features for PPO and exposes a
``project()`` method used by the companion ``CLTTReward`` intrinsic reward
adapter. The reward adapter drives temporal consistency: it computes an NT-Xent
contrastive loss between encoder projections of consecutive observations
(observations vs. next_observations) and updates only the projection head
(not the backbone, which the PPO actor-critic gradient already trains).

Architecture (single-frame input, 64×64 RGB):
    Backbone     : Conv(3→32)→BN→ReLU → Conv(32→32)→BN→ReLU → Conv(32→64)→BN→ReLU
                   → AdaptiveAvgPool(4,4) → Flatten → Linear(1024→features_dim) → ReLU
    Projector    : Linear(features_dim→64) → ReLU → Linear(64→64)   [not in RL path]

Parameter budget (features_dim=128):
    Backbone     :  ~61 K
    Linear head  : ~131 K
    Projector    :  ~12 K (auxiliary; excluded from RL output path)
    Total encoder: ~204 K
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor


class SimCLRCLTT(HWCFeatureExtractor):
    """SimCLR-style CNN backbone with a CLTT projection head.

    Call ``project(obs)`` to obtain L2-normalised contrastive embeddings used
    by ``CLTTReward``; the standard ``forward()`` returns backbone features for
    PPO without touching the projection head.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 128, **_) -> None:
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)

        # SimCLR backbone: three conv blocks with batch normalisation
        self.backbone = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.backbone(torch.zeros(1, channels, height, width)).shape[1]

        # RL feature head (used by PPO actor-critic)
        self.fc = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

        # CLTT projection head: maps features to a normalised embedding space.
        # Kept separate so freezing or detaching is straightforward.
        proj_dim = 64
        self.projector = nn.Sequential(
            nn.Linear(features_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._prepare_image(observations)
        h = self.backbone(x)
        return self.fc(h)

    def project(self, observations: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projection vectors for the CLTT contrastive loss.

        Backbone and fc are run WITHOUT gradients (detached) so CLTT only
        trains the projector head — preventing CLTT gradients from interfering
        with the PPO policy gradients in the shared backbone.
        """
        with torch.no_grad():
            x = self._prepare_image(observations)
            h = self.backbone(x)
            f = self.fc(h)
        # Only the projector gets CLTT gradients
        with torch.set_grad_enabled(True):
            z = self.projector(f.detach())
        return F.normalize(z, dim=-1)
