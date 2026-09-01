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

Architecture (single-frame input; the pool grid ADAPTS to the sensor):
    Backbone     : Conv(3→32)→BN→ReLU → Conv(32→32)→BN→ReLU → Conv(32→64)→BN→ReLU
                   → DeterministicAvgPool(grid) → Flatten → Linear(n_flat→features_dim) → ReLU
    Projector    : Linear(features_dim→64) → ReLU → Linear(64→64)   [not in RL path]

⚠ The grid was a hardcoded ``(4, 4)``, which was safe only while the sensor was
square. Three stride-2 convs take 128→→→16 but 80→→→10, and 10 % 4 != 0, so at the
128×80 campaign eye this encoder RAISED AT CONSTRUCTION -- before a single step, so
a wave that queued the arm lost the slot rather than the run. It is now the divisor
of each axis closest to 4 (ties to the larger), which reproduces the historical
(4, 4)/n_flat=1024 exactly on a square sensor.

Parameter budget (features_dim=128), per sensor:
    128×128 -> map 16×16, grid (4, 4), n_flat 1024 : Linear head ~131 K, total ~204 K
    128×80  -> map 10×16, grid (5, 4), n_flat 1280 : Linear head ~164 K, total ~237 K
    Backbone  ~61 K · Projector ~12 K (auxiliary; excluded from RL output path)
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.pool import DeterministicAvgPool2d, pool_grid_for


class SimCLRCLTT(HWCFeatureExtractor):
    """SimCLR-style CNN backbone with a CLTT projection head.

    Call ``project(obs)`` to obtain L2-normalised contrastive embeddings used
    by ``CLTTReward``; the standard ``forward()`` returns backbone features for
    PPO without touching the projection head.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 128,
                 conv_dim: int = 64, **_) -> None:
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)

        # ``conv_dim`` (final conv channels) scales the flatten size and thus the
        # RL head Linear's params, tuning capacity WITHOUT changing features_dim.
        # Default 64 preserves the original architecture.
        conv_layers = [
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, conv_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
        ]
        # MEASURE the map the convs actually produce; do not derive it. A derived
        # grid goes silently wrong the moment a stride or padding is edited, and
        # the failure mode is a construction-time raise on one sensor only.
        with torch.no_grad():
            _, _, feat_h, feat_w = nn.Sequential(*conv_layers)(
                torch.zeros(1, channels, height, width)).shape
        self.pool_grid = pool_grid_for(feat_h, feat_w)

        # Kept FLAT -- conv_layers is splatted, not nested in its own Sequential --
        # so state_dict keys stay backbone.0 .. backbone.10 exactly as before.
        self.backbone = nn.Sequential(
            *conv_layers,
            DeterministicAvgPool2d(self.pool_grid),
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
