"""Demonstrate registering a project-specific NETT feature extractor.

Heavy encoders such as DINO/ViT/SAM are not bundled with ``nett_skrl``. Wrap
them in a ``NETTFeatureExtractor`` subclass, expose ``features_dim``, and
register the adapter before constructing ``Brain`` or ``NETT``.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from nett_skrl.brain import Brain
from nett_skrl.brain.encoders import NETTFeatureExtractor
from nett_skrl.brain.registry import register_encoder
from nett_skrl.body.observation import image_channels_hw, prepare_image_tensor


class DinoV1Adapter(NETTFeatureExtractor):
    """Tiny stand-in showing the adapter shape expected by ``nett_skrl``."""

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        channels, _, _ = image_channels_hw(observation_space)
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, self.features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(prepare_image_tensor(observations, self.observation_space))


def main() -> int:
    register_encoder("DinoV1", DinoV1Adapter)

    brain = Brain(
        encoder="DinoV1",
        algorithm="PPO",
        reward="ICM",
    )
    assert brain.encoder is DinoV1Adapter, f"encoder did not resolve: got {brain.encoder!r}"
    print(f"Brain.encoder = {brain.encoder.__name__}")
    print(f"Brain.reward  = {brain.reward.__name__}")
    print("registration OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
