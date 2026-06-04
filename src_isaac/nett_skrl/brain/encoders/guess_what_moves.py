"""GuessWhatMoves dual-pathway encoder for 2-frame observations (~146 K parameters).

Inspired by Pathak et al. (2019) "Self-Supervised Exploration via Disagreement"
and the broader two-stream visual processing literature (Simonyan & Zisserman,
2014 "Two-Stream Convolutional Networks for Action Recognition in Videos").

The architecture uses two parallel pathways:

  * **What pathway** (appearance/content CNN): A standard 2D CNN operating on
    the *current* frame (last C channels of the stacked input). Captures static
    object identity and appearance.

  * **Moves pathway** (motion 3D CNN): A Conv3d layer operating on *both*
    frames, followed by 2D spatial refinement. Captures short-range motion,
    frame differences, and temporal dynamics.

The pathway outputs are concatenated and projected to ``features_dim``. This
forces the policy to jointly attend to *what* is present and *how it moves*,
consistent with the ventral/dorsal dual-stream hypothesis in biological vision.

Expects a 2-frame FrameStack body wrapper so observations arrive as
(C*2, H, W) = (6, 64, 64) in CHW format.

Parameter budget (64×64 RGB 2-frame, features_dim=128):
    What pathway    :  ~10 K
    Moves pathway   :   ~5 K
    Fusion linear   : ~131 K
    Total encoder   : ~146 K
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor


class GuessWhatMoves(HWCFeatureExtractor):
    """Dual-pathway encoder: appearance CNN + motion 3D CNN, fused via linear projection.

    What pathway  : 2D CNN on the most-recent (current) frame.
    Moves pathway : Conv3d on both frames → 2D CNN spatial refinement.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        num_frames: int = 2,
        **_,
    ) -> None:
        super().__init__(observation_space, features_dim)
        total_channels, height, width = image_channels_hw(observation_space)
        self.num_frames = int(num_frames)
        self.base_channels = total_channels // self.num_frames   # 3 for RGB

        # --- What pathway: 2D CNN on the current frame ----------------------
        self.what_cnn = nn.Sequential(
            nn.Conv2d(self.base_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # --- Moves pathway: 3D CNN on both frames → 2D refinement -----------
        # Conv3d collapses temporal dim fully (kernel depth = num_frames)
        self.moves_3d = nn.Conv3d(
            self.base_channels, 16,
            kernel_size=(self.num_frames, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.moves_2d = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Infer flat sizes
        with torch.no_grad():
            dummy_2d = torch.zeros(1, self.base_channels, height, width)
            what_flat = self.what_cnn(dummy_2d).shape[1]

            dummy_3d = torch.zeros(1, self.base_channels, self.num_frames, height, width)
            moves_flat = self.moves_2d(self.moves_3d(dummy_3d).squeeze(2)).shape[1]

        # Fusion projection
        self.fusion = nn.Sequential(
            nn.Linear(what_flat + moves_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._prepare_image(observations)   # (B, C*T, H, W)
        B, CT, H, W = x.shape
        C = self.base_channels
        T = self.num_frames

        # What pathway: current (most-recent) frame
        current_frame = x[:, -C:, :, :]       # (B, 3, H, W)
        what_feat = self.what_cnn(current_frame)

        # Moves pathway: reshape to (B, C, T, H, W) and run 3D + 2D conv
        x_3d = x.view(B, C, T, H, W)
        moves_out = self.moves_3d(x_3d).squeeze(2)   # (B, 16, H', W')
        moves_feat = self.moves_2d(moves_out)

        combined = torch.cat([what_feat, moves_feat], dim=-1)
        return self.fusion(combined)
