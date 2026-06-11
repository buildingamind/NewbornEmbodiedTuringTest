"""CompactCNN with pre-trained weights loaded from a checkpoint.

Loads weights saved by ``pretrain_encoder.py`` (supervised imprint-vs-variant
classification). Used with ``trainable=False`` so the encoder is frozen during
PPO training — the policy head then learns to use the discriminative features
to find the imprint monitor.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch

from .compact_cnn import CompactCNN


class CompactCNNPretrained(CompactCNN):
    """CompactCNN initialized from a pre-trained checkpoint.

    Args:
        observation_space: gym observation space (64×64 RGB).
        features_dim: must match the pre-training checkpoint (default 256).
        pretrained_path: path to the ``.pt`` file saved by ``pretrain_encoder.py``.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        pretrained_path: str | Path = "/tmp/pretrained_compact_cnn.pt",
        **kwargs,
    ) -> None:
        super().__init__(observation_space, features_dim, **kwargs)
        path = Path(pretrained_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Pre-trained encoder checkpoint not found: {path}. "
                "Run pretrain_encoder.py first."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
