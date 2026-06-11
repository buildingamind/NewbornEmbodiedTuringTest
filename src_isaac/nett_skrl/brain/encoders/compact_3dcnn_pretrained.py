"""Compact3DCNN with pre-trained weights loaded from a checkpoint."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch

from .compact_3dcnn import Compact3DCNN


class Compact3DCNNPretrained(Compact3DCNN):
    """Compact3DCNN initialized from a pre-trained checkpoint.

    Args:
        observation_space: gym observation space (64×64×6 for 2-frame stack).
        features_dim: must match the pre-training checkpoint (default 256).
        num_frames: number of stacked frames (default 2).
        pretrained_path: path to the ``.pt`` file saved by pretrain_3dcnn_encoder.py.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        num_frames: int = 2,
        pretrained_path: str | Path = "/tmp/pretrained_compact_3dcnn.pt",
        **kwargs,
    ) -> None:
        super().__init__(observation_space, features_dim, num_frames, **kwargs)
        path = Path(pretrained_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Pre-trained 3DCNN checkpoint not found: {path}. "
                "Run pretrain_3dcnn_encoder.py first."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
