"""Base feature extractor contract for NETT skrl models."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn


class NETTFeatureExtractor(nn.Module):
    """Base feature extractor contract for NETT skrl models.

    Implementations must expose ``features_dim`` and accept Isaac camera
    observations as HWC, CHW, or flattened tensors.
    """

    features_dim: int

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__()
        self.observation_space = observation_space
        self.features_dim = int(features_dim)
        # When True, ``_prepare_image`` returns its input unchanged. Used by the
        # auxiliary SimCLR loss to re-encode an already-prepared (and augmented)
        # (B, C*T, H, W) float image through the encoder's own forward() without
        # re-applying the HWC->CHW permute or the /255 normalization.
        self._skip_prepare: bool = False

    def encode_spatial(self, observations) -> torch.Tensor:
        """Return the unpooled spatial feature map (B, C, H, W)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement encode_spatial")

    def encode_spatial_prepared(self, prepared_image) -> torch.Tensor:
        """Encode a normalized BCHW image without preparing it a second time."""
        prev = self._skip_prepare
        self._skip_prepare = True
        try:
            return self.encode_spatial(prepared_image)
        finally:
            self._skip_prepare = prev

    def encode_prepared(self, prepared_image):
        """Run ``forward`` on an already-prepared (B, C*T, H, W) float image.

        Bypasses ``_prepare_image``'s permute/normalize so callers can feed
        GPU-augmented, already-normalized images straight through the encoder
        backbone (gradients flow exactly as in the normal forward path).
        """
        prev = self._skip_prepare
        self._skip_prepare = True
        try:
            return self.forward(prepared_image)
        finally:
            self._skip_prepare = prev
