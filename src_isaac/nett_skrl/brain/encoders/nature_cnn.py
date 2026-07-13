"""NatureCNN-style image encoder.

Mirrors the three-convolution stack of the SB3-default ``NatureCNN`` (Mnih et
al. 2015), then collapses the final feature map to a fixed 4x4 grid via
``DeterministicAvgPool2d`` before flattening, so the linear head's input size
(and param count) is independent of ``input_resolution``. At res=64 the conv
map is already 4x4, so the pool is an identity (the original behaviour is
preserved exactly); at higher resolutions it caps the flatten at 4x4xconv_dim.
A 4x4 grid still preserves coarse left/right spatial layout (e.g. "is the
bright thing on the left or right of frame").

The pool is ``DeterministicAvgPool2d`` (not ``nn.AdaptiveAvgPool2d``) because
adaptive_avg_pool2d's CUDA backward is nondeterministic; see
``encoders/utils/pool.py``.
"""

from __future__ import annotations

import os

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.pool import DeterministicAvgPool2d


def _amp_dtype() -> torch.dtype | None:
    """Optional autocast dtype for the encoder conv/linear (NETT_AMP, default off).

    NETT_AMP=bf16 -> torch.bfloat16 (recommended: fp32 range, no GradScaler,
    replay-deterministic — measured run-to-run grad diff 0, ~2.17x on the CNN
    fwd+bwd vs fp32). NETT_AMP=fp16 -> torch.float16 (needs care re: underflow).
    Unset -> None (fp32, unchanged). Value shift vs fp32 is larger than TF32, so
    validate rest before adopting."""
    v = os.environ.get("NETT_AMP", "").lower()
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16", "half"):
        return torch.float16
    return None


class NatureCNN(HWCFeatureExtractor):
    """Three-layer CNN, flatten-to-linear, no spatial pooling."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512,
                 conv_dim: int = 64, **_):
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)

        # ``conv_dim`` is the final conv layer's channel count; it sets the
        # flatten size (conv_dim*4*4) and thus the dominant Linear's params, so
        # it scales encoder capacity WITHOUT touching features_dim (the policy
        # head input stays fixed). Default 64 preserves the original architecture.
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, conv_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Pool the final conv map to a fixed 4x4 grid before flattening so the
            # linear head's input size (and thus param count) is independent of
            # input_resolution. At res=64 the conv map is already 4x4, so this is
            # a no-op (validated behaviour preserved exactly); at res=256 it caps
            # the flatten at 4*4*64=1024 instead of 28*28*64, keeping params
            # <700k. A 4x4 grid still preserves coarse left/right spatial layout.
            DeterministicAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, channels, height, width)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._prepare_image(observations)
        amp = _amp_dtype()
        if amp is not None and x.is_cuda:
            # Run the (dominant) conv stack + projection in bf16/fp16 on the tensor
            # cores, then cast the 512-d feature back to fp32 so the downstream skrl
            # action/value heads and the distribution stats stay full precision.
            # autocast need only wrap the forward; backward reuses the recorded
            # precision. One place, our code -- no skrl patch, no GradScaler (bf16).
            with torch.autocast("cuda", dtype=amp):
                out = self.linear(self.cnn(x))
            return out.float()
        return self.linear(self.cnn(x))
