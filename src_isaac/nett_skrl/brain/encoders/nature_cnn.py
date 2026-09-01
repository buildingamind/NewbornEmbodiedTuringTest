"""NatureCNN-style image encoder.

Mirrors the three-convolution stack of the SB3-default ``NatureCNN`` (Mnih et
al. 2015). By default it collapses the final feature map to a fixed 4x4 grid
via ``DeterministicAvgPool2d`` before flattening, so the linear head's input
size (and param count) is independent of ``input_resolution``. At res=64 the
conv map is already 4x4, so the pool is an identity (the original behaviour is
preserved exactly); at higher resolutions it caps the flatten at
4x4xconv_dim. A 4x4 grid still preserves coarse left/right spatial layout
(e.g. "is the bright thing on the left or right of frame").

``spatial_pool=False`` is the SB3-legacy form: the complete post-convolution feature
map goes to the linear projection, with no pooling anywhere. Pooling remains the
module default so existing configs and checkpoints keep their exact architecture.

⚠ WITHOUT POOLING THE SENSOR SIZE IS AN ARCHITECTURE DECISION, AND IT DECIDED THE RUN.
The flatten scales with resolution, so the unpooled parameter count is set by the eye:

    128x80,  spatial_pool=False -> flatten  4096 ->  2,435,744 params   <- SHIPPED
    256x160, spatial_pool=False -> flatten 25088 -> 14,756,512 params
    256x160, spatial_pool=True  -> pooled 64x4x4 ->    600,736 params
    (84x84, SB3's reference size -> flatten  3136 ->  1,682,080 params)

At 256x160 essentially all of the growth is one ``Linear(25088, 512)`` ~= 12.8M, about
8x the layer the architecture is "standard" with. MEASURED 2026-08-03, 1000 episodes
x 4 seeds, rest correct_pct:

    128x80 unpooled            1.000 0.998 1.000 1.000   -> 4/4
    256x160 unpooled lr 3e-4   1.000 0.500 0.500 0.500   -> 1/4
    256x160 unpooled lr 1e-4   0.920 0.500 0.583 0.751   -> 1/4
    256x160 POOLED             1.000 1.000 1.000 1.000   -> 4/4

So unpooled-at-256x160 does not merely cost memory, it costs the training
consistency, and dropping the learning rate recovers only part of it. The fisheye
field depends on the ASPECT alone, so 128x80 keeps the identical 300 x 148.5 deg view
(verified: 148.5425 deg vertical at both) and only gives up acuity. If you raise the
resolution, either enable pooling or re-run the acceptance wave -- do not assume the
recipe carries over.

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
from .utils.pool import DeterministicAvgPool2d, pool_grid_for


def _amp_dtype() -> torch.dtype | None:
    """Autocast dtype for the encoder conv/linear (NETT_AMP). DEFAULT = bf16.

    bf16 is the validated speedup default: ~2.17x on the CNN fwd+bwd vs fp32,
    replay-deterministic (measured run-to-run grad diff 0), and rest-preserving
    (1000ep 4-seed = 1.0,1.0,1.0,0.997 = 4/4, matching fp32). It has fp32 range
    (no GradScaler needed). It shifts values ~1e-2 vs fp32 -> runs are NOT
    bit-identical to fp32 (a one-time change, like the Fabric adoption); replay
    determinism holds. Opt OUT to full fp32 with NETT_AMP in
    {off,0,none,fp32,float32,""}. NETT_AMP=fp16 selects float16 (needs care re:
    underflow). Only wraps the NatureCNN encoder (the recipe encoder)."""
    v = os.environ.get("NETT_AMP", "bf16").lower()
    if v in ("off", "0", "none", "fp32", "float32", ""):
        return None
    if v in ("fp16", "float16", "half"):
        return torch.float16
    return torch.bfloat16


class NatureCNN(HWCFeatureExtractor):
    """Three-layer CNN with optional deterministic spatial pooling."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        conv_dim: int = 64,
        spatial_pool: bool = True,
        **_,
    ):
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)
        if not isinstance(spatial_pool, bool):
            raise TypeError(
                "NatureCNN spatial_pool must be a bool, got "
                f"{type(spatial_pool).__name__}"
            )
        self.spatial_pool = spatial_pool

        # ``conv_dim`` is the final conv layer's channel count; it sets the
        # flatten size and thus the dominant Linear's params, so it scales
        # encoder capacity WITHOUT touching features_dim (the policy head
        # input stays fixed). Default 64 preserves the original architecture.
        layers: list[nn.Module] = [
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, conv_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        ]
        if spatial_pool:
            # Pool the final conv map to a small grid before flattening so the linear
            # head's input size (and thus param count) is independent of
            # input_resolution: at res=256 it caps the flatten at ~16*conv_dim instead
            # of 28*28*conv_dim, keeping params <700k, while still preserving coarse
            # left/right spatial layout.
            #
            # ⛔ THE GRID WAS HARDCODED (4, 4) AND THAT MADE THIS ENCODER -- THE
            # CAMPAIGN'S BASELINE -- UNCONSTRUCTIBLE AT THE REAL EYE. 128x80 gives a
            # (6, 12) map and 6 % 4 != 0, so it raised before a single step; 128x128
            # gives (12, 12) and builds, so every square probe returned a clean PASS.
            # The comparator every other arm is read against could not be built.
            self.pool_grid = self.pool_grid_for_layers(
                layers, channels, height, width)
            layers.append(DeterministicAvgPool2d(self.pool_grid))
        else:
            # ⚠ EXPOSED EVEN WHEN THERE IS NO POOL. A missing attribute makes an
            # assertion return None, which reads as "no grid" and not as "this
            # encoder does not pool" -- and a caller using getattr(..., None)
            # would skip the check entirely rather than fail it.
            self.pool_grid = None
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, channels, height, width)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    @staticmethod
    def pool_grid_for_layers(layers, channels, height, width):
        """Measure the conv stack's output map, then pick a grid that divides it.

        MEASURED, not derived: a derived map goes silently wrong the moment a
        stride or padding is edited, and its failure mode is a construction-time
        raise on one sensor only -- i.e. the bug this replaces, again.
        """
        with torch.no_grad():
            _, _, feat_h, feat_w = nn.Sequential(*layers)(
                torch.zeros(1, channels, height, width)).shape
        return pool_grid_for(feat_h, feat_w)

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
