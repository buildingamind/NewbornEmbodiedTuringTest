"""Per-frame luminance standardisation: remove the global brightness cue.

⛔ WHY THIS EXISTS, AND THE MEASUREMENT THAT MOTIVATES IT (2026-09-15).
Measured directly on the 18 real NETT parsing clips at full resolution:

    swapping the OBJECT (ship <-> fork), same background .... 3.7% of pixels differ
    swapping the BACKGROUND, same object ................... 93-96% of pixels differ

    mean luminance, 1A/2A = 118.12 / 117.99   (objects matched to 0.1 grey levels)
                    1B/2B = 103.01 / 102.95
                    1C/2C = 172.36 / 172.28   (backgrounds differ by 50-70)

So object identity is deliberately UN-cued by brightness -- good design -- while the
background is trivially cued by it. An encoder trained on reward finds the background,
because it is ~25x larger and linearly separable. Measured consequence: a rule of
"approach the familiar background" fits 10 of 12 condition x pose cells, including
0.13 on Novel Familiar (where following background is exactly wrong) and 0.84 on
Imprinted Object Familiar (where it is exactly right).

This wrapper standardises each frame per colour channel to a fixed mean and standard
deviation, which removes the between-background brightness and contrast difference
while leaving local structure -- edges, shape, relative layout -- intact.

⚠ WHAT IT DOES NOT DO. It does not remove the background. Texture, layout and colour
RATIO survive standardisation, so an encoder can still key on background; this removes
only the cheapest cue, not the category. Read a null from a lumnorm arm as "the global
brightness cue was not the whole story", never as "background was controlled for".

⚠ It is applied to the OBSERVATION, so it shifts the input distribution for every
downstream consumer -- including any segmentation wrapper ordered after it. Arms using
both are a different condition from either alone, not a sum of the two.

Settings:
    NETT_LUMNORM_MEAN   0.45  target per-channel mean, in [0, 1]
    NETT_LUMNORM_STD    0.25  target per-channel standard deviation, in [0, 1]
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

from ..observation import image_layout


class LumNorm(gym.ObservationWrapper):
    """Standardise each frame per channel to a fixed mean/std, preserving dtype and layout."""

    def __init__(self, env):
        super().__init__(env)
        self.target_mean = float(os.environ.get("NETT_LUMNORM_MEAN", "0.45"))
        self.target_std = float(os.environ.get("NETT_LUMNORM_STD", "0.25"))
        if not 0.0 < self.target_std:
            raise ValueError("NETT_LUMNORM_STD must be positive")

    def observation(self, obs):
        arr = np.asarray(obs)
        if arr.ndim not in (3, 4):
            raise ValueError(f"LumNorm expects HWC/NHWC or CHW/NCHW, got shape {arr.shape}")
        chw = image_layout(arr.shape[-3:]) == "chw"
        # Reduce over the SPATIAL axes only, per channel and per frame in a batch, so a
        # channel stack of several frames is standardised frame-consistently rather than
        # having one frame's statistics imposed on another.
        axes = (-2, -1) if chw else (-3, -2)
        x = arr.astype(np.float32) / 255.0
        mean = x.mean(axis=axes, keepdims=True)
        std = x.std(axis=axes, keepdims=True)
        # A constant channel (std 0) carries no structure to preserve; leaving it at the
        # target mean is correct and avoids a divide-by-zero that would produce NaN and
        # poison every downstream consumer silently.
        scale = np.where(std > 1e-6, self.target_std / np.maximum(std, 1e-6), 0.0)
        y = (x - mean) * scale + self.target_mean
        return (np.clip(y, 0.0, 1.0) * 255.0).round().astype(arr.dtype)
