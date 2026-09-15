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
import torch

from ..observation import image_layout


def _policy_obs(obs):
    """Identical to ``framestack._policy_obs``. Kept byte-for-byte rather than imported so the two
    wrappers cannot silently drift apart; if this ever needs to change, change BOTH."""
    return obs.get("policy", obs) if isinstance(obs, dict) else obs


def _replace_policy_obs(obs, policy):
    """Identical to ``framestack._replace_policy_obs``."""
    if not isinstance(obs, dict):
        return policy
    out = dict(obs)
    out["policy"] = policy
    return out


class LumNorm(gym.ObservationWrapper):
    """Standardise each frame per channel to a fixed mean/std, preserving dtype and layout."""

    def __init__(self, env):
        super().__init__(env)
        self.target_mean = float(os.environ.get("NETT_LUMNORM_MEAN", "0.45"))
        self.target_std = float(os.environ.get("NETT_LUMNORM_STD", "0.25"))
        if not 0.0 < self.target_std:
            raise ValueError("NETT_LUMNORM_STD must be positive")

    def _axes(self, shape):
        """Spatial reduction axes. Reduce over the SPATIAL axes only, per channel and per frame in a
        batch, so a channel stack of several frames is standardised frame-consistently rather than
        having one frame's statistics imposed on another."""
        if len(shape) not in (3, 4):
            raise ValueError(f"LumNorm expects HWC/NHWC or CHW/NCHW, got shape {tuple(shape)}")
        return (-2, -1) if image_layout(tuple(shape[-3:])) == "chw" else (-3, -2)

    def observation(self, obs):
        # ⛔ THE REAL ENV YIELDS ``{"policy": <CUDA torch.Tensor>}`` -- a dict, and a TENSOR inside it.
        # Two separate defects, found one after the other by live smoke of wave row 03 (seat:insect,
        # 2026-09-15). First this read ``np.asarray(obs)``, and ``np.asarray({"policy": arr})`` is a
        # 0-d OBJECT array -> "got shape ()". With the dict unwrapped it then reached
        # ``np.asarray(<cuda tensor>)`` -> "can't convert cuda:0 device type tensor to numpy".
        # ⭐ THE SECOND DEFECT SURVIVED THE FIRST FIX BECAUSE EVERY FIXTURE WAS A NUMPY ARRAY. A
        # red-then-green test proves the code handles the input THE TEST supplies; it says nothing
        # about the input the env supplies. The tests now cover cuda and cpu tensors.
        # ⚠ Type-preserving ON PURPOSE: a tensor in yields a tensor out, on the same device and
        # dtype. LumNorm is ordered FIRST in the chain and does not know what follows it --
        # FrameStack would convert to numpy itself (``_obs_to_numpy``), but a wrapper must not
        # depend on its successor to repair its output type.
        # ⚠ Non-``policy`` keys ride through untouched: only the policy observation is an image.
        policy = _policy_obs(obs)
        if isinstance(policy, torch.Tensor):
            return _replace_policy_obs(obs, self._normalise_torch(policy))
        return _replace_policy_obs(obs, self._normalise_numpy(np.asarray(policy)))

    def _normalise_numpy(self, arr):
        axes = self._axes(arr.shape)
        x = arr.astype(np.float32) / 255.0
        mean = x.mean(axis=axes, keepdims=True)
        std = x.std(axis=axes, keepdims=True)
        # A constant channel (std 0) carries no structure to preserve; leaving it at the
        # target mean is correct and avoids a divide-by-zero that would produce NaN and
        # poison every downstream consumer silently.
        scale = np.where(std > 1e-6, self.target_std / np.maximum(std, 1e-6), 0.0)
        y = (x - mean) * scale + self.target_mean
        return (np.clip(y, 0.0, 1.0) * 255.0).round().astype(arr.dtype)

    def _normalise_torch(self, t):
        """The numpy path's twin. Stays on-device: LumNorm runs on every frame, and a round trip
        through host memory here would be paid per step."""
        axes = self._axes(t.shape)
        x = t.to(torch.float32) / 255.0
        mean = x.mean(dim=axes, keepdim=True)
        # ⛔ ``torch.std`` DEFAULTS TO THE UNBIASED (ddof=1) ESTIMATOR AND ``np.std`` DOES NOT
        # (ddof=0). Ported naively the two backends would disagree by a factor of
        # sqrt(n/(n-1)) -- tiny per pixel, systematic across every frame, and invisible to any
        # test that exercises only one backend. ``correction=0`` is what makes them the same
        # function. test_lumnorm.py pins the two paths to identical output.
        std = x.std(dim=axes, keepdim=True, correction=0)
        scale = torch.where(std > 1e-6, self.target_std / torch.clamp(std, min=1e-6),
                            torch.zeros_like(std))
        y = (x - mean) * scale + self.target_mean
        return (torch.clamp(y, 0.0, 1.0) * 255.0).round().to(t.dtype)
