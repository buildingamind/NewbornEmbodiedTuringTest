"""Foveated eye: a crisp centre with a small colour spot, a slightly blurred grey surround.

⭐ OWNER 2026-10-15 (chat, workspace DECISIONS §117): "a retinal filter where the surroundings are
slightly blurred and the center is crisp with a very small area in the center where color is present.
The surroundings would be 3 duplicated greyscale channels."

⛔ NOT `retina.py`. That wrapper is a CPU, motion-difference (`process(image, next_image)`) model that
stacks frames itself and has never run on this fleet. This one is a static, per-frame GPU transform.

## What the frame becomes

With r = the pixel's distance from the image centre ((H-1)/2, (W-1)/2) and y = BT.601 luminance
(``dvs_polarity._LUMA``, imported so the weights cannot drift):

    grey    g  = (y, y, y)                                  three duplicated channels
    surround s = gaussian_blur(g, sigma)                    "slightly blurred"
    w_sharp(r) = 1 inside SHARP_RADIUS, 0 beyond SHARP_RADIUS + SHARP_RADIUS/2, cosine ramp between
    w_color(r) = the same, at COLOR_RADIUS
    out        = w_color * rgb + (1 - w_color) * (w_sharp * g + (1 - w_sharp) * s)

So the colour spot is crisp and in colour, the ring out to SHARP_RADIUS is crisp and grey, and the rest
is blurred grey. The ramps avoid a hard circular edge that a convolution would learn as a feature.
The output is uint8 with 3 channels, so the encoder is unchanged (nature_cnn reads the count from the
space).

## Knobs -- ALL REQUIRED, no defaults (a fovea of unstated size is not a recipe)

    NETT_FOVEA_COLOR_RADIUS  colour-spot radius, pixels of the eye image   (> 0)
    NETT_FOVEA_SHARP_RADIUS  crisp-region radius, pixels                   (>= COLOR_RADIUS)
    NETT_FOVEA_SIGMA         surround Gaussian sigma, pixels               (> 0)

Radii are in pixels of the frame the wrapper receives (the eye's NETT_EYE_RES), so the same numbers mean
a different visual angle at another resolution or FOV. The row states them; the boot banner prints them.

⚠ ORDER: `pre`, innermost, like `grayscale`: every downstream consumer sees the foveated frame. It
refuses anything but 3 channels (a stacking wrapper ran first, or the frame is already grey).
"""

from __future__ import annotations

import math
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from ..observation import image_layout
from .dvs_polarity import _LUMA
from .grayscale import _policy_obs, _replace_policy_obs

ENV_COLOR = "NETT_FOVEA_COLOR_RADIUS"
ENV_SHARP = "NETT_FOVEA_SHARP_RADIUS"
ENV_SIGMA = "NETT_FOVEA_SIGMA"


def _env_required_float(name: str, unset: str = "") -> float:
    """Reads ``name``; ``unset`` is what an absent variable reads as (always refused: no default size)."""
    raw = os.environ.get(name, unset).strip()
    if not raw:
        raise ValueError(f"{name} is unset. The fovea has no default size (workspace DECISIONS §117): "
                         f"the row must state {ENV_COLOR}, {ENV_SHARP} and {ENV_SIGMA}.")
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: expected a number of pixels") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name}={raw!r}: expected a positive number of pixels")
    return v


def fovea_params_from_env() -> dict:
    color = _env_required_float(ENV_COLOR, "")
    sharp = _env_required_float(ENV_SHARP, "")
    sigma = _env_required_float(ENV_SIGMA, "")
    if sharp < color:
        raise ValueError(f"{ENV_SHARP}={sharp} < {ENV_COLOR}={color}: the colour spot must sit inside the "
                         f"crisp region.")
    return {"color_radius": color, "sharp_radius": sharp, "sigma": sigma}


def _ramp(r: torch.Tensor, radius: float) -> torch.Tensor:
    """1 for r <= radius, 0 for r >= 1.5 radius, raised-cosine between."""
    width = radius / 2.0
    t = ((r - radius) / width).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * t))


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    half = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


class Fovea(gym.ObservationWrapper):
    """RGB (HWC/NHWC or CHW/NCHW, uint8, numpy or torch on any device) -> foveated 3-channel uint8."""

    def __init__(self, env, color_radius: float | None = None, sharp_radius: float | None = None,
                 sigma: float | None = None):
        super().__init__(env)
        if color_radius is None or sharp_radius is None or sigma is None:
            p = fovea_params_from_env()
            color_radius = p["color_radius"] if color_radius is None else color_radius
            sharp_radius = p["sharp_radius"] if sharp_radius is None else sharp_radius
            sigma = p["sigma"] if sigma is None else sigma
        if not (0 < color_radius <= sharp_radius) or sigma <= 0:
            raise ValueError(f"Fovea: need 0 < color_radius <= sharp_radius and sigma > 0; got "
                             f"{color_radius}, {sharp_radius}, {sigma}")
        self.color_radius, self.sharp_radius, self.sigma = float(color_radius), float(sharp_radius), float(sigma)
        self._check_space(env.observation_space)
        self.observation_space = env.observation_space
        self._weights: dict = {}      # (H, W, device) -> (w_color, w_sharp), each (H, W, 1)
        print(f"[NETT fovea] colour radius {self.color_radius:g} px, crisp radius {self.sharp_radius:g} px, "
              f"surround sigma {self.sigma:g} px; surround = BT.601 grey x3 (workspace DECISIONS §117)",
              flush=True)

    def _check_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            for v in space.spaces.values():
                self._check_space(v)
            return
        shape = tuple(getattr(space, "shape", None) or ())
        if len(shape) < 3:
            return                              # not an image; rides through untouched
        chw = image_layout(shape[-3:]) == "chw"
        self._check_channels(shape[-3] if chw else shape[-1])

    @staticmethod
    def _check_channels(c):
        if c != 3:
            raise ValueError(
                f"Fovea received {c} channels; it consumes RAW RGB frames (3). Anything else means a "
                "stacking, grey or event wrapper ran first -- put fovea in `pre`, innermost.")

    def _masks(self, h: int, w: int, device):
        key = (h, w, str(device))
        if key not in self._weights:
            yy = torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2.0
            xx = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2.0
            r = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
            self._weights[key] = (_ramp(r, self.color_radius)[..., None], _ramp(r, self.sharp_radius)[..., None])
        return self._weights[key]

    def _blur(self, g: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian over (H, W) of a (..., H, W, 1) float tensor, reflect padding."""
        lead = g.shape[:-3]
        h, w = g.shape[-3], g.shape[-2]
        x = g.reshape(-1, 1, h, w)
        k = _gaussian_kernel(self.sigma, g.device)
        p = k.numel() // 2
        x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), k.view(1, 1, -1, 1))
        return x.reshape(*lead, h, w, 1)

    def observation(self, obs):
        policy = _policy_obs(obs)
        is_torch = isinstance(policy, torch.Tensor)
        t = policy if is_torch else torch.as_tensor(np.asarray(policy))
        chw = image_layout(tuple(t.shape[-3:])) == "chw"
        x = t.movedim(-3, -1) if chw else t     # -> (..., H, W, C)
        self._check_channels(x.shape[-1])
        xf = x.to(torch.float32)
        lw = torch.tensor(_LUMA, device=x.device, dtype=torch.float32)
        y = (xf * lw).sum(-1, keepdim=True)                       # (..., H, W, 1)
        s = self._blur(y)
        w_color, w_sharp = self._masks(x.shape[-3], x.shape[-2], x.device)
        grey = w_sharp * y + (1.0 - w_sharp) * s                  # (..., H, W, 1), broadcast to 3 below
        out = w_color * xf + (1.0 - w_color) * grey
        out = out.round().clamp(0, 255).to(t.dtype)
        out = out.movedim(-1, -3) if chw else out
        return _replace_policy_obs(obs, out if is_torch else out.numpy())
