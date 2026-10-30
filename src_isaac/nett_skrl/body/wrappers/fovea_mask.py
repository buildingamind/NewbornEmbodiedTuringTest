"""Soft-mask foveated retina: a small colour spot at the image centre, grey everywhere else.

⭐ OWNER 2026-10-30 (presenter chat): "a 3 pixel radius foveated retina", "the default foveated retina
setup", by these steps:
  * start with the RGB image at 128 x 80; normalise to [0, 1] if stored as 0-255;
  * fovea centre at the middle of the image, approximately (64, 40);
  * binary circular mask: 1 inside radius 3, 0 outside;
  * Gaussian-blur THE MASK (not the colour image), giving a soft boundary;
  * grey image: luminance-weighted RGB, copied into all 3 channels;
  * blend: out = mask * colour + (1 - mask) * grey;
  * keep the output at 128 x 80 (no resize).
Mask blur sigma 1 px: the owner's choice (2026-10-30, "sigma 1 px").

⛔ NOT `fovea.py` (workspace DECISIONS §117). That wrapper blurs the grey SURROUND image, has a crisp grey
ring out to a second radius and a cosine-ramped mask, and has no default size. This one blurs only the
mask and has no crisp ring. Its knobs have their own prefix (NETT_FOVEAM_*), so a §117 row's NETT_FOVEA_*
can never change it.

## What the frame becomes

With r = the pixel's distance from the image centre ((H-1)/2, (W-1)/2) -- the exact middle; (39.5, 63.5)
at 128 x 80 -- and y = BT.601 luminance (``dvs_polarity._LUMA``, the fleet's grey weights):

    disk  d(r) = 1 if r <= RADIUS else 0
    mask  m    = gaussian_blur(d, MASK_SIGMA)               zero padding; the disk sits far from the border
    grey  g    = (y, y, y)
    out        = m * rgb + (1 - m) * g

The blend is done on 0-255 floats and rounded back to uint8, which is the same as the owner's [0, 1]
blend scaled by 255: the encoders keep their uint8 input and their own /255. The blurred mask's centre
value is below 1 (about 0.99 at radius 3, sigma 1); it is printed at boot and not renormalised.

## Knobs (defaults = the owner's 2026-10-30 recipe)

    NETT_FOVEAM_RADIUS      colour-disk radius, pixels of the frame the wrapper receives   (default 3)
    NETT_FOVEAM_MASK_SIGMA  Gaussian sigma of the mask blur, pixels                        (default 1)

⚠ ORDER: `pre`, innermost, like `fovea` and `grayscale`: every downstream consumer (frame stacks
included) sees the foveated frame. It refuses anything but 3 channels.
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

ENV_RADIUS = "NETT_FOVEAM_RADIUS"
ENV_SIGMA = "NETT_FOVEAM_MASK_SIGMA"
DEFAULT_RADIUS = 3.0
DEFAULT_SIGMA = 1.0


def _env_positive(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: expected a number of pixels") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name}={raw!r}: expected a positive number of pixels")
    return v


def fovea_mask_params_from_env() -> dict:
    return {"radius": _env_positive(ENV_RADIUS, DEFAULT_RADIUS), "sigma": _env_positive(ENV_SIGMA, DEFAULT_SIGMA)}


def soft_mask(h: int, w: int, radius: float, sigma: float, device=None) -> torch.Tensor:
    """(H, W) float32: the radius-`radius` binary disk at the image centre, Gaussian-blurred with `sigma`."""
    yy = torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2.0
    xx = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2.0
    disk = ((yy[:, None] ** 2 + xx[None, :] ** 2) <= radius * radius).to(torch.float32)
    half = max(1, int(math.ceil(3.0 * sigma)))
    k = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(k * k) / (2.0 * sigma * sigma))
    k = k / k.sum()
    x = disk.view(1, 1, h, w)
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, half))
    x = F.conv2d(x, k.view(1, 1, -1, 1), padding=(half, 0))
    return x.view(h, w)


class FoveaMask(gym.ObservationWrapper):
    """RGB (HWC/NHWC or CHW/NCHW, uint8, numpy or torch on any device) -> soft-mask foveated 3-channel uint8."""

    def __init__(self, env, radius: float | None = None, sigma: float | None = None):
        super().__init__(env)
        p = fovea_mask_params_from_env()
        self.radius = float(p["radius"] if radius is None else radius)
        self.sigma = float(p["sigma"] if sigma is None else sigma)
        if self.radius <= 0 or self.sigma <= 0:
            raise ValueError(f"FoveaMask: need radius > 0 and sigma > 0; got {self.radius}, {self.sigma}")
        self._check_space(env.observation_space)
        self.observation_space = env.observation_space
        self._masks: dict = {}        # (H, W, device) -> (H, W, 1) mask
        hw = self._image_hw(env.observation_space)
        centre = "" if hw is None else (f"; frame {hw[1]}x{hw[0]}, mask centre value "
                                       f"{float(soft_mask(hw[0], hw[1], self.radius, self.sigma).max()):.3f}")
        print(f"[NETT fovea_mask] colour radius {self.radius:g} px, mask blur sigma {self.sigma:g} px, "
              f"surround = BT.601 grey x3, no surround blur (owner 2026-10-30){centre}", flush=True)

    def _image_hw(self, space):
        if isinstance(space, gym.spaces.Dict):
            for v in space.spaces.values():
                hw = self._image_hw(v)
                if hw is not None:
                    return hw
            return None
        shape = tuple(getattr(space, "shape", None) or ())
        if len(shape) < 3:
            return None
        s = shape[-3:]
        return (s[1], s[2]) if image_layout(s) == "chw" else (s[0], s[1])

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
                f"FoveaMask received {c} channels; it consumes RAW RGB frames (3). Anything else means a "
                "stacking, grey or event wrapper ran first -- put fovea_mask in `pre`, innermost.")

    def _mask(self, h: int, w: int, device):
        key = (h, w, str(device))
        if key not in self._masks:
            self._masks[key] = soft_mask(h, w, self.radius, self.sigma, device)[..., None]
        return self._masks[key]

    def observation(self, obs):
        policy = _policy_obs(obs)
        is_torch = isinstance(policy, torch.Tensor)
        t = policy if is_torch else torch.as_tensor(np.asarray(policy))
        chw = image_layout(tuple(t.shape[-3:])) == "chw"
        x = t.movedim(-3, -1) if chw else t     # -> (..., H, W, C)
        self._check_channels(x.shape[-1])
        xf = x.to(torch.float32)
        lw = torch.tensor(_LUMA, device=x.device, dtype=torch.float32)
        y = (xf * lw).sum(-1, keepdim=True)                       # (..., H, W, 1), broadcast to 3 below
        m = self._mask(x.shape[-3], x.shape[-2], x.device)
        out = m * xf + (1.0 - m) * y
        out = out.round().clamp(0, 255).to(t.dtype)
        out = out.movedim(-1, -3) if chw else out
        return _replace_policy_obs(obs, out if is_torch else out.numpy())
