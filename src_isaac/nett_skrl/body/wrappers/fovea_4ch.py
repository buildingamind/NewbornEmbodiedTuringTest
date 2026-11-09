"""Four-channel foveated eye: colour only inside the fovea, plus a separate whole-image grey channel.

⭐ OWNER 2026-11-09 (FOVEA-4CH pilot, via the planning session): "does a separate grey channel let the CNN weight
greyscale apart from colour?". The same fovea as `fovea_mask` (radius-3 disk at the image centre, mask Gaussian-
blurred with sigma 1, BT.601 luma on the encoded uint8 RGB -- workspace FINDINGS add. 383), but the channels are laid
out separately instead of blended:

    m   = soft_mask(H, W, RADIUS, MASK_SIGMA)                     fovea_mask.soft_mask, the same window and edge
    y   = BT.601 luma of the WHOLE frame (fovea included)         dvs_polarity._LUMA, as fovea_mask.py
    ch0..2 = m * rgb      (colour inside the fovea, exactly 0 where m == 0)
    ch3    = y            (whole-image grey)

Both are computed on 0-255 floats and rounded back to uint8 (fovea_mask's rounding), so the encoders keep their
uint8 input and their own /255. The observation space says 4 channels, so a single-frame encoder builds its first
convolution with in_channels=4 and a frame-stacked one sees 4 per frame (8 for a 2-frame stack).

⚠ ORDER: `pre`, innermost, like `fovea_mask`: frame stacks stack the 4-channel frame. It refuses anything but 3
input channels.
⚠ A CLTT aux on a stacked encoder must be told the per-frame depth: rows set NETT_AUX_CLTT_CHANNELS_PER_FRAME=4
(left at the default 3, an 8-channel stack fails loudly: 8 % 3 != 0).

## Knobs

    NETT_FOVEAM_RADIUS, NETT_FOVEAM_MASK_SIGMA   shared with fovea_mask (defaults = the owner's 2026-10-30 recipe)
    NETT_FOVEA4CH_DUMP                           optional directory: the FIRST non-blank observation this process wraps
                                                 (input RGB and the 4-channel output, env 0) is saved there once, as
                                                 fovea4ch_first_obs_<pid>.npz, for the pilot's smoke check.

At boot it prints the mask centre value; on the first non-blank observation it prints a self-check on the real frame:
periphery colour (where m == 0) must be 0, the fovea's colour must equal round(m * rgb), and ch3 must equal the
BT.601 luma of the whole input frame.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import torch

from ..observation import image_layout
from .dvs_polarity import _LUMA
from .fovea_mask import fovea_mask_params_from_env, soft_mask
from .grayscale import _policy_obs, _replace_policy_obs

ENV_DUMP = "NETT_FOVEA4CH_DUMP"


class Fovea4ch(gym.ObservationWrapper):
    """RGB (HWC/NHWC or CHW/NCHW, uint8, numpy or torch) -> 4-channel uint8: masked colour (3) + whole-image grey (1)."""

    def __init__(self, env, radius: float | None = None, sigma: float | None = None):
        super().__init__(env)
        p = fovea_mask_params_from_env()
        self.radius = float(p["radius"] if radius is None else radius)
        self.sigma = float(p["sigma"] if sigma is None else sigma)
        if self.radius <= 0 or self.sigma <= 0:
            raise ValueError(f"Fovea4ch: need radius > 0 and sigma > 0; got {self.radius}, {self.sigma}")
        self.observation_space = self._out_space(env.observation_space)
        self._masks: dict = {}
        self._checked = False
        self._dump = os.environ.get(ENV_DUMP, "").strip()
        hw = self._image_hw(env.observation_space)
        centre = "" if hw is None else (f"; frame {hw[1]}x{hw[0]}, mask centre value "
                                       f"{float(soft_mask(hw[0], hw[1], self.radius, self.sigma).max()):.3f}")
        print(f"[NETT fovea_4ch] 4 channels: ch0-2 = mask x RGB (colour radius {self.radius:g} px, mask blur sigma "
              f"{self.sigma:g} px, 0 outside), ch3 = BT.601 grey of the whole frame (owner 2026-11-09){centre}",
              flush=True)

    # ── spaces ──────────────────────────────────────────────────────────────────────────────────────────────────
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

    def _out_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict({k: self._out_space(v) for k, v in space.spaces.items()})
        shape = tuple(getattr(space, "shape", None) or ())
        if len(shape) < 3:
            return space                        # not an image; rides through untouched
        chw = image_layout(shape[-3:]) == "chw"
        self._check_channels(shape[-3] if chw else shape[-1])
        out = shape[:-3] + ((4,) + shape[-2:] if chw else shape[-3:-1] + (4,))
        return gym.spaces.Box(low=0, high=255, shape=out, dtype=np.uint8)

    @staticmethod
    def _check_channels(c):
        if c != 3:
            raise ValueError(
                f"Fovea4ch received {c} channels; it consumes RAW RGB frames (3). Anything else means a "
                "stacking, grey or event wrapper ran first -- put fovea_4ch in `pre`, innermost.")

    def _mask(self, h: int, w: int, device):
        key = (h, w, str(device))
        if key not in self._masks:
            self._masks[key] = soft_mask(h, w, self.radius, self.sigma, device)[..., None]
        return self._masks[key]

    # ── the transform ───────────────────────────────────────────────────────────────────────────────────────────
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """(..., H, W, 3) uint8 -> (..., H, W, 4) uint8."""
        self._check_channels(x.shape[-1])
        xf = x.to(torch.float32)
        lw = torch.tensor(_LUMA, device=x.device, dtype=torch.float32)
        y = (xf * lw).sum(-1, keepdim=True)
        m = self._mask(x.shape[-3], x.shape[-2], x.device)
        out = torch.cat([m * xf, y], dim=-1)
        return out.round().clamp(0, 255).to(torch.uint8)

    def observation(self, obs):
        policy = _policy_obs(obs)
        is_torch = isinstance(policy, torch.Tensor)
        t = policy if is_torch else torch.as_tensor(np.asarray(policy))
        chw = image_layout(tuple(t.shape[-3:])) == "chw"
        x = t.movedim(-3, -1) if chw else t     # -> (..., H, W, C)
        out = self.transform(x).to(t.dtype)
        if not self._checked and bool(x.max() > 0):    # the first NON-BLANK frame (a reset frame can be black)
            self._checked = True
            self._self_check(x, out)
        out = out.movedim(-1, -3) if chw else out
        return _replace_policy_obs(obs, out if is_torch else out.numpy())

    # ── first-frame self-check and optional dump ────────────────────────────────────────────────────────────────
    def _self_check(self, x: torch.Tensor, out: torch.Tensor) -> None:
        xf = x.to(torch.float32)
        m = self._mask(x.shape[-3], x.shape[-2], x.device)[..., 0]
        outside = m == 0
        periph_max = int((out[..., :3].to(torch.int32) * outside[..., None]).max())
        ref_col = (m[..., None] * xf).round().clamp(0, 255)
        col_err = float((out[..., :3].to(torch.float32) - ref_col).abs().max())
        lw = torch.tensor(_LUMA, device=x.device, dtype=torch.float32)
        ref_y = (xf * lw).sum(-1).round().clamp(0, 255)
        grey_err = float((out[..., 3].to(torch.float32) - ref_y).abs().max())
        fovea_max = int((out[..., :3].to(torch.int32) * (m > 0.5)[..., None]).max())
        ok = periph_max == 0 and col_err == 0 and grey_err == 0
        print(f"[NETT fovea_4ch] first-frame check: periphery RGB max {periph_max} (want 0), fovea RGB max {fovea_max}, "
              f"|colour - round(m*rgb)| max {col_err:g}, |ch3 - luma(whole frame)| max {grey_err:g} -> "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise RuntimeError("Fovea4ch first-frame self-check failed (see the line above).")
        if self._dump:
            os.makedirs(self._dump, exist_ok=True)
            first_in = x.reshape(-1, *x.shape[-3:])[0].cpu().numpy()
            first_out = out.reshape(-1, *out.shape[-3:])[0].cpu().numpy()
            path = os.path.join(self._dump, f"fovea4ch_first_obs_{os.getpid()}.npz")
            np.savez_compressed(path, input_rgb=first_in, out_4ch=first_out,
                                mask=m.cpu().numpy(), radius=self.radius, sigma=self.sigma)
            print(f"[NETT fovea_4ch] first observation saved to {path}", flush=True)
