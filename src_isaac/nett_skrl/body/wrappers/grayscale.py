"""Black-and-white eye: RGB -> ONE luminance channel, in training, test and record alike.

⭐ OWNER 2026-09-28 (chat): "if backgrounds matter so much, is this because of the texture of the
background or its color? A B&W vision filter on successful model configurations in Isaac could
answer this. We take naive models in Isaac, give them a B&W filter (adjusting the input channels
down from 3 to 1), and train and test them".

WHAT IT REMOVES AND WHAT IT KEEPS, measured on the 18 parsing clips (FINDINGS §4dh.44e.3):

    background   mean lum   saturation   edge density
    A desert       119         .37          .138
    B forest       107         .18          .261     (fork-2's imprint background)
    C beach        175         .60          .070

Hue and saturation go. Luminance and texture stay. In grey, A and B are 12 levels apart and
differ mainly by texture (2x the edges); C stays apart by brightness. So an arm with this
filter asks whether colour carries the background choice. It does not remove the background,
and a null reads "colour was not the whole story", never "background was controlled for".
`lumnorm` is the brightness counterpart.

Luminance is ITU-R BT.601, y = 0.299 R + 0.587 G + 0.114 B: cv2.COLOR_RGB2GRAY, PIL "L", and the
weights `dvs_polarity` already uses (`_LUMA`, imported, so the two cannot drift). The output is
uint8 with ONE channel, and the observation space says so, so the encoder builds its first
convolution with in_channels=1 (nature_cnn reads the count from the space).

⚠ ORDER: `pre`, innermost. Every downstream consumer must see the grey frame. It refuses more
than 3 channels, which would mean a stacking wrapper ran first.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from ..observation import image_layout
from .dvs_polarity import _LUMA


def _policy_obs(obs):
    """Identical to ``framestack._policy_obs`` (and lumnorm's copy)."""
    return obs.get("policy", obs) if isinstance(obs, dict) else obs


def _replace_policy_obs(obs, policy):
    """Identical to ``framestack._replace_policy_obs``."""
    if not isinstance(obs, dict):
        return policy
    out = dict(obs)
    out["policy"] = policy
    return out


class Grayscale(gym.ObservationWrapper):
    """RGB (HWC/NHWC or CHW/NCHW, uint8, numpy or torch on any device) -> 1-channel luminance."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self._out_space(env.observation_space)

    def _out_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict({k: self._out_space(v) for k, v in space.spaces.items()})
        shape = tuple(getattr(space, "shape", None) or ())
        if len(shape) < 3:
            return space                        # not an image; ride through untouched
        chw = image_layout(shape[-3:]) == "chw"
        c = shape[-3] if chw else shape[-1]
        self._check_channels(c)
        out = shape[:-3] + ((1,) + shape[-2:] if chw else shape[-3:-1] + (1,))
        return gym.spaces.Box(low=0, high=255, shape=out, dtype=np.uint8)

    @staticmethod
    def _check_channels(c):
        if c not in (1, 3):
            raise ValueError(
                f"Grayscale received {c} channels; it consumes RAW RGB frames (3, or 1 already grey). "
                "Anything else means a stacking or event wrapper ran first -- put grayscale in `pre`.")

    def observation(self, obs):
        policy = _policy_obs(obs)
        is_torch = isinstance(policy, torch.Tensor)
        t = policy if is_torch else torch.as_tensor(np.asarray(policy))
        chw = image_layout(tuple(t.shape[-3:])) == "chw"
        x = t.movedim(-3, -1) if chw else t     # -> (..., H, W, C)
        self._check_channels(x.shape[-1])
        if x.shape[-1] == 3:
            w = torch.tensor(_LUMA, device=x.device, dtype=torch.float32)
            y = (x.to(torch.float32) * w).sum(-1, keepdim=True).round().clamp(0, 255).to(t.dtype)
        else:
            y = x
        y = y.movedim(-1, -3) if chw else y
        return _replace_policy_obs(obs, y if is_torch else y.numpy())
