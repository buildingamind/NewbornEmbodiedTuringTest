"""Two-channel event (DVS) representation: ON and OFF polarity as separate channels.

⛔ WHY THIS EXISTS RATHER THAN `wrappers/dvs.py`. The existing `DVS` wrapper is unusable on this
fleet's path and has never been run by any arm (0 references outside its own test):

  * it emits ONE ternary channel -- ``{0, 128, 255}`` for (decrease, none, increase) -- or three
    colour channels, never a polarity PAIR;
  * it is explicitly CPU/NumPy (`dvs.py:198`) and calls ``.detach().cpu().numpy()`` per step, a
    host round-trip for all 112 envs inside the rollout loop;
    ⚠ BE PRECISE ABOUT WHAT THIS WRAPPER BUYS THERE, BECAUSE IT IS LESS THAN IT LOOKS. This one
    is type- and device-PRESERVING, but on a framestacked arm it is followed by ``framestack``,
    which converts to NumPy itself (``framestack._obs_to_numpy``) -- measured: ``[framestack]``
    and ``[dvs_polarity, framestack]`` both emit ``ndarray``, ``[dvs_polarity]`` alone emits a
    Tensor. So the pipeline does NOT stay on the GPU end to end, and it does not for the
    incumbent ViT-CLTT-Ref either. What this wrapper avoids is an ADDITIONAL round trip and a
    second conversion of its own -- not the one framestack already performs.
  * it wraps ``gym.wrappers.FrameStackObservation`` internally, which collides with this body's own
    ``FrameStack`` and does not handle the batched ``(N, H, W, C)`` Isaac layout.

⚠ Leaving it in place: it is referenced by `tests/test_dvs.py` and removing a thing in order to add
a thing turns one reviewable change into two.

## What the representation is

A silicon retina reports per-pixel BRIGHTNESS CHANGE, not brightness. Splitting the sign into two
channels rather than encoding it along one axis matters: in a ternary channel the two event types
sit at OPPOSITE ENDS of a scalar with "no event" between them, so a linear readout must learn a
non-monotone function to separate them, and a convolution averages an ON pixel against an adjacent
OFF pixel to produce "no event". As two channels both are non-negative and independently summable.

    luminance   y = 0.299 R + 0.587 G + 0.114 B      (ITU-R BT.601, matching cv2.COLOR_RGB2GRAY
                                                      as used by the existing dvs.py)
    change      d = blur(y_t) - blur(y_{t-1})
    channel 0   ON  = 255 where d >= +threshold
    channel 1   OFF = 255 where d <= -threshold

⇒ COLOUR IS DISCARDED BY CONSTRUCTION, and that is a deliberate part of the hypothesis rather than
a cost of it: the parsing task's backgrounds are cued by brightness and colour while its objects are
matched for them, so an encoder that can only see motion cannot solve the task with the background
shortcut. ⚠ It also cannot see a STATIONARY object at all -- if the agent and object are both still,
every channel is zero. That is the representation's central risk and the falsifier should name it.

## Ordering

⚠ MUST PRECEDE `framestack` in the wrapper list. This wrapper is itself temporal -- it holds the
previous frame -- so it consumes raw frames and emits events; framestack then stacks N event frames
the way it would stack N raw ones. Reversed, it would difference two already-stacked tensors and
the channel axis would no longer mean what either wrapper thinks it means.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import torch

from ..observation import image_layout
# ⚠ Imported rather than copied. `lumnorm.py` already carries a second copy of these two helpers
# with the comment "Identical to framestack._replace_policy_obs"; a third copy is how a shared
# convention diverges silently, so this takes the dependency instead.
from .framestack import _done_mask, _policy_obs, _replace_policy_obs

#: ITU-R BT.601 luma weights -- the same transform `cv2.COLOR_RGB2GRAY` applies in `dvs.py`.
_LUMA = (0.299, 0.587, 0.114)


def _env_flag(name: str, default: bool = False) -> bool:
    """Same spellings the fleet's other boolean knobs accept."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class DVSPolarity(gym.Wrapper):
    """Replace each frame with a 2-channel (ON, OFF) event map. Batched- and device-preserving.

    Knobs:
        ``NETT_DVS_THRESHOLD``  change magnitude, in 0-255 luminance units (default 30, matching
                                ``dvs.py``'s ``change_threshold``).
        ``NETT_DVS_BLUR``       3x3 Gaussian pre-blur, on by default, as in ``dvs.py``. Off makes
                                the wrapper a pure difference and much noisier.
    """

    def __init__(self, env, threshold: float | None = None):
        super().__init__(env)
        self.threshold = float(
            threshold if threshold is not None else os.environ.get("NETT_DVS_THRESHOLD", 30.0)
        )
        # ⛔ NOT `>= 0`. At threshold exactly 0 the two comparisons are `d >= 0` and `d <= 0`,
        # which are BOTH TRUE wherever d == 0 -- every unchanged pixel would report an ON *and*
        # an OFF event, i.e. the degenerate input would return a full-field success rather than
        # an error. A zero-threshold event camera is meaningless anyway (infinite shot noise).
        if not self.threshold > 0:
            raise ValueError(
                f"NETT_DVS_THRESHOLD must be > 0; got {self.threshold}. At 0 every unchanged "
                "pixel fires BOTH polarities."
            )
        self.blur = _env_flag("NETT_DVS_BLUR", True)
        self._prev = None                       # previous BLURRED luminance, (N,H,W) or (H,W)
        self.observation_space = self._out_space(env.observation_space)

    # -- spaces ---------------------------------------------------------------------------
    def _out_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict({k: self._out_space(v) for k, v in space.spaces.items()})
        shape = tuple(space.shape)
        if len(shape) < 3:
            return space                        # not an image; ride through untouched
        if image_layout(shape[-3:]) == "chw":
            out = shape[:-3] + (2,) + shape[-2:]
        else:
            out = shape[:-3] + shape[-3:-1] + (2,)
        return gym.spaces.Box(low=0, high=255, shape=out, dtype=np.uint8)

    # -- the transform --------------------------------------------------------------------
    @staticmethod
    def _to_torch(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))

    def _luma(self, t: torch.Tensor):
        """(..., H, W, C) or (..., C, H, W) -> (..., H, W) float32 luminance in 0-255."""
        chw = image_layout(tuple(t.shape[-3:])) == "chw"
        x = t.to(torch.float32)
        if chw:
            x = x.movedim(-3, -1)               # -> (..., H, W, C)
        c = x.shape[-1]
        if c == 1:
            return x[..., 0], chw
        if c > 3:
            # ⛔ The ordering rule, enforced rather than merely documented. More than three
            # channels means this wrapper is downstream of something that already stacked --
            # framestack, most likely -- and it is about to difference two tensors whose
            # channel axis no longer means "colour". Fail loudly at construction-time cost
            # rather than emit events computed across a frame boundary.
            raise ValueError(
                f"DVSPolarity received {c} channels; it consumes RAW frames (1 or 3 channels). "
                "More than 3 means a stacking wrapper ran first -- order dvs_polarity BEFORE "
                "framestack in the arm's wrapper list."
            )
        w = torch.tensor(_LUMA[:c], device=x.device, dtype=x.dtype)
        w = w / w.sum()                         # a 2-channel input must still average to 0-255
        return (x[..., :c] * w).sum(-1), chw

    def _blur(self, y: torch.Tensor) -> torch.Tensor:
        """3x3 binomial [1,2,1] blur, replicate-padded.

        ⚠ THIS IS NOT cv2's sigma=1 KERNEL AND THE DOCSTRING USED TO CLAIM IT WAS.
        ``cv2.getGaussianKernel(3, 1)`` is [0.2741, 0.4519, 0.2741]; this is [0.25, 0.5, 0.25] --
        the standard binomial approximation, up to 0.0481 away on the centre tap (measured). Close,
        deliberately chosen (exact in integer arithmetic, no kernel table), and NOT interchangeable if
        anyone ever wants to reproduce `dvs.py`'s cv2 output numerically.
        """
        if not self.blur:
            return y
        k1 = torch.tensor([1.0, 2.0, 1.0], device=y.device, dtype=y.dtype)
        k = torch.outer(k1, k1)
        k = (k / k.sum()).view(1, 1, 3, 3)
        flat = y.reshape(-1, 1, *y.shape[-2:])
        pad = torch.nn.functional.pad(flat, (1, 1, 1, 1), mode="replicate")
        return torch.nn.functional.conv2d(pad, k).reshape(y.shape)

    def _events(self, policy):
        """Frame -> (ON, OFF) uint8, and update the previous-frame buffer."""
        as_tensor = isinstance(policy, torch.Tensor)
        t = self._to_torch(policy)
        y, chw = self._luma(t)
        y = self._blur(y)
        # ⛔ NO PREVIOUS FRAME => NO EVENTS, never a difference against zeros. Differencing the
        # first frame of an episode against a zero buffer reports the entire image as one giant
        # ON event -- a full-field flash at every reset, perfectly correlated with episode start.
        if self._prev is None or self._prev.shape != y.shape:
            d = torch.zeros_like(y)
        else:
            d = y - self._prev
        self._prev = y
        on = (d >= self.threshold).to(torch.uint8) * 255
        off = (d <= -self.threshold).to(torch.uint8) * 255
        out = torch.stack((on, off), dim=-3 if chw else -1)
        if not as_tensor:
            return out.cpu().numpy().astype(np.uint8)
        return out

    # -- gym API --------------------------------------------------------------------------
    def reset(self, **kwargs):
        self._prev = None
        obs, info = self.env.reset(**kwargs)
        return _replace_policy_obs(obs, self._events(_policy_obs(obs))), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        policy = _policy_obs(obs)
        events = self._events(policy)
        # ⚠ Per-env boundary: an episode that just reset has a previous frame from the OLD
        # episode, and the difference across that cut is a teleport, not motion. Zero those
        # envs' events and seed their buffer with the new first frame. Same hazard FrameStack
        # handles at `framestack.py:step`; a stateful wrapper that ignores `done` reports one
        # spurious full-field event per episode, on every env, forever.
        done = _done_mask(terminated, truncated)
        if done is not None and self._prev is not None and self._prev.ndim == 3:
            idx = np.where(np.asarray(done).ravel())[0]
            if len(idx):
                sel = torch.as_tensor(idx, device=events.device if isinstance(events, torch.Tensor)
                                      else "cpu", dtype=torch.long)
                if isinstance(events, torch.Tensor):
                    events.index_fill_(0, sel, 0)
                else:
                    events[idx] = 0
        return _replace_policy_obs(obs, events), reward, terminated, truncated, info
