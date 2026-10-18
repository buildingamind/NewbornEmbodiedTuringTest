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

⚠ ``NETT_DVS_COLOR=1`` (owner 2026-10-16, workspace DECISIONS §131: "a 6 channel DVS (a positive and negative
channel for each color)") replaces the luma line with the same rule applied to R, G and B separately:

    channel 2k    ON  of colour k = 255 where blur(c_k,t) - blur(c_k,t-1) >= +threshold     (k = R, G, B)
    channel 2k+1  OFF of colour k = 255 where that difference <= -threshold

i.e. (R+, R-, G+, G-, B+, B-), six channels, the same threshold and blur per colour. Unset keeps the
two luma channels bit for bit. The paragraph below describes the luma default.

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

import math
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
        ``NETT_DVS_COLOR``      off by default (2 luma channels). On: 6 channels (R+, R-, G+, G-, B+, B-),
                                the luma rule applied to each colour separately; needs a 3-channel frame.
        ``NETT_DVS_BLUR_KERNEL``  side of the pre-blur's binomial kernel, odd and >= 3 (default unset = 3, the
                                [1,2,1] kernel, bit for bit). Owner 2026-10-17 (workspace DECISIONS §134): "a single
                                passed blur that is just a larger gaussian filter". Side 2N+1 is the binomial
                                kernel of order 2N, which equals N passes of the 3x3 kernel exactly in the
                                interior (9 = 4 passes, sigma sqrt(2) px; 5 = 2 passes). Within N px of the frame
                                edge it differs, because one replicate pad by N is not N replicate pads by 1.
                                Refused with ``NETT_DVS_BLUR`` off, where it would do nothing.
        ``NETT_DVS_WINDOW``     integer W >= 2 (default unset = one tick's events, bit for bit). Owner 2026-10-18
                                (workspace DECISIONS §149): "the video would play at 1 frame a tick and the new DVS
                                would receive 5 frames but convert it into a single 6-channel observation". The
                                wrapper keeps the last W per-tick event maps and emits, every tick,
                                    out = 255 * sum_k lam^k e_{t-k} / sum_k lam^k,   k = 0..W-1,  e in {0, 1}
                                rounded to uint8, same channels as one tick. Needs NETT_DECISION_LOOP=P >= W: the
                                loop returns the LAST tick's observation, and at that tick the window is the
                                decision's last W ticks. W < P drops the first P-W ticks' changes -- the agent moves
                                on tick 0 of a decision, so W = P-1 keeps only held ticks (owner, DECISIONS §149
                                add. 1: "take in 6 frames, drop the first frame and feed the other 5"). W > P would
                                sum across two decisions and refuses.
        ``NETT_DVS_WINDOW_TAU`` decay constant tau > 0 in ticks, lam = exp(-1/tau) (FINDINGS §4dh.44h.183). Unset =
                                lam 1, a plain per-polarity event count (W events = 255). Needs NETT_DVS_WINDOW.
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
        self.blur_kernel = self._blur_kernel_side(self.blur)
        self.color = _env_flag("NETT_DVS_COLOR", False)
        self.n_out = 6 if self.color else 2
        # ⛔ A consumer that reads the per-frame channel count from its own knob would split a 6-channel
        # event frame as three 2-channel frames. Refuse the mismatch here rather than downstream.
        cpf = os.environ.get("NETT_AUX_CLTT_CHANNELS_PER_FRAME")
        if self.color and cpf is not None and cpf.strip() != "6":
            raise ValueError(f"NETT_DVS_COLOR=1 makes one frame 6 channels; NETT_AUX_CLTT_CHANNELS_PER_FRAME={cpf!r}")
        self.window, self.tau = self._window_knobs()
        if self.window is not None:
            lam = 1.0 if self.tau is None else math.exp(-1.0 / self.tau)
            w = [lam ** k for k in range(self.window)]          # k = 0 is the NEWEST tick
            self._weights = [x / sum(w) for x in w]
            print(f"[NETT dvs window] W={self.window} tau={'none (plain count)' if self.tau is None else self.tau} "
                  f"weights newest->oldest {[round(x / w[0], 3) for x in w]} (workspace DECISIONS §149)", flush=True)
        self._hist: list = []                   # last W per-tick event maps (uint8 0/255), NEWEST LAST
        self._phase_c = 0                       # maps emitted since reset(): decision ticks are c = 1 + kP
        self._ref = None                        # blurred frame just before the window (diagnostic only)
        self._dens: list = []                   # (window non-zero, window intermediate, single-diff non-zero) per decision
        self._prev = None                       # previous BLURRED luma (N,H,W)|(H,W), or colour (N,3,H,W)|(3,H,W)
        self._batched = False                   # whether the last frame carried a leading env axis
        self.observation_space = self._out_space(env.observation_space)

    # -- spaces ---------------------------------------------------------------------------
    def _out_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict({k: self._out_space(v) for k, v in space.spaces.items()})
        shape = tuple(space.shape)
        if len(shape) < 3:
            return space                        # not an image; ride through untouched
        if image_layout(shape[-3:]) == "chw":
            out = shape[:-3] + (self.n_out,) + shape[-2:]
        else:
            out = shape[:-3] + shape[-3:-1] + (self.n_out,)
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

    @staticmethod
    def _window_knobs():
        """``NETT_DVS_WINDOW`` / ``NETT_DVS_WINDOW_TAU`` -> (W | None, tau | None). Refuses anything ambiguous."""
        raw_w = os.environ.get("NETT_DVS_WINDOW", "").strip()
        raw_t = os.environ.get("NETT_DVS_WINDOW_TAU", "").strip()
        if raw_w == "":
            if raw_t != "":
                raise ValueError(f"NETT_DVS_WINDOW_TAU={raw_t!r} without NETT_DVS_WINDOW: there is no window to weight.")
            return None, None
        if not raw_w.isdigit() or int(raw_w) < 2:
            raise ValueError(f"NETT_DVS_WINDOW={raw_w!r}: expected an integer >= 2 (leave it unset for one tick).")
        w = int(raw_w)
        # ⛔ The window must BE the decision window. The decision loop hands the agent only the last tick's
        # observation; a shorter window drops the move tick, a longer one sums across two decisions, and with no
        # loop at all every tick is a decision, so W ticks would mean W decisions of history.
        loop = os.environ.get("NETT_DECISION_LOOP", "").strip()
        if not loop.isdigit() or int(loop) < w:
            raise ValueError(f"NETT_DVS_WINDOW={w} needs NETT_DECISION_LOOP=P with P >= {w} (the window is the decision's "
                             f"last W ticks); got NETT_DECISION_LOOP={loop!r}.")
        tau = None
        if raw_t != "":
            try:
                tau = float(raw_t)
            except ValueError as exc:
                raise ValueError(f"NETT_DVS_WINDOW_TAU={raw_t!r}: expected a number > 0.") from exc
            if not (tau > 0 and math.isfinite(tau)):
                raise ValueError(f"NETT_DVS_WINDOW_TAU={raw_t!r}: expected a finite number > 0 (unset = plain count).")
        return w, tau

    #: decisions averaged for the one-time density line
    DENSITY_DECISIONS = 50

    def _density(self, y: torch.Tensor, res: torch.Tensor) -> None:
        """Diagnostic only (never changes an output): at decision ticks, the window map's density beside the density of ONE
        difference over the same span (the frame just before the window -> this frame, same threshold) -- what a
        P-frames-per-tick arm would show for those frames. Averaged over the first DENSITY_DECISIONS decisions, printed once."""
        if len(self._dens) >= self.DENSITY_DECISIONS:
            return
        p = int(os.environ["NETT_DECISION_LOOP"])
        phase = (self._phase_c - 1) % p
        if phase == 0 and self._phase_c > 1 and self._ref is not None and self._ref.shape == y.shape:
            d = y - self._ref
            single = 0.5 * float((d >= self.threshold).float().mean() + (d <= -self.threshold).float().mean())
            self._dens.append((float((res > 0).float().mean()), float(((res > 0) & (res < 255)).float().mean()), single))
        # AFTER the decision-tick read: with W = P the reference IS the decision-tick frame, for the next decision.
        if phase == (p - self.window) % p:
            self._ref = y.clone()
        if len(self._dens) == self.DENSITY_DECISIONS and not getattr(self, "_dens_printed", False):
            self._dens_printed = True
            nz, mid, one = (sum(x[i] for x in self._dens) / len(self._dens) for i in range(3))
            print(f"[NETT dvs window] density over the first {len(self._dens)} decisions (decision ticks, P={p}, W={self.window}): "
                  f"window non-zero {nz:.6f}, intermediate (0<v<255) {mid:.6f}; one {self.window}-frame difference over the same "
                  f"frames, same threshold: non-zero {one:.6f} (fractions of pixels x channels)", flush=True)

    def _accumulate(self, out: torch.Tensor) -> torch.Tensor:
        """One tick's uint8 0/255 event map -> the W-tick weighted map (uint8). Episode start = fewer maps, never padding."""
        if self._hist and self._hist[-1].shape != out.shape:
            self._hist = []
        self._hist.append(out)
        if len(self._hist) > self.window:
            self._hist.pop(0)
        acc = torch.zeros(out.shape, dtype=torch.float32, device=out.device)
        for k, e in enumerate(reversed(self._hist)):            # k = 0 is this tick
            acc += self._weights[k] * e.to(torch.float32)
        return torch.round(acc).clamp_(0, 255).to(torch.uint8)

    def _clear_hist(self, idx) -> None:
        """Zero the history of the envs in ``idx`` (an episode boundary): a new episode never sums the old one's events."""
        for e in self._hist:
            e[idx] = 0

    @staticmethod
    def _blur_kernel_side(blur_on: bool) -> int:
        """``NETT_DVS_BLUR_KERNEL`` -> the kernel side (unset = 3). Refuses even, < 3, non-integer, or a non-3 side with blur off."""
        raw = os.environ.get("NETT_DVS_BLUR_KERNEL", "3")
        try:
            k = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"NETT_DVS_BLUR_KERNEL must be an odd integer >= 3; got {raw!r}.") from exc
        if k < 3 or k % 2 == 0:
            raise ValueError(f"NETT_DVS_BLUR_KERNEL must be an odd integer >= 3; got {raw!r}.")
        # ⛔ A non-default kernel with the blur off would change nothing and still name a blurred condition.
        if not blur_on and k != 3:
            raise ValueError(f"NETT_DVS_BLUR_KERNEL={raw!r} with NETT_DVS_BLUR off: the kernel would never run.")
        return k

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
        n = self.blur_kernel
        # Binomial row of length n (n = 3 -> [1, 2, 1], the original kernel; same ops, so bit-identical).
        k1 = torch.tensor([float(math.comb(n - 1, i)) for i in range(n)], device=y.device, dtype=y.dtype)
        k = torch.outer(k1, k1)
        k = (k / k.sum()).view(1, 1, n, n)
        flat = y.reshape(-1, 1, *y.shape[-2:])
        r = n // 2
        pad = torch.nn.functional.pad(flat, (r, r, r, r), mode="replicate")
        return torch.nn.functional.conv2d(pad, k).reshape(y.shape)

    def _colour(self, t: torch.Tensor):
        """(..., H, W, 3) or (..., 3, H, W) -> (..., 3, H, W) float32 colour planes in 0-255."""
        chw = image_layout(tuple(t.shape[-3:])) == "chw"
        x = t.to(torch.float32)
        if not chw:
            x = x.movedim(-1, -3)               # -> (..., C, H, W)
        if x.shape[-3] != 3:
            raise ValueError(
                f"NETT_DVS_COLOR=1 needs a 3-channel RGB frame; got {x.shape[-3]} channels. More than 3 "
                "means a stacking wrapper ran first -- order dvs_polarity BEFORE framestack."
            )
        return x, chw

    def _events(self, policy):
        """Frame -> (ON, OFF) uint8 (or 3 such pairs under NETT_DVS_COLOR), and update the previous-frame buffer."""
        as_tensor = isinstance(policy, torch.Tensor)
        t = self._to_torch(policy)
        self._batched = t.ndim == 4
        if self.color:
            y, chw = self._colour(t)
            y = self._blur(y)                   # per plane: _blur flattens every leading axis
        else:
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
        if self.color:
            # (..., 3, H, W) x2 -> (..., 3, 2, H, W) -> (..., 6, H, W): R+, R-, G+, G-, B+, B-
            out = torch.stack((on, off), dim=-3).flatten(-4, -3)
            if not chw:
                out = out.movedim(-3, -1)
        else:
            out = torch.stack((on, off), dim=-3 if chw else -1)
        if self.window is not None:
            out = self._accumulate(out)
            self._phase_c += 1
            self._density(y, out)
        if not as_tensor:
            return out.cpu().numpy().astype(np.uint8)
        return out

    # -- gym API --------------------------------------------------------------------------
    def reset(self, **kwargs):
        self._prev = None
        self._hist = []
        self._phase_c = 0
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
        # ⚠ `_batched`, not `_prev.ndim == 3`: a colour buffer is (N,3,H,W) batched and (3,H,W) not.
        if self._prev is not None and self._batched:
            idx = np.where(np.asarray(done).ravel())[0]
            if len(idx):
                sel = torch.as_tensor(idx, device=events.device if isinstance(events, torch.Tensor)
                                      else "cpu", dtype=torch.long)
                if isinstance(events, torch.Tensor):
                    events.index_fill_(0, sel, 0)
                else:
                    events[idx] = 0
                if self.window is not None:
                    self._clear_hist(torch.as_tensor(idx, dtype=torch.long))
        return _replace_policy_obs(obs, events), reward, terminated, truncated, info
