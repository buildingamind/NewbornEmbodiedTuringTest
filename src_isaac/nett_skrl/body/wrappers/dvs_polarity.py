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

⚠ FOUR PLANES, ONLY ON fovea_4ch (owner 2026-11-09, "Option 1", relayed by the presenter). Directly on top of the
``fovea_4ch`` wrapper (ch0-2 = mask x RGB, ch3 = whole-frame BT.601 luma) the same per-plane rule runs on all four
planes: (R+, R-, G+, G-, B+, B-, L+, L-), EIGHT channels. Any other 4-channel input is still refused -- four channels
anywhere else most likely means a stacking wrapper ran first -- so the 8-channel mode cannot be entered by accident.

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
                                the luma rule applied to each colour separately; needs a 3-channel frame
                                (or, directly on fovea_4ch, its 4 planes -> 8 channels; see above).
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
        ``NETT_DVS_MOVE_GAIN``  g in [0, 1] (default unset = no change, bit for bit). FINDINGS add. 351 (owner
                                2026-11-06, egomotion DVS pilot, head-bob arm): the events of a tick on which the env
                                applied the agent's motion (repo B ``_last_decision_mask``, read from the unwrapped
                                env, never tick parity: this wrapper also sees the pre-roll ticks) are multiplied by
                                g, so 255 -> round(255 g). Needs NETT_DECISION_LOOP=P >= 2 (without a window every
                                tick is a move tick) and refuses NETT_DVS_WINDOW.

    The three egomotion knobs below (FINDINGS add. 351, the probe's picks for arms MC / GATE / GFS; the definitions are
    the probe's own, ego/probe/evlib.py + ana.py) are default unset = unchanged, bit for bit. At most one is set, and
    none combines with NETT_DVS_WINDOW or NETT_DVS_MOVE_GAIN. Reset / done handling is the plain wrapper's.

        ``NETT_DVS_MOTION_COMP``  ``yaw``. Before differencing, the previous (blurred, per-plane) frame is warped into
                                the current camera by the yaw change about the WORLD vertical between the two ticks:
                                each current pixel's ray (300 deg equisolid lens, image u right = camera -Y, v down =
                                camera -Z, the probe's measured convention) is rotated by R_prev^T Rz(dyaw) R_prev into
                                the previous camera, reprojected, and the previous frame is sampled there bilinearly
                                (``grid_sample``, border padding, align_corners off). A pixel outside the rim (r >= cx-1)
                                or whose source falls outside the previous view emits no events. Pose = the unwrapped
                                env's ``_cameras["center"].data.quat_w_world`` (wxyz, forward +X), read after every
                                reset() and step(). The lens is the env's (``LensSpec.from_observation_cfg``); the probe
                                ran only at equisolid 300 deg 128x80, so any other lens refuses.
        ``NETT_DVS_GATE_K``     k > 0 (the arm: 0.5). A tick's events x max(0, 1 - k |clip(a, -1, 1)| / sqrt 2), with
                                a = the action passed to step() (repo B stores it as ``_actions`` unchanged), rounded to
                                uint8 as NETT_DVS_MOVE_GAIN does. Every tick is gated, so under a decision loop the hold
                                ticks are scaled by the decision's action too (the probe measured move ticks only).
        ``NETT_DVS_GLOBAL_SUB`` ``median:R``, integer R >= 1 (the arm: 4). Per plane, the signed difference d minus its
                                (2R+1)x(2R+1) median (nearest = replicate padding, scipy ``median_filter`` mode
                                "nearest"), then ON / OFF at +/- threshold.
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
        # Colour planes: 3 (RGB), or 4 ONLY when the wrapped env is fovea_4ch (see the module note).
        from .fovea_4ch import Fovea4ch
        self.n_planes = 4 if self.color and isinstance(env, Fovea4ch) else 3
        self.n_out = 2 * self.n_planes if self.color else 2
        # ⛔ A consumer that reads the per-frame channel count from its own knob would split a 6-channel
        # event frame as three 2-channel frames. Refuse the mismatch here rather than downstream.
        cpf = os.environ.get("NETT_AUX_CLTT_CHANNELS_PER_FRAME")
        if self.color and cpf is not None and cpf.strip() != str(self.n_out):
            raise ValueError(f"NETT_DVS_COLOR=1 makes one frame {self.n_out} channels; NETT_AUX_CLTT_CHANNELS_PER_FRAME={cpf!r}")
        self.window, self.tau = self._window_knobs()
        self.move_gain = self._move_gain_knob(self.window)
        self.motion_comp, self.gate_k, self.global_sub = self._ego_knobs(self.window, self.move_gain)
        self._mc = None                         # _YawWarp for the env's lens (motion comp only; built at first reset)
        self._mc_raw = None                     # the unwrapped env carrying _cameras (motion comp only)
        self._pose = None                       # previous tick's camera rotation (N,3,3) float64 (motion comp only)
        self._raw = None                        # the unwrapped env carrying _last_decision_mask (move gain only)
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
        self._dens_plain: list = []             # no window: per-step event non-zero fraction (diagnostic only)
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

    @staticmethod
    def _move_gain_knob(window):
        """``NETT_DVS_MOVE_GAIN`` -> g | None (unset). Refuses anything ambiguous."""
        raw = os.environ.get("NETT_DVS_MOVE_GAIN", "").strip()
        if raw == "":
            return None
        try:
            g = float(raw)
        except ValueError as exc:
            raise ValueError(f"NETT_DVS_MOVE_GAIN={raw!r}: expected a number in [0, 1].") from exc
        if not (0.0 <= g <= 1.0):
            raise ValueError(f"NETT_DVS_MOVE_GAIN={raw!r}: expected a number in [0, 1].")
        loop = os.environ.get("NETT_DECISION_LOOP", "").strip()
        if not (loop.isdigit() and int(loop) >= 2):
            raise ValueError(f"NETT_DVS_MOVE_GAIN={raw} needs NETT_DECISION_LOOP=P >= 2 (got {loop!r}): without a "
                             f"window every tick is a move tick and the gain would scale every event.")
        if window is not None:
            raise ValueError(f"NETT_DVS_MOVE_GAIN with NETT_DVS_WINDOW={window}: a window sums ticks of both kinds. "
                             f"Refusing.")
        print(f"[NETT dvs move gain] g={g:g}: events of move ticks (the env's decision mask) x {g:g}, hold ticks "
              f"unchanged (FINDINGS add. 351)", flush=True)
        return g

    def _move_mask(self):
        """The env's decision mask for the tick just stepped (True = the agent's motion was applied)."""
        if self._raw is None:
            current, seen = self.env, set()
            while current is not None and id(current) not in seen and not hasattr(current, "_last_decision_mask"):
                seen.add(id(current))
                current = getattr(current, "_env", None) or getattr(current, "env", None)
            if current is None or not hasattr(current, "_last_decision_mask"):
                raise RuntimeError("NETT_DVS_MOVE_GAIN: no env under dvs_polarity carries _last_decision_mask.")
            self._raw = current
        return torch.as_tensor(self._raw._last_decision_mask).reshape(-1).bool()

    def _apply_move_gain(self, events):
        mask = self._move_mask()
        if not self._batched or int(mask.numel()) != int(events.shape[0]):
            raise RuntimeError(f"NETT_DVS_MOVE_GAIN: decision mask has {int(mask.numel())} envs, events "
                               f"{tuple(events.shape)} (batched={self._batched}).")
        if not bool(mask.any()):
            return events
        if isinstance(events, torch.Tensor):
            m = mask.to(events.device)
            events[m] = (events[m].to(torch.float32) * self.move_gain).round().to(torch.uint8)
        else:
            m = mask.cpu().numpy()
            events[m] = np.round(events[m].astype(np.float32) * self.move_gain).astype(np.uint8)
        return events

    @staticmethod
    def _ego_knobs(window, move_gain):
        """``NETT_DVS_MOTION_COMP`` / ``NETT_DVS_GATE_K`` / ``NETT_DVS_GLOBAL_SUB`` -> (``"yaw"`` | None, k | None,
        R | None). Refuses anything ambiguous, any two together, and either of NETT_DVS_WINDOW / NETT_DVS_MOVE_GAIN."""
        raw_mc = os.environ.get("NETT_DVS_MOTION_COMP", "").strip()
        raw_k = os.environ.get("NETT_DVS_GATE_K", "").strip()
        raw_gs = os.environ.get("NETT_DVS_GLOBAL_SUB", "").strip()
        mc = k = r = None
        if raw_mc != "":
            if raw_mc.lower() != "yaw":
                raise ValueError(f"NETT_DVS_MOTION_COMP={raw_mc!r}: expected 'yaw' (leave it unset for none).")
            mc = "yaw"
        if raw_k != "":
            try:
                k = float(raw_k)
            except ValueError as exc:
                raise ValueError(f"NETT_DVS_GATE_K={raw_k!r}: expected a finite number > 0.") from exc
            if not (math.isfinite(k) and k > 0):
                raise ValueError(f"NETT_DVS_GATE_K={raw_k!r}: expected a finite number > 0 (leave it unset for none).")
        if raw_gs != "":
            kind, sep, rr = raw_gs.lower().partition(":")
            if kind != "median" or sep != ":" or not (rr.isascii() and rr.isdigit()) or int(rr) < 1:
                raise ValueError(f"NETT_DVS_GLOBAL_SUB={raw_gs!r}: expected 'median:R' with an integer R >= 1.")
            r = int(rr)
        named = [n for n, v in (("NETT_DVS_MOTION_COMP", mc), ("NETT_DVS_GATE_K", k), ("NETT_DVS_GLOBAL_SUB", r))
                 if v is not None]
        if len(named) > 1:
            raise ValueError(f"{' and '.join(named)}: the egomotion knobs are one arm each (FINDINGS add. 351). "
                             f"Set at most one.")
        if named and window is not None:
            raise ValueError(f"{named[0]} with NETT_DVS_WINDOW={window}: the probe defined it on one tick's events. "
                             f"Refusing.")
        if named and move_gain is not None:
            raise ValueError(f"{named[0]} with NETT_DVS_MOVE_GAIN={move_gain:g}: two egomotion treatments on one arm. "
                             f"Refusing.")
        if mc is not None:
            print("[NETT dvs motion comp] yaw: the previous frame is warped into the current camera by the world-"
                  "vertical yaw change (300 deg equisolid, bilinear; out-of-view sources emit nothing) "
                  "(FINDINGS add. 351)", flush=True)
        if k is not None:
            print(f"[NETT dvs gate] k={k:g}: every tick's events x max(0, 1 - {k:g} |clip(a,-1,1)|/sqrt 2) "
                  f"(FINDINGS add. 351)", flush=True)
        if r is not None:
            print(f"[NETT dvs global sub] median:{r}: per-plane difference minus its {2 * r + 1}x{2 * r + 1} median "
                  f"(replicate padding) before thresholding (FINDINGS add. 351)", flush=True)
        return mc, k, r

    def _apply_gate(self, events, action):
        """A tick's events x max(0, 1 - k |clip(a, -1, 1)| / sqrt 2), per env. The gain is the probe's own NumPy
        expression on the float32 action (ana.py, section (c)), so the dtype path is the probe's."""
        a = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        a = a.astype(np.float32)
        n = int(events.shape[0]) if self._batched else 1
        if a.ndim == 0 or (self._batched and (a.ndim != 2 or a.shape[0] != n)):
            raise RuntimeError(f"NETT_DVS_GATE_K: action {tuple(a.shape)} does not give one action vector per env "
                               f"for events {tuple(events.shape)} (batched={self._batched}).")
        a = a.reshape(n, -1)
        g = np.maximum(0.0, 1 - self.gate_k * (np.linalg.norm(np.clip(a, -1, 1), axis=-1) / np.sqrt(2)))
        g = g.astype(np.float64)
        if not self._batched:
            g = g[0]
        if isinstance(events, torch.Tensor):
            gt = torch.as_tensor(g, dtype=torch.float64, device=events.device)
            gt = gt.reshape((-1,) + (1,) * (events.ndim - 1)) if self._batched else gt
            return (events.to(torch.float64) * gt).round().to(torch.uint8)
        gn = g.reshape((-1,) + (1,) * (events.ndim - 1)) if self._batched else g
        return np.round(events.astype(np.float64) * gn).astype(np.uint8)

    def _find_unwrapped(self, attr: str, knob: str):
        """Walk ``.env`` / ``._env`` down to the env that carries ``attr`` (as ``_move_mask`` does)."""
        current, seen = self.env, set()
        while current is not None and id(current) not in seen and not hasattr(current, attr):
            seen.add(id(current))
            current = getattr(current, "_env", None) or getattr(current, "env", None)
        if current is None or not hasattr(current, attr):
            raise RuntimeError(f"{knob}: no env under dvs_polarity carries {attr}.")
        return current

    def _mc_pose(self) -> np.ndarray:
        """The centre camera's rotation (N,3,3) float64, body -> world, from the unwrapped env (the probe's hook read
        ``_cameras["center"].data.quat_w_world``). Builds the warp for the env's lens on first use."""
        if self._mc is None:
            raw = self._find_unwrapped("_cameras", "NETT_DVS_MOTION_COMP")
            if "center" not in raw._cameras:
                raise RuntimeError(f"NETT_DVS_MOTION_COMP: the env has no 'center' camera ({sorted(raw._cameras)}); "
                                   f"the probe warped the centre eye only.")
            from nett_isaac.lens import LensSpec

            lens = LensSpec.from_observation_cfg(raw.cfg.observation)
            got = (lens.method, lens.fov_h_deg, lens.width, lens.height)
            if got != _YawWarp.PROBE_LENS:
                raise ValueError(f"NETT_DVS_MOTION_COMP: the env's lens is {got}; the add. 351 probe defined and "
                                 f"checked the warp only at {_YawWarp.PROBE_LENS} (method, fov, width, height).")
            self._mc_raw, self._mc = raw, _YawWarp(lens)
        q = self._mc_raw._cameras["center"].data.quat_w_world
        q = q.detach().cpu().numpy() if isinstance(q, torch.Tensor) else np.asarray(q)
        return _quat_to_R(q.reshape(-1, 4).astype(np.float64))

    def _warp_prev(self, y: torch.Tensor, pose: np.ndarray):
        """The previous blurred frame resampled into the current camera, and the (same-shape) mask of pixels whose
        source lies inside the previous view."""
        n = int(pose.shape[0])
        h, w = int(y.shape[-2]), int(y.shape[-1])
        if (w, h) != (self._mc.W, self._mc.H) or y.numel() % (n * h * w):
            raise RuntimeError(f"NETT_DVS_MOTION_COMP: frame {tuple(y.shape)} vs lens {self._mc.W}x{self._mc.H} and "
                               f"{n} camera poses.")
        grid, ok = self._mc.grid(self._pose, pose)
        prev = self._prev.reshape(n, -1, h, w)
        out = torch.nn.functional.grid_sample(prev, grid.to(prev.device), mode="bilinear", padding_mode="border",
                                              align_corners=False)
        ok = ok.to(prev.device)[:, None].expand(prev.shape)
        return out.reshape(y.shape), ok.reshape(y.shape)

    #: planes per median chunk: 48 x (9*9) x 128x80 float32 = 160 MB of unfolded windows
    MEDIAN_CHUNK_PLANES = 48

    def _median_sub(self, d: torch.Tensor) -> torch.Tensor:
        """d minus its per-plane (2R+1)^2 median, replicate padding (= scipy median_filter mode 'nearest'). The window
        count is odd, so the median is one of the window's values and the result equals scipy's exactly."""
        r = self.global_sub
        h, w = d.shape[-2:]
        flat = d.reshape(-1, 1, h, w)
        med = torch.empty_like(flat)
        for s in range(0, flat.shape[0], self.MEDIAN_CHUNK_PLANES):
            p = torch.nn.functional.pad(flat[s:s + self.MEDIAN_CHUNK_PLANES], (r, r, r, r), mode="replicate")
            cols = torch.nn.functional.unfold(p, 2 * r + 1)             # (b, (2R+1)^2, H*W)
            med[s:s + self.MEDIAN_CHUNK_PLANES] = cols.median(dim=1).values.reshape(-1, 1, h, w)
        return (flat - med).reshape(d.shape)

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

    def _density_plain(self, out: torch.Tensor) -> None:
        """Diagnostic only (never changes an output), NO window: the event map's non-zero fraction (pixels x channels) over
        the first DENSITY_DECISIONS steps that have a previous frame, printed once -- the density the §162 add. 3 floor
        (mean non-zero >= 1e-3) is read against when a row steps the threshold down (workspace DECISIONS §162 add. 3)."""
        if len(self._dens_plain) >= self.DENSITY_DECISIONS:
            return
        self._dens_plain.append(float((out > 0).float().mean()))
        if len(self._dens_plain) == self.DENSITY_DECISIONS:
            print(f"[NETT dvs] density over the first {len(self._dens_plain)} steps (no window, threshold {self.threshold:g}): "
                  f"non-zero {sum(self._dens_plain) / len(self._dens_plain):.6f} (fractions of pixels x channels)", flush=True)

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
        if x.shape[-3] != self.n_planes:
            raise ValueError(
                f"NETT_DVS_COLOR=1 needs a {self.n_planes}-channel frame here; got {x.shape[-3]} channels "
                "(3 = RGB; 4 only directly on fovea_4ch). More "
                "means a stacking wrapper ran first -- order dvs_polarity BEFORE framestack."
            )
        return x, chw

    def _events(self, policy, pose=None):
        """Frame -> (ON, OFF) uint8 (or 3 such pairs under NETT_DVS_COLOR), and update the previous-frame buffer.
        ``pose`` = this tick's camera rotation (N,3,3), motion comp only."""
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
        had_prev = not (self._prev is None or self._prev.shape != y.shape)
        if not had_prev:
            d = torch.zeros_like(y)
        elif self.motion_comp is not None:
            warped, ok = self._warp_prev(y, pose)
            d = torch.where(ok, y - warped, torch.zeros_like(y))       # = the probe's events x ok (threshold > 0)
        elif self.global_sub is not None:
            d = self._median_sub(y - self._prev)
        else:
            d = y - self._prev
        self._prev = y
        if self.motion_comp is not None:
            self._pose = pose
        on = (d >= self.threshold).to(torch.uint8) * 255
        off = (d <= -self.threshold).to(torch.uint8) * 255
        if self.color:
            # (..., P, H, W) x2 -> (..., P, 2, H, W) -> (..., 2P, H, W): R+, R-, G+, G-, B+, B- (then L+, L- at P = 4)
            out = torch.stack((on, off), dim=-3).flatten(-4, -3)
            if not chw:
                out = out.movedim(-3, -1)
        else:
            out = torch.stack((on, off), dim=-3 if chw else -1)
        if self.window is not None:
            out = self._accumulate(out)
            self._phase_c += 1
            self._density(y, out)
        elif had_prev:
            self._density_plain(out)
        if not as_tensor:
            return out.cpu().numpy().astype(np.uint8)
        return out

    # -- gym API --------------------------------------------------------------------------
    def reset(self, **kwargs):
        self._prev = None
        self._hist = []
        self._phase_c = 0
        obs, info = self.env.reset(**kwargs)
        pose = self._mc_pose() if self.motion_comp is not None else None
        return _replace_policy_obs(obs, self._events(_policy_obs(obs), pose)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        policy = _policy_obs(obs)
        pose = self._mc_pose() if self.motion_comp is not None else None
        events = self._events(policy, pose)
        if self.move_gain is not None:
            events = self._apply_move_gain(events)
        if self.gate_k is not None:
            events = self._apply_gate(events, action)
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


# -- NETT_DVS_MOTION_COMP geometry: the add. 351 probe's own functions (ego/probe/evlib.py), NumPy float64 as there ------
def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz -> (N,3,3) rotation (body -> world)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _yaw_of(R: np.ndarray) -> np.ndarray:
    """World yaw (about +Z) of the body +X axis."""
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def _Rz(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, i = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, o], -1), np.stack([s, c, o], -1), np.stack([o, o, i], -1)], -2)


class _YawWarp:
    """Pixel <-> camera ray for the equisolid eye, Isaac 'world' camera convention (forward +X, left +Y, up +Z), image
    u right = camera -Y and v down = camera -Z (SU = SV = -1: the probe fitted the u sign, residual 9.76 vs 23.57
    flipped and 17.69 unwarped; a yaw cannot tell the v sign apart and the physical convention was kept)."""

    #: (method, fov_h_deg, width, height) the probe ran and checked; anything else refuses
    PROBE_LENS = ("equisolid", 300.0, 128, 80)
    SU = SV = -1.0

    def __init__(self, lens):
        self.W, self.H = lens.width, lens.height
        self.cx, self.cy = lens.cx, lens.cy
        self.theta_max_h = lens.theta_max_h
        v, u = np.meshgrid(np.arange(self.H) + 0.5, np.arange(self.W) + 0.5, indexing="ij")
        du, dv = u - self.cx, v - self.cy
        r = np.hypot(du, dv)
        th = np.array([lens.theta_of_r(float(x)) for x in r.ravel()]).reshape(r.shape)
        phi = np.arctan2(dv, du)
        st = np.sin(th)
        self.rays = np.stack([np.cos(th), self.SU * st * np.cos(phi), self.SV * st * np.sin(phi)], -1)  # (H,W,3)
        self.valid = r < (self.cx - 1.0)        # inside the rim (rim mask margin)

    def project(self, d: np.ndarray):
        """(...,3) camera-frame rays -> (u, v) pixel coords and a validity mask (equisolid r = cx sin(t/2)/sin(tmax/2))."""
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        th = np.arccos(np.clip(d[..., 0], -1, 1))
        phi = np.arctan2(self.SV * d[..., 2], self.SU * d[..., 1])
        r = self.cx * np.sin(th / 2.0) / math.sin(self.theta_max_h / 2.0)
        u = self.cx + r * np.cos(phi)
        v = self.cy + r * np.sin(phi)
        ok = (r < self.cx - 1.0) & (u >= 0) & (u <= self.W) & (v >= 0) & (v <= self.H)
        return u, v, ok

    def grid(self, R_prev: np.ndarray, R_cur: np.ndarray):
        """Yaw-only: the current camera = the previous one turned about the WORLD vertical by the yaw change. Returns the
        ``grid_sample`` grid (N,H,W,2) float32 into the previous frame and the in-view mask (N,H,W) bool."""
        dyaw = _yaw_of(R_cur) - _yaw_of(R_prev)
        Mrel = np.einsum("nji,njk,nkl->nil", R_prev, _Rz(dyaw), R_prev)   # R_prev^T Rz(dyaw) R_prev
        d = np.einsum("nij,hwj->nhwi", Mrel, self.rays)                    # current ray in the previous camera
        u, v, ok = self.project(d)
        ok &= self.valid[None]
        gx = torch.as_tensor(u / self.W * 2 - 1, dtype=torch.float32)
        gy = torch.as_tensor(v / self.H * 2 - 1, dtype=torch.float32)
        return torch.stack((gx, gy), -1), torch.as_tensor(ok)
