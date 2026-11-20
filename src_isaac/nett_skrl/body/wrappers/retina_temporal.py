"""Retina-like TEMPORAL front-ends: fixed, stateful per-env filters over the tick stream (RETINA program, owner 2026-11-17).

One wrapper, one mode per arm, chosen by ``NETT_RETINA`` (required; unset refuses). Every mode runs on EVERY tick (the
decision loop sits outside the body wrappers), keeps its own per-env state across ticks, and is reseeded from the
current frame on an episode boundary, so no output ever straddles a reset. Input: raw RGB (3 channels, HWC/NHWC or
CHW/NCHW, uint8 or 0-255 float, numpy or torch on any device). Output: uint8, same layout, ``n_out`` channels.

Modes (``RET_MODES``):

  tr<K>     K-tick colour TRANSIENT, the K-frame generalisation of NETT_DVS_TRANSIENT (FINDINGS add. 420): per colour,
            up = min_{j != c} (f_c - f_j), down = min_{j != c} (f_j - f_c) over the last K ticks, c = the middle tick;
            kept where >= threshold, the value. 6 ch R+,R-,G+,G-,B+,B-. K odd, 3..9 (tr3 == the add. 420 rule).
  mtap<K>   MULTI-TAP reference DVS: per colour, d = f_t - sum_{j=1..K-1} w_j f_{t-j}, w_j ~ 0.6^(j-1) (normalised),
            i.e. change against an exponentially weighted K-1-tick past, not the single previous tick. ON/OFF where
            |d| >= threshold, the value. 6 ch. K 3..9.
  edr       multi-timescale log-ratio ON/OFF on luminance (Liu et al. 2018, "EDR"): r_h = 255 * ln((L+1)/(E_h+1)),
            E_h an EMA with half-life h in (1, 3, 8) ticks; ON = r - 255 ln 1.05, OFF = -r - 255 ln 1.05, clamped. 6 ch
            (ON1,OFF1,ON3,OFF3,ON8,OFF8).
  edrrgb    RGB (3) + edr (6) = 9 ch (EDR helped RL only when ADDED to the frames).
  tsurf     colour DVS events (single-tick difference, threshold) with an exponentially DECAYING time surface:
            S = max(S * exp(-1/tau), 255 * event). 6 ch. ``NETT_RETINA_TAU`` (default 2 ticks).
  smear     RGB exponential motion smear: E = EMA(RGB) with half-life ``NETT_RETINA_HALF`` (default 2 ticks). 3 ch.
  sustrans  chick-like two streams (Seifert et al. 2023): sustained colour = EMA(RGB, half-life 3) (3 ch) + fast
            luminance transient = EMA(L, 1) - EMA(L, 3), ON / OFF x 4 (gain), clamped (2 ch) = 5 ch.

MOTION-SENSOR modes (owner 2026-11-20: "There are lots of motion-based sensors. Try them out instead of giving up or trying
slight variations"). Each is a different CLASS of biological motion sensor, computed on luminance L (0-255), and each is
ADDED to the colour frame (RGB first, then the sensor maps), the two-stream layout: colour/form + motion.

  emd       Hassenstein-Reichardt correlation detector (fly EMD; Hassenstein & Reichardt 1956, Borst 2014): a temporally
            high-passed input h = L - EMA(L, half 4); per axis and offset d = 2 px,
            R = LP(h)(x) h(x+d) - h(x) LP(h)(x+d), LP = EMA half-life 1 tick; opponent, so R > 0 = motion toward +axis;
            CONTRAST-NORMALISED, R / (64 + Gaussian_2(h^2 + LP(h)^2)) (Borst & Helmstaedter 2015; without it the map
            tracks contrast x size, the failure of the change arms). 4 maps (right, left, down, up) = rectified x 255. 7 ch.
  me        Adelson-Bergen MOTION ENERGY (1985): luminance through an even/odd Gabor quadrature pair (sigma 2, period 8 px)
            per axis, each through a fast biphasic (EMA1 - EMA3) and a slow monophasic (EMA2 - EMA8) temporal filter;
            directional energies (Se Ts +/- So Tf)^2 + (So Ts -/+ Se Tf)^2, DIVISIVELY NORMALISED by the local energy pooled
            over both axes and directions (Simoncelli & Heeger 1998: N = E / (100 + Gaussian_3(sum E))), opponent
            difference, 4 maps (right, left, down, up) = rectified x 255. 7 ch.
  oms       OBJECT-MOTION-SENSITIVE retina (Olveczky, Baccus & Meister 2003; Baccus et al. 2008): rectified ON / OFF
            transient subunits b = relu(+-(L - EMA(L, half 1))); centre = Gaussian(b, sigma 1.5), surround = Gaussian(b,
            sigma 8) (the wide-field amacrine pool); OMS = relu(centre / (surround + 4) - 2) x 64, ON and OFF (divisive:
            a subtractive surround leaves textured global motion through wherever local contrast is high). Uniform
            (global, self-generated) image motion drives centre and surround alike and is suppressed; motion that
            differs from the background survives. 5 ch.
  loom      LGMD-style LOOMING / approach detector (Rind & Bramwell 1996): excitation E = |L_t - L_{t-1}|, inhibition
            I = the previous tick's E spread over the 8 neighbours (weight 1/8 each, x 0.8); S = relu(E - I) x 2, plus
            the per-env mean of S broadcast (the LGMD summation) x 8. 5 ch.
  flow      dense LUCAS-KANADE optic flow (Gaussian window sigma 2, Tikhonov 25) on Gaussian-smoothed L, with the per-env
            MEDIAN flow subtracted (the egomotion estimate), i.e. RELATIVE motion, the flow-discontinuity cue for
            figure-ground; 4 maps (u+, u-, v+, v-) of the residual, px/tick x 64. 7 ch.
  rand      CONTROL (capacity): 4 FIXED RANDOM spatiotemporal filters, channel-matched to emd / me / flow. Each is a
            seeded random 9x9 zero-mean spatial kernel on luminance, through a random signed mix of temporal states
            (EMA half-lives 1, 2, 4, 8 minus the frame itself), rectified, scaled to the EMD's output range. 7 ch.
  dvsgm     CONTROL (is egomotion cancellation alone the active ingredient?): luminance change d = L_t - L_{t-1} minus its
            per-env frame mean, ON / OFF where |.| >= threshold, the value. 5 ch.

``NETT_RETINA_DIAG=<path>`` (diagnostics only; never set in a registered arm): every 25 ticks append one JSON line with,
per env, the median and 95th-percentile |Lucas-Kanade flow| (background vs fastest image motion, px/tick) and the mean
of every added channel -- the pre-pilot checks (background image motion, pixel speed, fork / ship energy bias).

Knobs: ``NETT_RETINA_THRESHOLD`` (default 4, the DVS6 rows' threshold, on 0-255), ``NETT_RETINA_TAU``,
``NETT_RETINA_HALF``. ⛔ Put ``retina_temporal`` in ``pre``, innermost, BEFORE any framestack; it refuses non-RGB input.
"""
from __future__ import annotations

import json
import math
import os
import re

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from ..observation import image_layout
from .framestack import _done_mask, _policy_obs, _replace_policy_obs

RET_MODES = ("tr", "mtap", "edr", "edrrgb", "tsurf", "smear", "sustrans", "emd", "me", "oms", "loom", "flow", "dvsgm", "rand")
MOTION_MODES = ("emd", "me", "oms", "loom", "flow", "dvsgm", "rand")
_EDR_HALVES = (1.0, 3.0, 8.0)
_LUMA = (0.299, 0.587, 0.114)


def _alpha(half: float) -> float:
    """EMA weight on the NEW frame for a half-life of ``half`` ticks."""
    return 1.0 - 2.0 ** (-1.0 / half)


def parse_mode(raw: str):
    """``NETT_RETINA`` -> (base, K). Refuses anything unknown."""
    s = (raw or "").strip().lower()
    m = re.fullmatch(r"(tr|mtap)([0-9]+)", s)
    if m:
        k = int(m.group(2))
        if m.group(1) == "tr" and (k < 3 or k > 9 or k % 2 == 0):
            raise ValueError(f"NETT_RETINA={raw!r}: tr<K> needs an odd K in 3..9.")
        if m.group(1) == "mtap" and (k < 3 or k > 9):
            raise ValueError(f"NETT_RETINA={raw!r}: mtap<K> needs K in 3..9.")
        return m.group(1), k
    if s in ("edr", "edrrgb", "tsurf", "smear", "sustrans") + MOTION_MODES:
        return s, None
    raise ValueError(f"NETT_RETINA={raw!r}: expected one of tr<K>, mtap<K>, edr, edrrgb, tsurf, smear, sustrans, "
                     f"{', '.join(MOTION_MODES)}.")


def _float_knob(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r}: not a number.") from exc
    if not (lo <= v <= hi):
        raise ValueError(f"{name}={raw!r}: must be in [{lo}, {hi}].")
    return v


N_OUT = {"tr": 6, "mtap": 6, "edr": 6, "edrrgb": 9, "tsurf": 6, "smear": 3, "sustrans": 5,
         "emd": 7, "me": 7, "oms": 5, "loom": 5, "flow": 7, "dvsgm": 5, "rand": 7}


def _gauss1d(sigma: float, device, dtype) -> torch.Tensor:
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _sep_conv(x: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """(..., C, H, W) separable 'same' convolution (replicate padding) with 1-D kernels kx (along W) and ky (along H)."""
    lead, (c, h, w) = x.shape[:-3], x.shape[-3:]
    y = x.reshape(-1, 1, h, w)
    rx, ry = kx.numel() // 2, ky.numel() // 2
    y = F.conv2d(F.pad(y, (rx, rx, 0, 0), mode="replicate"), kx.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, ry, ry), mode="replicate"), ky.view(1, 1, -1, 1))
    return y.reshape(*lead, c, h, w)


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gauss1d(sigma, x.device, x.dtype)
    return _sep_conv(x, k, k)


def _shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """out(x) = in(x + d), replicate at the border: the neighbour d px along +x (right) / +y (down)."""
    h, w = x.shape[-2:]
    if dx:
        idx = (torch.arange(w, device=x.device) + dx).clamp(0, w - 1)
        x = x.index_select(-1, idx)
    if dy:
        idx = (torch.arange(h, device=x.device) + dy).clamp(0, h - 1)
        x = x.index_select(-2, idx)
    return x


def _lk(prev: torch.Tensor, cur: torch.Tensor):
    """Dense Lucas-Kanade flow (px/tick) between two Gaussian-smoothed luminance frames (..., 1, H, W): central-difference
    gradients of their mean, Gaussian window sigma 2, Tikhonov 25 on the structure tensor's diagonal."""
    avg = 0.5 * (cur + prev)
    ix = 0.5 * (_shift(avg, 1, 0) - _shift(avg, -1, 0))
    iy = 0.5 * (_shift(avg, 0, 1) - _shift(avg, 0, -1))
    it = cur - prev
    a, b, c = _blur(ix * ix, 2.0) + 25.0, _blur(ix * iy, 2.0), _blur(iy * iy, 2.0) + 25.0
    p, q = -_blur(ix * it, 2.0), -_blur(iy * it, 2.0)
    det = a * c - b * b
    return (c * p - b * q) / det, (a * q - b * p) / det


def _gabor_pair(sigma: float, period: float, device, dtype):
    r = int(math.ceil(2.5 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    even, odd = g * torch.cos(2 * math.pi * x / period), g * torch.sin(2 * math.pi * x / period)
    even = even - even.mean()
    n = even.abs().sum()
    return even / n, odd / n, _gauss1d(sigma, device, dtype)


class RetinaTemporal(gym.Wrapper):
    """Stateful per-env temporal filter; see the module docstring."""

    def __init__(self, env):
        super().__init__(env)
        raw = os.environ.get("NETT_RETINA")
        if raw is None or raw.strip() == "":
            raise ValueError("retina_temporal needs NETT_RETINA (tr<K> | mtap<K> | edr | edrrgb | tsurf | smear | sustrans).")
        self.mode, self.k = parse_mode(raw)
        self.threshold = _float_knob("NETT_RETINA_THRESHOLD", 4.0, 0.5, 255.0)
        self.tau = _float_knob("NETT_RETINA_TAU", 2.0, 0.25, 64.0)
        self.half = _float_knob("NETT_RETINA_HALF", 2.0, 0.25, 64.0)
        self.n_out = N_OUT[self.mode]
        if self.mode == "mtap":
            w = [0.6 ** (j - 1) for j in range(1, self.k)]
            self._mtap_w = [x / sum(w) for x in w]          # j = 1 (previous tick) first
        self._check_space(env.observation_space)
        self.observation_space = self._out_space(env.observation_space)
        self._diag = os.environ.get("NETT_RETINA_DIAG") or None   # diagnostics only, never in a registered arm
        self._diag_tick = 0
        self._state: dict = {}
        self._batched = False
        self._printed = False
        desc = {
            "tr": f"K={self.k} colour transient, middle tick vs the other {(self.k or 1) - 1}, threshold {self.threshold:g}, value",
            "mtap": f"K={self.k} multi-tap reference DVS, past weights {[round(x, 3) for x in getattr(self, '_mtap_w', [])]}, "
                    f"threshold {self.threshold:g}, value",
            "edr": f"log-ratio ON/OFF on luma, EMA half-lives {_EDR_HALVES} ticks, nu = ln 1.05",
            "edrrgb": f"RGB + edr (half-lives {_EDR_HALVES})",
            "tsurf": f"colour DVS time surface, threshold {self.threshold:g}, tau {self.tau:g} ticks",
            "smear": f"RGB EMA smear, half-life {self.half:g} ticks",
            "sustrans": "sustained RGB EMA (half-life 3) + luma EMA1-EMA3 ON/OFF x4",
            "emd": "RGB + Hassenstein-Reichardt EMD (HP half 4, delay EMA half 1, d 2 px), R/L/D/U",
            "me": "RGB + Adelson-Bergen motion energy (Gabor sigma 2 period 8; EMA1-EMA3 / EMA2-EMA8), R/L/D/U opponent",
            "oms": "RGB + object-motion-sensitive ON/OFF (transient subunits, centre sigma 1.5 / (surround sigma 8 + 4) - 2)",
            "loom": "RGB + LGMD looming S-layer (|dL| - 0.8 x spread previous) + its per-env mean",
            "flow": "RGB + Lucas-Kanade flow minus the per-env median (relative motion) u+/u-/v+/v-",
            "dvsgm": f"RGB + luma change minus its frame mean, ON/OFF, threshold {self.threshold:g} (control)",
            "rand": "RGB + 4 fixed random spatiotemporal filters, rectified (capacity control, seed 20261120)",
        }[self.mode]
        print(f"[NETT retina] mode {raw.strip()} -> {self.n_out} ch: {desc} ({'MOTION program, owner 2026-11-20' if self.mode in MOTION_MODES else 'RETINA program, owner 2026-11-17'})", flush=True)

    # -- spaces --------------------------------------------------------------------------
    def _check_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            for v in space.spaces.values():
                self._check_space(v)
            return
        shape = tuple(getattr(space, "shape", None) or ())
        if len(shape) < 3:
            return
        chw = image_layout(shape[-3:]) == "chw"
        c = shape[-3] if chw else shape[-1]
        if c != 3:
            raise ValueError(f"retina_temporal received {c} channels; it consumes raw RGB (3). Put it in `pre`, innermost.")

    def _out_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict({k: self._out_space(v) for k, v in space.spaces.items()})
        shape = tuple(space.shape)
        if len(shape) < 3:
            return space
        if image_layout(shape[-3:]) == "chw":
            out = shape[:-3] + (self.n_out,) + shape[-2:]
        else:
            out = shape[:-3] + shape[-3:-1] + (self.n_out,)
        return gym.spaces.Box(low=0, high=255, shape=out, dtype=np.uint8)

    # -- per-env state -------------------------------------------------------------------
    def _seed(self, y: torch.Tensor, idx=None) -> None:
        """(Re)seed the state from this frame: every env (idx None) or the envs in ``idx`` (an episode boundary)."""
        st = self._state

        def put(name, value):
            if idx is None or name not in st:
                st[name] = value.clone()
            else:
                st[name][idx] = value[idx]

        if self.mode == "tr" or self.mode == "mtap":
            if idx is None or "buf" not in st:
                st["buf"] = [y.clone() for _ in range(self.k)]           # oldest first
            else:
                for b in st["buf"]:
                    b[idx] = y[idx]
        elif self.mode in ("edr", "edrrgb"):
            lum = self._lum(y)
            for h in _EDR_HALVES:
                put(f"e{h}", lum)
        elif self.mode == "tsurf":
            put("prev", y)
            z = torch.zeros((*y.shape[:-3], 6, *y.shape[-2:]), dtype=torch.float32, device=y.device)
            put("surf", z)
        elif self.mode == "smear":
            put("ema", y)
        elif self.mode == "sustrans":
            put("ema3", y)
            lum = self._lum(y)
            put("l1", lum)
            put("l3", lum)
        elif self.mode == "emd":
            lum = self._lum(y)
            put("hp_ref", lum)
            put("lp", torch.zeros_like(lum))
        elif self.mode == "me":
            s4 = self._me_spatial(self._lum(y))
            for name in ("f1", "f3", "f2", "f8"):
                put(name, s4)
        elif self.mode == "oms":
            put("l1", self._lum(y))
        elif self.mode == "loom":
            lum = self._lum(y)
            put("prev", lum)
            put("e_prev", torch.zeros_like(lum))
        elif self.mode == "flow":
            put("prev", _blur(self._lum(y), 1.0))
        elif self.mode == "dvsgm":
            put("prev", self._lum(y))
        elif self.mode == "rand":
            lum = self._lum(y)
            for h in (1.0, 2.0, 4.0, 8.0):
                put(f"r{h}", lum)
        if self._diag and (idx is None or "diag_prev" not in st):
            st["diag_prev"] = _blur(self._lum(y), 1.0)
        elif self._diag:
            st["diag_prev"][idx] = _blur(self._lum(y), 1.0)[idx]

    @staticmethod
    def _lum(y: torch.Tensor) -> torch.Tensor:
        """(..., 3, H, W) -> (..., 1, H, W)."""
        w = torch.tensor(_LUMA, device=y.device, dtype=y.dtype).view(3, 1, 1)
        return (y * w).sum(-3, keepdim=True)

    # -- the transform ---------------------------------------------------------------------
    def _filter(self, y: torch.Tensor) -> torch.Tensor:
        """This tick's colour planes (..., 3, H, W) float 0-255 -> (..., n_out, H, W) float 0-255; updates state."""
        st, thr = self._state, self.threshold
        if self.mode == "tr":
            buf = st["buf"]
            buf.pop(0)
            buf.append(y.clone())
            c = self.k // 2
            fc = buf[c]
            others = torch.stack([b for j, b in enumerate(buf) if j != c], 0)
            up = (fc.unsqueeze(0) - others).amin(0)
            down = (others - fc.unsqueeze(0)).amin(0)
            up = torch.where(up >= thr, up, torch.zeros_like(up))
            down = torch.where(down >= thr, down, torch.zeros_like(down))
            return torch.stack((up, down), -3).flatten(-4, -3)
        if self.mode == "mtap":
            buf = st["buf"]
            ref = sum(w * buf[-j] for j, w in enumerate(self._mtap_w, start=1))   # buf[-1] = previous tick
            d = y - ref
            buf.pop(0)
            buf.append(y.clone())
            on = torch.where(d >= thr, d, torch.zeros_like(d))
            off = torch.where(-d >= thr, -d, torch.zeros_like(d))
            return torch.stack((on, off), -3).flatten(-4, -3)
        if self.mode in ("edr", "edrrgb"):
            lum = self._lum(y)
            nu = 255.0 * math.log(1.05)
            chans = []
            for h in _EDR_HALVES:
                e = st[f"e{h}"]
                r = 255.0 * (torch.log(lum + 1.0) - torch.log(e + 1.0))
                chans += [(r - nu).clamp(min=0.0), (-r - nu).clamp(min=0.0)]
                st[f"e{h}"] = e + _alpha(h) * (lum - e)
            out = torch.cat(chans, -3)
            return torch.cat((y, out), -3) if self.mode == "edrrgb" else out
        if self.mode == "tsurf":
            d = y - st["prev"]
            st["prev"] = y.clone()
            ev = torch.stack(((d >= thr), (d <= -thr)), -3).flatten(-4, -3).to(torch.float32) * 255.0
            st["surf"] = torch.maximum(st["surf"] * math.exp(-1.0 / self.tau), ev)
            return st["surf"]
        if self.mode == "smear":
            st["ema"] = st["ema"] + _alpha(self.half) * (y - st["ema"])
            return st["ema"]
        if self.mode in MOTION_MODES:
            return torch.cat((y, getattr(self, "_m_" + self.mode)(self._lum(y))), -3)
        # sustrans
        st["ema3"] = st["ema3"] + _alpha(3.0) * (y - st["ema3"])
        lum = self._lum(y)
        st["l1"] = st["l1"] + _alpha(1.0) * (lum - st["l1"])
        st["l3"] = st["l3"] + _alpha(3.0) * (lum - st["l3"])
        tr = 4.0 * (st["l1"] - st["l3"])
        return torch.cat((st["ema3"], tr.clamp(min=0.0), (-tr).clamp(min=0.0)), -3)

    # -- motion sensors (MOTION program, owner 2026-11-20); input L (..., 1, H, W) 0-255, output 0-255 maps ---------
    def _m_emd(self, lum):
        st = self._state
        st["hp_ref"] = st["hp_ref"] + _alpha(4.0) * (lum - st["hp_ref"])
        h = lum - st["hp_ref"]                                    # temporally high-passed input (photoreceptor/LMC)
        lp = st["lp"]                                             # delayed arm = LP(h) from the previous ticks
        out = []
        norm = 64.0 + _blur(h * h + lp * lp, 2.0)                 # contrast normalisation
        for dx, dy in ((2, 0), (0, 2)):
            hn, lpn = _shift(h, dx, dy), _shift(lp, dx, dy)
            r = (lp * hn - h * lpn) / norm                        # > 0: a feature moved from x to x + d
            out += [r.clamp(min=0.0), (-r).clamp(min=0.0)]
        st["lp"] = lp + _alpha(1.0) * (h - lp)
        return torch.cat(out, -3) * 255.0                         # right, left, down, up

    def _me_spatial(self, lum):
        """(..., 1, H, W) -> (..., 4, H, W): even_x, odd_x, even_y, odd_y (Gabor along x / along y)."""
        if not hasattr(self, "_gab"):
            self._gab = _gabor_pair(2.0, 8.0, lum.device, lum.dtype)
        e, o, g = self._gab
        return torch.cat((_sep_conv(lum, e, g), _sep_conv(lum, o, g), _sep_conv(lum, g, e), _sep_conv(lum, g, o)), -3)

    def _m_me(self, lum):
        st = self._state
        s4 = self._me_spatial(lum)
        for name, half in (("f1", 1.0), ("f3", 3.0), ("f2", 2.0), ("f8", 8.0)):
            st[name] = st[name] + _alpha(half) * (s4 - st[name])
        tf = st["f1"] - st["f3"]                                  # fast, biphasic
        ts = st["f2"] - st["f8"]                                  # slow, monophasic (DC removed)
        en = []
        for a in (0, 2):                                          # x axis (even 0, odd 1), y axis (even 2, odd 3)
            se_tf, so_tf, se_ts, so_ts = tf[..., a, :, :], tf[..., a + 1, :, :], ts[..., a, :, :], ts[..., a + 1, :, :]
            en += [(se_ts + so_tf) ** 2 + (so_ts - se_tf) ** 2, (se_ts - so_tf) ** 2 + (so_ts + se_tf) ** 2]
        en = torch.stack(en, -3)                                  # +x, -x, +y, -y energies
        n = en / (100.0 + _blur(en.sum(-3, keepdim=True), 3.0))  # Simoncelli-Heeger divisive normalisation
        out = []
        for a in (0, 2):
            opp = (n[..., a, :, :] - n[..., a + 1, :, :]).unsqueeze(-3)
            out += [opp.clamp(min=0.0), (-opp).clamp(min=0.0)]
        return torch.cat(out, -3) * 255.0

    def _m_oms(self, lum):
        st = self._state
        d = lum - st["l1"]
        st["l1"] = st["l1"] + _alpha(1.0) * d
        b = torch.cat((d.clamp(min=0.0), (-d).clamp(min=0.0)), -3)   # ON / OFF transient subunits
        return (_blur(b, 1.5) / (_blur(b, 8.0) + 4.0) - 2.0).clamp(min=0.0) * 64.0   # divisive surround

    def _m_loom(self, lum):
        st = self._state
        e = (lum - st["prev"]).abs()
        k = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], device=lum.device, dtype=lum.dtype) / 8.0
        h, w = lum.shape[-2:]
        spread = F.conv2d(F.pad(st["e_prev"].reshape(-1, 1, h, w), (1, 1, 1, 1), mode="replicate"), k.view(1, 1, 3, 3))
        s_ = (e - 0.8 * spread.reshape(e.shape)).clamp(min=0.0)
        st["prev"], st["e_prev"] = lum.clone(), e
        mean = s_.mean(dim=(-2, -1), keepdim=True).expand_as(s_)
        return torch.cat((s_ * 2.0, mean * 8.0), -3)

    def _m_rand(self, lum):
        st = self._state
        if not hasattr(self, "_rand_w"):
            g = torch.Generator().manual_seed(20261120)
            k = torch.randn(4, 1, 9, 9, generator=g)
            k = k - k.mean(dim=(-2, -1), keepdim=True)
            self._rand_k = (k / k.abs().sum(dim=(-2, -1), keepdim=True)).to(lum.device, lum.dtype)
            self._rand_w = torch.randn(4, 4, generator=g).to(lum.device, lum.dtype)   # filter x temporal state
        temps = []
        for h in (1.0, 2.0, 4.0, 8.0):
            st[f"r{h}"] = st[f"r{h}"] + _alpha(h) * (lum - st[f"r{h}"])
            temps.append(st[f"r{h}"] - lum)                       # zero on a static scene
        tmp = torch.cat(temps, -3)                                # (..., 4, H, W)
        h_, w_ = lum.shape[-2:]
        lead = lum.shape[:-3]
        mixed = torch.einsum("ft,...thw->...fhw", self._rand_w, tmp)  # 4 random temporal mixes
        x = F.pad(mixed.reshape(-1, 4, h_, w_), (4, 4, 4, 4), mode="replicate")
        out = F.conv2d(x, self._rand_k, groups=4).reshape(*lead, 4, h_, w_)
        return out.clamp(min=0.0) * 4.0

    def _m_dvsgm(self, lum):
        st = self._state
        d = lum - st["prev"]
        st["prev"] = lum.clone()
        d = d - d.mean(dim=(-2, -1), keepdim=True)                # cancel the frame-wide (self-motion) component
        thr = self.threshold
        return torch.cat((torch.where(d >= thr, d, torch.zeros_like(d)), torch.where(-d >= thr, -d, torch.zeros_like(d))), -3)

    def _m_flow(self, lum):
        st = self._state
        cur = _blur(lum, 1.0)
        prev = st["prev"]
        st["prev"] = cur
        u, v = _lk(prev, cur)
        lead = u.shape[:-3]
        mu = u.reshape(*lead, -1).median(-1).values.view(*lead, 1, 1, 1)
        mv = v.reshape(*lead, -1).median(-1).values.view(*lead, 1, 1, 1)
        u, v = (u - mu) * 64.0, (v - mv) * 64.0
        return torch.cat((u.clamp(min=0.0), (-u).clamp(min=0.0), v.clamp(min=0.0), (-v).clamp(min=0.0)), -3)

    def _diag_write(self, y, out):
        st = self._state
        self._diag_tick += 1
        cur = _blur(self._lum(y), 1.0)
        prev = st.get("diag_prev", cur)
        st["diag_prev"] = cur
        if self._diag_tick % 25:
            return
        u, v = _lk(prev, cur)
        mag = (u * u + v * v).sqrt().reshape(*u.shape[:-3], -1)
        ch = out[..., 3:, :, :].mean(dim=(-2, -1)) if out.shape[-3] > 3 else out.mean(dim=(-2, -1))
        rec = {"tick": self._diag_tick, "mode": self.mode,
               "flow_median": mag.median(-1).values.reshape(-1).tolist(),
               "flow_p95": mag.quantile(0.95, dim=-1).reshape(-1).tolist(),
               "chan_mean": ch.reshape(-1, ch.shape[-1]).tolist()}
        with open(self._diag, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _planes(self, policy):
        t = policy if isinstance(policy, torch.Tensor) else torch.as_tensor(np.asarray(policy))
        chw = image_layout(tuple(t.shape[-3:])) == "chw"
        x = t.to(torch.float32)
        if not chw:
            x = x.movedim(-1, -3)
        if x.shape[-3] != 3:
            raise ValueError(f"retina_temporal needs a 3-channel RGB frame; got {x.shape[-3]} channels.")
        return t, x, chw

    def _emit(self, t, out, chw):
        o = out.round().clamp(0, 255).to(torch.uint8)
        if not chw:
            o = o.movedim(-3, -1)
        o = o.contiguous()
        if not self._printed:
            self._printed = True
            print(f"[NETT retina] first output: in {tuple(t.shape)} {t.dtype} -> out {tuple(o.shape)}; "
                  f"non-zero fraction {float((o > 0).float().mean()):.4f}", flush=True)
        return o if isinstance(t, torch.Tensor) else o.cpu().numpy()

    # -- gym API ---------------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        t, y, chw = self._planes(_policy_obs(obs))
        self._batched = t.ndim == 4
        self._state = {}
        self._seed(y)
        out = self._filter(y)
        return _replace_policy_obs(obs, self._emit(t, out, chw)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        t, y, chw = self._planes(_policy_obs(obs))
        self._batched = t.ndim == 4
        if not self._state:
            self._seed(y)
        done = _done_mask(terminated, truncated)
        idx = None
        if self._batched:
            nz = np.where(np.asarray(done).ravel())[0]
            if len(nz):
                idx = torch.as_tensor(nz, dtype=torch.long, device=y.device)
                # ⚠ The new episode's first frame: reseed BEFORE filtering, so nothing differences across the cut.
                self._seed(y, idx)
        elif bool(np.asarray(done).any()):
            self._seed(y)
        out = self._filter(y)
        if self._diag:
            self._diag_write(y, out)
        return _replace_policy_obs(obs, self._emit(t, out, chw)), reward, terminated, truncated, info
