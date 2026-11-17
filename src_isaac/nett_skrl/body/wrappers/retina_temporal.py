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

Knobs: ``NETT_RETINA_THRESHOLD`` (default 4, the DVS6 rows' threshold, on 0-255), ``NETT_RETINA_TAU``,
``NETT_RETINA_HALF``. ⛔ Put ``retina_temporal`` in ``pre``, innermost, BEFORE any framestack; it refuses non-RGB input.
"""
from __future__ import annotations

import math
import os
import re

import gymnasium as gym
import numpy as np
import torch

from ..observation import image_layout
from .framestack import _done_mask, _policy_obs, _replace_policy_obs

RET_MODES = ("tr", "mtap", "edr", "edrrgb", "tsurf", "smear", "sustrans")
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
    if s in ("edr", "edrrgb", "tsurf", "smear", "sustrans"):
        return s, None
    raise ValueError(f"NETT_RETINA={raw!r}: expected one of tr<K>, mtap<K>, edr, edrrgb, tsurf, smear, sustrans.")


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


N_OUT = {"tr": 6, "mtap": 6, "edr": 6, "edrrgb": 9, "tsurf": 6, "smear": 3, "sustrans": 5}


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
        }[self.mode]
        print(f"[NETT retina] mode {raw.strip()} -> {self.n_out} ch: {desc} (RETINA program, owner 2026-11-17)", flush=True)

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
        # sustrans
        st["ema3"] = st["ema3"] + _alpha(3.0) * (y - st["ema3"])
        lum = self._lum(y)
        st["l1"] = st["l1"] + _alpha(1.0) * (lum - st["l1"])
        st["l3"] = st["l3"] + _alpha(3.0) * (lum - st["l3"])
        tr = 4.0 * (st["l1"] - st["l3"])
        return torch.cat((st["ema3"], tr.clamp(min=0.0), (-tr).clamp(min=0.0)), -3)

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
        return _replace_policy_obs(obs, self._emit(t, out, chw)), reward, terminated, truncated, info
