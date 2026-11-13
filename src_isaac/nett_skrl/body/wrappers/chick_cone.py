"""Chick-cone colour input: each RGB pixel re-expressed as chicken single-cone catches (CHICK-CONE, step 2).

⭐ OWNER 2026-11-13 (presenter relay; workspace FINDINGS add. 421): "CHICK-CONE input for CLTT agents", step 2 scope
"just do it for the core four without CLTT, DVS, or Fovea". The transform is the step 1 proposal (workspace
notes/presenter/chickcone/STEP1_DRAFT.md §5, reviewed by researcher2 in notes/researcher2/chickcone/), computed by
the workspace tool tools/chick_cone.py and copied here as constants: repo A carries no spectra.

## What the frame becomes

Per pixel, with no neighbourhood term:

    u8 sRGB -> linear (IEC 61966-2-1) -> T3 @ rgb_lin -> sRGB-encode -> round -> u8

T3's rows are the white-normalised photon catches of the chicken LWS cone, the MWS cone, and VS + SWS pooled 1 : 2 by
receptor abundance, for a display with stand-in primary spectra. Its columns are linear R, G, B. They go out in the R,
G, B slots. Every row sums to 1 and every weight is >= 0, so:
  * a grey pixel (v, v, v) maps to itself exactly, for every v;
  * nothing clips;
  * the encoders keep their uint8 input and their own /255.
Saturated colours are desaturated; e.g. (252, 182, 20) -> (241, 182, 93).

## Exactness

The map is a fixed function of 2^24 inputs, so the wrapper builds it once as a uint8 lookup table, (2^24, 3), 48 MiB,
one per device. It does this in float64 numpy with the dot product in a fixed order and no BLAS, then gathers per
frame. The table is identical on CPU and GPU, and it equals tools/chick_cone.py `apply_transform` on all 2^24 colours
(tests/test_chick_cone.py pins the table's sha256). The boot banner prints that sha256 and refuses to run if:
  * it differs from LUT_SHA256;
  * a grey fails to map to itself;
  * the reference colour fails to map to its value.

No knobs: the label says which ran.

⚠ ORDER: `pre`, innermost, like `fovea_mask`: every downstream consumer (frame stacks included) sees the cone frame.
It refuses anything but 3 channels.
"""

from __future__ import annotations

import hashlib

import gymnasium as gym
import numpy as np
import torch

from ..observation import image_layout
from .grayscale import _policy_obs, _replace_policy_obs

# T3, rows (LWS, MWS, VS+SWS 1:2), cols linear (R, G, B); tools/chick_cone.py transform_matrix(default_model()), primary
# parameter set (workspace origin/main ad944ff97), float64 repr.
T3 = (
    (0.8077463376250353, 0.19193094621604365, 0.000322716158921003),
    (0.0400201199641955, 0.9202665741392116, 0.03971330589659291),
    (0.0120556364793512, 0.19549047835904163, 0.7924538851616072),
)
#: sha256 of the (2^24, 3) uint8 table, row index = (R << 16) | (G << 8) | B; equal to tools/chick_cone.py apply_transform.
LUT_SHA256 = "dae59f6ad05ac31d9b506ea835a84515188a3912d08a6be9a3c3f868096017d3"
#: one saturated reference colour (add. 383's Object1 yellow face, source mean) and its cone value
REF_IN, REF_OUT = (252, 182, 20), (241, 182, 93)

_LUT_NP: np.ndarray | None = None


def _lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _enc(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def build_lut() -> np.ndarray:
    """(2^24, 3) uint8: the cone value of every u8 RGB triple. Fixed-order float64; ~2 s, ~1 MB working set per slice."""
    T = np.asarray(T3, np.float64)
    L = _lin(np.arange(256))
    g = np.repeat(np.arange(256), 256)
    b = np.tile(np.arange(256), 256)
    l1, l2 = L[g], L[b]
    out = np.empty((1 << 24, 3), np.uint8)
    for r in range(256):
        l0 = L[r]
        q = np.stack([(l0 * T[k, 0] + l1 * T[k, 1]) + l2 * T[k, 2] for k in range(3)], -1)
        out[r << 16:(r + 1) << 16] = np.clip(np.round(255.0 * _enc(q)), 0, 255).astype(np.uint8)
    return out


def lut_np() -> np.ndarray:
    global _LUT_NP
    if _LUT_NP is None:
        _LUT_NP = build_lut()
    return _LUT_NP


def lut_sha256(lut: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(lut).tobytes()).hexdigest()


def check_lut(lut: np.ndarray) -> str:
    """Refuse a table that is not the registered one. Returns the banner's check text."""
    sha = lut_sha256(lut)
    if sha != LUT_SHA256:
        raise RuntimeError(f"ChickCone: lookup table sha256 {sha} != registered {LUT_SHA256} (float64 / libm drift?)")
    v = np.arange(256)
    grey = lut[(v << 16) | (v << 8) | v]
    if not np.array_equal(grey, np.stack([v, v, v], -1)):
        raise RuntimeError("ChickCone: a grey pixel does not map to itself")
    r, g, b = REF_IN
    got = tuple(int(x) for x in lut[(r << 16) | (g << 8) | b])
    if got != REF_OUT:
        raise RuntimeError(f"ChickCone: {REF_IN} -> {got}, expected {REF_OUT}")
    return f"LUT sha256 {sha} OK; grey 0..255 -> itself OK; {REF_IN} -> {got} OK"


class ChickCone(gym.ObservationWrapper):
    """RGB (HWC/NHWC or CHW/NCHW, uint8, numpy or torch on any device) -> chick-cone (LWS, MWS, VS+SWS) uint8."""

    def __init__(self, env):
        super().__init__(env)
        self._check_space(env.observation_space)
        self.observation_space = env.observation_space
        self._luts: dict = {}         # device -> (2^24, 3) uint8 tensor
        self._first_seen = False
        msg = check_lut(lut_np())
        rows = "; ".join(f"{n} " + " ".join(f"{w:.4f}" for w in r) for n, r in zip(("LWS", "MWS", "VS+SWS"), T3))
        print(f"[NETT chick_cone] per-pixel u8 sRGB -> lin -> T3 -> sRGB-encode -> u8 (CHICK-CONE add. 421); T3 rows "
              f"(cols R G B linear): {rows}; first-frame check: {msg}", flush=True)

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
                f"ChickCone received {c} channels; it consumes RAW RGB frames (3). Anything else means a "
                "stacking, grey or event wrapper ran first -- put chick_cone in `pre`, innermost.")

    def _lut(self, device):
        key = str(device)
        if key not in self._luts:
            self._luts[key] = torch.as_tensor(lut_np()).to(device)
        return self._luts[key]

    def observation(self, obs):
        policy = _policy_obs(obs)
        is_torch = isinstance(policy, torch.Tensor)
        t = policy if is_torch else torch.as_tensor(np.asarray(policy))
        chw = image_layout(tuple(t.shape[-3:])) == "chw"
        x = t.movedim(-3, -1) if chw else t     # -> (..., H, W, C)
        self._check_channels(x.shape[-1])
        # uint8 frames index directly; a float frame on the 0-255 scale (prepare_image_tensor accepts both) is rounded
        # and clamped first, and the output goes back in the input's dtype, as fovea_mask does
        xi = (x if x.dtype == torch.uint8 else x.round().clamp(0, 255)).to(torch.int64)
        idx = (xi[..., 0] << 16) | (xi[..., 1] << 8) | xi[..., 2]
        out = self._lut(x.device)[idx]          # (..., H, W, 3) uint8
        if not self._first_seen:
            self._first_seen = True
            grey = (xi[..., 0] == xi[..., 1]) & (xi[..., 1] == xi[..., 2])
            same = (out.to(torch.int64) == xi).all(-1)
            print(f"[NETT chick_cone] first frame: dtype {t.dtype}, shape {tuple(t.shape)}, grey px {int(grey.sum())} "
                  f"unchanged {int((grey & same).sum())}, non-grey px {int((~grey).sum())} changed "
                  f"{int((~grey & ~same).sum())}", flush=True)
        out = out.to(t.dtype)
        out = out.movedim(-1, -3) if chw else out
        out = out.contiguous()
        return _replace_policy_obs(obs, out if is_torch else out.numpy())
