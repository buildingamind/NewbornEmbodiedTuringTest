"""CLTT-Win: CLTT-Ref with a SAMPLED positive-pair window (CLTT-WINDOW program, owner 2026-11-17).

Owner: "Given the success of random frames with CLTT on ViT, I would be interested in different ways to form positive
pairs (wider windows, windows with gradual drop offs, windows that include images from further away in time, etc)."

Everything that is not the pair rule is ``cltt_ref``'s: the projector (Linear-BN-ReLU-Linear, L2), the temperature
(``NETT_AUX_CLTT_REF_TEMP``, 0.5), the views (``_make_views``: the CURRENT frame, repeated over a video encoder's T
slots), one-stream negatives (one episode-contiguous window of one env), frame-IDENTITY masking of the anchor's and the
positive's own frames out of the candidate set (both directions), backbone gradients, and summing M positive draws per
anchor (cltt_ref sums its 2 offsets, so M defaults to 2).

What changes: every anchor in the window gets its positive at a SIGNED offset d (|d| from the scheme's weights, d != 0,
both directions allowed, truncated to the window), or -- under ``+x<P>`` -- with probability P a CROSS-ENV positive: a
random stored step of a different env (same imprint, different place and time; Aubret et al. 2023's "same object,
other context" positives). The window is the longest episode-contiguous span of one env, capped at NETT_AUX_BATCH
(BL2: 256 -> ~50 = one episode), so every anchor is used.

``NETT_AUX_CLTT_WIN`` (required) = ``<base>[+x<P>]``, base one of
  ``uni:W``      |d| uniform on 1..W
  ``lin:W``      |d| with weight W + 1 - |d| on 1..W (linear drop-off; CVRL's best shape)
  ``exp:T:W``    |d| with weight exp(-(|d| - 1) / T) on 1..W (gradual exponential drop-off)
  ``epi``        |d| uniform over the whole window (any other step of the episode)
``NETT_AUX_CLTT_WIN_M`` = positive draws per anchor (default 2, cltt_ref's offset count).
"""

from __future__ import annotations

import math
import os
import re

import torch
import torch.nn.functional as F
from skrl import logger

from .cltt_ref_aux import (CLTTReferenceAuxLoss, dedup_is_exact, episode_window_batch, draw_episode_window,
                           nt_xent_diagnostics_absent)

WIN_ENV = "NETT_AUX_CLTT_WIN"
M_ENV = "NETT_AUX_CLTT_WIN_M"


def parse_scheme(raw: str):
    """-> (base, params, p_xenv). Refuses anything unknown."""
    s = (raw or "").strip().lower()
    m = re.fullmatch(r"(uni:\d+|lin:\d+|exp:[0-9.]+:\d+|epi)(?:\+x([0-9.]+))?", s)
    if not m:
        raise ValueError(f"{WIN_ENV}={raw!r}: expected uni:W | lin:W | exp:T:W | epi, optionally +x<P> (0 < P < 1).")
    base, *rest = m.group(1).split(":")
    params = [float(x) for x in rest]
    if base in ("uni", "lin") and not (1 <= params[0] <= 64):
        raise ValueError(f"{WIN_ENV}={raw!r}: W must be 1..64.")
    if base == "exp" and not (params[0] > 0 and 1 <= params[1] <= 64):
        raise ValueError(f"{WIN_ENV}={raw!r}: exp:T:W needs T > 0 and W in 1..64.")
    p = float(m.group(2)) if m.group(2) else 0.0
    if m.group(2) and not (0.0 < p < 1.0):
        raise ValueError(f"{WIN_ENV}={raw!r}: +x<P> needs 0 < P < 1.")
    return base, params, p


def offset_weights(base, params, length):
    """Weight of |d| = 1 .. length-1 (index 0 = |d| 1)."""
    k = torch.arange(1, max(length, 2), dtype=torch.float64)
    if base == "epi":
        w = torch.ones_like(k)
    else:
        wmax = int(params[-1])
        if base == "uni":
            w = torch.ones_like(k)
        elif base == "lin":
            w = (wmax + 1 - k).clamp(min=0)
        else:
            w = torch.exp(-(k - 1) / params[0])
        w = torch.where(k <= wmax, w, torch.zeros_like(w))
    return w


def sample_offsets(base, params, length, generator=None):
    """(length,) signed offsets d_i, each with i + d_i in [0, length), d_i != 0, |d_i| ~ the scheme's weights
    restricted to what is in the window for anchor i (both directions)."""
    w = offset_weights(base, params, length)                         # (length-1,)
    d = torch.arange(-(length - 1), length, dtype=torch.long)       # all signed offsets
    wd = torch.cat([w.flip(0), torch.zeros(1, dtype=w.dtype), w])   # weight of each signed offset
    i = torch.arange(length)[:, None]
    tgt = i + d[None, :]
    ok = (tgt >= 0) & (tgt < length)
    W = wd[None, :].expand(length, -1) * ok
    if bool((W.sum(1) <= 0).any()):
        raise ValueError(f"CLTT-Win: some anchor has no admissible positive (window {length}, scheme {base}{params}).")
    pick = torch.multinomial(W.float(), 1, generator=generator).squeeze(1)
    return d[pick]


def exclusion_mask(keys_a: torch.Tensor, keys_p: torch.Tensor) -> torch.Tensor:
    """(2B, 2B) bool: candidates that leave row r -- frames IDENTICAL to r's own or to its positive's, other than r and
    its positive themselves -- made symmetric (row r drops c iff row c drops r). Self and positive are NOT set here."""
    b = keys_a.shape[0]
    keys = torch.cat([keys_a, keys_p])
    partner = (torch.arange(2 * b, device=keys.device) + b) % (2 * b)
    same_self = keys[:, None] == keys[None, :]
    same_pos = keys[partner][:, None] == keys[None, :]
    ex = same_self | same_pos
    eye = torch.eye(2 * b, dtype=torch.bool, device=keys.device)
    ex &= ~eye
    ex[torch.arange(2 * b, device=keys.device), partner] = False
    ex = ex | ex.t()
    ex &= ~eye
    ex[torch.arange(2 * b, device=keys.device), partner] = False
    return ex


def nt_xent_keyed(za, zp, temperature, keys_a, keys_p):
    """NT-Xent over 2B rows with frame-identity exclusions. -> (loss, n_excluded, mean ln|candidates|)."""
    b = za.shape[0]
    z = torch.cat([za, zp])
    sim = torch.mm(z, z.t()) / temperature
    ex = exclusion_mask(keys_a, keys_p)
    eye = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(ex | eye, float("-inf"))
    labels = (torch.arange(2 * b, device=z.device) + b) % (2 * b)
    cand = (2 * b - 1) - ex.sum(1)
    return F.cross_entropy(sim, labels), int(ex.sum()), float(torch.log(cand.double()).mean())


class CLTTWindowAuxLoss(CLTTReferenceAuxLoss):
    """``cltt_ref`` with a sampled signed-offset (and optional cross-env) positive; see the module docstring."""

    def __init__(self, encoder) -> None:
        super().__init__(encoder)
        raw = os.environ.get(WIN_ENV)
        if raw is None or raw.strip() == "":
            raise ValueError(f"cltt_win needs {WIN_ENV} (uni:W | lin:W | exp:T:W | epi [+x<P>]).")
        self.scheme_raw = raw.strip()
        self.base, self.params, self.p_xenv = parse_scheme(raw)
        m = os.environ.get(M_ENV, "2").strip()
        if not m.isdigit() or not (1 <= int(m) <= 8):
            raise ValueError(f"{M_ENV}={m!r}: 1..8.")
        self.m = int(m)
        self._logged = False

    def _encode(self, prepared, encoder):
        return encoder.encode_prepared(self._make_views([prepared], encoder)[0])

    def compute(self, encoder, observations):
        if self._memory is None:
            raise RuntimeError("CLTTWindowAuxLoss draws its own windows; call attach_memory(memory) first.")
        memory = self._memory
        raw = memory.tensors["observations"]
        t_max = memory.memory_size if memory.filled else memory.memory_index
        # The window = the longest episode-contiguous run of one env, capped at NETT_AUX_BATCH (offsets (0,) = the anchor
        # alone must be valid; every anchor in [t0, t0+L) is then in one episode with every other).
        L, starts = episode_window_batch(memory, (0,), min(self.max_samples, t_max))
        env, t0 = draw_episode_window(starts, draw_single_start=True)
        device = next(encoder.parameters()).device
        n_env = raw.shape[1]
        with torch.no_grad():
            span = encoder._prepare_image(raw[t0:t0 + L, env].to(device))
        keys_a = (env * t_max + t0 + torch.arange(L)).to(device)
        dedup, _ = dedup_is_exact(encoder)
        z_span = self._encode(span, encoder)
        h_a = self.head(z_span)
        total, excluded, floor, mean_abs, n_x = 0.0, 0, 0.0, 0.0, 0
        for _ in range(self.m):
            d = sample_offsets(self.base, self.params, L)
            idx = torch.arange(L) + d
            xmask = torch.zeros(L, dtype=torch.bool)
            if self.p_xenv > 0 and n_env > 1:
                xmask = torch.rand(L) < self.p_xenv
            n_xi = int(xmask.sum())
            keys_p = (env * t_max + t0 + idx).to(device)
            if dedup:
                z_in = z_span[idx.to(device)]
            else:
                with torch.no_grad():
                    pos = encoder._prepare_image(raw[(t0 + idx).tolist(), env].to(device))
                z_in = self._encode(pos, encoder)
            if n_xi:
                other = torch.randint(n_env - 1, (n_xi,))
                other = other + (other >= env).long()                    # any env but this one
                tt = torch.randint(t_max, (n_xi,))
                with torch.no_grad():
                    xp = encoder._prepare_image(raw[tt.tolist(), other.tolist()].to(device))
                z_x = self._encode(xp, encoder)
                xi = xmask.nonzero().flatten().to(device)
                z_in = z_in.clone()
                z_in[xi] = z_x
                keys_p[xi] = (other * t_max + tt).to(device)
            h_p = self.head(z_in)
            loss, ex, fl = nt_xent_keyed(h_a, h_p, self.temperature, keys_a, keys_p)
            total = total + loss
            excluded += ex
            floor += fl
            sel = ~xmask
            mean_abs += float(d[sel].abs().double().mean()) if bool(sel.any()) else 0.0
            n_x += n_xi
        if not self._logged:
            self._logged = True
            logger.info("CLTTWindowAuxLoss: scheme %s (base %s %s, cross-env P=%s), M=%s draws, window L=%s of env %s "
                        "(t_max=%s, NETT_AUX_BATCH=%s), dedup=%s; mean |d| first update %.2f",
                        self.scheme_raw, self.base, self.params, self.p_xenv, self.m, L, env, t_max,
                        self.max_samples, dedup, mean_abs / self.m)
        self.last_scalars = {"B": float(L), "t_max": float(t_max), "encode_dedup": float(dedup),
                             "same_frame_masked": 1.0, "mask_regime": 3.0,       # 3 = keyed frame-identity (cltt_win)
                             "masked_pairs": float(excluded), "masked_frac": float(excluded) / float(self.m * (2 * L) ** 2),
                             "chance": floor, "win_mean_abs_d": mean_abs / self.m,
                             "win_xenv_frac": n_x / float(self.m * L)}
        self.last_scalars.update(nt_xent_diagnostics_absent(L))
        return total
