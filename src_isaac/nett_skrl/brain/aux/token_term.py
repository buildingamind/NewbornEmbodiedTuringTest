"""Shared machinery for wave-17 token-level auxiliary TERMS.

A "term" is the new half of a wave-17 row: `WithCLTTRef(encoder, term, name)` adds it to the
incumbent `cltt_ref`, owns the single EMA teacher, and exposes one `head`. Everything the four
terms do identically lives here:

* the per-aux batch/offset knobs (⛔ never `NETT_AUX_BATCH`, which `cltt_ref` reads with default
  512 and `slot_contrast` with default 32 -- in a composition row setting it for one term
  silently resizes the other);
* drawing ONE action-aligned (t, t+k) window through the P0 sampler, so every term inherits the
  episode/ring-seam safety and the verified action alignment, and no second sampler exists;
* the parked/transit stratification every diagnostic in the research spec is split on;
* the sentinel discipline: a statistic that cannot be computed emits `NOT_MEASURED`, never
  silence and never a plausible-looking zero.

⚠ THE DEFAULT OFFSET IS 8 STEPS FOR EVERY TERM, and it is the load-bearing hyperparameter of the
whole wave (research spec §0.1). At k = 1 a patch moves ~0.1 of a patch width in transit and
~0.02 parked, so `argmax_j sim(z_i^t, z_j^{t+k}) = i` for essentially every token: the affinity
target becomes the identity, the ego forward model learns the identity whatever the action was,
and the DenseCL correspondence degenerates to same-location. Every term therefore reports its
IDENTITY FRACTION, split parked/transit -- that is the I77 "engaged vs not engaged" gate, and it
is required, not optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .action_windows import draw_action_window
from .knobs import _env_positive_int

#: A statistic that could not be computed (empty stratum, degenerate null, diagnostic off).
#: ⛔ Negative and distinctive: every real quantity these terms emit is either a probability, a
#: fraction, a norm or a correlation, so a plausible-looking 0.0 would be indistinguishable from
#: "the objective is perfectly degenerate", which is exactly the reading it must not permit.
NOT_MEASURED = -9.0

#: No window drawn yet (vicreg_tt's MASK_UNSET, same value so track_transit_mask agrees).
MASK_UNSET = -4.0


@dataclass
class TokenWindow:
    """One (t, t+k) window, prepared for the encoder, with its aligned cumulative action."""

    prepared_t: torch.Tensor      # (B, C, H, W) normalized
    prepared_tk: torch.Tensor     # (B, C, H, W) normalized
    actions: torch.Tensor         # (B, k, A) a_t .. a_{t+k-1}
    a_bar: torch.Tensor           # (B, A) cumulative command over [t, t+k)
    mean_turn: float              # the sampler's state/value (>= 0 engaged, else a sentinel)
    env: int
    t0: int

    @property
    def turn(self) -> torch.Tensor:
        """|cumulative turn| per sample -- the quantity parked/transit is split on."""
        return self.a_bar[:, 0].abs()


def parked_transit(turn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks (parked, transit), split at the batch MEDIAN of |cumulative turn|.

    ⛔ A median split, not a threshold: the action->degrees gain is not known here and any
    constant would be an uncalibrated magic number (the reasoning vicreg_tt's transit mask
    already records). The split is therefore always balanced, and a batch with no motion at all
    is caught by the callers' `MASK_NO_MOTION` sentinel rather than by a silent empty stratum.
    """
    if turn.numel() < 2:
        z = torch.zeros_like(turn, dtype=torch.bool)
        return z, z
    med = turn.median()
    transit = turn > med
    parked = ~transit
    return parked, transit


def stratum_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of `values` over `mask`, or NOT_MEASURED when the stratum is empty."""
    if mask.dtype != torch.bool or not bool(mask.any()):
        return NOT_MEASURED
    return float(values[mask].mean())


def rank_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation of two 1-D tensors, or NOT_MEASURED if either is constant.

    ⚠ A zero-variance input has no correlation, and returning 0.0 there would read as "measured,
    and there is no relationship" -- the false negative this whole file's sentinel rule exists
    to prevent.
    """
    if x.numel() < 3 or y.numel() != x.numel():
        return NOT_MEASURED
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = float(x.norm() * y.norm())
    if denom < 1e-12:
        return NOT_MEASURED
    return float((x * y).sum() / denom)


def column_shift(match: torch.Tensor, n_w: int) -> torch.Tensor:
    """MEDIAN horizontal displacement, in token columns, of a token->token correspondence.

    ⛔ MEDIAN, NOT MEAN, AND THIS IS NOT ROBUSTNESS COSMETICS. Columns are a bounded index: a
    token pushed off one edge is matched somewhere on the other side, contributing −(n_w−1)
    against the +1 of its neighbours. On a fixture whose content wraps exactly, the MEAN
    displacement is 0 for EVERY shift -- the statistic has zero variance across the batch and the
    correlation below reads "not measured" on a fixture built to carry the signal. (Measured: a
    one-column roll of a 5x8 grid gives mean 0.0, median +1.0.) Real frames do not wrap, but they
    do produce edge mismatches and near-tied argmaxes on uniform background, which is the same
    contamination in smaller doses.

    `match` is (B, N) with `match[b, i] = j`, the destination token index matched to source i.
    On a coplanar monitor, turning shifts the image horizontally, so this is the free
    ground-truth check: correlate it with the window's cumulative turn (research spec §1.5).
    The SIGN convention is UNVERIFIED -- the magnitude against the permuted-action null is what
    is read, not the sign.
    """
    src_col = (torch.arange(match.shape[1], device=match.device) % n_w).float()
    dst_col = (match % n_w).float()
    return (dst_col - src_col).median(dim=1).values


class TokenWindowTerm(nn.Module):
    """Base for a wave-17 term: knobs, one window draw, teacher access, sentinel discipline.

    Subclasses set the four class attributes, build `self.head` (⛔ EVERY trainable parameter of
    the term, and nothing of the encoder: `head` is the only thing AuxLossPPO optimizes), and
    implement `_core(encoder, window) -> (loss, scalars)`.
    """

    needs_memory = True
    #: The composite builds one EMA teacher and attaches it when this is True.
    needs_teacher = True
    #: Sampling: whether the window draw is weighted by mean |turn| (research spec §0.2).
    TRANSIT_WEIGHTED = True

    BATCH_ENV = "NETT_AUX_BATCH"       # subclasses MUST override; see the module docstring
    OFFSET_ENV = "NETT_AUX_OFFSET"
    DEFAULT_BATCH = 32
    DEFAULT_OFFSET = 8

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        if self.BATCH_ENV == TokenWindowTerm.BATCH_ENV:
            raise TypeError(
                f"{type(self).__name__} must define its OWN batch knob: NETT_AUX_BATCH is read "
                f"by cltt_ref (default 512) and slot_contrast (default 32), so a composition "
                f"row that sets it for one term silently resizes the other.")
        self.batch = _env_positive_int(self.BATCH_ENV, self.DEFAULT_BATCH)
        self.offset = _env_positive_int(self.OFFSET_ENV, self.DEFAULT_OFFSET)
        self.n_tokens, self.grid, self.token_dim = self._probe(encoder)
        self._memory = None
        self._teacher = None
        self.last_scalars: dict = {}
        #: Read by ppo_aux.track_transit_mask through the composite.
        self.last_window_turn: float = MASK_UNSET

    @staticmethod
    def _probe(encoder: nn.Module) -> tuple[int, tuple[int, int], int]:
        """N, (n_h, n_w) and D from the encoder itself -- never hardcoded.

        The campaign eye gives 40 tokens of 144 dims at patch 16, but `patch_size` and
        `embed_dim` are row knobs (wave 15 moves both), and a term that assumed 40x144 would
        build a head of the wrong shape for the P8 row and fail at its first forward.
        """
        from ...body.observation import image_channels_hw
        from .token_features import spatial_tokens
        c, h, w = image_channels_hw(getattr(encoder, "observation_space", None))
        device = next(encoder.parameters()).device
        with torch.no_grad():
            tokens, grid = spatial_tokens(encoder, torch.zeros(1, c, h, w, device=device))
        return int(tokens.shape[1]), (int(grid[0]), int(grid[1])), int(tokens.shape[2])

    def attach_memory(self, memory) -> None:
        self._memory = memory

    def attach_teacher(self, teacher) -> None:
        self._teacher = teacher

    @property
    def teacher(self):
        if self._teacher is None:
            raise RuntimeError(
                f"{type(self).__name__} needs the shared EMA teacher: its target must come from "
                f"a stop-grad copy of the trunk, or the objective's optimum is a constant z. "
                f"Build it through WithCLTTRef, which owns the one teacher.")
        return self._teacher

    def draw(self, encoder: nn.Module, *, transit_weighted: bool | None = None,
             batch: int | None = None) -> TokenWindow:
        """One episode-contiguous (t, t+k) window with its aligned actions, prepared for input."""
        if self._memory is None:
            raise RuntimeError(
                f"{type(self).__name__} draws its own temporal windows; "
                f"call attach_memory(memory) before compute().")
        weighted = self.TRANSIT_WEIGHTED if transit_weighted is None else bool(transit_weighted)
        device = next(encoder.parameters()).device
        win = draw_action_window(self._memory, self.offset, batch or self.batch,
                                 transit_weighted=weighted, device=device)
        # ⛔ Prepared under no_grad and re-encoded through `_skip_prepare`, as every other
        # memory-drawing aux does: `_prepare_image` owns the HWC->CHW permute and the single
        # /255, and applying it twice reaches the trunk at 1/255 magnitude.
        with torch.no_grad():
            prepared_t = encoder._prepare_image(win.obs_t)
            prepared_tk = encoder._prepare_image(win.obs_tk)
        return TokenWindow(prepared_t=prepared_t, prepared_tk=prepared_tk, actions=win.actions,
                           a_bar=win.actions.sum(dim=1), mean_turn=win.mean_turn,
                           env=win.env, t0=win.t0)

    def draw_mixed(self, encoder: nn.Module, *, transit_frac: float,
                   batch: int | None = None) -> TokenWindow:
        """Half the batch transit-weighted, half uniform, as TWO draws of the shared sampler.

        ⛔ TWO HALF-SLABS, NOT ONE COIN FLIP PER UPDATE. A uniform slab contains zero transit
        steps 63.3% of the time (measured over 35 archived parsing arms; vicreg_tt_aux.py:151),
        so a per-update Bernoulli choice would leave most updates with one mode only -- and every
        diagnostic here is split parked/transit at the batch median, which is degenerate when a
        stratum is empty. Mixing WITHIN the batch keeps both strata populated every call.
        """
        total = batch or self.batch
        n_transit = int(round(float(transit_frac) * total))
        parts, turns = [], []
        for size, weighted in ((n_transit, True), (total - n_transit, False)):
            if size <= 0:
                continue
            win = self.draw(encoder, transit_weighted=weighted, batch=size)
            parts.append(win)
            if weighted:
                turns.append(win.mean_turn)
        if len(parts) == 1:
            return parts[0]
        return TokenWindow(
            prepared_t=torch.cat([p.prepared_t for p in parts]),
            prepared_tk=torch.cat([p.prepared_tk for p in parts]),
            actions=torch.cat([p.actions for p in parts]),
            a_bar=torch.cat([p.a_bar for p in parts]),
            # The transit half's sampler state: that is the half whose weighting can fall back.
            mean_turn=turns[0] if turns else parts[0].mean_turn,
            env=parts[0].env, t0=parts[0].t0)

    def _core(self, encoder: nn.Module, window: TokenWindow):
        raise NotImplementedError

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw a temporal window and score it."""
        window = self.draw(encoder)
        loss, scalars = self._core(encoder, window)
        self.last_window_turn = float(window.mean_turn)
        self.last_scalars = {"B": float(window.prepared_t.shape[0]),
                             "k": float(self.offset),
                             "window_turn": float(window.mean_turn),
                             **{k: float(v) for k, v in scalars.items()}}
        return loss

    # -- diagnostics shared by more than one term ----------------------------------------

    def identity_fraction(self, match: torch.Tensor, turn: torch.Tensor) -> dict:
        """The I77 engaged-vs-not gate: how often the correspondence is "the same token".

        `match[b, i] = j`. Engaged ⇔ the TRANSIT value is clearly below the PARKED value: at
        k too small (or while the agent is parked) the teacher's argmax is j = i everywhere and
        the term teaches "stay where you are". Emitted split, because a pooled value averages a
        parked mode and a transit mode and describes neither.
        """
        idx = torch.arange(match.shape[1], device=match.device)
        same = (match == idx).float().mean(dim=1)               # (B,)
        parked, transit = parked_transit(turn)
        return {"identity_frac": float(same.mean()),
                "identity_frac_parked": stratum_mean(same, parked),
                "identity_frac_transit": stratum_mean(same, transit)}

    def shift_correlation(self, match: torch.Tensor, turn_signed: torch.Tensor) -> dict:
        """ρ(column shift of the correspondence, cumulative turn) and its permuted-action null.

        Free ground truth on this apparatus: the stimulus is a video on a coplanar monitor, so
        ego rotation is a horizontal image shift. If the correspondence tracks it, ρ is large
        against the null; if the term is image-blind or stuck on the identity, both are ~0.
        """
        shift = column_shift(match, self.grid[1])
        perm = torch.randperm(turn_signed.shape[0], device=turn_signed.device)
        return {"shift_rho": rank_corr(shift, turn_signed),
                "shift_rho_null": rank_corr(shift, turn_signed[perm])}
