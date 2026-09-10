"""Temporal-pair VICReg auxiliary loss on the shared encoder, never a reward.

Reuse the incumbent objective, expander, coefficients, and augmentations; only
the view construction changes to aug(t) and aug(t+k). Draw contiguous windows
from one rollout stream just as cltt_ref does. The default single offset keeps
the objective strictly two-view; explicitly configured extra offsets are summed.

WHY THE DEFAULT OFFSET IS 8 AND NOT cltt_ref's 2. This loss is only different
from ``vicreg`` to the extent that t and t+k actually show different viewpoints,
so the offset was measured rather than inherited. From agent.angle in an existing
parsing test log (ViViT+VICReg fork-1 off0, 448 episodes, within-episode, wrapped
to +-180): median |d angle| is 4.0 deg at lag 2 and 6.0 deg at lag 8, mean 8.5 vs
17.6. At the live eye's 2.34 deg/px field average (300 deg over 128 px, equisolid
so this is an average) lag 2 is UNDER TWO PIXELS of pan -- far inside _augment's
own crop, which reaches 41% linear zoom at scale_min=0.5. At offset 2 the temporal
signal would be invisible under the augmentation and a null would mean nothing.
Rotation saturates by lag ~16 (mean 8.5 -> 17.6 -> 19.7 -> 20.7 at lags 2, 8, 16,
32) and translation is negligible at every lag (0.46 at lag 8 against a chamber
half-width of 33.15, i.e. 1.4%), so heading decorrelation is the only binding
limit. 8 takes 84% of the available rotation while staying short of saturation;
16 is the first retry if 8 returns a null. Offsets must stay multiples of the
realised stack depth T -- see the guard in compute(). Full derivation:
notes/researcher/vicreg-tt-plus.md section 6a in the fleet workspace.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import logger

from .simclr_aux import _augment
from .vicreg_aux import VICRegExpander, _off_diagonal, vicreg_loss


def vicreg_terms(
    z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unweighted (invariance, variance, covariance), matching vicreg_loss."""
    inv = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    var = torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))
    B, D = z1.shape
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / (B - 1)
    cov2 = (z2c.T @ z2c) / (B - 1)
    cov = _off_diagonal(cov1).pow(2).sum() / D + _off_diagonal(cov2).pow(2).sum() / D
    return inv, var, cov


class VICRegTemporalAuxLoss(nn.Module):
    """VICReg over independently augmented temporal windows from one stream."""

    needs_memory = True

    #: Sentinels for ``last_window_turn``. Negative so they can never collide with a
    #: real mean |turn|, which is an absolute value and therefore >= 0.
    MASK_OFF = -1.0
    MASK_NO_ACTIONS = -2.0
    MASK_NO_MOTION = -3.0
    #: No window has been selected yet. Distinct from MASK_OFF on purpose: "the knob
    #: is off" and "the loss has not run" are different facts, and reading the first
    #: where the second holds is how an inert component looks configured.
    MASK_UNSET = -4.0

    def __init__(
        self,
        encoder: nn.Module,
        *,
        crop_scale_min: float = 0.5,
        jitter: float = 0.2,
        max_samples: int = 48,
    ) -> None:
        super().__init__()
        self.head = VICRegExpander(int(encoder.features_dim), 512, 512)
        self.head.to(next(encoder.parameters()).device)
        offsets = os.environ.get("NETT_AUX_VICREG_TT_OFFSETS", "8")
        try:
            self.offsets = tuple(int(x) for x in offsets.split(","))
            if not self.offsets or any(k <= 0 for k in self.offsets):
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                "NETT_AUX_VICREG_TT_OFFSETS must be a comma-separated list of "
                f"at least one positive integer; got {offsets!r}."
            ) from exc
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))
        # Preserve the incumbent's owner-set balance (2026-06-15).
        self.inv = float(os.environ.get("NETT_VICREG_INV", "3"))
        self.var = float(os.environ.get("NETT_VICREG_VAR", "30"))
        self.cov = float(os.environ.get("NETT_VICREG_COV", "10"))
        self.crop_scale_min = float(crop_scale_min)
        self.jitter = float(jitter)
        self._memory = None
        self._warned_no_motion = False
        self.num_frames: int | None = None
        # GATE A diagnostic, OFF unless asked for. Computes the invariance term a
        # SECOND time against another augmentation of the anchor -- i.e. exactly the
        # incumbent `vicreg` construction -- on the same real batch. The ratio
        # temporal/control answers "does the pairing reach the loss at all?", which is
        # logically prior to whether its coefficient is large enough. Costs one extra
        # encoder forward per update, so it is opt-in and a real arm pays nothing.
        from ...nett import _env_flag  # local import: avoids a package-level cycle
        self.diag = _env_flag("NETT_AUX_VICREG_TT_DIAG")
        self.last_inv_temporal: float | None = None
        self.last_inv_control: float | None = None
        # TRANSIT MASK, OFF unless asked for. See _select_window for why.
        self.transit_mask = _env_flag("NETT_AUX_TRANSIT_MASK")
        # ⛔ EMITTED ON EVERY PATH, NOT ONLY THE ONE THAT WORKS. If this were set only
        # where the mask succeeds, "mask engaged" and "mask silently fell through"
        # would BOTH present as absent -- and absent reads as benign. A missing value
        # and a fallback are the same observation unless the fallback writes something
        # distinguishable, so each outcome writes its own number:
        #   >= 0.0  the mask ran; the value is the selected window's mean |turn|
        #   MASK_UNSET        no window selected yet (the loss has not run)
        #   MASK_OFF          the knob is off (expected; not a fault)
        #   MASK_NO_ACTIONS   set, but no usable actions tensor -- fell back to uniform
        #   MASK_NO_MOTION    set, actions present, but ZERO commanded rotation anywhere
        # The last is the one that would otherwise be invisible: it is the case where
        # the mask is on, nothing is wrong, and it still samples uniformly.
        self.last_window_turn: float = self.MASK_UNSET
        # Sum unweighted terms across offsets, matching the returned total loss.
        self.last_terms: tuple[float, float, float] | None = None

    def attach_memory(self, memory) -> None:
        self._memory = memory

    def _select_window(self, memory, n_env: int, avail: int, batch: int):
        """Choose (env, t0) for the contiguous window this update trains on.

        Default: uniform, which is what Gate A measured. Under NETT_AUX_TRANSIT_MASK:
        sample t0 with probability proportional to the window's mean |turn| command.

        WHY THIS EXISTS, and why it is a SAMPLING fix rather than a loss change.
        The agent is trained on a closeness reward, so it approaches the monitor and
        stops: measured over 35 archived parsing arms, it reaches a collision boundary
        3.23 from the screen at a median step 43 of 500 and **81.1% of scored steps are
        parked** (range 59.9-89.8%). Parked, heading moves 3.5 deg per 8 steps (2.1 px
        on-axis); in transit it moves ~26 deg (12.1 px, ViViT 20.4). Because this loss
        draws ONE CONTIGUOUS SLAB of `batch` steps, simulating that sampler on the real
        trajectories gives: **63.3% of updates draw a slab containing zero transit
        steps**, median slab shift 2.4 px, and only 11.0% of updates draw a slab whose
        shift clears the augmentation's ~26 px. So the temporal pair is two nearly
        identical frames about four updates in five.

        ⛔ This is the defect behind the Gate A withdrawal of this very candidate. That
        gate read "8 steps of agent motion is ~2.6 px against a 41% zoom, so the
        manipulation is ~0.05% of the objective" -- but 2.6 px was a MEDIAN OVER A
        MIXTURE of a 2.1 px parked mode and a 12.1 px transit mode and describes
        neither. The kill stands as scoped (that configuration could not test the
        hypothesis) but is a sampling defect, not the offset and not the mechanism.
        Full measurement: notes/researcher/the-parked-agent.md in the fleet workspace.

        WHY |turn| AND NOT |move|, AND NOT A DISTANCE THRESHOLD.
        - `turn` is actions[..., 0] and `move` is actions[..., 1] (motor_system.apply:107).
          A parked agent still COMMANDS forward motion into a wall it cannot pass, so
          |move| reports "moving" for exactly the steps this mask exists to exclude.
          Rotation is never blocked, so |turn| does not have that failure mode.
        - Heading is also the term that dominates the signal: translation at lag 8 is
          0.46 against a chamber half-width of 33.15 (1.4%), rotation is ~26 deg.
        - Weighted sampling, not a threshold, because the action->degrees gain is not
          known here and any constant would be an uncalibrated magic number. Weighting
          is scale-free and needs no calibration.

        ⚠ This CHANGES THE INPUT DISTRIBUTION the auxiliary objective sees; it is not a
        variance reduction. It answers "is there a channel at all", which is prior to
        "does the mechanism work" -- the latter still needs an arm.
        """
        uniform_env = int(torch.randint(n_env, ()).item())
        uniform_t0 = (
            int(torch.randint(avail - batch + 1, ()).item())
            if avail >= self.max_samples
            else 0
        )
        if not self.transit_mask:
            self.last_window_turn = self.MASK_OFF
            return uniform_env, uniform_t0
        actions = memory.tensors.get("actions")
        n_windows = avail - batch + 1
        if actions is None or actions.ndim < 3 or actions.shape[-1] < 1 or n_windows < 1:
            # No action channel to weight by (or no choice to make): stay uniform
            # rather than fail. A silently-uniform mask would be undetectable, so
            # say so once.
            logger.warning(
                "VICRegTemporalAuxLoss: NETT_AUX_TRANSIT_MASK set but no usable "
                "'actions' tensor (%s); falling back to UNIFORM sampling.",
                None if actions is None else tuple(actions.shape),
            )
            self.last_window_turn = self.MASK_NO_ACTIONS
            return uniform_env, uniform_t0
        turn = actions[:avail, :n_env, 0].abs().float().cpu()      # (avail, n_env)
        # Mean |turn| over every contiguous window, via cumulative sum.
        csum = torch.cat([torch.zeros(1, turn.shape[1]), turn.cumsum(0)], dim=0)
        win = (csum[batch:] - csum[:n_windows]) / float(batch)     # (n_windows, n_env)
        weights = win.flatten().clamp_min(0.0)
        total = float(weights.sum())
        if not (total > 0.0) or not torch.isfinite(weights).all():
            # A rollout with no commanded rotation anywhere: uniform is the honest
            # answer, and a degenerate multinomial would raise. ⛔ This branch was
            # previously SILENT -- no warning, no value -- so a mask that was on and
            # sampling uniformly looked exactly like a mask that was working. It now
            # says so once and writes its own sentinel.
            if not self._warned_no_motion:
                self._warned_no_motion = True
                logger.warning(
                    "VICRegTemporalAuxLoss: NETT_AUX_TRANSIT_MASK set and 'actions' "
                    "usable, but the rollout carries ZERO commanded rotation -- "
                    "falling back to UNIFORM sampling. The mask is on and inert."
                )
            self.last_window_turn = self.MASK_NO_MOTION
            return uniform_env, uniform_t0
        flat = int(torch.multinomial(weights, 1).item())
        t0, env = divmod(flat, win.shape[1])
        self.last_window_turn = float(win[t0, env])
        return int(env), int(t0)

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw temporal windows from attached memory."""
        if self._memory is None:
            raise RuntimeError(
                "VICRegTemporalAuxLoss draws its own temporal windows; "
                "call attach_memory(memory) before compute()."
            )
        memory = self._memory
        # Slice BEFORE transferring: HybridDeviceMemory.get_tensor_by_name would
        # copy the WHOLE ~2 GB observation buffer to the GPU on every call.
        raw = memory.tensors["observations"]
        t_max = memory.memory_size if memory.filled else memory.memory_index
        avail = t_max - max(self.offsets)
        batch = min(self.max_samples, avail)
        if batch < 2:
            raise ValueError(
                f"VICRegTemporalAuxLoss needs B_eff >= 2, got {batch} "
                f"(t_max={t_max}, offsets={self.offsets}, NETT_AUX_BATCH={self.max_samples}). "
                "VICReg variance/covariance terms are degenerate at B=1: "
                "single-row variance is zero or NaN and covariance divides by B-1."
            )
        env, t0 = self._select_window(memory, raw.shape[1], avail, batch)
        device = next(encoder.parameters()).device
        with torch.no_grad():
            views = [
                encoder._prepare_image(raw[t0 + k : t0 + batch + k, env].to(device))
                for k in (0, *self.offsets)
            ]
        if self.num_frames is None:
            self.num_frames = views[0].shape[1] // 3
            logger.info(
                "VICRegTemporalAuxLoss: offsets=%s, stack depth T=%s",
                self.offsets, self.num_frames,
            )
        if any(k % self.num_frames for k in self.offsets):
            raise ValueError(
                f"VICRegTemporalAuxLoss: realised stack depth T={self.num_frames}, "
                f"offsets={self.offsets}. Offsets not aligned to the stack depth "
                "can make positive views share a literally identical frame, "
                "allowing a shared-frame matching shortcut. Choose "
                f"NETT_AUX_VICREG_TT_OFFSETS that are multiples of T={self.num_frames}."
            )

        prepared = views
        views = [
            _augment(view, scale_min=self.crop_scale_min, jitter=self.jitter)
            for view in prepared
        ]
        z_anchor = self.head(encoder.encode_prepared(views[0]))  # backbone grad ON
        losses = []
        terms = []
        for view in views[1:]:
            z_view = self.head(encoder.encode_prepared(view))
            losses.append(vicreg_loss(z_anchor, z_view, self.inv, self.var, self.cov))
            with torch.no_grad():
                terms.append(torch.stack(vicreg_terms(z_anchor, z_view)))
        self.last_terms = tuple(torch.stack(terms).sum(dim=0).tolist())
        if self.diag:
            with torch.no_grad():
                # Control view: a second augmentation of the ANCHOR frames. Identical
                # pipeline, identical batch, differing ONLY in that no time has passed.
                z_ctrl = self.head(encoder.encode_prepared(
                    _augment(prepared[0], scale_min=self.crop_scale_min, jitter=self.jitter)
                ))
                z_temporal = self.head(encoder.encode_prepared(views[1]))
                self.last_inv_temporal = float(vicreg_terms(z_anchor, z_temporal)[0])
                self.last_inv_control = float(vicreg_terms(z_anchor, z_ctrl)[0])
        return sum(losses)
