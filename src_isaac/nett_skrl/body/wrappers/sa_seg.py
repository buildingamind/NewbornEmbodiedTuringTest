"""Slot Attention segmenters (Locatello et al. 2020), as body wrappers, trained from scratch.

Owner request 2026-10-05 (U35): "For slot attention, I was thinking something more along the
lines of this paper: https://arxiv.org/pdf/2006.15055". The model is
``brain/aux/slot_attention_ae.py`` (the paper's CLEVR object-discovery auto-encoder; its
docstring lists every paper value kept and every divergence). These wrappers put it in the
MoTokSeg form: a separately optimised network whose mask MULTIPLIES THE OBSERVATION, no
gradient shared with the policy, and the ``masks.requires_grad`` guard still raises.

    SASeg      reconstruction only -- the paper's objective.
    SAFlowSeg  reconstruction + NETT_SEG_MOTION_WEIGHT x the GWM flow-reconstruction loss on the
               decoder ALPHAS of the raw current frame, against expert block flow prev -> curr
               (units, reset handling and storage: ``motion_seg.py``).

Both run through the SAME pair machinery (``MotionPairSegmenter``), so the motion term is the
only difference between them; SASeg simply never computes it.

⛔ FOREGROUND SELECTION IS PER SAMPLE, BY AREA. Slots are i.i.d. draws from one shared Gaussian
and the module is permutation-equivariant, so a slot INDEX names nothing across images; the
base wrapper's batch-mean area rule (right for MoTok's fixed learned slot init) would pick an
index and apply it to frames where that index holds something else. So, per frame:
    K > 2 (default, "not_background"): background = that frame's LARGEST-area alpha; the policy
          sees 1 - M_bg, i.e. every other slot -- both test alternatives survive (base rationale).
    K <= 2 ("foreground"): keep that frame's SMALLEST-area alpha.
NETT_SEG_MASK_RULE overrides as in the base. An integer NETT_SEG_FG_SLOT is REFUSED: it would
name an index that carries no identity. The legacy stat keys are all still emitted (batch mode
of the per-frame choice), plus ``seg/selected_slot_agree`` = fraction of frames whose choice is
that mode -- 1.0 means the indices happen to be stable, which is a measurement, not a premise.

DEFAULT K = 4 (NETT_SEG_SA_SLOTS). The parsing test frame holds four kinds of region: the two
monitor images (object on its background), the monitor bezels/screen surround, and the chamber
(walls + floor). The paper sets K = #objects + 1 or more (Tetrominoes 3+1, CLEVR6 6+1); with
the not_background rule, any K >= 3 can keep both alternatives, and K = 4 is the largest that
fits MoTok's activation budget at the paper's widths (K is the decoder's batch multiplier:
measured 260 MB at K=4 vs MoTok's 379 MB; see the model docstring).

OPTIMISER = THE PAPER'S (Table 11), NOT THE SEGMENTER DEFAULT. Adam, lr 4e-4, betas (0.9, 0.999),
eps 1e-8, no weight decay; linear warm-up from 0 over 10k steps, then lr x 0.5^(step/100k).
The step count is ``seg/train_steps`` (persisted), so a resumed chunk continues the schedule.
NETT_SEG_LR / NETT_SEG_WD are REFUSED for this family: read-and-ignored would be the silent no-op.
⚠ Kept from the base and NOT in the paper: grad-norm clipping at 1.0 (every segmenter here
clips). ⚠ Batch 8 (NETT_SEG_BATCH) vs the paper's 64. Under the U30 recipe (rollout cadence,
3072 frames per update, 8000 episodes of 192 steps) the segmenter takes ~192k steps, so the
warm-up is ~5% of training (paper 2%) and the final lr factor is ~0.26 (paper 0.03).
⚠ The schedule assumes NETT_SEG_CADENCE=rollout. Online cadence takes ~3 steps per rollout and
would spend the whole run inside the warm-up.

COST (CPU, batch 8, 280x448 input, tensors saved for backward; MoTokSeg = 379 MB): SASeg 260 MB;
SAFlowSeg 274 MB under TRAIN_ON=raw (the training forward's alphas ARE the raw frame's) and 529 MB
under TRAIN_ON=masked (a second forward on the raw frame). 890,308 parameters at any K (MoTok
60,811): the paper's widths, not MoTok's. ⚠ Prefer TRAIN_ON=raw for this family: it is the paper's
objective, it keeps SAFlow inside MoTok's budget, and GwmSeg's docstring records why training on the
segmenter's own masked output risks a self-reinforcing shrinkage loop.

NETT_SEG_SA_SLOTS       4     K
NETT_SEG_MOTION_WEIGHT  1.0   SAFlowSeg only; REFUSED on SASeg (it is the recon-only control)
"""

from __future__ import annotations

import logging
import os

import torch
from torch.nn import functional as F

from ...brain.aux.knobs import _env_nonneg_float, _env_positive_int
from .motion_seg import MotionPairSegmenter

SA_LR = 4e-4
SA_WARMUP_STEPS = 10_000
SA_DECAY_STEPS = 100_000
SA_DECAY_RATE = 0.5


class SASeg(MotionPairSegmenter):
    """Multiply observations by a from-scratch Slot Attention mask (reconstruction only)."""

    KIND = "sa"
    MOTION = False

    def __init__(self, env) -> None:
        super().__init__(env)
        if self.fg_slot != "auto":
            raise ValueError(
                f"NETT_SEG_FG_SLOT={self.fg_slot!r}: Slot Attention slots are i.i.d. draws from one "
                "shared Gaussian, so a slot index carries no identity across frames. Unset it; the "
                "per-frame area rule is used (see sa_seg.py).")
        self.last_stats.update({"seg/lr": float("nan"), "seg/selected_slot_agree": float("nan")})

    def _configure(self) -> None:
        for name in ("NETT_SEG_LR", "NETT_SEG_WD"):
            if name in os.environ:
                raise ValueError(
                    f"{name} is set, but {type(self).__name__} uses the paper's optimiser (Adam "
                    f"lr {SA_LR:g} with warm-up and decay, no weight decay). Unset it.")
        self.kind = self.KIND
        self.num_slots = _env_positive_int("NETT_SEG_SA_SLOTS", 4)
        if self.MOTION:
            self.motion_weight = _env_nonneg_float("NETT_SEG_MOTION_WEIGHT", 1.0)
        else:
            if "NETT_SEG_MOTION_WEIGHT" in os.environ:
                raise ValueError(
                    "NETT_SEG_MOTION_WEIGHT is set on SASeg, the reconstruction-only control. "
                    "Use SAFlowSeg (sa-flow label) for the motion term.")
            self.motion_weight = 0.0

    def _ensure(self, in_ch: int) -> None:
        if self._model is not None:
            return
        from ...brain.aux.slot_attention_ae import SA_RES, SlotAttentionAutoEncoder

        self._model = SlotAttentionAutoEncoder(num_slots=self.num_slots, in_ch=in_ch).to(self.device).eval()
        self._optim = torch.optim.Adam(self._model.parameters(), lr=SA_LR, betas=(0.9, 0.999), eps=1e-8)
        logging.getLogger(f"nett.body.{self.kind}_seg").info(
            "%s_seg: Slot Attention AE on %s, K=%d, working res %dx%d, %d params, Adam lr %g "
            "(warm-up %d, x%g per %d steps), motion weight %g, cadence=%s, train_on=%s",
            self.kind, self.device, self.num_slots, *SA_RES,
            sum(p.numel() for p in self._model.parameters()), SA_LR, SA_WARMUP_STEPS,
            SA_DECAY_RATE, SA_DECAY_STEPS, self.motion_weight, self.cadence, self.train_on)

    def _lr_at(self, step: float) -> float:
        """Table 11 / E.6: linear warm-up from 0, then exponential decay."""
        warm = min(1.0, step / SA_WARMUP_STEPS) if SA_WARMUP_STEPS > 0 else 1.0
        return SA_LR * warm * SA_DECAY_RATE ** (step / SA_DECAY_STEPS)

    def _opt_step(self, batch: torch.Tensor) -> float | None:
        lr = self._lr_at(self.last_stats["seg/train_steps"])
        for group in self._optim.param_groups:
            group["lr"] = lr
        self.last_stats["seg/lr"] = lr
        return super()._opt_step(batch)

    def _keep_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """Per-frame slot choice; see module docstring. Returns (B,1,H,W)."""
        rule = self.mask_rule
        if rule == "auto":
            rule = "foreground" if masks.shape[1] <= 2 else "not_background"
        areas = masks.mean(dim=(2, 3))                             # (B, K)
        fg = areas.argmin(dim=1)                                   # (B,)
        h, w = masks.shape[-2:]
        if rule == "foreground":
            slot = fg
            m = masks.gather(1, slot.view(-1, 1, 1, 1).expand(-1, 1, h, w))
        elif rule == "not_background":
            slot = areas.argmax(dim=1)
            m = 1.0 - masks.gather(1, slot.view(-1, 1, 1, 1).expand(-1, 1, h, w))
        else:
            raise ValueError(f"NETT_SEG_MASK_RULE={rule!r} unknown; expected "
                             "'auto', 'foreground' or 'not_background'.")
        mode = int(torch.mode(slot).values.item())
        self.last_stats["seg/fg_slot"] = float(torch.mode(fg).values.item())
        self.last_stats["seg/fg_area"] = float(areas.gather(1, fg.view(-1, 1)).mean().item())
        self.last_stats["seg/mask_rule_not_background"] = float(rule == "not_background")
        self.last_stats["seg/selected_slot"] = float(mode)
        self.last_stats["seg/selected_slot_agree"] = float((slot == mode).float().mean().item())
        self.last_stats["seg/kept_area"] = float(m.mean().item())
        return m

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        prev, curr, target, valid = self._split(batch)
        net = self._model
        x = net.prepare(target)
        recon, alphas = net.decode(x)
        recon_loss = F.mse_loss(recon, x)                          # paper: MSE in [-1, 1]
        loss = recon_loss
        motion = None
        if self.motion_weight > 0 and bool(valid.any()):
            if self.train_on == "raw":
                m = alphas[valid]                                  # target IS the raw curr
            else:
                _, m = net.decode(net.prepare(curr[valid]))
            motion = self._motion_term(m, prev[valid], curr[valid])
            loss = loss + self.motion_weight * motion
        with torch.no_grad():
            self.last_stats.update(self.slot_diagnostics(alphas))
            self.last_stats.update({
                "seg/recon_loss": float(recon_loss.item()),
                "seg/motion_loss": float(motion.item()) if motion is not None else float("nan"),
                "seg/motion_valid_frac": float(valid.float().mean().item()),
            })
        return loss


class SAFlowSeg(SASeg):
    """SASeg + NETT_SEG_MOTION_WEIGHT x flow-reconstruction loss on the decoder alphas."""

    KIND = "sa_flow"
    MOTION = True
