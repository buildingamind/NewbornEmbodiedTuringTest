"""MoTokSeg — the Isaac equivalent of Unity's ``GwmSegWrapper``.

PORTED FROM, read at the file:
    scripts/wrappers/seg_wrappers.py:122-215   ``GwmSegWrapper``
    scripts/gwmEventBased1/trainParsing.py:328-400  ``GwmPPO_MoTokNet._train_ventral``

⛔★★★★★ WHY THIS IS A WRAPPER AND NOT AN AUXILIARY LOSS
------------------------------------------------------
The owner's high-conviction MoTok result (``_gwm_train_rA10_parsing_motok``, 8
brains at chick-level parsing) ran MoTok as a **separately-optimised network
whose foreground mask MULTIPLIES THE OBSERVATION** — ``seg_wrappers.py:213``,
``masked = obs * M_np``. The policy never shares a gradient with it.

That is a DIFFERENT HYPOTHESIS from an auxiliary loss. An aux loss asks *"does an
object-centric objective SHAPE THE REPRESENTATION usefully?"*; this asks *"does
handing the policy a SEGMENTED IMAGE help?"*. **The Unity evidence is for the
second**, so replicating the second is what replicates the result. An arm running
the aux-loss form is a different arm and must be named for what it tests.

⛔ THE GRADIENT ISOLATION IS THE PROPERTY THAT MAKES THE ARM FAITHFUL, so the
reference's ``RuntimeError`` at ``seg_wrappers.py:190`` is kept verbatim in
spirit: if the mask ever carries grad, this RAISES. A mask that quietly carried
grad would let PPO's return backpropagate into the segmenter and silently convert
this arm into a (differently-parameterised) aux-loss arm — the two hypotheses
would merge and neither would have been tested.

DIVERGENCES FROM THE REFERENCE, DECLARED RATHER THAN DISCOVERED
---------------------------------------------------------------
1. ⚠ **WHEN THE SEGMENTER TRAINS.** Unity trained it in ``_train_ventral`` at each
   PPO update, off the rollout buffer. Isaac exposes no such hook to a body
   wrapper, so this keeps a small internal ring buffer and trains every
   ``train_every`` observations. ``train_step()`` is also PUBLIC so a runner can
   drive it at true update boundaries later; the internal cadence is a
   stand-in, not a claim of equivalence. **The optimiser and its schedule are the
   part a reviewer should check first.** ⭐ ``NETT_SEG_CADENCE=rollout`` +
   ``NETT_SEG_TRAIN_ON=masked`` + ``NETT_SEG_QUANTIZE=floor`` now reproduce the
   reference schedule (one ordered batch-8 pass over every rollout frame, on the
   masked uint8 observation); see ``segmentation.py``. The online default is
   ~3 optimiser steps per 3072-frame rollout against the reference's ~384-500.
   ⚠ One-frame offset, declared not fixed: the pass fires on the rollout's LAST stored
   observation, so the frame that opens the next rollout is masked with the NEW weights;
   SB3 masked it with the old ones.
2. ⚠ **LAYOUT.** Body wrappers run BEFORE ``ChannelsFirst``, so observations here
   are HWC ``(H,W,C)`` or batched ``(N,H,W,C)`` — the reference took CHW. Converted
   internally; the wrapper returns the layout it was given.
3. ⚠ **FRAMESTACK ORDER.** If this runs AFTER framestack the input carries
   ``C = 3*T`` channels. Each frame is then masked INDEPENDENTLY with its own
   mask, which is exactly equivalent to running this BEFORE framestack — not a
   guess, but it costs T forward passes. Declaring ``motok_seg`` before
   ``framestack`` is cheaper and is the recommended order.
4. ⚠ **NO ``num_queries`` SWEEP.** Fixed by env var, default 2 (fg/bg), matching
   ``gwm_num_queries: 2`` in the winning config.

⛔ SLOT 1 IS FOREGROUND BY CONVENTION ONLY
------------------------------------------
``MoTokNet.get_mask`` returns ``get_masks(frame)[:, 1:2]`` unconditionally
(``model.py:645``) and the reference does the same. **Nothing in a
reconstruction+VQ objective binds slot 1 to the object**, and with 2 slots and a
permutation-symmetric init which slot lands on the object is a COIN FLIP PER
SEED. This wrapper therefore offers ``NETT_SEG_FG_SLOT=auto`` (default), which
picks the slot by a stated, seed-independent rule and RECORDS which one it chose
in ``last_stats`` — so the choice is auditable instead of assumed. Any parsing
score computed downstream must still be permutation-invariant; see
``permutation_invariant_iou`` below.

CONFIG (env vars — the launcher constructs body wrappers with the env ALONE)
---------------------------------------------------------------------------
    NETT_SEG_MODEL        motok                  (only value implemented)
    NETT_SEG_FG_SLOT      auto | 0 | 1           (default auto; 'auto' = smaller-area slot)
    NETT_SEG_QUERIES      2                      gwm_num_queries
    NETT_SEG_UPSAMPLE     0                      gwm_upsample -- KEEP 0 on 128x80
    NETT_SEG_VQ_COEF      0.1                    gwm_vq_coef
    NETT_SEG_LR           1e-4                   gwm_lr
    NETT_SEG_WD           1e-4                   gwm_wd
    NETT_SEG_BATCH        8                      gwm_batch_size
    NETT_SEG_TRAIN_EVERY  64                     observations between train steps
    NETT_SEG_BUFFER       256                    ring-buffer capacity (frames)
    NETT_SEG_DEVICE       cuda | cpu             (default: cuda if available)
    NETT_SEG_CADENCE      online | rollout       rollout = one ordered pass over each rollout (reference)
    NETT_SEG_TRAIN_ON     raw | masked           masked = the policy's own observation (reference)
    NETT_SEG_QUANTIZE     round | floor          floor = the reference's astype(uint8)
    NETT_SEG_ROLLOUT_ORDER   time | env          env = consecutive frames of one env per batch (the
                                                 reference's single-env content); time = envs at one step
    NETT_SEG_ROLLOUT_FRAMES  (NETT_ROLLOUTS)     frames per update; ONE BRAIN PER PROCESS --
                                                 campaign_train refuses rollout at brains>1
"""

from __future__ import annotations

import logging
import os

import torch

from .segmentation import SegmentationObservationWrapper

logger = logging.getLogger("nett.body.motok_seg")


def permutation_invariant_iou(masks: torch.Tensor, gt: torch.Tensor) -> tuple[float, int]:
    """Best IoU over ALL slot->foreground assignments. Returns ``(iou, n)``.

    ⛔ THIS IS NOT A REFINEMENT, IT IS THE ONLY VALID FORM. Nothing binds slot 1
    to the object, so a metric that reads slot 1 measures which slot the seed
    happened to put the object in. Across seeds that returns a plausible mean
    with pure noise underneath — the same shape as every degenerate pass this
    campaign has found.

    Args:
        masks: (B, K, H, W) slot masks, K slots.
        gt:    (B, 1, H, W) or (B, H, W) binary ground-truth foreground.
    Returns:
        (mean best-IoU over the batch, B) — **report N with the number.**
    """
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    b, k = masks.shape[0], masks.shape[1]
    g = (gt > 0.5).float()
    best = torch.full((b,), -1.0, device=masks.device)
    for s in range(k):
        m = (masks[:, s : s + 1] > 0.5).float()
        inter = (m * g).flatten(1).sum(1)
        union = ((m + g) > 0).float().flatten(1).sum(1).clamp(min=1e-8)
        best = torch.maximum(best, inter / union)
    return float(best.mean()), int(b)


class MoTokSeg(SegmentationObservationWrapper):
    """Multiply observations by a separately trained MoTok foreground mask."""

    UNITY_TRAINING_KNOBS = True

    def _configure(self) -> None:
        kind = os.environ.get("NETT_SEG_MODEL", "motok").strip().lower()
        if kind != "motok":
            raise ValueError(
                f"NETT_SEG_MODEL={kind!r} is not implemented. Only 'motok' exists. "
                "Refusing rather than silently running an unsegmented arm that "
                "would look like a segmentation result."
            )
        self.kind = kind
        self.num_queries = int(os.environ.get("NETT_SEG_QUERIES", "2"))
        self.upsample = int(os.environ.get("NETT_SEG_UPSAMPLE", "0"))
        self.vq_coef = float(os.environ.get("NETT_SEG_VQ_COEF", "0.1"))

    def _ensure(self, in_ch: int) -> None:
        if self._model is not None:
            return
        from ...brain.aux.motok_aux import MoTokNet

        self._model = MoTokNet(
            num_queries=self.num_queries, upsample_size=self.upsample, in_ch=in_ch
        ).to(self.device).eval()
        self._optim = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        # ⚠ NOTE THE DIVERGENCE, DELIBERATE: the reference put its never-trained
        # dorsal stream in this optimiser at a separate LR (trainParsing.py:145),
        # where AdamW's decoupled weight decay shrank 75,906 unused parameters
        # (55.5% of MoTokNet) toward zero. This port has no dorsal, so there is a
        # single param group and gwm_backbone_lr has no meaning here.
        logger.info(
            "motok_seg: MoTok on %s, %d queries, lr=%g wd=%g, cadence=%s (%s), train_on=%s, "
            "quantize=%s, fg_slot=%s",
            self.device, self.num_queries, self.lr, self.wd, self.cadence,
            f"{self.rollout_frames} frames/update" if self.cadence == "rollout"
            else f"every {self.train_every} obs", self.train_on, self.quantize, self.fg_slot,
        )

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Reference reconstruction + VQ objective (trainParsing.py:371)."""
        import torch.nn.functional as F

        recon, commit = self._model.reconstruct(batch)
        return F.mse_loss(recon, batch) + self.vq_coef * commit
