"""Slots from motion: segmenters that also see (previous, current) RAW frame pairs.

Owner request 2026-10-05 (U35): "try out the slots-from-motion MoTok you suggested on a card".
This module holds the pair machinery shared by every motion segmenter and the first of them,
``MoTokFlowSeg``. The Slot Attention pair (``sa_seg.py``) builds on the same base.

⛔ SAME HYPOTHESIS AS MoTokSeg: a SEPARATELY OPTIMISED segmenter whose mask MULTIPLIES THE
OBSERVATION. The policy never shares a gradient with it, and the ``masks.requires_grad`` guard
still raises. What changes is only what the segmenter is trained on.

MoTokFlowSeg = MoTokNet UNCHANGED (same layers, same state_dict, same AdamW, same recon + VQ
objective) PLUS the GWM flow-reconstruction loss on MoTok's own inference masks:

    loss = MSE(recon(target), target) + vq_coef * commit                    (MoTokSeg._loss)
         + NETT_SEG_MOTION_WEIGHT * flow_reconstruction_loss(masks(curr), flow(prev -> curr))

``masks(curr)`` is ``MoTokNet.get_masks`` on the RAW current frame -- the very distribution the
wrapper multiplies the observation by -- area-pooled to the flow grid. ``flow`` is the
parameter-free ``ExpertBlockFlow`` (exhaustive patch search, no weights, no gradient: a fixed
target that the scale-equivariant loss cannot shrink toward zero), on the RAW pair.

⛔ THE PREVIOUS FRAME AND EPISODE RESETS. Isaac auto-resets: the observation returned on a done
step is ALREADY the next episode's first frame (the same fact FrameStack's scrub exists for).
Pairing it with the cached last frame of the old episode would hand the flow a teleport -- a
whole-frame displacement no object made. So ``step()`` reads ``terminated | truncated`` BEFORE
``observation()`` runs, and those envs' pairs are marked INVALID: the reconstruction still trains
on their frame, the motion term skips them, and the new frame becomes the cache, so the NEXT
step's pair is within-episode and valid. ``reset()`` clears the cache (every env invalid once).
A direct ``observation()`` call with no ``step()`` (a probe, a test) assumes no env ended.
⚠ This wrapper must therefore sit where ``step()`` sees the env's own done flags, i.e. BEFORE
framestack, which is where a single-frame segmenter sits anyway (registry ORDER note).

FLOW UNITS = GwmSeg's: PIXELS OF THE 80 x 128 GRID
    The expert flow is computed on 80 x 128 area-downsampled frames, the grid its radius 12 was
    calibrated on (at 448x280 the real motion would exceed the search window) and the grid
    GwmSeg's loss ran on. The vectors stay in those pixels; a consumer whose masks live on
    another grid (Slot Attention, 64 x 96) gets the FIELD area-resampled to its grid with the
    VECTORS UNCHANGED, so the loss is in the same units for every motion segmenter.
    ⚠ MEASURED, and why not the fit's [-1,1] grid units: on the synthetic moving-square clip
    (tests/test_motion_slots_seg.py) MoTok's recon term is ~0.2 and the flow residual in grid
    units was ~2e-4 -- at weight 1.0 the motion term was 1/1000 of the loss, i.e. MoTokFlow
    would have been MoTok. In pixels it is the same order as recon there. In the real scene
    ego-motion moves the whole frame, so the residual can be larger; read seg/recon_loss
    against seg/motion_loss in the first rollouts before trusting the default weight.
    ``seg/flow_spatial_std`` / ``seg/flow_absmax`` are logged in the same pixels, so GwmSeg's
    ratio bar (<0.02 collapsed, 0.03-0.12 structured) reads unchanged. NETT_SEG_FLOW_REG keeps
    GwmSeg's 1e-4.

STORAGE (why ``_mask_one`` is overridden, not extended)
    The base wrapper buffers ONE tensor per call and, under NETT_SEG_TRAIN_ON=masked, insists
    that tensor be the frame. A motion segmenter needs three things per frame: the RAW previous
    and current frames (flow and masks) and the reconstruction target (masked, under TRAIN_ON=
    masked; the raw frame otherwise). The target cannot be regenerated at training time because
    the weights move during the pass. So this class stores uint8 ``cur`` (+ ``tgt`` when masked)
    and a per-env validity flag per call, and rebuilds ``prev`` from the sequence (step t's prev
    is step t-1's cur; the rollout's first step pairs with the cached frame), which costs no
    frame copy. ``_mask_one`` below is the base body with that storage swapped in; the base file
    is UNCHANGED, so MoTokSeg and GwmSeg are byte-identical. Rollout host RAM at 448x280:
    1.16 GB per 3072 frames for ``cur``, doubled to 2.3 GB under TRAIN_ON=masked.
    ⚠ Pairs need ONE batch size per call; both rollout orders refuse a change.

COST (measured on CPU, batch 8 at 280x448, tensors autograd saves for backward)
    MoTokSeg 379 MB. MoTokFlowSeg 434 MB under TRAIN_ON=raw (one MoTok pass serves recon and
    masks, ``_recon_and_masks``) and 602 MB under TRAIN_ON=masked (the target is the masked frame,
    so the masks of the raw frame need a second MoTok forward). The expert flow is no-grad and
    the 80x128 least-squares fit is small. Parameters: 60,811, identical to MoTokSeg.

NETT_SEG_MOTION_WEIGHT   1.0   weight of the flow term (>= 0; 0 = recon only, still pairs)
NETT_SEG_FLOW_REG        1e-4  QR Tikhonov regulariser, GwmSeg's value
NETT_EXPERT_FLOW_RADIUS/STRIDE/PATCH   12/2/8 at the 80x128 flow grid (expert_flow.py)
Everything else is MoTokSeg's (NETT_SEG_QUERIES, VQ_COEF, LR, WD, CADENCE, TRAIN_ON, ...).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from torch.nn import functional as F

from ...brain.aux.expert_flow import ExpertBlockFlow
from ...brain.aux.gwm_dual_loss import flow_reconstruction_loss
from ...brain.aux.knobs import _env_nonneg_float
from .framestack import _done_mask
from .motok_seg import MoTokSeg
from .segmentation import SegmentationObservationWrapper

FLOW_HW = (80, 128)   # the grid ExpertBlockFlow's radius 12 was calibrated on


def expert_flow_px(flow_model, prev: torch.Tensor, curr: torch.Tensor,
                   out_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """RAW [0,1] frames -> (flow field at ``out_hw``, flow at FLOW_HW), both in PIXELS OF THE
    80 x 128 GRID (GwmSeg's units). Resampling to ``out_hw`` moves the field, not the vectors.

    No gradient: the expert flow has no parameters and is a fixed target.
    """
    with torch.no_grad():
        p = F.interpolate(prev, size=FLOW_HW, mode="area")
        c = F.interpolate(curr, size=FLOW_HW, mode="area")
        px = flow_model.forward_single(p, c)                         # (B, 2, 80, 128)
        flow = px if tuple(out_hw) == FLOW_HW else F.interpolate(px, size=out_hw, mode="area")
    return flow, px


class MotionPairSegmenter(SegmentationObservationWrapper):
    """Base for segmenters trained on (prev RAW, curr RAW, target) triples. See module doc."""

    UNITY_TRAINING_KNOBS = True

    def __init__(self, env) -> None:
        super().__init__(env)
        self._prev: torch.Tensor | None = None           # uint8 (B,3,H,W), last RAW frame
        self._pending_done: np.ndarray | None = None
        self._roll_tgt: list[torch.Tensor] = []
        self._roll_valid: list[torch.Tensor] = []
        self._roll_prev0: torch.Tensor | None = None
        self._flow = ExpertBlockFlow()
        self.flow_reg = float(os.environ.get("NETT_SEG_FLOW_REG", "1e-4"))
        self.last_stats.update({
            "seg/recon_loss": float("nan"),
            "seg/motion_loss": float("nan"),
            # fraction of the last train batch whose pair was within one episode
            "seg/motion_valid_frac": float("nan"),
            "seg/flow_spatial_std": float("nan"),
            "seg/flow_absmax": float("nan"),
        })

    # ------------------------------------------------------------------ episode bookkeeping
    def reset(self, **kwargs):
        self._prev = None
        self._pending_done = None
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # ⛔ BEFORE observation(): a done env's obs is already the NEXT episode's first frame.
        self._pending_done = _done_mask(terminated, truncated)
        return self.observation(obs), reward, terminated, truncated, info

    def _pair_valid(self, cur: torch.Tensor) -> torch.Tensor:
        """(B,) bool: does ``self._prev`` hold this env's previous frame of the SAME episode?"""
        b = cur.shape[0]
        if self._prev is None or self._prev.shape != cur.shape:
            valid = torch.zeros(b, dtype=torch.bool)
        else:
            valid = torch.ones(b, dtype=torch.bool)
        if self._pending_done is not None:
            done = torch.from_numpy(np.asarray(self._pending_done, dtype=bool).reshape(-1))
            if done.numel() != b:
                raise ValueError(f"{type(self).__name__}: {done.numel()} done flags for {b} frames.")
            valid &= ~done
        return valid

    # ------------------------------------------------------------------ masking + storage
    def _mask_one(self, obs):
        arr = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
        x, batched = self._to_bchw(arr)
        if x.shape[1] != 3:
            raise ValueError(
                f"{type(self).__name__} takes ONE RGB frame per env (got {x.shape[1]} channels): "
                "order it BEFORE framestack -- it keeps its own previous frame.")
        self._ensure(3)
        if self._pending_state is not None:
            self._apply_pending_state()
        fd = x.to(self.device)
        with torch.no_grad():
            masks = self._model.get_masks(fd)                     # (B,K,H,W)
        # ⛔ THE ISOLATION GUARD, unchanged from the base wrapper.
        if masks.requires_grad:
            raise RuntimeError(
                f"{type(self).__name__}: mask carries requires_grad=True. The segmenter "
                "must stay isolated from the policy gradient — that isolation "
                "is what makes this the 'segmented image' hypothesis rather "
                "than an auxiliary loss."
            )
        m = self._keep_mask(masks)
        masked = self._masked_levels(fd, m)                       # integer-valued, [0,255]
        cur = (x * 255.0).round().to(torch.uint8)                 # exact: x came from uint8
        valid = self._pair_valid(cur)
        if self.train_every > 0:
            tgt = masked.cpu().to(torch.uint8) if self.train_on == "masked" else None
            if self.cadence == "rollout":
                if not self._roll:
                    self._roll_prev0 = (self._prev if self._prev is not None
                                        and self._prev.shape == cur.shape else torch.zeros_like(cur))
                self._roll.append(cur)
                self._roll_valid.append(valid)
                if tgt is not None:
                    self._roll_tgt.append(tgt)
                self._roll_n += int(cur.shape[0])
            else:
                prev = (self._prev if self._prev is not None and self._prev.shape == cur.shape
                        else torch.zeros_like(cur))
                vplane = (valid.to(torch.uint8) * 255).view(-1, 1, 1, 1).expand(-1, 1, *cur.shape[2:])
                self._store_sample(torch.cat([prev, cur, cur if tgt is None else tgt, vplane], dim=1))
        self._prev = cur
        self._pending_done = None

        self._seen += 1
        if self.train_every > 0:
            if self.cadence == "rollout":
                if self._roll_n >= self.rollout_frames:
                    self.train_rollout()
            elif self._seen % self.train_every == 0:
                self.train_step()

        y = masked.cpu().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)   # back to NHWC
        if not batched:
            y = y[0]
        res = y.numpy()
        return torch.from_numpy(res) if isinstance(obs, torch.Tensor) else res

    def train_step(self) -> float | None:
        """Online cadence: ``batch`` random stored calls (uint8 10-channel samples)."""
        if self._model is None or len(self._buf) < max(2, self.batch):
            return None
        idx = torch.randperm(len(self._buf))[: self.batch]
        batch = torch.cat([self._buf[i] for i in idx.tolist()], dim=0)
        return self._opt_step(batch.to(self.device).float().div_(255.0))

    def train_rollout(self) -> float | None:
        """One ordered batch-``batch`` pass over the rollout, as the base, on 10-channel samples
        [prev(3) | cur(3) | target(3) | valid(1)]."""
        if self._model is None or not self._roll:
            return None
        if len({t.shape for t in self._roll}) != 1:
            raise RuntimeError(f"{type(self).__name__}: frame pairs need one batch size per call.")
        cur = torch.stack(self._roll, dim=0)                       # (T, E, 3, H, W) uint8
        tgt = torch.stack(self._roll_tgt, dim=0) if self._roll_tgt else cur
        valid = torch.stack(self._roll_valid, dim=0)               # (T, E)
        prev0 = self._roll_prev0
        self._roll, self._roll_tgt, self._roll_valid, self._roll_n = [], [], [], 0
        t_len, e_len = valid.shape
        j = torch.arange(t_len * e_len)
        if self.rollout_order == "env":                            # env after env, time inside
            t_idx, e_idx = j % t_len, j // t_len
        else:                                                      # time-major, as stored
            t_idx, e_idx = j // e_len, j % e_len
        hw = cur.shape[-2:]
        losses = []
        for i in range(0, j.numel(), self.batch):
            ti, ei = t_idx[i : i + self.batch], e_idx[i : i + self.batch]
            c = cur[ti, ei]
            p = torch.where((ti > 0).view(-1, 1, 1, 1), cur[(ti - 1).clamp(min=0), ei], prev0[ei])
            v = (valid[ti, ei].to(torch.uint8) * 255).view(-1, 1, 1, 1).expand(-1, 1, *hw)
            sample = torch.cat([p, c, tgt[ti, ei], v], dim=1)
            loss = self._opt_step(sample.to(self.device).float().div_(255.0))
            if loss is not None:
                losses.append(loss)
        self.last_stats["seg/rollout_steps"] = float(len(losses))
        if losses:
            self.last_stats["seg/loss"] = float(np.mean(losses))
        return float(np.mean(losses)) if losses else None

    # ------------------------------------------------------------------ the motion term
    @staticmethod
    def _split(batch: torch.Tensor):
        """(n,10,H,W) -> prev, curr, target, valid (n,) bool."""
        return batch[:, 0:3], batch[:, 3:6], batch[:, 6:9], batch[:, 9, 0, 0] > 0.5

    def _motion_term(self, masks: torch.Tensor, prev: torch.Tensor, curr: torch.Tensor):
        """Flow-reconstruction loss of ``masks`` (n,K,h,w; sums to 1) against expert flow on
        the RAW pair, on the masks' own grid. Records the flow statistics."""
        flow, px = expert_flow_px(self._flow, prev, curr, tuple(masks.shape[-2:]))
        loss = flow_reconstruction_loss(masks, flow, reg=self.flow_reg)
        with torch.no_grad():
            self.last_stats.update({
                "seg/flow_spatial_std": float(px.std(dim=(2, 3), correction=0).mean().item()),
                "seg/flow_absmax": float(px.abs().max().item()),
            })
        return loss


class MoTokFlowSeg(MotionPairSegmenter, MoTokSeg):
    """MoTokSeg + the GWM flow-reconstruction loss on its own masks. See module docstring."""

    def _configure(self) -> None:
        MoTokSeg._configure(self)                     # NETT_SEG_MODEL=motok, QUERIES, UPSAMPLE, VQ
        self.kind = "motok_flow"
        self.motion_weight = _env_nonneg_float("NETT_SEG_MOTION_WEIGHT", 1.0)
        logging.getLogger("nett.body.motok_flow_seg").info(
            "motok_flow_seg: MoTokNet unchanged + %g x flow_reconstruction_loss(masks, expert "
            "flow prev->curr, pixels of the %dx%d grid)", self.motion_weight, *FLOW_HW)

    def _recon_and_masks(self, frame: torch.Tensor):
        """ONE encoder pass -> (MoTokNet.reconstruct(frame), MoTokNet.get_masks(frame)).

        Used when the reconstruction target IS the raw frame (TRAIN_ON=raw), so the motion term
        costs no second MoTok forward. The arithmetic is MoTokNet's own (motok_aux.py
        ``reconstruct`` and ``get_masks``), composed from its methods; a test pins equality.
        """
        net = self._model
        h, w = frame.shape[2:]
        z_q, commit = net._encode_quantize(net._maybe_upsample(frame))
        slots, attn, (b, d, hf, wf) = net._slots_from_feats(z_q)
        agg = torch.bmm(attn.transpose(1, 2), slots).permute(0, 2, 1).reshape(b, d, hf, wf)
        recon = net.decoder(agg)
        if recon.shape[2:] != (h, w):
            recon = F.interpolate(recon, size=(h, w), mode="bilinear", align_corners=False)
        masks = F.softmax(attn.reshape(b, net.num_queries, hf, wf), dim=1)
        masks = F.softmax(F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False), dim=1)
        return (recon, commit), masks

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        prev, curr, target, valid = self._split(batch)
        use_motion = self.motion_weight > 0 and bool(valid.any())
        if self.train_on == "raw":
            (recon, commit), masks = self._recon_and_masks(target)
            masks = masks[valid]
        else:
            recon, commit = self._model.reconstruct(target)
            masks = self._model.get_masks(curr[valid]) if use_motion else None
        recon_loss = F.mse_loss(recon, target) + self.vq_coef * commit
        loss = recon_loss
        motion = None
        if use_motion:
            motion = self._motion_term(F.interpolate(masks, size=FLOW_HW, mode="area"),
                                       prev[valid], curr[valid])
            loss = loss + self.motion_weight * motion
        with torch.no_grad():
            self.last_stats.update({
                "seg/recon_loss": float(recon_loss.item()),
                "seg/motion_loss": float(motion.item()) if motion is not None else float("nan"),
                "seg/motion_valid_frac": float(valid.float().mean().item()),
            })
        return loss
