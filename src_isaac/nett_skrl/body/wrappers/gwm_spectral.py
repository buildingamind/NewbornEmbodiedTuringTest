"""GWM-Seg with K > 2 regions merged into figure/ground by a spectral normalized cut (U35).

Owner, 2026-10-05: "I think it might be valuable to try GWM-Seg with RAFT included using a CNN
instead of DINO with more than 2 regions." Everything not named here is ``GwmSeg`` unchanged:
the same from-scratch SmallCNNVentral, the same quadratic flow-reconstruction loss, the same
separate AdamW, raw unmasked pairs, frozen at test.

WHAT IS NEW
1. K mask queries (``NETT_SEG_QUERIES``, default 4 here, must be >= 3), MERGED into a 2-way
   figure/ground partition by a spectral normalized cut over the K regions, per image:
     - node k = region k; its feature is the k-mask-pooled ``enc3`` map of the segmenter's OWN
       ventral CNN (128 ch at 1/8 resolution) -- the CNN-in-place-of-DINO the owner asked for;
     - affinity W_ij = exp((cos(f_i, f_j) - 1) / tau), self-loops kept, tau = NETT_GWM_SPECTRAL_TAU;
     - L_sym = I - D^-1/2 W D^-1/2, eigendecomposed ON THE CPU IN FLOAT64; y = v_2 / sqrt(d)
       (Shi & Malik 2000's generalized Fiedler vector);
     - sweep cut over the K-1 splits of y's sorted order, keeping the one with minimum Ncut.
       Ties go to the canonical partition with the smallest bitmask, which makes the result
       independent of the eigenvector's sign. Both sides are non-empty by construction.
   ⛔ THE FIGURE RULE: the figure is the side with the SMALLER total mask area. Flow cannot
   decide it, because test-time masking is frozen and has no flow. Area is the same
   assumption ``_pick_fg``'s auto rule already makes (object < background). It is recorded
   every step (``seg/spectral_fg_rule_area`` = 1, plus the per-slot figure frequency) so a
   reader can check it rather than inherit it. The kept mask is the SUM of the figure-side
   slots, which is exact because the slots are a softmax partition.
   ⚠ RISK, STATED IN ADVANCE: if the two test-monitor objects fall on DIFFERENT sides, one test
   alternative is deleted (see ``SegmentationObservationWrapper._keep_mask``). Watch
   ``seg/spectral_fg_slots`` and the per-slot frequencies.
2. The flow source, ``NETT_GWM_FLOW``:
     expert          ``ExpertBlockFlow`` (parameter-free block matching), the GWM-Seg baseline.
     raft_scratch    RAFT-S (``brain/aux/raft_small.py``), NO checkpoint, trained IN THIS WRAPPER
                     by its own unsupervised photometric objective and its own AdamW group.
     raft_pretrained FROZEN torchvision raft_large C_T_V2 (``brain/aux/raft_pretrained.py``):
                     the owner's scoped exception to the no-pretrained-weights rule (DECISIONS §73,
                     2026-10-06: "A pretrained exception is warranted here. The idea is to see if
                     this approach works at all."). Flow TARGETS only: no gradient, no optimizer
                     group, never the policy's input, never in seg_state. Label ``...-RAFTPT``;
                     it is not a compliant solution.
   ⛔ RAFT IS NEVER TRAINED THROUGH THE SEGMENTATION LOSS AND NEVER BY PPO. The quadratic
   reconstruction loss is homogeneous of degree 2 in the flow, so a flow net trained through it
   shrinks its own output to zero (GWM-PAPER cell E). Each segmenter step is TWO separate
   optimiser steps on disjoint parameter sets: (a) RAFT on the unsupervised flow loss; (b) the
   ventral on flow_reconstruction_loss(masks, RAFT flow DETACHED). AdamW skips parameters whose
   grad is None, so one optimiser holds both groups, and the existing save/load carries the
   RAFT moments with no new persistence path.

Settings (beyond GwmSeg's NETT_SEG_*):
    NETT_GWM_FLOW            expert    expert | raft_scratch | raft_pretrained (DECISIONS §73)
    NETT_GWM_SPECTRAL_TAU    0.1       affinity temperature on cosine similarity
    NETT_GWM_RAFT_LR         4e-4      RAFT-S AdamW lr (weight decay NETT_SEG_WD); the reference's
                                       FlyingChairs lr. Measured on random-shift textures (80x128,
                                       4 pairs/step): 4e-4 reached held-out EPE 0.63 px (zero-flow
                                       2.27) by ~250 steps; 1e-4 had learned nothing at 300.
    NETT_GWM_RAFT_ITERS      8         GRU iterations (reference trains at 12)
    NETT_GWM_RAFT_BATCH      32        pairs per RAFT step, drawn from the segmenter's batch
    NETT_GWM_RAFT_SMOOTH     4.0       edge-aware smoothness weight
    NETT_GWM_RAFT_OCC_AFTER  300       segmenter steps before the occlusion mask switches on
    NETT_GWM_RAFT_CHUNK      256       pairs per no-grad chunk when computing the ventral's flow
                                       (both RAFT modes)
    NETT_GWM_RAFT_PT_ITERS   12        raft_pretrained only: RAFT update iterations (torchvision's
                                       default and the reference's eval setting)
    NETT_GWM_RAFT_PT_SCALE   2.0       raft_pretrained only: minimum UPsampling before RAFT (short
                                       side also >= 128, sides /8): 80x128 -> 160x256
NETT_GWM_RAFT_{LR,ITERS,BATCH,SMOOTH,OCC_AFTER} apply to raft_scratch only and are REFUSED under
raft_pretrained (a frozen net has no lr); NETT_GWM_RAFT_PT_* are refused under any other mode.
NETT_SEG_BACKBONE_LR does not apply (there is no slow learned dorsal) and is refused if set.
NETT_SEG_MASK_RULE and NETT_SEG_FG_SLOT must be 'auto': the merge IS the mask rule.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn

from ...brain.aux.dual_stream import SmallCNNVentral
from ...brain.aux.gwm_dual_loss import flow_reconstruction_loss
from .gwm_seg import GwmSeg

_FLOW_MODES = ("expert", "raft_scratch", "raft_pretrained")
_TRUTHY = {"1", "true", "yes", "on"}


def spectral_bipartition(masks: torch.Tensor, feats: torch.Tensor, tau: float):
    """Merge K soft regions into two groups by a spectral normalized cut, per image.

    masks (B,K,H,W) softmax slots; feats (B,C,h,w) ventral features. Returns
    (fg (B,K) bool, ncut (B,), eig2 (B,), exact_gap (B,)), all on the CPU. ``fg`` marks the
    figure side (smaller total area). ``exact_gap`` = sweep Ncut minus the exact minimum over
    all 2^(K-1)-1 bipartitions (a diagnostic for the relaxation, 0 when the sweep is optimal).
    """
    B, K = masks.shape[:2]
    if K < 3:
        raise ValueError(f"spectral_bipartition needs K >= 3 regions; got {K}")
    with torch.no_grad():
        m = F.adaptive_avg_pool2d(masks.float(), feats.shape[-2:]).flatten(2)     # (B,K,n)
        f = feats.float().flatten(2)                                              # (B,C,n)
        pooled = torch.einsum("bkn,bcn->bkc", m, f) / m.sum(-1, keepdim=True).clamp_min(1e-6)
        area = masks.float().mean(dim=(2, 3))                                     # (B,K)
    p = F.normalize(pooled.detach().cpu().double(), dim=-1)
    area = area.detach().cpu().double()
    # self-affinity KEPT (W_kk = 1, as in Shi & Malik's pixel graphs). With a zero diagonal a
    # region unlike all others has volume == its own cut, so cutting it off scores Ncut ~ 1 --
    # the most separable region would be the most expensive to separate (measured, 1-vs-3 case).
    W = torch.exp((p @ p.transpose(1, 2) - 1.0) / tau)
    d = W.sum(-1).clamp_min(1e-12)                                                # (B,K)
    dis = d.rsqrt()
    L = torch.eye(K, dtype=W.dtype) - dis[:, :, None] * W * dis[:, None, :]
    evals, evecs = torch.linalg.eigh(L)
    y = evecs[:, :, 1] * dis
    order = torch.sort(y, dim=1, stable=True).indices                            # (B,K)

    bits = 2 ** torch.arange(K, dtype=torch.long)

    def ncut(side):                                                               # side (B,K) bool
        s = side.double()
        cut = torch.einsum("bi,bij,bj->b", s, W, 1.0 - s)
        va, vb = (d * s).sum(-1), (d * (1.0 - s)).sum(-1)
        return cut / va.clamp_min(1e-12) + cut / vb.clamp_min(1e-12)

    def canon(side):                                                              # slot 0 never on the coded side
        side = torch.where(side[:, :1], ~side, side)
        return side, (side.long() * bits).sum(-1)

    best_side = torch.zeros(B, K, dtype=torch.bool)
    best_val = torch.full((B,), float("inf"), dtype=torch.float64)
    best_code = torch.full((B,), 1 << K, dtype=torch.long)
    for s in range(1, K):
        side = torch.zeros(B, K, dtype=torch.bool)
        side.scatter_(1, order[:, :s], True)
        side, code = canon(side)
        val = ncut(side)
        better = (val < best_val - 1e-12) | ((val - best_val).abs() <= 1e-12) & (code < best_code)
        best_side = torch.where(better[:, None], side, best_side)
        best_val = torch.where(better, val, best_val)
        best_code = torch.where(better, code, best_code)

    exact = torch.full((B,), float("inf"), dtype=torch.float64)
    for code in range(1, 1 << (K - 1)):                       # every partition, slot 0 off the coded side
        side = ((code << 1) & bits).bool().expand(B, K)
        exact = torch.minimum(exact, ncut(side))

    a_side = (area * best_side.double()).sum(-1)
    a_rest = (area * (~best_side).double()).sum(-1)
    # figure = smaller-area side; an exact tie goes to the coded side (slot 0 is never on it)
    fg = torch.where((a_side <= a_rest)[:, None], best_side, ~best_side)
    return fg, best_val, evals[:, 1], best_val - exact


class _GwmSpectralModel(nn.Module):
    def __init__(self, num_queries: int, dorsal: nn.Module):
        super().__init__()
        self.ventral = SmallCNNVentral(num_out_channels=num_queries)
        self.dorsal = dorsal
        self.last_feats: torch.Tensor | None = None

    def get_masks(self, frame):
        # SmallCNNVentral.forward, step for step, keeping the enc3 map for the merge
        # (test_gwm_spectral checks these logits equal ventral(frame) exactly).
        v = self.ventral
        output_size = frame.shape[2:]
        x = v.enc3(v.enc2(v.enc1(frame)))
        self.last_feats = x.detach()
        for block in (v.dec3, v.dec2, v.dec1):
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        logits = v.head(x)
        if logits.shape[2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits.softmax(dim=1)


class GwmSpectralSeg(GwmSeg):
    """GwmSeg with K >= 3 slots, a spectral figure/ground merge and a selectable flow source."""

    def __init__(self, env):
        super().__init__(env)
        if self.mask_rule != "auto" or self.fg_slot != "auto":
            raise ValueError(
                "GwmSpectralSeg: NETT_SEG_MASK_RULE and NETT_SEG_FG_SLOT must be 'auto' -- the "
                "spectral merge is this arm's mask rule; another rule would run under its label.")
        self._flow_cache: torch.Tensor | None = None
        for k in range(self.num_queries):
            self.last_stats[f"seg/spectral_fg_frac_slot{k}"] = float("nan")
        self.last_stats.update({
            "seg/spectral_fg_rule_area": 1.0,
            "seg/spectral_fg_slots": float("nan"),
            "seg/spectral_fg_area": float("nan"),
            "seg/spectral_ncut": float("nan"),
            "seg/spectral_ncut_exact_gap": float("nan"),
            "seg/spectral_eig2": float("nan"),
            "seg/flow_mode_raft": float(self.flow_mode == "raft_scratch"),
            "seg/flow_mode_raft_pretrained": float(self.flow_mode == "raft_pretrained"),
        })
        if self.flow_mode == "raft_scratch":
            self.last_stats.update({k: float("nan") for k in (
                "seg/raft_loss", "seg/raft_census", "seg/raft_smooth", "seg/raft_nonocc_frac",
                "seg/raft_occ_applied_frac", "seg/raft_inframe_frac", "seg/raft_flow_absmax")})
            self.last_stats["seg/raft_steps"] = 0.0

    def _configure(self):
        self.kind = "gwm_spectral"
        self.num_queries = int(os.environ.get("NETT_SEG_QUERIES", "4"))
        if self.num_queries < 3:
            raise ValueError(
                f"GwmSpectralSeg needs NETT_SEG_QUERIES >= 3 (got {self.num_queries}); a 2-way "
                "merge of 2 regions is plain GwmSeg -- use CNN2F+GWM-Seg.")
        if "NETT_SEG_BACKBONE_LR" in os.environ:
            raise ValueError(
                "GwmSpectralSeg: NETT_SEG_BACKBONE_LR does not apply (no slow learned dorsal); "
                "RAFT-S has its own NETT_GWM_RAFT_LR. Unset it for this arm.")
        # unused: set only so GwmSeg.__init__'s 10x ventral/dorsal ratio check holds at any NETT_SEG_LR
        self.backbone_lr = float(os.environ.get("NETT_SEG_LR", "1e-4")) / 10.0
        self.flow_reg = float(os.environ.get("NETT_SEG_FLOW_REG", "1e-4"))
        mode = os.environ.get("NETT_GWM_FLOW", "expert").strip().lower()
        if mode not in _FLOW_MODES:
            raise ValueError(f"NETT_GWM_FLOW={mode!r}: expected one of {_FLOW_MODES}.")
        scratch_only = [k for k in ("NETT_GWM_RAFT_LR", "NETT_GWM_RAFT_ITERS", "NETT_GWM_RAFT_BATCH",
                                    "NETT_GWM_RAFT_SMOOTH", "NETT_GWM_RAFT_OCC_AFTER") if k in os.environ]
        pt_only = [k for k in ("NETT_GWM_RAFT_PT_ITERS", "NETT_GWM_RAFT_PT_SCALE") if k in os.environ]
        if mode == "raft_pretrained" and scratch_only:
            raise ValueError(
                f"NETT_GWM_FLOW=raft_pretrained: {scratch_only} configure RAFT-S training and do not "
                "apply to the frozen pretrained RAFT; unset them (use NETT_GWM_RAFT_PT_*).")
        if mode != "raft_pretrained" and pt_only:
            raise ValueError(
                f"NETT_GWM_FLOW={mode!r}: {pt_only} apply to raft_pretrained only; unset them.")
        ef = os.environ.get("NETT_EXPERT_FLOW")
        if ef is not None and (ef.strip().lower() in _TRUTHY) != (mode == "expert"):
            raise ValueError(
                f"NETT_EXPERT_FLOW={ef!r} contradicts NETT_GWM_FLOW={mode!r}; this wrapper takes its "
                "flow source from NETT_GWM_FLOW only. Unset NETT_EXPERT_FLOW.")
        self.flow_mode = mode
        self.tau = float(os.environ.get("NETT_GWM_SPECTRAL_TAU", "0.1"))
        if not self.tau > 0:
            raise ValueError(f"NETT_GWM_SPECTRAL_TAU={self.tau} must be > 0")
        self.raft_lr = float(os.environ.get("NETT_GWM_RAFT_LR", "4e-4"))
        self.raft_iters = int(os.environ.get("NETT_GWM_RAFT_ITERS", "8"))
        self.raft_batch = int(os.environ.get("NETT_GWM_RAFT_BATCH", "32"))
        self.raft_smooth = float(os.environ.get("NETT_GWM_RAFT_SMOOTH", "4.0"))
        self.raft_occ_after = int(os.environ.get("NETT_GWM_RAFT_OCC_AFTER", "300"))
        self.raft_chunk = int(os.environ.get("NETT_GWM_RAFT_CHUNK", "256"))
        if min(self.raft_iters, self.raft_batch, self.raft_chunk) < 1:
            raise ValueError("NETT_GWM_RAFT_ITERS/BATCH/CHUNK must be >= 1")
        self.raft_pt_iters = int(os.environ.get("NETT_GWM_RAFT_PT_ITERS", "12"))
        self.raft_pt_scale = float(os.environ.get("NETT_GWM_RAFT_PT_SCALE", "2.0"))
        if self.raft_pt_iters < 1:
            raise ValueError(f"NETT_GWM_RAFT_PT_ITERS={self.raft_pt_iters} must be >= 1")
        if not self.raft_pt_scale >= 1.0:
            raise ValueError(f"NETT_GWM_RAFT_PT_SCALE={self.raft_pt_scale} must be >= 1 (an UPsampling factor)")

    def _ensure(self, in_ch):
        if self._model is not None:
            return
        log = logging.getLogger("nett.body.gwm_spectral")
        if self.flow_mode == "raft_scratch":
            from ...brain.aux.raft_small import RAFTSmall
            dorsal = RAFTSmall(iters=self.raft_iters)
        elif self.flow_mode == "raft_pretrained":
            from ...brain.aux.raft_pretrained import FrozenRAFTLarge, PretrainedFlow
            dorsal = PretrainedFlow(FrozenRAFTLarge(iters=self.raft_pt_iters, scale=self.raft_pt_scale,
                                                    chunk=self.raft_chunk, device=self.device))
        else:
            from ...brain.aux.expert_flow import ExpertBlockFlow
            dorsal = ExpertBlockFlow()
        self._model = _GwmSpectralModel(self.num_queries, dorsal).to(self.device).eval()
        groups = [{"params": self._model.ventral.parameters(), "lr": self.lr, "name": "ventral"}]
        raft_params = list(self._model.dorsal.parameters())
        if raft_params:
            groups.append({"params": raft_params, "lr": self.raft_lr, "name": "raft"})
        self._optim = torch.optim.AdamW(groups, weight_decay=self.wd)
        log.info(
            "gwm_spectral: %s, K=%d, flow=%s (%d params), tau=%g, ventral lr=%g, raft lr=%g iters=%d "
            "batch=%d smooth=%g occ_after=%d, train every %d obs; figure = smaller-area side",
            self.device, self.num_queries, self.flow_mode, sum(p.numel() for p in raft_params),
            self.tau, self.lr, self.raft_lr, self.raft_iters, self.raft_batch, self.raft_smooth,
            self.raft_occ_after, self.train_every)

    # ------------------------------------------------------------------ mask rule
    def _keep_mask(self, masks):
        feats = self._model.last_feats
        if feats is None or feats.shape[0] != masks.shape[0]:
            raise RuntimeError("GwmSpectralSeg: ventral features do not match the masks being merged")
        fg, ncut, eig2, gap = spectral_bipartition(masks, feats, self.tau)
        sel = fg.to(masks.device, masks.dtype)[:, :, None, None]
        m = (masks * sel).sum(dim=1, keepdim=True)
        # legacy keys, unconditionally (base _keep_mask's contract); there is no single
        # selected slot, so selected_slot is -1 and the figure set is reported per slot below
        areas_fg = self._pick_fg(masks)
        self.last_stats["seg/fg_slot"] = float(areas_fg)
        self.last_stats["seg/fg_area"] = float(masks[:, areas_fg: areas_fg + 1].mean().item())
        self.last_stats["seg/mask_rule_not_background"] = 0.0
        self.last_stats["seg/selected_slot"] = -1.0
        self.last_stats["seg/kept_area"] = float(m.mean().item())
        fgf = fg.double()
        for k in range(masks.shape[1]):
            self.last_stats[f"seg/spectral_fg_frac_slot{k}"] = float(fgf[:, k].mean())
        self.last_stats["seg/spectral_fg_slots"] = float(fgf.sum(1).mean())
        self.last_stats["seg/spectral_fg_area"] = float(m.mean().item())
        self.last_stats["seg/spectral_ncut"] = float(ncut.mean())
        self.last_stats["seg/spectral_ncut_exact_gap"] = float(gap.mean())
        self.last_stats["seg/spectral_eig2"] = float(eig2.mean())
        return m

    # ------------------------------------------------------------------ training
    def _raft_step(self, batch):
        from ...brain.aux.raft_small import unsupervised_flow_loss
        raft = self._model.dorsal
        idx = torch.randperm(batch.shape[0])[: self.raft_batch].to(batch.device)
        sub = batch.index_select(0, idx)
        prev, curr = sub[:, :3], sub[:, -3:]
        use_occ = self.last_stats["seg/train_steps"] >= self.raft_occ_after
        with torch.enable_grad():
            raft.train()
            self._optim.zero_grad(set_to_none=True)
            loss, sc = unsupervised_flow_loss(prev, curr, raft(prev, curr), raft(curr, prev),
                                              smooth=self.raft_smooth, use_occlusion=use_occ)
            if not torch.isfinite(loss):
                raft.eval()
                logging.getLogger("nett.body.gwm_spectral").warning(
                    "gwm_spectral: non-finite RAFT loss, RAFT step skipped")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raft.parameters(), max_norm=1.0)
            self._optim.step()            # the ventral's grads are None, so only RAFT moves
            self._optim.zero_grad(set_to_none=True)
        raft.eval()
        self.last_stats["seg/raft_loss"] = float(loss.item())
        self.last_stats["seg/raft_steps"] += 1.0
        self.last_stats.update({f"seg/{k}": v for k, v in sc.items()})

    def _opt_step(self, batch):
        if self.flow_mode != "raft_scratch":
            return super()._opt_step(batch)
        self._raft_step(batch)
        raft = self._model.dorsal
        with torch.no_grad():
            self._flow_cache = torch.cat([
                raft.forward_single(batch[i: i + self.raft_chunk, :3], batch[i: i + self.raft_chunk, -3:])
                for i in range(0, batch.shape[0], self.raft_chunk)]).detach()
        try:
            return super()._opt_step(batch)
        finally:
            self._flow_cache = None

    def _loss(self, batch):
        if self.flow_mode != "raft_scratch":
            return super()._loss(batch)
        prev, curr = batch[:, :3], batch[:, -3:]
        masks = self._model.get_masks(curr)
        flow = self._flow_cache
        if flow is None or flow.shape[0] != batch.shape[0] or flow.requires_grad:
            raise RuntimeError("GwmSpectralSeg: RAFT flow must be precomputed, detached, one per pair")
        loss = flow_reconstruction_loss(masks, flow, reg=self.flow_reg)
        with torch.no_grad():
            self.last_stats.update(self.slot_diagnostics(masks))
            self.last_stats.update({
                "seg/flow_spatial_std": float(flow.std(dim=(2, 3), correction=0).mean().item()),
                "seg/flow_absmax": float(flow.abs().max().item()),
                "seg/recon_loss": float(loss.item()),
            })
        return loss
