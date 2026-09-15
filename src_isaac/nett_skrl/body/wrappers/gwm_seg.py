"""GWM segmentation OBSERVATION wrapper, never a policy auxiliary loss.

Reproduces the Unity GwmSegWrapper / GwmPPO integration with the two-query
SmallCNNVentral + Small3DCNNDorsal model, initialized entirely from scratch.
The separate AdamW uses ventral LR 1e-4 and dorsal LR 1e-5. This 10x ratio
is load-bearing: a faster dorsal can bend toward spatially constant flow,
which the quadratic basis reconstructs with near-zero loss.

⛔ TO DETECT THAT COLLAPSE, MONITOR seg/flow_spatial_std / seg/flow_absmax --
THE RATIO, NEVER seg/flow_spatial_std ALONE. An earlier version of this
docstring said to watch the raw statistic. That instruction was wrong and was
followed. ``flow_reconstruction_loss`` is a least-squares residual against a
per-slot quadratic basis, so it is homogeneous of degree 2 in the flow:
measured loss(c*flow) / (c**2 * loss(flow)) = 1.0000 at c = 1.0, 0.5, 0.1,
0.01. The dorsal stream can therefore drive the loss arbitrarily low by
SHRINKING THE MAGNITUDE of its output while learning nothing, and a plain
``flow.std(dim=(2,3))`` falls in exact lockstep (1.0002 -> 0.0100 at c=0.01).
A falling raw value is equally consistent with (a) the collapse, (b) uniform
magnitude shrinkage, and (c) healthy flow that is simply small.

The ratio is scale-free by construction -- invariant at 0.26743 across three
decades of scaling, and exactly 0.0 for a spatially constant field. Reference
values from Farneback optical flow (a classical algorithm with no learned
weights, used as a measuring instrument and NOT as a model component, so it
does not touch the no-pretrained-weights constraint) on the 18 real NETT
parsing clips at the live 80x128 eye:

    spatially constant flow (the collapse) .... 0.0000
    real NETT scene flow ..................... median 0.0645 (0.0397-0.0918)
    unstructured Gaussian noise .............. 0.2556

⇒ Registered bar, on the RATIO only: < 0.02 collapsed; 0.03-0.12 structured;
> 0.15 unstructured; 0.02-0.03 and 0.12-0.15 declared indeterminate in advance.
See notes/researcher/flow-collapse-detector-is-scale-confounded.md (workspace
25afda75), registered before any arm value was read.

Must run AFTER framestack. The raw pair comes from the first and last RGB
channels, reusing FrameStack's episode-reset handling without a previous-frame
cache. Each frame in the policy's stack receives its own foreground mask.
Slot 1 matches reference get_mask(); NETT_SEG_FG_SLOT=auto (default) retains
MoTok's smaller-area rule and records the selected slot.

Deliberate divergence from Unity: buffer RAW, UNMASKED pairs. Unity trained on
buf.observations, which were already masked; learning from the segmenter's own
masked output creates a self-reinforcing shrinkage loop. As in MoTokSeg, a
small ring buffer and periodic train_step stand in for Unity's PPO-update hook.

Shared NETT_SEG_* settings are the same as MoTokSeg (device, queries, foreground
slot, LR, WD, batch, train cadence, buffer capacity). Buffer capacity counts
pair batches. Additional settings:
    NETT_SEG_BACKBONE_LR  1e-5  (dorsal; must remain 10x slower than ventral)
    NETT_SEG_FLOW_REG     1e-4  (Unity value, not the loss module's 0.01 default)
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch
from torch import nn

from ...brain.aux.dual_stream import SmallCNNVentral, Small3DCNNDorsal
from ...brain.aux.gwm_dual_loss import flow_reconstruction_loss
from ..observation import image_layout
from .segmentation import SegmentationObservationWrapper


class _GwmModel(nn.Module):
    def __init__(self, num_queries: int):
        super().__init__()
        self.ventral = SmallCNNVentral(num_out_channels=num_queries)
        self.dorsal = Small3DCNNDorsal()

    def get_masks(self, frame):
        return self.ventral(frame).softmax(dim=1)


class GwmSeg(SegmentationObservationWrapper):
    """Multiply the frame stack by masks with zero policy-gradient coupling."""

    def __init__(self, env):
        super().__init__(env)
        if self.backbone_lr <= 0 or not math.isclose(self.lr / self.backbone_lr, 10.0):
            raise ValueError(
                "GwmSeg requires NETT_SEG_LR / NETT_SEG_BACKBONE_LR = 10 "
                "(ventral 1e-4, dorsal 1e-5 by default); a slower dorsal is load-bearing."
            )
        self.last_stats.update({
            "seg/flow_spatial_std": float("nan"),
            "seg/flow_absmax": float("nan"),
            "seg/recon_loss": float("nan"),
        })

    def _configure(self):
        self.kind = "gwm"
        self.num_queries = int(os.environ.get("NETT_SEG_QUERIES", "2"))
        if self.num_queries < 2:
            raise ValueError("GwmSeg requires at least two mask slots")
        self.backbone_lr = float(os.environ.get("NETT_SEG_BACKBONE_LR", "1e-5"))
        self.flow_reg = float(os.environ.get("NETT_SEG_FLOW_REG", "1e-4"))

    def _ensure(self, in_ch):
        if self._model is not None:
            return
        self._model = _GwmModel(self.num_queries).to(self.device).eval()
        # DO NOT combine these groups or learning rates. The measured constant-
        # flow degeneracy is much worse when dorsal learns as fast as ventral.
        self._optim = torch.optim.AdamW([
            {"params": self._model.ventral.parameters(), "lr": self.lr, "name": "ventral"},
            {"params": self._model.dorsal.parameters(), "lr": self.backbone_lr, "name": "dorsal"},
        ], weight_decay=self.wd)
        logging.getLogger("nett.body.gwm_seg").info(
            "gwm_seg: %s, %d queries, ventral lr=%g dorsal lr=%g wd=%g flow_reg=%g, "
            "train every %d obs; for collapse watch seg/flow_spatial_std / seg/flow_absmax "
            "(the RATIO -- the raw std confounds structure with scale; bar <0.02 collapsed, "
            "0.03-0.12 structured)",
            self.device, self.num_queries, self.lr, self.backbone_lr, self.wd,
            self.flow_reg, self.train_every,
        )

    def _frames_and_samples(self, x):
        if x.shape[1] < 6:
            raise ValueError(
                "GwmSeg needs framestack=True and must be ordered AFTER framestack"
            )
        if x.shape[1] % 3:
            raise ValueError("GwmSeg expects a channel stack of complete RGB frames")
        frames, _ = super()._frames_and_samples(x)
        pair = torch.cat((x[:, :3], x[:, -3:]), dim=1)
        return frames, [None] * (len(frames) - 1) + [pair]

    def _mask_one(self, obs):
        # MoTok's legacy HWC/NHWC path stays unchanged. Also accept CHW/NCHW
        # here for direct wrapper use, preserving layout and numpy/torch type.
        arr = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
        if arr.ndim not in (3, 4):
            raise ValueError(f"GwmSeg expects HWC/NHWC or CHW/NCHW, got shape {arr.shape}")
        chw = image_layout(arr.shape[-3:]) == "chw"
        if chw:
            arr = np.moveaxis(arr, -3, -1)
        result = super()._mask_one(arr)
        if chw:
            result = np.moveaxis(result, -1, -3)
        return torch.from_numpy(result) if isinstance(obs, torch.Tensor) else result

    def _loss(self, batch):
        prev, curr = batch[:, :3], batch[:, -3:]
        masks = self._model.get_masks(curr)
        flow = self._model.dorsal.forward_single(prev, curr)
        loss = flow_reconstruction_loss(masks, flow, reg=self.flow_reg)
        with torch.no_grad():
            self.last_stats.update(self.slot_diagnostics(masks))
            self.last_stats.update({
                # Spatial variation per example and flow component: differences
                # between examples or u/v offsets must NOT hide constant flow.
                "seg/flow_spatial_std": float(flow.std(dim=(2, 3), correction=0).mean().item()),
                "seg/flow_absmax": float(flow.abs().max().item()),
                "seg/recon_loss": float(loss.item()),
            })
        return loss
