"""Block-matching optical flow: an ALGORITHM used as a model component.

⛔ WHY THIS IS NOT A PRETRAINED-WEIGHTS VIOLATION. The reference GWM/EoO use a FROZEN
RAFT, which is pretrained weights and is forbidden here. This module has NO parameters
at all -- it is exhaustive patch correspondence search, the optical-flow equivalent of a
sort. Nothing was fit on any dataset. It is the compliant substitute for frozen RAFT and
it should be described that way rather than as "our flow model".

⭐ WHAT IT IS FOR. Our learned dorsal never improves: measured across three arms, the
spatial-structure ratio's MAXIMUM is its first logged value and it declines thereafter,
and flow_absmax falls 3.8-7.8x because `flow_reconstruction_loss` is exactly
scale-equivariant and so contains no scale anchor. An expert flow separates two
questions the learned version conflates:

    "is segmentation-by-motion the wrong idea in this scene?"   vs
    "is our FLOW too poor for segmentation-by-motion to be testable?"

⚠ WHAT IT COSTS. Block matching is discrete, so the output is quantised to the search
stride and has no sub-pixel precision. It is also NOT differentiable -- there is no
gradient path back through it, by construction. That is fine for the wrapper form, where
the dorsal is never trained anyway, and it makes this UNUSABLE as a drop-in for any arm
that expects to train its flow model. Such an arm must be a different condition, not a
silent substitution.

Settings:
    NETT_EXPERT_FLOW_RADIUS   12  search radius in pixels (matches the measured motion)
    NETT_EXPERT_FLOW_STRIDE    2  search stride in pixels
    NETT_EXPERT_FLOW_PATCH     8  patch size the cost is pooled over
"""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn import functional as F


class ExpertBlockFlow(nn.Module):
    """Two frames -> (B, 2, H, W) flow, by exhaustive patch search. No parameters."""

    def __init__(self, radius: int | None = None, stride: int | None = None,
                 patch: int | None = None):
        super().__init__()
        self.radius = int(os.environ.get("NETT_EXPERT_FLOW_RADIUS", "12")) if radius is None else radius
        self.stride = int(os.environ.get("NETT_EXPERT_FLOW_STRIDE", "2")) if stride is None else stride
        self.patch = int(os.environ.get("NETT_EXPERT_FLOW_PATCH", "8")) if patch is None else patch
        if self.stride < 1 or self.radius < 1 or self.patch < 1:
            raise ValueError("expert flow radius/stride/patch must all be >= 1")

    @torch.no_grad()
    def forward_single(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        """Forward flow frame_t -> frame_t1. Same signature as Small3DCNNDorsal."""
        if frame_t.shape != frame_t1.shape:
            raise ValueError(f"frame shapes differ: {frame_t.shape} vs {frame_t1.shape}")
        B, _, H, W = frame_t.shape
        p = self.patch
        offs = list(range(-self.radius, self.radius + 1, self.stride))

        a = frame_t.mean(dim=1, keepdim=True)          # luminance; correspondence needs no colour
        b = frame_t1.mean(dim=1, keepdim=True)

        costs, shifts = [], []
        for dy in offs:
            for dx in offs:
                # roll is a cyclic shift; the wrapped band is a small fraction of the frame
                # at radius 12 on 80x128 and costs far less than padding every candidate.
                shifted = torch.roll(b, shifts=(dy, dx), dims=(2, 3))
                sad = (a - shifted).abs()
                costs.append(F.avg_pool2d(sad, kernel_size=p, stride=p))
                shifts.append((dx, dy))

        cost = torch.cat(costs, dim=1)                  # (B, n_shifts, H/p, W/p)
        best = cost.argmin(dim=1)                       # (B, H/p, W/p)
        # ⛔ NEGATE. `shifts` records how far frame_t1 had to be rolled BACK to align with
        # frame_t; the forward flow t -> t1 is the opposite sign. Verified against known
        # translations: a true +4 px shift recovers +4, not -4. Getting this backwards is
        # silent -- the magnitudes and the spatial structure are identical either way, and
        # only the direction of motion is wrong.
        table = -torch.tensor(shifts, dtype=frame_t.dtype, device=frame_t.device)  # (n_shifts, 2)
        flow_coarse = table[best].permute(0, 3, 1, 2)   # (B, 2, H/p, W/p), channel 0 = dx
        return F.interpolate(flow_coarse, size=(H, W), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def forward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor):
        """Bidirectional flows, matching ``Small3DCNNDorsal.forward``'s 2-TUPLE contract.

        ⛔ THIS USED TO BE ``forward = forward_single``, and that alias was wrong for half
        the callers. ``Small3DCNNDorsal`` has TWO contracts on purpose: ``forward_single``
        returns one (B,2,H,W) tensor and GWM/``gwm_seg`` call that, while ``forward``
        returns ``(F_fwd, F_bwd)`` and EoO calls THAT (``eoo_dual_loss`` unpacks the pair).
        Aliasing the two made this a drop-in for the GWM path only; an EoO arm would have
        unpacked a (B,2,H,W) tensor along its batch axis and failed -- or worse, silently
        succeeded at B=2. The bidirectional flow is computed by swapping the frame order,
        exactly as the learned dorsal does; block matching is symmetric, so the backward
        pass is a second search rather than a negation of the first (they differ wherever
        a correspondence is occluded, which is the signal EoO's consistency term reads).
        """
        return self.forward_single(frame_t, frame_t1), self.forward_single(frame_t1, frame_t)
