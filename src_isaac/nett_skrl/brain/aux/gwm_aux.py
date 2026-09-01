"""GWM (Guess What Moves) auxiliary loss — motion-coherence segmentation.

⛔★★★★★ THIS DELIBERATELY DOES NOT REIMPLEMENT THE REFERENCE LOSS, AND THE REASON
IS A VERIFIED FINDING IN THIS CAMPAIGN
(``from-orchestrator/2026-08-25T1607Z_..._the-GWM-flow-loss-optimises-its-own-supervision...md``):

    In the reference (karazijal/guess-what-moves, BMVC 2022) flow is a FIXED
    EXTERNAL TARGET that the masks cannot influence. In the vendored variant it is
    a FREE VARIABLE optimised by the same loss that consumes it -- and the code
    declares that coupling as INTENT, two lines above it.

    A TRIVIAL JOINT OPTIMUM THEREFORE EXISTS: the dorsal stream emits any globally
    smooth field -- constant, or zero -- the masks reconstruct it exactly, the loss
    goes to 0, AND NO SEGMENTATION IS LEARNED.

★ Same family as the CLTT zero-loss: an objective whose global optimum is reachable
WITHOUT DOING THE TASK. Neither was visible in the paper; both needed the code.

THE FIX, and it is one call
---------------------------
The flow field is trained ONLY by a photometric warping term -- grounded in the
actual next frame, which no branch can alter -- and the mask branch consumes
``flow.detach()``. That single detach restores exactly the property the reference
had and the variant lost: THE MASK CANNOT INFLUENCE ITS OWN SUPERVISION.

⚠ Without the detach this file would score, train, and converge, and its loss curve
would look BETTER than the correct version's, because zero is easy to reach.

The objective
-------------
"Guess what moves": partition the frame so that motion is COHERENT WITHIN each
slot. Two slots (fg/bg), matching MoTok's ``gwm_num_queries=2``. Per slot we take
the slot-weighted mean flow and penalise the slot-weighted variance around it. A
mask that separates a moving object from a differently-moving background lowers
this; a mask that mixes them cannot.

See ``eoo_aux.py`` for where the temporal pair comes from and why the T-major
channel order is not optional.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eoo_aux import _charbonnier, smoothness, warp


class GWMHead(nn.Module):
    """Coarse flow field + K-slot segmentation from encoder features."""

    def __init__(self, in_dim: int, grid_h: int, grid_w: int,
                 slots: int = 2, hidden: int = 32) -> None:
        super().__init__()
        self.grid_h, self.grid_w, self.hidden = int(grid_h), int(grid_w), int(hidden)
        # ⛔ See EoOHead: an unbounded flow escapes the grid and the photometric term
        # goes EXACTLY zero in value AND gradient. THIS loss is the one that was
        # measured diverging -- 2 of 8 seeds, |flow| to 14.89 on a 10x16 grid.
        self.max_disp = max(min(self.grid_h, self.grid_w) / 2.0, 1.0)
        self.slots = int(slots)
        self.project = nn.Linear(in_dim, hidden * self.grid_h * self.grid_w)
        # ⛔★★★★★ TWO TRUNKS, NOT ONE. seat:verifier measured that a single shared
        # trunk makes the detach a lie: with the loss set to `coh` alone, |grad| on
        # flow_out was 0.000e+00 -- the detach does exactly what a detach does -- but
        # refine 9.49e-06, project 8.83e-06, ENCODER 6.78e-06. flow and mask came off
        # ONE trunk, so the coherence term trained THREE OF THE FOUR STAGES THAT
        # PRODUCE FLOW.
        # ★ THE DETACH SEVERED THE STEP, NOT THE LOOP. The reference's property is
        # that flow is a tensor NO PARAMETER OF THE MODEL PRODUCES; mine was weakened
        # to a ONE-STEP LAG while the docstring claimed the strong version. On
        # verifier's stimulus the degenerate optimum was not visited, so it was A FALSE
        # DOCSTRING ON CODE THAT WORKS -- the kind that survives longest, because the
        # docstring is the reason nobody looks again.
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.mask_trunk = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.flow_out = nn.Conv2d(hidden, 2, 3, padding=1)
        self.mask_out = nn.Conv2d(hidden, self.slots, 3, padding=1)
        # Zero-init the flow head only: a random flow field warps to noise on step 1.
        # The mask head must NOT be zero-init -- identical logits give a uniform
        # softmax whose slots are symmetric, and symmetric slots never separate.
        nn.init.zeros_(self.flow_out.weight)
        nn.init.zeros_(self.flow_out.bias)

    def forward(self, feats: torch.Tensor):
        B = feats.shape[0]
        x = self.project(feats).view(B, self.hidden, self.grid_h, self.grid_w)
        # The mask trunk reads the SHARED projection but is detached from it, so no
        # mask-side gradient reaches `project` or the encoder through this path.
        flow = torch.tanh(self.flow_out(self.refine(x))) * self.max_disp
        mask = F.softmax(self.mask_out(self.mask_trunk(x.detach())), dim=1)
        return flow, mask


class GWMAuxLoss(nn.Module):
    """Motion-coherence segmentation loss with GROUNDED flow supervision."""

    def __init__(self, encoder: nn.Module, *, grid_div: int = 8, slots: int = 2,
                 hidden: int = 32, max_samples: int = 32) -> None:
        super().__init__()
        from ...body.observation import image_channels_hw
        total_c, H, W = image_channels_hw(getattr(encoder, "observation_space", None))
        if total_c % 3 != 0 or total_c // 3 < 2:
            raise ValueError(
                f"GWMAuxLoss needs a framestacked observation (>=2 RGB frames); got "
                f"{total_c} channels from {H}x{W}. Declare framestack=True on this arm."
            )
        self.grid_h = max(H // grid_div, 2)
        self.grid_w = max(W // grid_div, 2)
        self.slots = int(os.environ.get("NETT_GWM_SLOTS", slots))
        self.head = GWMHead(int(encoder.features_dim), self.grid_h, self.grid_w,
                            self.slots, hidden)
        self.head.to(next(encoder.parameters()).device)
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))
        self.w_photo = float(os.environ.get("NETT_GWM_PHOTO", "1.0"))
        self.w_coh = float(os.environ.get("NETT_GWM_COH", "1.0"))
        self.w_smooth = float(os.environ.get("NETT_GWM_SMOOTH", "0.1"))
        self.w_bal = float(os.environ.get("NETT_GWM_BAL", "0.1"))

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        from ...body.observation import prepare_image_tensor
        obs = observations[: self.max_samples]
        imgs = prepare_image_tensor(obs, encoder.observation_space)
        prev, curr = imgs[:, -6:-3], imgs[:, -3:]     # T-MAJOR; see eoo_aux docstring

        feats = encoder(obs)                          # clean gradient path to backbone
        flow, mask = self.head(feats)

        small_prev = F.interpolate(prev, (self.grid_h, self.grid_w),
                                   mode="bilinear", align_corners=False)
        small_curr = F.interpolate(curr, (self.grid_h, self.grid_w),
                                   mode="bilinear", align_corners=False)

        # --- flow branch: GROUNDED. Its only supervision is the real next frame. ---
        photo = _charbonnier(warp(small_prev, flow) - small_curr).mean()
        smooth = smoothness(flow, small_curr)

        # --- mask branch: consumes flow.detach() ---
        # ⛔ THE DETACH IS THE WHOLE POINT. Remove it and the mask can drive the flow
        # to a constant to make its own coherence term vanish -- the degenerate joint
        # optimum this module exists to avoid. It would train, converge, and report a
        # LOWER loss than the correct version.
        #
        # ⛔★★★★★ BUT THE ISOLATION IS ONE-DIRECTIONAL, AND THE UNPROTECTED DIRECTION
        # IS THE ONE THAT FIRED (seat:insect, 2 of 8 seeds, measured). The detach stops
        # the MASK corrupting its own target. It does NOT stop the FLOW corrupting the
        # MASK: coherence is computed on flow.detach(), so as |flow| grows the per-slot
        # variance grows with it and NOTHING PUSHES BACK -- the photometric term, which
        # is supposed to be the ground, has gone blind past the frame edge.
        # ✅ Fixed at the source by bounding the flow in GWMHead rather than by adding
        # another penalty here. A bound cannot be outrun; a penalty can.
        f = flow.detach().unsqueeze(1)                      # (B,1,2,h,w)
        m = mask.unsqueeze(2)                               # (B,K,1,h,w)
        w = m.sum(dim=(3, 4), keepdim=True).clamp_min(1e-6)
        mean = (m * f).sum(dim=(3, 4), keepdim=True) / w    # per-slot mean flow
        # Slot-weighted variance around each slot's own mean motion.
        coh = ((m * (f - mean).pow(2)).sum(dim=(3, 4)) / w.squeeze(-1).squeeze(-1)).mean()

        # ⚠ ANTI-COLLAPSE. With no balance term one slot claims every pixel; the
        # within-slot variance is then just the global flow variance and the mask has
        # learned nothing while the loss still falls. Push the slot occupancies
        # toward uniform.
        occ = mask.mean(dim=(2, 3))                         # (B,K)
        bal = (occ - 1.0 / self.slots).pow(2).sum(dim=1).mean()

        # ⚠ A SLOT-COLLAPSE GATE REPORTS A DIVERGING ARM HEALTHY. On the diverging
        # seeds the mask did not collapse -- it FROZE at a locked 30/70 split with
        # `bal` pinned at exactly 0.08 and minOcc at exactly 0.30. Watch flow_absmax
        # against max_disp: pinned AT the bound is the tell, not the occupancies.
        self.last = {"photo": float(photo.detach()), "smooth": float(smooth.detach()),
                     "coh": float(coh.detach()), "bal": float(bal.detach()),
                     "flow_absmax": float(flow.detach().abs().max()),
                     "max_disp": float(self.head.max_disp),
                     "minOcc": float(occ.min().detach())}
        return (self.w_photo * photo + self.w_smooth * smooth
                + self.w_coh * coh + self.w_bal * bal)
