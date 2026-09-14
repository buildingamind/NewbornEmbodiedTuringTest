"""EoO (Emergence of Objectness) auxiliary loss — self-supervised optical flow.

Reference implementation: ``scripts/eoo/losses.py::unflow_loss`` and
``scripts/eoo/train.py:197``. There a single-frame ventral stream predicts mask
``M`` and a separate frame-pair dorsal stream predicts a forward/backward flow
pair from ``(frame_t, frame_t1)``. The loss warps ``frame_t`` by the flow and
compares it against ``frame_t1``.

★ THE POINT, AND WHY EoO BELONGS HERE RATHER THAN AS A REWARD: the supervision is
THE NEXT FRAME ITSELF. Nothing external labels anything. An earlier reading of EoO
in this campaign objected that it "estimates flow but does not compensate
ego-motion" -- that answers an ARM-DESIGN question. As a LOSS, estimating flow IS
the training signal, and the objection does not apply.

WHERE THE TEMPORAL PAIR COMES FROM
----------------------------------
⛔ There is NO ``next_observations`` in the PPO sample tuple; ``memory.sample``
yields observations only. A motion loss gets its pair from FRAMESTACK: an arm
declaring ``framestack=True`` has both frames concatenated on the channel axis by
``observation.py:96`` (``torch.cat``), i.e. **T-MAJOR**:

    [t-1 R, t-1 G, t-1 B, t R, t G, t B]

⚠★★★★★ GETTING THIS ORDER WRONG IS NOT HYPOTHETICAL. Both shipped motion arms
(``compact_3dcnn:83``, ``guess_what_moves:125``) viewed this tensor C-MAJOR, which
pairs (t-1 R, t-1 G) and (t G, t B) -- BOTH FROM A SINGLE FRAME -- so two of three
channels carried no temporal information at all. A reshape moves no weights, so
the parameter count is identical either way and no capacity or construction check
could ever have caught it. This module splits T-major explicitly and asserts it.

RESOLUTION
----------
The encoder returns a flat ``features_dim`` vector, not a spatial map, so flow is
predicted at a COARSE grid and the photometric loss is evaluated on frames
downsampled to match. That is standard multi-scale flow practice and it keeps the
head small; it also means this loss shapes the encoder through the only tensor the
encoder actually exposes.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Robust L1 (``scripts/eoo/losses.py:68``)."""
    return torch.sqrt(diff.pow(2) + eps * eps)


def warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp ``frame`` by ``flow`` (B,2,H,W) with a normalised sampling grid.

    Flow is in PIXELS; ``grid_sample`` wants [-1,1], hence the 2/(W-1) scaling.
    ``align_corners=True`` matches that normalisation exactly -- the two must agree
    or the warp is silently offset by half a pixel and the photometric loss simply
    reports a slightly worse number rather than an error.
    """
    B, _, H, W = frame.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=frame.device, dtype=frame.dtype),
        torch.arange(W, device=frame.device, dtype=frame.dtype),
        indexing="ij",
    )
    x = xs.unsqueeze(0) + flow[:, 0]
    y = ys.unsqueeze(0) + flow[:, 1]
    gx = 2.0 * x / max(W - 1, 1) - 1.0
    gy = 2.0 * y / max(H - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(frame, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def smoothness(flow: torch.Tensor, img: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """Edge-aware first-order flow smoothness (``scripts/eoo/losses.py:128``).

    Flow gradients are penalised WHERE THE IMAGE IS FLAT and forgiven at image
    edges -- that edge-awareness is what lets a flow field break at an object
    boundary, which is the whole objectness signal.
    """
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()
    wx = torch.exp(-alpha * (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(1, keepdim=True))
    wy = torch.exp(-alpha * (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(1, keepdim=True))
    return (wx * dx).mean() + (wy * dy).mean()


class EoOHead(nn.Module):
    """Decodes encoder features into a coarse flow field and an objectness mask."""

    def __init__(self, in_dim: int, grid_h: int, grid_w: int, hidden: int = 32) -> None:
        super().__init__()
        self.grid_h, self.grid_w, self.hidden = int(grid_h), int(grid_w), int(hidden)
        self.max_disp = max(min(self.grid_h, self.grid_w) / 2.0, 1.0)
        self.project = nn.Linear(in_dim, hidden * self.grid_h * self.grid_w)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        # 2 flow channels + 1 mask channel.
        self.out = nn.Conv2d(hidden, 3, 3, padding=1)
        # ⚠ Start at ZERO FLOW. A randomly-initialised flow field warps the frame
        # to noise on step 1, and the photometric term is then dominated by an
        # arbitrary displacement rather than by anything the encoder did.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: torch.Tensor):
        B = feats.shape[0]
        x = self.project(feats).view(B, self.hidden, self.grid_h, self.grid_w)
        x = self.out(self.refine(x))
        # ⛔★★★★★ THE FLOW IS BOUNDED STRUCTURALLY, NOT PENALISED.
        # seat:insect measured GWMAuxLoss (same warp, same grid) diverging on 2 of 8
        # seeds by letting |flow| reach 14.89 on a 10x16 grid. Past ~15px every sample
        # is outside the frame, padding_mode="border" clamps it, and the photometric
        # term goes EXACTLY ZERO in BOTH value and gradient -- bit-identical at
        # |flow|=15 and |flow|=30. The grounding does not weaken out there, IT GOES
        # BLIND, and a blind grounding cannot pull the flow back.
        # A tanh bound makes escape impossible rather than expensive. max_disp is half
        # the smaller grid axis: displacements beyond that are not physical at this
        # resolution, and EoO's measured-clean runs never approached it, so this
        # changes nothing that was already working.
        flow = torch.tanh(x[:, :2]) * self.max_disp
        return flow, torch.sigmoid(x[:, 2:3])


class EoOAuxLoss(nn.Module):
    """Self-supervised flow loss that shapes the encoder backbone.

    Same plug-in contract as ``VICRegAuxLoss``: ``compute(encoder, observations)``
    returns a scalar, and ``.head`` parameters are registered with the PPO
    optimizer by ``AuxLossPPO``.
    """

    def __init__(self, encoder: nn.Module, *, grid_div: int = 8, hidden: int = 32,
                 max_samples: int = 32) -> None:
        super().__init__()
        self.encoder = encoder
        obs_space = getattr(encoder, "observation_space", None)
        from ...body.observation import image_channels_hw
        total_c, H, W = image_channels_hw(obs_space)

        # ⛔ REFUSE A NON-FRAMESTACKED ARM RATHER THAN INVENTING A SECOND FRAME.
        # With 3 channels there is no t-1 to compare against. Silently comparing a
        # frame with itself would make this loss identically zero -- a no-op that
        # trains as plain PPO while logging aux=eoo, which is precisely the failure
        # ppo_aux.py's registry exists to stop.
        if total_c % 3 != 0 or total_c // 3 < 2:
            raise ValueError(
                f"EoOAuxLoss needs a framestacked observation (>=2 RGB frames); got "
                f"{total_c} channels from {H}x{W}. Declare framestack=True on this arm."
            )
        self.num_frames = total_c // 3
        self.grid_h = max(H // grid_div, 2)
        self.grid_w = max(W // grid_div, 2)
        self.head = EoOHead(int(encoder.features_dim), self.grid_h, self.grid_w, hidden)
        self.head.to(next(encoder.parameters()).device)
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))
        self.w_photo = float(os.environ.get("NETT_EOO_PHOTO", "1.0"))
        self.w_smooth = float(os.environ.get("NETT_EOO_SMOOTH", "0.1"))
        self.w_mask = float(os.environ.get("NETT_EOO_MASK", "0.01"))

    def split_frames(self, x: torch.Tensor):
        """Split a T-MAJOR framestacked tensor into (frame_prev, frame_curr).

        ⛔ T-major, NOT C-major -- see the module docstring. ``x[:, -3:]`` is the
        current frame, which is also how ``guess_what_moves.py:121`` reads it.
        """
        return x[:, -6:-3], x[:, -3:]

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        from ...body.observation import prepare_image_tensor
        obs = observations[: self.max_samples]
        imgs = prepare_image_tensor(obs, encoder.observation_space)
        prev, curr = self.split_frames(imgs)

        # Features come from encoder.forward -- the CLEAN gradient path. Do not use
        # any project()/no_grad variant here: the whole purpose is to reach the
        # backbone.
        feats = encoder(obs)
        flow, mask = self.head(feats)

        small_prev = F.interpolate(prev, size=(self.grid_h, self.grid_w),
                                   mode="bilinear", align_corners=False)
        small_curr = F.interpolate(curr, size=(self.grid_h, self.grid_w),
                                   mode="bilinear", align_corners=False)
        warped = warp(small_prev, flow)

        # LEGACY EQUATION: preserve this ungated reconstruction for corpus
        # reproducibility. It does not supervise objectness: only the regularizer
        # below trains the mask, with an all-foreground optimum. `eoo_dual`
        # restores the reference's normalized mask-weighted reconstruction using
        # separate spatial streams. This legacy arm must not be called objectness.
        photo = _charbonnier(warped - small_curr).mean()
        smooth = smoothness(flow, small_curr)
        # This frame-independent term is NOT constant: its mask derivative is
        # negative everywhere, so it drives foreground saturation.
        mask_reg = (1.0 - mask).mean()

        # Report components separately: Charbonnier has an epsilon floor and
        # mask_reg decreases as the mask approaches one, regardless of objects.
        self.last = {"photo": float(photo.detach()), "smooth": float(smooth.detach()),
                     "mask_reg": float(mask_reg.detach()),
                     "mask_mean": float(mask.detach().mean()),
                     "flow_absmax": float(flow.detach().abs().max()),
                     # seat:insect: GWM reports max_disp, EoO did not, so the
                     # "is it PINNED at the bound?" check could not be run on EoO
                     # from the dict at all. Symmetry matters here because pinned-at-
                     # the-bound is the divergence tell, not the flow magnitude.
                     "max_disp": float(self.head.max_disp)}
        return self.w_photo * photo + self.w_smooth * smooth + self.w_mask * mask_reg
