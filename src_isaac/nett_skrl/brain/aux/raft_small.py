"""RAFT-S (Teed & Deng, ECCV 2020), the small variant, trained FROM SCRATCH WITHOUT LABELS.

U35 (owner 2026-10-05): "GWM-Seg with RAFT included using a CNN instead of DINO with more than 2
regions". The reference GWM (Choudhury et al. 2022) feeds its segmenter flow from a FROZEN RAFT
pretrained on FlyingChairs/Things with SUPERVISED flow labels. Both are excluded in this campaign
(no pretrained weights, no supervised labels) unless the owner rules otherwise, so this module is
the RAFT ARCHITECTURE with NO checkpoint, trained inside the run by an UNSUPERVISED objective
(UFlow/SMURF-style, minus SMURF's supervised-teacher stage):

    L = sum_i gamma^(n-1-i) * [ census(I1, warp(I2, f_i)) on non-occluded, in-frame pixels
                                + w_s * edge-aware first-order smoothness(f_i) ]

over BOTH directions, with occlusion from forward-backward consistency (UnFlow, Meister et al.
2018: |f12 + f21(x + f12)|^2 < 0.01 (|f12|^2 + |f21(x + f12)|^2) + 0.5), computed on the final
iteration's flows and DETACHED, switched on after a warm-up and floored per sample (see
``unsupervised_flow_loss``). SSIM is not used: census is UFlow's illumination-robust term.

⛔ NO PRETRAINED PATH EXISTS HERE AND NONE MAY BE ADDED. This module is RAFT-S from scratch only.
The owner's scoped pretrained exception (DECISIONS §73) is a SEPARATE module, ``raft_pretrained.py``
(frozen torchvision raft_large C_T_V2, flow targets only), reached by `NETT_GWM_FLOW=raft_pretrained`.

⛔ WHY ITS OWN OBJECTIVE AND NEVER THE SEGMENTATION LOSS. gwm_seg.py's docstring: the quadratic
flow-reconstruction loss is homogeneous of degree 2 in the flow, so a flow net trained THROUGH it
drives the loss to zero by shrinking its output (GWM-PAPER cell E collapsed exactly so, absmax
~0.51 against the expert's 12). Census + smoothness has a scale anchor: the photometric term is
minimised by the TRUE displacement, not by zero. The segmentation loss only ever sees this net's
flow DETACHED.

ARCHITECTURE, FROM THE REFERENCE (princeton-vl/RAFT core/, `--small`), with the deviations stated:
- feature encoder: SmallEncoder(128, instance norm), context encoder: SmallEncoder(96+64, no norm),
  both at 1/8 resolution; hidden 96, context 64.
- all-pairs correlation, 4-level pyramid, radius 3 (4 x 49 = 196 planes), bilinear lookup.
- SmallMotionEncoder + ConvGRU + FlowHead; flow upsampled x8 bilinearly (RAFT-S has no convex
  upsampler). coords1 is detached at the start of every iteration, as in the reference.
- ⚠ DEVIATION: iterations default to 8 (`NETT_GWM_RAFT_ITERS`), not the reference's 12 at train
  time, to bound training memory at the GWM-Seg packing; any value can be set.
- ⚠ DEVIATION: inputs are padded (replicate) to a multiple of 8 and unpadded, so any eye size runs.
- ⚠ NONDETERMINISM, STATED: the bilinear lookup and the photometric warp use `grid_sample`, whose
  CUDA backward has no deterministic kernel. The segmenter trains on the ROLLOUT path, where the
  ambient policy is `use_deterministic_algorithms(True, warn_only=True)` (runtime/task.py), so this
  WARNS rather than raises; it never runs inside the PPO update's strict scope. Under
  NETT_STRICT_DETERMINISM=1 (a diagnostic mode) it raises, as it should.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Encoders (reference core/extractor.py: BottleneckBlock, SmallEncoder)
# ─────────────────────────────────────────────────────────────────────────────

def _norm(kind: str, ch: int) -> nn.Module:
    if kind == "instance":
        return nn.InstanceNorm2d(ch)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"RAFT-S norm {kind!r}: expected 'instance' or 'none'")


class _Bottleneck(nn.Module):
    def __init__(self, in_planes: int, planes: int, norm: str, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes // 4, 1)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, 3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, 1)
        self.n1, self.n2, self.n3 = _norm(norm, planes // 4), _norm(norm, planes // 4), _norm(norm, planes)
        self.down = None
        if stride != 1:
            self.down = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride), _norm(norm, planes))

    def forward(self, x):
        y = F.relu(self.n1(self.conv1(x)))
        y = F.relu(self.n2(self.conv2(y)))
        y = F.relu(self.n3(self.conv3(y)))
        if self.down is not None:
            x = self.down(x)
        return F.relu(x + y)


class SmallEncoder(nn.Module):
    """(B, 3, H, W) in [-1, 1] -> (B, out, H/8, W/8)."""

    def __init__(self, out_dim: int, norm: str):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.n1 = _norm(norm, 32)
        self.layer1 = nn.Sequential(_Bottleneck(32, 32, norm, 1), _Bottleneck(32, 32, norm, 1))
        self.layer2 = nn.Sequential(_Bottleneck(32, 64, norm, 2), _Bottleneck(64, 64, norm, 1))
        self.layer3 = nn.Sequential(_Bottleneck(64, 96, norm, 2), _Bottleneck(96, 96, norm, 1))
        self.conv2 = nn.Conv2d(96, out_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.n1(self.conv1(x)))
        return self.conv2(self.layer3(self.layer2(self.layer1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Correlation pyramid (reference core/corr.py: CorrBlock)
# ─────────────────────────────────────────────────────────────────────────────

def _coords_grid(b: int, h: int, w: int, device, dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    return torch.stack([xs, ys], dim=0).unsqueeze(0).expand(b, -1, -1, -1)   # (B, 2, h, w), x first


def _bilinear(img: torch.Tensor, coords: torch.Tensor, padding: str = "zeros") -> torch.Tensor:
    """Sample img (N, C, H, W) at pixel coords (N, h, w, 2) (x, y); ``padding`` as grid_sample."""
    h, w = img.shape[-2:]
    gx = 2.0 * coords[..., 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * coords[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(img, torch.stack([gx, gy], dim=-1), mode="bilinear",
                         padding_mode=padding, align_corners=True)


class CorrPyramid:
    def __init__(self, f1: torch.Tensor, f2: torch.Tensor, levels: int = 4, radius: int = 3):
        b, d, h, w = f1.shape
        corr = torch.einsum("bdn,bdm->bnm", f1.flatten(2), f2.flatten(2)) / (d ** 0.5)
        corr = corr.reshape(b * h * w, 1, h, w)
        self.levels, self.radius, self.shape = levels, radius, (b, h, w)
        self.pyramid = [corr]
        for _ in range(levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2, ceil_mode=True)
            self.pyramid.append(corr)
        r = radius
        d1 = torch.linspace(-r, r, 2 * r + 1, device=f1.device, dtype=f1.dtype)
        dy, dx = torch.meshgrid(d1, d1, indexing="ij")
        self.delta = torch.stack([dx, dy], dim=-1).view(1, 2 * r + 1, 2 * r + 1, 2)

    @property
    def planes(self) -> int:
        return self.levels * (2 * self.radius + 1) ** 2

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        b, h, w = self.shape
        c = coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
        out = []
        for i, corr in enumerate(self.pyramid):
            win = c / (2 ** i) + self.delta                          # (BHW, 2r+1, 2r+1, 2)
            out.append(_bilinear(corr, win).view(b, h, w, -1))
        return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Update block (reference core/update.py: SmallMotionEncoder, ConvGRU, FlowHead)
# ─────────────────────────────────────────────────────────────────────────────

class _SmallUpdate(nn.Module):
    def __init__(self, corr_planes: int, hidden: int = 96, context: int = 64):
        super().__init__()
        self.convc1 = nn.Conv2d(corr_planes, 96, 1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)
        gin = 82 + context
        self.convz = nn.Conv2d(hidden + gin, hidden, 3, padding=1)
        self.convr = nn.Conv2d(hidden + gin, hidden, 3, padding=1)
        self.convq = nn.Conv2d(hidden + gin, hidden, 3, padding=1)
        self.fh1 = nn.Conv2d(hidden, 128, 3, padding=1)
        self.fh2 = nn.Conv2d(128, 2, 3, padding=1)
        # ⚠ DEVIATION: the last flow-head conv starts at zero, so the untrained net predicts zero
        # flow (the reference's random init gave absmax ~11 px at 80x128 here). Its weights still
        # receive gradient from the first step, through fh1's activations.
        nn.init.zeros_(self.fh2.weight)
        nn.init.zeros_(self.fh2.bias)

    def forward(self, net, inp, corr, flow):
        c = F.relu(self.convc1(corr))
        f = F.relu(self.convf2(F.relu(self.convf1(flow))))
        motion = torch.cat([F.relu(self.conv(torch.cat([c, f], dim=1))), flow], dim=1)   # 82
        x = torch.cat([inp, motion], dim=1)
        hx = torch.cat([net, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * net, x], dim=1)))
        net = (1 - z) * net + z * q
        return net, self.fh2(F.relu(self.fh1(net)))


class RAFTSmall(nn.Module):
    """Two frames in [0, 1] -> list of (B, 2, H, W) flows in PIXELS (x, y), one per iteration.

    ``forward_single`` returns the final flow only, matching ``ExpertBlockFlow`` /
    ``Small3DCNNDorsal`` so GWM code can treat it as a dorsal.
    """

    HIDDEN, CONTEXT = 96, 64

    def __init__(self, iters: int = 8, levels: int = 4, radius: int = 3):
        super().__init__()
        if iters < 1:
            raise ValueError(f"RAFT-S iters must be >= 1; got {iters}")
        self.iters, self.levels, self.radius = int(iters), int(levels), int(radius)
        self.fnet = SmallEncoder(128, "instance")
        self.cnet = SmallEncoder(self.HIDDEN + self.CONTEXT, "none")
        self.update = _SmallUpdate(levels * (2 * radius + 1) ** 2, self.HIDDEN, self.CONTEXT)

    def forward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor, iters: int | None = None):
        if frame_t.shape != frame_t1.shape:
            raise ValueError(f"frame shapes differ: {tuple(frame_t.shape)} vs {tuple(frame_t1.shape)}")
        H, W = frame_t.shape[-2:]
        ph, pw = (-H) % 8, (-W) % 8
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        i1 = F.pad(2.0 * frame_t - 1.0, pad, mode="replicate")
        i2 = F.pad(2.0 * frame_t1 - 1.0, pad, mode="replicate")
        f = self.fnet(torch.cat([i1, i2], dim=0))
        f1, f2 = f.float().chunk(2, dim=0)
        corr = CorrPyramid(f1, f2, self.levels, self.radius)
        ctx = self.cnet(i1)
        net, inp = torch.split(ctx, [self.HIDDEN, self.CONTEXT], dim=1)
        net, inp = torch.tanh(net), F.relu(inp)
        b, _, h, w = f1.shape
        coords0 = _coords_grid(b, h, w, f1.device, f1.dtype)
        coords1 = coords0.clone()
        flows = []
        for _ in range(int(iters or self.iters)):
            coords1 = coords1.detach()
            flow = coords1 - coords0
            net, delta = self.update(net, inp, corr(coords1), flow)
            coords1 = coords1 + delta
            up = 8.0 * F.interpolate(coords1 - coords0, scale_factor=8, mode="bilinear", align_corners=True)
            flows.append(up[..., pad[2]:pad[2] + H, pad[0]:pad[0] + W])
        return flows

    def forward_single(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        return self.forward(frame_t, frame_t1)[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Unsupervised objective (UFlow-style: census + edge-aware smoothness, FB occlusion)
# ─────────────────────────────────────────────────────────────────────────────

def warp(img: torch.Tensor, flow: torch.Tensor, padding: str = "border") -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-warp img (B,C,H,W) by flow (B,2,H,W): out(x) = img(x + flow(x)). Also the in-frame mask.

    ⛔ BORDER PADDING, AND THE IN-FRAME MASK IS REPORTED, NOT USED TO EXCLUDE. Measured on the
    synthetic translating texture: excluding out-of-frame (and occluded) pixels from a loss
    normalised by the kept count gave the optimiser an exit -- within 3 steps the flow ran to
    dx ~ -17 px, the forward-backward check failed everywhere, the kept set was empty and the
    loss was exactly 0.0 with zero gradient from then on. With border replication an
    out-of-frame match reads a smeared edge, which costs census like any wrong match.
    """
    b, _, h, w = img.shape
    coords = _coords_grid(b, h, w, img.device, img.dtype) + flow
    inside = ((coords[:, 0] >= 0) & (coords[:, 0] <= w - 1) & (coords[:, 1] >= 0) & (coords[:, 1] <= h - 1))
    return _bilinear(img, coords.permute(0, 2, 3, 1), padding), inside.unsqueeze(1).to(img.dtype)


def _census(img: torch.Tensor, patch: int = 7) -> torch.Tensor:
    gray = img.mean(dim=1, keepdim=True) * 255.0
    k = torch.eye(patch * patch, device=img.device, dtype=img.dtype).view(patch * patch, 1, patch, patch)
    neigh = F.conv2d(gray, k, padding=patch // 2)
    diff = neigh - gray
    return diff / torch.sqrt(0.81 + diff * diff)


def census_loss(img1: torch.Tensor, img2_warped: torch.Tensor, mask: torch.Tensor, patch: int = 7) -> torch.Tensor:
    """UFlow's soft-Hamming census distance, robust-penalised, mean over ``mask`` (B,1,H,W)."""
    t1, t2 = _census(img1, patch), _census(img2_warped, patch)
    d2 = (t1 - t2) ** 2
    dist = (d2 / (0.1 + d2)).sum(dim=1, keepdim=True)
    pen = (dist + 0.01) ** 0.4
    # the census window at the border reads zero padding: exclude a patch//2 frame
    r = patch // 2
    border = torch.zeros_like(mask)
    border[..., r:-r or None, r:-r or None] = 1.0
    m = mask * border
    return (pen * m).sum() / m.sum().clamp_min(1.0)


def smoothness_loss(flow: torch.Tensor, img: torch.Tensor, edge: float = 150.0) -> torch.Tensor:
    """First-order edge-aware smoothness, flow normalised by image size."""
    h, w = flow.shape[-2:]
    f = flow / torch.tensor([w, h], device=flow.device, dtype=flow.dtype).view(1, 2, 1, 1)
    gx = (f[..., :, 1:] - f[..., :, :-1]).abs()
    gy = (f[..., 1:, :] - f[..., :-1, :]).abs()
    ix = (img[..., :, 1:] - img[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    iy = (img[..., 1:, :] - img[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    return (gx * torch.exp(-edge * ix)).mean() + (gy * torch.exp(-edge * iy)).mean()


@torch.no_grad()
def fb_occlusion(f12: torch.Tensor, f21: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) 1 = non-occluded under UnFlow's forward-backward check."""
    f21_w, _ = warp(f21, f12, padding="zeros")
    lhs = ((f12 + f21_w) ** 2).sum(dim=1, keepdim=True)
    rhs = 0.01 * ((f12 ** 2).sum(dim=1, keepdim=True) + (f21_w ** 2).sum(dim=1, keepdim=True)) + 0.5
    return (lhs < rhs).to(f12.dtype)


def unsupervised_flow_loss(img1, img2, flows12, flows21, *, smooth: float = 4.0, gamma: float = 0.8,
                           use_occlusion: bool = True, min_nonocc: float = 0.5):
    """Sum over iterations and both directions. Returns (loss, scalars).

    Occlusion (UnFlow forward-backward, from the final iteration, detached) is applied only when
    ``use_occlusion`` and only to a sample whose non-occluded fraction is >= ``min_nonocc``; any
    other sample is scored on every pixel. ⚠ The floor is NOT in UFlow (which instead switches
    occlusion on after a step count, as the wrapper also does via NETT_GWM_RAFT_OCC_AFTER). It is
    here because a detached mask that drops most of a sample zeroes that sample's gradient, and
    on the synthetic texture that state was absorbing (see ``warp``).
    """
    if len(flows12) != len(flows21) or not flows12:
        raise ValueError("unsupervised_flow_loss needs equal, non-empty forward/backward iteration lists")
    b = img1.shape[0]
    ones = img1.new_ones(b, 1, *img1.shape[-2:])
    occ12, occ21, applied = ones, ones, 0.0
    nonocc = float("nan")
    if use_occlusion:
        o12 = fb_occlusion(flows12[-1].detach(), flows21[-1].detach())
        o21 = fb_occlusion(flows21[-1].detach(), flows12[-1].detach())
        nonocc = float(o12.mean())
        keep = ((o12.mean(dim=(1, 2, 3)) >= min_nonocc) & (o21.mean(dim=(1, 2, 3)) >= min_nonocc)).to(img1.dtype)
        k = keep.view(b, 1, 1, 1)
        occ12, occ21 = k * o12 + (1 - k) * ones, k * o21 + (1 - k) * ones
        applied = float(keep.mean())
    n = len(flows12)
    total = img1.new_zeros(())
    last_c = last_s = None
    inside = None
    def term(a, bk):
        w2, in2 = warp(img2, a)
        w1, _ = warp(img1, bk)
        c = 0.5 * (census_loss(img1, w2, occ12) + census_loss(img2, w1, occ21))
        s = 0.5 * (smoothness_loss(a, img1) + smoothness_loss(bk, img2))
        return c, s, in2

    for i, (a, bk) in enumerate(zip(flows12, flows21)):
        wgt = gamma ** (n - 1 - i)
        # ⛔ CHECKPOINTED: the census transform holds 49-plane maps at full resolution, and kept
        # for every iteration and both directions it was ~190 of the 210 MiB per training pair
        # (measured, 80x128, 8 iterations). Recomputing it in backward costs one extra census.
        if torch.is_grad_enabled() and (a.requires_grad or bk.requires_grad):
            c, s, in2 = checkpoint(term, a, bk, use_reentrant=False)
        else:
            c, s, in2 = term(a, bk)
        total = total + wgt * (c + smooth * s)
        last_c, last_s, inside = c, s, in2
    scalars = {
        "raft_census": float(last_c.detach()),
        "raft_smooth": float(last_s.detach()),
        "raft_nonocc_frac": nonocc,
        "raft_occ_applied_frac": applied,
        "raft_inframe_frac": float(inside.mean()),
        "raft_flow_absmax": float(flows12[-1].detach().abs().max()),
    }
    return total, scalars
