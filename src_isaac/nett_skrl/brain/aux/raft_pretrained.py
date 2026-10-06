"""FROZEN, PRETRAINED RAFT as a flow TARGET for GWM-Seg -- the owner's scoped exception (U35).

⛔ THIS IS A RULE EXCEPTION, NOT A DEFAULT. The campaign excludes pretrained weights. The owner
granted ONE exception (DECISIONS §73, 2026-10-06, verbatim: "A pretrained exception is warranted
here. The idea is to see if this approach works at all."): frozen pretrained RAFT as the flow
source of GWM-Seg in U35, which is the published GWM's own flow prior. Scope, as recorded:
  - RAFT gets NO gradient and produces flow TARGETS only. It never feeds the policy and never
    touches the encoder; it is reachable only through ``NETT_GWM_FLOW=raft_pretrained`` in
    ``body/wrappers/gwm_spectral.py``.
  - A cell that uses it is labelled ``...-RAFTPT`` and is not a compliant solution.

WEIGHTS. torchvision ``raft_large`` / ``Raft_Large_Weights.C_T_V2``: trained from scratch on
FlyingChairs + FlyingThings3D (synthetic rendered scenes with ground-truth flow; no real video,
no NETT data). 5,257,536 parameters. The weight file's sha256 is checked against
``EXPECTED_SHA256`` before use, so a different file cannot run under this label, and its first
12 hex digits are printed once at construction for close-time verification:
    [NETT pretrained] RAFT raft_large C_T_V2 frozen sha256=1bb1363a43b4 iters=12

FREEZING, ENFORCED THREE WAYS: ``eval()``; ``requires_grad_(False)`` on every parameter; every
call under ``torch.no_grad()``. The net is held OUTSIDE the nn.Module tree of the shim the
segmenter sees (``PretrainedFlow`` has no parameters and an empty state_dict), so it cannot reach
an optimizer through ``.parameters()`` and is not written into seg_state checkpoints.

RESOLUTION AND UNITS. The GWM loss takes flow in PIXELS OF THE SEGMENTER'S FRAME (80x128 at the
live eye), channel 0 = dx (right +), channel 1 = dy (down +) -- exactly ``ExpertBlockFlow``'s
output. RAFT works on a 1/8 grid, so 80x128 would give it a 10x16 correlation grid. Frames are
therefore upsampled by s = max(NETT_GWM_RAFT_PT_SCALE, 128 / min(H, W)), each side rounded UP to
a multiple of 8 (80x128 at the default 2.0 -> 160x256, a 20x32 grid), mapped [0,1] -> [-1,1] as
torchvision's ``OpticalFlow`` transform does, and the flow is resized back to (H, W) bilinearly
with dx scaled by W/W' and dy by H/H'. Measured on CPU, textured random-shift pairs at 80x128,
12 iterations, interior EPE: 0.03-0.07 px at s=1.6, 0.03-0.04 px at 2.0, 0.01-0.02 px at 3.0.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

EXPECTED_SHA256 = "1bb1363a43b40f8dea96c530217a7e8b1804ffc77506058db2ec0afa428ad4f6"
WEIGHTS_NAME = "C_T_V2"
MIN_SIDE = 128


def _weights_file(weights) -> str:
    return os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(weights.url))


def raft_input_size(h: int, w: int, scale: float) -> tuple[int, int]:
    """(H', W') RAFT runs at: at least ``scale``x and at least MIN_SIDE on the short side, /8."""
    s = max(float(scale), MIN_SIDE / min(h, w))
    return 8 * math.ceil(h * s / 8 - 1e-9), 8 * math.ceil(w * s / 8 - 1e-9)


class FrozenRAFTLarge:
    """torchvision raft_large (C_T_V2), frozen. Deliberately NOT an nn.Module (see module doc)."""

    def __init__(self, iters: int = 12, scale: float = 2.0, chunk: int = 256, device="cpu"):
        if iters < 1 or chunk < 1:
            raise ValueError(f"FrozenRAFTLarge: iters ({iters}) and chunk ({chunk}) must be >= 1")
        if not scale >= 1.0:
            raise ValueError(f"FrozenRAFTLarge: scale ({scale}) must be >= 1 (it is an UPsampling factor)")
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        weights = getattr(Raft_Large_Weights, WEIGHTS_NAME)
        net = raft_large(weights=weights, progress=False)       # downloads to the hub cache if absent
        path = _weights_file(weights)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest != EXPECTED_SHA256:
            raise RuntimeError(
                f"pretrained RAFT weight file {path} has sha256 {digest}, expected {EXPECTED_SHA256}; "
                "refusing to run a different file under the -RAFTPT label (DECISIONS §73).")
        net.eval()
        net.requires_grad_(False)
        self.net = net.to(device)
        self.iters, self.scale, self.chunk = int(iters), float(scale), int(chunk)
        self.sha256 = digest
        self.num_params = sum(p.numel() for p in net.parameters())
        msg = (f"[NETT pretrained] RAFT raft_large {WEIGHTS_NAME} frozen sha256={digest[:12]} "
               f"iters={self.iters}")
        print(msg, flush=True)
        logging.getLogger("nett.brain.raft_pretrained").info(
            "%s scale>=%g chunk=%d params=%d (no grad, no optimizer, not checkpointed)",
            msg, self.scale, self.chunk, self.num_params)

    @torch.no_grad()
    def __call__(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        """frames (B,3,H,W) in [0,1] -> forward flow t -> t1, (B,2,H,W), pixels of the input frame."""
        if frame_t.shape != frame_t1.shape or frame_t.dim() != 4 or frame_t.shape[1] != 3:
            raise ValueError(f"FrozenRAFTLarge expects two (B,3,H,W) frames; got "
                             f"{tuple(frame_t.shape)} and {tuple(frame_t1.shape)}")
        if self.net.training:                                   # nothing may have flipped it
            self.net.eval()
        B, _, H, W = frame_t.shape
        Hs, Ws = raft_input_size(H, W, self.scale)
        out = []
        for i in range(0, B, self.chunk):
            a = F.interpolate(frame_t[i: i + self.chunk].float(), (Hs, Ws), mode="bilinear", align_corners=False)
            b = F.interpolate(frame_t1[i: i + self.chunk].float(), (Hs, Ws), mode="bilinear", align_corners=False)
            f = self.net(a * 2.0 - 1.0, b * 2.0 - 1.0, num_flow_updates=self.iters)[-1]
            f = F.interpolate(f, (H, W), mode="bilinear", align_corners=False)
            out.append(torch.stack((f[:, 0] * (W / Ws), f[:, 1] * (H / Hs)), dim=1))
        return torch.cat(out).to(frame_t.dtype).detach()


class PretrainedFlow(nn.Module):
    """The segmenter's ``dorsal``: ``forward_single`` like ExpertBlockFlow, NO parameters, NO state.

    The frozen net lives in ``__dict__`` (not ``_modules``), so ``parameters()`` is empty and
    ``state_dict()`` is ``{}``: it cannot be optimised through this shim and is not checkpointed.
    """

    def __init__(self, raft: FrozenRAFTLarge):
        super().__init__()
        self.__dict__["raft"] = raft

    @torch.no_grad()
    def forward_single(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        return self.raft(frame_t, frame_t1)

    @torch.no_grad()
    def forward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor):
        """Bidirectional pair, matching ExpertBlockFlow.forward's 2-tuple contract."""
        return self.raft(frame_t, frame_t1), self.raft(frame_t1, frame_t)
