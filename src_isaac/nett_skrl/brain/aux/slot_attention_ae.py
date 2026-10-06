"""Slot Attention object-discovery auto-encoder (Locatello et al. 2020), trained from scratch.

PORTED FROM, read at the source:
    Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020,
    arXiv:2006.15055 -- Algorithm 1, Appendix E.1-E.3 (Tables 4 and 6: the CLEVR object-discovery
    encoder and decoder), Table 11 (shared hyperparameters). The official TF code
    (google-research/slot_attention/model.py) settles what the tables leave open; where it does,
    that is said at the line.

⛔ THE MASKS ARE THE DECODER ALPHAS, NOT THE ATTENTION WEIGHTS. Each slot is decoded by a shared
spatial-broadcast decoder to RGB + one unnormalised alpha; the alphas are softmaxed ACROSS SLOTS
and are the mixture weights that recombine the per-slot images (E.3). That softmax is a partition
of unity per pixel, which is what ``flow_reconstruction_loss`` and the wrapper's
``1 - M_background`` rule require. The attention map lives at the encoder grid and is not what the
paper evaluates.

⛔ SLOT INDICES CARRY NO IDENTITY. Slots are drawn i.i.d. from ONE shared Gaussian and every step
is permutation-equivariant (Appendix D), so "slot 2" names nothing across images. A consumer must
pick slots per sample by a stated rule; see ``SASeg._keep_mask``.

PAPER VALUES, KEPT
    Slot dim D = 64; GRU hidden 64; residual MLP hidden 128, ReLU; T = 3 iterations; attention
    epsilon 1e-8; softmax over SLOTS, then a weighted mean over inputs; three LayerNorms
    (inputs, slots, MLP). Encoder: 4 x Conv 5x5, 64 ch, stride 1, SAME, ReLU; soft position
    embedding (4-channel [0,1] ramp to each border, learned linear map, ADDED); flatten;
    LayerNorm; per-location MLP 64 ReLU -> 64. Decoder: broadcast each slot to a grid, add a
    position embedding, 4 x stride-2 5x5 64 ReLU, one stride-1 5x5 64 ReLU, 3x3 -> 4 (RGB +
    alpha); softmax the alphas over slots; recon = sum_k alpha_k * rgb_k. Pixels in [-1, 1]
    (E.8); MSE reconstruction. Slot init mu, log_sigma shared and learned, glorot-uniform as in
    the official code (shape [1,1,D] -> bound sqrt(6 / (1 + D))).

DIVERGENCES, DECLARED
    1. ⚠ WORKING RESOLUTION 64 x 96, not 128 x 128. The 448x280 eye is area-downsampled to
       64 x 96 (H x W) before the encoder, so the decoder starts from a 4 x 6 broadcast grid and
       its four stride-2 layers land exactly on 64 x 96 (the paper starts from 8 x 8 -> 128 x 128).
       Aspect 1.5 against the eye's 1.6: a 6% horizontal squeeze, undone when the alphas are
       resampled to the eye. WHY: activation memory per train step. Measured on CPU at batch 8:
       MoTokNet at 280x448 saves 379 MB for backward; this net saves 431 MB at 80 x 128 and
       260 MB at 64 x 96 (K=4). The paper's own 4-layer stride-1 encoder at full 280x448 would
       be ~30x MoTok.
    2. ⚠ THE RECONSTRUCTION TARGET IS THE DOWNSAMPLED FRAME. The paper reconstructs its input at
       its input resolution; so does this, and its input is the 64 x 96 copy.
    3. ⚠ DETERMINISTIC INFERENCE. Training draws fresh slot noise each forward (the paper). In
       eval() the draw is a FIXED persistent buffer ``eval_noise`` -- one sample from N(0, I),
       saved with the weights -- so the mask handed to the policy is a function of the frame
       alone and the test phase uses the same draw as train. It is a valid sample of the
       distribution the model was trained on, not a different init.
    4. ⚠ Table 6 writes the decoder rows as "Conv 5x5 ... stride 2"; the official code uses
       Conv2DTranspose for every decoder layer, and so does this (stride 1 transposed conv is a
       conv with flipped padding; output_padding=1 makes each stride-2 layer double exactly).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

SA_RES = (64, 96)          # working (H, W); see divergence 1
SA_BROADCAST = (4, 6)      # SA_RES / 2**4


def _ramp_grid(h: int, w: int) -> torch.Tensor:
    """(h, w, 4): [y, x, 1-y, 1-x], each in [0, 1] -- official ``build_grid`` order."""
    ys = torch.linspace(0.0, 1.0, h)
    xs = torch.linspace(0.0, 1.0, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gy, gx, 1.0 - gy, 1.0 - gx], dim=-1)


class SoftPositionEmbed(nn.Module):
    """E.2: a learned linear map of the 4-channel border-distance ramp, ADDED to features."""

    def __init__(self, dim: int, hw: tuple[int, int]) -> None:
        super().__init__()
        self.proj = nn.Linear(4, dim)
        self.register_buffer("grid", _ramp_grid(*hw), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # x: (B, C, H, W)
        return x + self.proj(self.grid).permute(2, 0, 1).unsqueeze(0)


class SlotAttention(nn.Module):
    """Algorithm 1. Returns slots (B, K, D)."""

    def __init__(self, num_slots: int, dim: int = 64, iters: int = 3,
                 mlp_hidden: int = 128, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_slots, self.dim, self.iters, self.eps = num_slots, dim, iters, eps
        self.scale = dim ** -0.5
        bound = math.sqrt(6.0 / (1 + dim))                        # TF glorot on [1, 1, D]
        self.slots_mu = nn.Parameter(torch.empty(1, 1, dim).uniform_(-bound, bound))
        self.slots_log_sigma = nn.Parameter(torch.empty(1, 1, dim).uniform_(-bound, bound))
        self.register_buffer("eval_noise", torch.randn(1, num_slots, dim))   # divergence 3
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:      # inputs: (B, N, D)
        b, k, d = inputs.shape[0], self.num_slots, self.dim
        if self.training:
            noise = torch.randn(b, k, d, device=inputs.device, dtype=inputs.dtype)
        else:
            noise = self.eval_noise.expand(b, -1, -1)
        slots = self.slots_mu + self.slots_log_sigma.exp() * noise
        inputs = self.norm_inputs(inputs)
        keys, values = self.to_k(inputs), self.to_v(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            attn = torch.softmax(torch.einsum("bnd,bkd->bkn", keys, q) * self.scale, dim=1)  # over SLOTS
            attn = attn + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)          # weighted mean over inputs
            updates = torch.einsum("bkn,bnd->bkd", attn, values)
            slots = self.gru(updates.reshape(b * k, d), slots_prev.reshape(b * k, d)).reshape(b, k, d)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class SlotAttentionAutoEncoder(nn.Module):
    """Frame (B,3,H,W) in [0,1] -> slots -> per-slot RGB + alpha at SA_RES."""

    def __init__(self, num_slots: int = 4, in_ch: int = 3, ch: int = 64, dim: int = 64) -> None:
        super().__init__()
        self.num_slots, self.in_ch = num_slots, in_ch
        enc = []
        for i in range(4):
            enc += [nn.Conv2d(in_ch if i == 0 else ch, ch, 5, padding=2), nn.ReLU()]
        self.encoder_cnn = nn.Sequential(*enc)
        self.encoder_pos = SoftPositionEmbed(ch, SA_RES)
        self.layer_norm = nn.LayerNorm(ch)
        self.mlp = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, dim))
        self.slot_attention = SlotAttention(num_slots, dim=dim)
        self.decoder_pos = SoftPositionEmbed(dim, SA_BROADCAST)
        dec, c = [], dim
        for _ in range(4):
            dec += [nn.ConvTranspose2d(c, ch, 5, stride=2, padding=2, output_padding=1), nn.ReLU()]
            c = ch
        dec += [nn.ConvTranspose2d(ch, ch, 5, stride=1, padding=2), nn.ReLU(),
                nn.ConvTranspose2d(ch, in_ch + 1, 3, stride=1, padding=1)]
        self.decoder_cnn = nn.Sequential(*dec)

    @staticmethod
    def prepare(frame: torch.Tensor) -> torch.Tensor:
        """[0,1] frame at any size -> the paper's [-1,1] input at SA_RES (area resample)."""
        if tuple(frame.shape[-2:]) != SA_RES:
            frame = F.interpolate(frame, size=SA_RES, mode="area")
        return frame * 2.0 - 1.0

    def decode(self, x: torch.Tensor):
        """x: prepared (B,C,*SA_RES). Returns (recon (B,C,*SA_RES), alphas (B,K,*SA_RES))."""
        b = x.shape[0]
        f = self.encoder_pos(self.encoder_cnn(x))                  # (B, ch, H, W)
        f = self.mlp(self.layer_norm(f.flatten(2).transpose(1, 2)))  # (B, N, D)
        slots = self.slot_attention(f)                             # (B, K, D)
        z = slots.reshape(b * self.num_slots, -1, 1, 1).expand(-1, -1, *SA_BROADCAST)
        out = self.decoder_cnn(self.decoder_pos(z))                # (B*K, C+1, H, W)
        out = out.reshape(b, self.num_slots, self.in_ch + 1, *SA_RES)
        alphas = torch.softmax(out[:, :, self.in_ch], dim=1)       # over SLOTS
        recon = (out[:, :, : self.in_ch] * alphas.unsqueeze(2)).sum(dim=1)
        return recon, alphas

    def get_masks(self, frame: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) in [0,1] -> (B,K,H,W) alpha masks resampled to the frame (sum to 1)."""
        _, alphas = self.decode(self.prepare(frame))
        if tuple(frame.shape[-2:]) != SA_RES:
            # bilinear weights sum to 1, so the partition of unity survives the resample
            alphas = F.interpolate(alphas, size=frame.shape[-2:], mode="bilinear", align_corners=False)
        return alphas
