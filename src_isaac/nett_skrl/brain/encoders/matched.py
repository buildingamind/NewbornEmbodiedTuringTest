"""The owner's six MATCHED encoders (workspace DECISIONS §92; spec, verbatim:
``docs/specs/matched_architectures_owner_2026-10-12.md``).

CNN, ViT, CNN2F, ViT2F, 3DCNN and ViViT share everything except the token MIXER and how many frames
it sees at once, so the comparison is attention vs convolution and single-frame vs two-frame vs
video -- not six unrelated trunks. One class, one ``variant`` key:

    variant  frames  mixer                        mixer scope             temporal pos  readout D
    cnn      1       Conv2d(96,96,3,p1)           one frame               no            128
    vit      1       attention 3 x 32             one frame, N 160        no            128
    cnn2f    2       Conv2d(96,96,3,p1)           each frame alone        no            64 / frame
    vit2f    2       attention 3 x 32             each frame alone, N 160 no            64 / frame
    3dcnn    2       Conv3d(96,96,(3,3,3),p1)     both frames             yes           64 / frame
    vivit    2       attention 3 x 32             both frames, N 320      yes           64 / frame
    3dcnn_g6 2       Conv3d(96,96,(3,3,3),p1,g6)  both frames             yes           64 / frame

Shared by all six: per-frame patch projection Conv2d(3,96,16,s16) -> [B,T,10,16,96]; learned spatial
positions [1,1,10,16,96]; three pre-norm residual blocks ``u = x + mixer(norm1(x)); y = u +
ffn(norm2(u))`` with ffn Linear(96,384)-GELU-Linear(384,96); LayerNorm(96); a per-frame readout with
shared weights, Conv2d(96,32,4,s2)-ReLU-Conv2d(32,32,(4,3))-ReLU-flatten(160)-Linear(160,D); two-frame
models concatenate [previous, current]; LayerNorm(128). All LayerNorms eps 1e-6; bias everywhere; no
pooling, CLS token, dropout or batch norm.

ENCODER PARAMETERS (the spec's table, pinned by tests/test_matched_encoders.py):
    cnn 644,640   vit 507,264   cnn2f 634,336   vit2f 496,960   3dcnn 1,132,192   vivit 497,152
    3dcnn_g6 510,112
⚠ 3dcnn is 2.3x vivit: the spec's own Conv3d mixer is 27 x 96 x 96 per block. Reported, not hidden.
3dcnn_g6 is the owner's COMPACT 3DCNN (DECISIONS §92, answer 2): the same model with groups=6 in the
three backbone Conv3d mixers ("nn.Conv3d(96, 96, kernel_size=3, padding=1, groups=6)"), width 96,
three blocks, FFNs and readout unchanged -- 41,568 params per mixer instead of 248,928.

⛔ THE EYE IS FIXED AT 160 x 256 (H x W). The spatial positions are [10,16] patches of 16 and the
readout's 4x7 -> 1x5 arithmetic only closes there, so any other eye REFUSES at construction rather
than silently re-gridding (NETT_EYE_RES=256x160).

⛔ FRAME ORDER. The framestack wrapper concatenates channels oldest-first, so a 6-channel observation
is [previous RGB, current RGB] -- the spec's order. ``validate_framestack_depth`` is called in its
STRICT form (channels_per_frame=3): a 3-frame stack, or a DVS pair, refuses to build.

INIT (spec "Initialization"): conv/linear weights and biases U(+-1/sqrt(fan_in)) -- which IS PyTorch's
default for Conv*/Linear (kaiming_uniform a=sqrt(5) and the matching bias bound), so the constructor
leaves them alone; LayerNorm 1/0 (default); positions Normal(0, .02), set here. ``OWNS_INIT = True``
so the builder and backbone skip the fleet's orthogonal pass (models/utils/init.encoder_owns_init),
which would otherwise overwrite every conv and linear.
"""

from __future__ import annotations

import os

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.temporal import validate_framestack_depth

DIM, BLOCKS, HEADS, HEAD_DIM, FFN, PATCH = 96, 3, 3, 32, 384, 16
EYE_HW = (160, 256)
GRID = (EYE_HW[0] // PATCH, EYE_HW[1] // PATCH)          # (10, 16)
OUT_DIM = 128
LN_EPS = 1e-6

#: variant -> (frames, mixer, joint over time, temporal positions); "conv3d_g6" = Conv3d with groups=6
VARIANTS: dict[str, tuple[int, str, bool, bool]] = {
    "cnn":   (1, "conv2d", False, False),
    "vit":   (1, "attn",   False, False),
    "cnn2f": (2, "conv2d", False, False),
    "vit2f": (2, "attn",   False, False),
    "3dcnn": (2, "conv3d", True,  True),
    "vivit": (2, "attn",   True,  True),
    "3dcnn_g6": (2, "conv3d_g6", True, True),
}


class _Attention(nn.Module):
    """The spec's attention mixer: qkv Linear(96,288), 3 heads x 32, no mask, out Linear(96,96)."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(DIM, 3 * HEADS * HEAD_DIM)
        self.out_proj = nn.Linear(HEADS * HEAD_DIM, DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        M, N, _ = tokens.shape
        q, k, v = self.qkv_proj(tokens).reshape(M, N, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4).unbind(0)
        # == softmax(q k^T / sqrt(32)) v (default scale 1/sqrt(head_dim)); pinned against the spec's
        # explicit form in tests/test_matched_encoders.py.
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(y.transpose(1, 2).reshape(M, N, HEADS * HEAD_DIM))


class _Mixer(nn.Module):
    """Shape-preserving token mixer on the internal layout [B, T, 10, 16, 96]."""

    def __init__(self, kind: str, joint: bool) -> None:
        super().__init__()
        self.kind, self.joint = kind, joint
        if kind == "conv2d":
            self.op = nn.Conv2d(DIM, DIM, kernel_size=3, stride=1, padding=1)
        elif kind == "conv3d":
            self.op = nn.Conv3d(DIM, DIM, kernel_size=(3, 3, 3), stride=1, padding=1)
        elif kind == "conv3d_g6":
            self.op = nn.Conv3d(DIM, DIM, kernel_size=(3, 3, 3), stride=1, padding=1, groups=6)
        else:
            self.op = _Attention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, Hp, Wp, D = x.shape
        if self.kind == "conv2d":                          # [B*T, 96, 10, 16]
            y = self.op(x.reshape(B * T, Hp, Wp, D).permute(0, 3, 1, 2))
            return y.permute(0, 2, 3, 1).reshape(B, T, Hp, Wp, D)
        if self.kind in ("conv3d", "conv3d_g6"):          # [B, 96, T, 10, 16]
            return self.op(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        if self.joint:                                     # M = B, N = T*160, order t, h, w
            return self.op(x.reshape(B, T * Hp * Wp, D)).reshape(B, T, Hp, Wp, D)
        return self.op(x.reshape(B * T, Hp * Wp, D)).reshape(B, T, Hp, Wp, D)   # M = B*T, N = 160


class _Block(nn.Module):
    def __init__(self, kind: str, joint: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=LN_EPS)
        self.mixer = _Mixer(kind, joint)
        self.norm2 = nn.LayerNorm(DIM, eps=LN_EPS)
        self.ffn = nn.Sequential(nn.Linear(DIM, FFN), nn.GELU(approximate="none"), nn.Linear(FFN, DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x + self.mixer(self.norm1(x))
        return u + self.ffn(self.norm2(u))


def matched_amp_dtype() -> torch.dtype | None:
    """NETT_MATCHED_AMP: autocast dtype for the matched encoders' forward. DEFAULT OFF (fp32/TF32, the
    registered recipe). "bf16" runs the trunk under CUDA autocast and returns fp32 features, as NatureCNN
    does under NETT_AMP; owner-directed speed test (DECISIONS §103), NOT a recipe change until ruled.
    Anything else refuses, so a typo cannot silently run fp32."""
    v = os.environ.get("NETT_MATCHED_AMP", "off").strip().lower()
    if v in ("off", "0", "fp32", ""):
        return None
    if v == "bf16":
        return torch.bfloat16
    raise ValueError(f"NETT_MATCHED_AMP={v!r}; expected 'off' or 'bf16'")


class MatchedEncoder(HWCFeatureExtractor):
    """One of the owner's six matched encoders, chosen by ``variant`` (see the module docstring)."""

    OWNS_INIT = True

    def __init__(self, observation_space: gym.Space, variant: str, features_dim: int = OUT_DIM) -> None:
        # Explicit keywords only (no ``**_``): a cfg key this encoder does not read refuses at
        # construction instead of being dropped.
        if variant not in VARIANTS:
            raise ValueError(f"MatchedEncoder variant={variant!r}; expected one of {sorted(VARIANTS)}")
        if int(features_dim) != OUT_DIM:
            raise ValueError(f"MatchedEncoder outputs {OUT_DIM} features by spec; features_dim={features_dim}")
        super().__init__(observation_space, OUT_DIM)
        frames, kind, joint, temporal = VARIANTS[variant]
        total_channels, height, width = image_channels_hw(observation_space)
        if (height, width) != EYE_HW:
            raise ValueError(
                f"MatchedEncoder[{variant}] is specified at H x W = {EYE_HW[0]} x {EYE_HW[1]} (10 x 16 patches "
                f"of 16; the readout closes to 1 x 5 only there); got {height} x {width}. Set NETT_EYE_RES=256x160.")
        self.variant, self.num_frames = variant, validate_framestack_depth(
            total_channels, frames, f"MatchedEncoder[{variant}]", channels_per_frame=3)
        self.base_channels = 3
        self.patch = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH, padding=0)
        self.pos_spatial = nn.Parameter(torch.empty(1, 1, *GRID, DIM))
        nn.init.normal_(self.pos_spatial, mean=0.0, std=0.02)
        if temporal:
            self.pos_temporal = nn.Parameter(torch.empty(1, frames, 1, 1, DIM))
            nn.init.normal_(self.pos_temporal, mean=0.0, std=0.02)
        else:
            self.register_parameter("pos_temporal", None)
        self.blocks = nn.ModuleList(_Block(kind, joint) for _ in range(BLOCKS))
        self.norm = nn.LayerNorm(DIM, eps=LN_EPS)
        d = OUT_DIM // frames
        self.readout = nn.Sequential(
            nn.Conv2d(DIM, 32, kernel_size=(4, 4), stride=(2, 2), padding=0), nn.ReLU(),   # [32, 4, 7]
            nn.Conv2d(32, 32, kernel_size=(4, 3), stride=1, padding=0), nn.ReLU(),         # [32, 1, 5]
            nn.Flatten(start_dim=1),                                                        # 160, C-major
            nn.Linear(160, d),
        )
        self.out_norm = nn.LayerNorm(OUT_DIM, eps=LN_EPS)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._prepare_image(observations)              # (B, 3T, 160, 256), [prev, cur], /255 once
        amp = matched_amp_dtype()
        if amp is not None and x.is_cuda:
            with torch.autocast("cuda", dtype=amp):
                out = self._trunk(x)
            return out.float()
        return self._trunk(x)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[0], self.num_frames
        f = self.patch(x.reshape(B * T, 3, *EYE_HW))       # same weights on every frame -> [B*T, 96, 10, 16]
        x = f.permute(0, 2, 3, 1).reshape(B, T, *GRID, DIM) + self.pos_spatial
        if self.pos_temporal is not None:
            x = x + self.pos_temporal
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        z = self.readout(x.reshape(B * T, *GRID, DIM).permute(0, 3, 1, 2))   # [B*T, D], shared per frame
        return self.out_norm(z.reshape(B, T * z.shape[1]))                   # [prev | cur] -> [B, 128]
