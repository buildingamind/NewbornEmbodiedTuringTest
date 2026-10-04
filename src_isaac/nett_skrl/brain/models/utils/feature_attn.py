"""Feature-token attention trunk: self-attention ACROSS the coordinates of the encoder's feature vector.

★ WHAT IT IS (owner request, 2026-10-04). The encoder emits one feature vector f (B, F) -- for the
ViT family the post-LayerNorm CLS token projected to ``features_dim``. The default trunk reads it
with an MLP (``mlp_trunk``; ``hidden_sizes=[]`` is the campaign's linear head). This trunk instead
treats each coordinate (or each group of ``group`` adjacent coordinates) as a TOKEN and runs
transformer blocks over the F/group tokens, so the head can form an attention matrix between
feature positions (FT-Transformer, Gorishniy et al. 2021, "Revisiting Deep Learning Models for
Tabular Data").

    tokenize : t_i = f[i*g:(i+1)*g] @ W_i + b_i          W_i (g, dim), b_i (dim), one pair PER POSITION
    blocks   : ``blocks`` x pre-norm (MHSA + FFN), the SAME ``_TransformerBlock`` the ViT encoder uses
    readout  : LayerNorm, mean over tokens -> (B, dim)   (``last`` = dim; the actor/critic Linear follows)

⚠ A lone scalar cannot form a useful query or key with a SHARED projection (q_i would be x_i times
one vector, so every token would point the same way). The per-position W_i, b_i are what make
position i distinguishable; b_i IS the position embedding, so there is no separate one.

⚠ The coordinates are LEARNED FEATURE AXES, not image locations. This tests pairwise interaction
between features, not spatial attention.

⚠ MEMORY. The attention matrix is (B, heads, N, N) per block per model (policy and value each own a
trunk; features_forward caches only the encoder). At N=512, heads=4, B=384 that is ~1.6 GB per
block per model in fp32, before autograd's saved copies. The attention call is the encoder's own
(``nn.MultiheadAttention`` with its default ``need_weights``), the path already proven under the
PPO update's ``use_deterministic_algorithms(True)``. ``group`` > 1 cuts N by that factor.

★ INIT. This module initialises itself, ViT-style: every Linear trunc_normal(0.02) with zero bias,
LayerNorm (1, 0), nn.MultiheadAttention's own in_proj init kept (xavier), the tokenizer W_i
normal(0, 1/sqrt(g)) so a unit-scale coordinate gives a unit-scale token, b_i trunc_normal(0.02).
``FeatureBackbone`` does NOT apply its orthogonal pass to this trunk (it would re-init out_proj and
the FFN but not in_proj_weight, the asymmetry compact_vit.py records for its own blocks).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from ...encoders.compact_vit import _TransformerBlock

# Keys a feature_attn cfg may carry, with their defaults. Anything else refuses.
FEATURE_ATTN_DEFAULTS: dict[str, Any] = {"dim": 64, "heads": 4, "blocks": 1, "group": 1, "mlp_ratio": 2.0}


def resolve_feature_attn(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and validate a feature_attn cfg. Raises on an unknown key or a bad value."""
    unknown = sorted(set(cfg) - set(FEATURE_ATTN_DEFAULTS))
    if unknown:
        raise ValueError(f"feature_attn: unknown key(s) {unknown}; allowed {sorted(FEATURE_ATTN_DEFAULTS)}")
    out = {**FEATURE_ATTN_DEFAULTS, **cfg}
    for k in ("dim", "heads", "blocks", "group"):
        v = out[k]
        if isinstance(v, bool) or int(v) != v or int(v) < 1:
            raise ValueError(f"feature_attn: {k} must be a positive integer, got {v!r}")
        out[k] = int(v)
    if not float(out["mlp_ratio"]) > 0:
        raise ValueError(f"feature_attn: mlp_ratio must be > 0, got {out['mlp_ratio']!r}")
    out["mlp_ratio"] = float(out["mlp_ratio"])
    if out["dim"] % out["heads"]:
        raise ValueError(f"feature_attn: dim={out['dim']} is not divisible by heads={out['heads']}")
    return out


class FeatureTokenAttention(nn.Module):
    """(B, F) -> (B, dim): transformer blocks over F/group per-position-embedded feature tokens."""

    def __init__(self, in_dim: int, dim: int, heads: int, blocks: int, group: int, mlp_ratio: float) -> None:
        super().__init__()
        if in_dim % group:
            raise ValueError(f"feature_attn: features_dim={in_dim} is not divisible by group={group}")
        self.in_dim, self.dim, self.group = int(in_dim), int(dim), int(group)
        self.num_tokens = self.in_dim // self.group
        self.tok_weight = nn.Parameter(torch.empty(self.num_tokens, self.group, self.dim))
        self.tok_bias = nn.Parameter(torch.empty(self.num_tokens, self.dim))
        self.blocks = nn.ModuleList(
            _TransformerBlock(self.dim, heads, mlp_ratio=mlp_ratio) for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(self.dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.tok_weight, std=1.0 / math.sqrt(self.group))
        nn.init.trunc_normal_(self.tok_bias, std=0.02)

    def tokens(self, f: torch.Tensor) -> torch.Tensor:
        """(B, F) -> (B, N, dim) input tokens, before any block."""
        x = f.reshape(f.shape[0], self.num_tokens, self.group)
        return torch.einsum("bng,ngd->bnd", x, self.tok_weight) + self.tok_bias

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = self.tokens(f)
        for block in self.blocks:
            x = block(x)
        return self.norm(x).mean(dim=1)


def feature_attn_trunk(in_dim: int, cfg: dict[str, Any]) -> tuple[FeatureTokenAttention, int]:
    """Same contract as ``mlp_trunk``: returns (module, output width)."""
    c = resolve_feature_attn(cfg)
    module = FeatureTokenAttention(in_dim, c["dim"], c["heads"], c["blocks"], c["group"], c["mlp_ratio"])
    return module, module.dim
