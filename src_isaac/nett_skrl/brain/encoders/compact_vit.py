"""Compact Vision Transformer (ViT) encoder (~313 K encoder parameters).

Implements a small ViT-B/8 variant following Dosovitskiy et al. (2021)
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".
Uses patch_size=8 on 64×64 inputs → 64 patches + 1 CLS token (65 tokens
total). Attention and MLP dimensions are reduced to stay under 600 K total
model parameters with PPO heads.

Architecture:
    Patch embed   : Conv2d(C, embed_dim, patch_size, patch_size)
    Position embed: learnable, shape (1, num_patches+1, embed_dim)
    Transformer   : depth × (LayerNorm + MHSA + LayerNorm + FFN)
    Head          : LayerNorm on CLS token → Linear → features_dim
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block (MHSA + FFN, residual connections)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CompactViT(HWCFeatureExtractor):
    """Compact ViT encoder for NETT visual RL.

    Parameter budget (64×64 RGB, embed_dim=128, depth=2, features_dim=128):
        Patch embed  :  ~25 K
        Pos embed    :   ~8 K
        2 Transformer: ~264 K
        Head         :  ~16 K
        Total encoder: ~313 K
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        **_,
    ) -> None:
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)

        assert height % patch_size == 0 and width % patch_size == 0, (
            f"Image size ({height}×{width}) must be divisible by patch_size={patch_size}"
        )
        num_patches = (height // patch_size) * (width // patch_size)

        # Patch embedding via a strided convolution (equivalent to linear projection of patches)
        self.patch_embed = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # Project from transformer dim to features_dim (identity if equal)
        self.head = nn.Linear(embed_dim, features_dim) if embed_dim != features_dim else nn.Identity()

        # Weight init following ViT paper
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._prepare_image(observations)   # (B, C, H, W)
        B = x.shape[0]

        # Patch embed → (B, embed_dim, nH, nW) → (B, N, embed_dim)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend CLS token and add position embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)[:, 0]  # CLS token
        return self.head(x)
