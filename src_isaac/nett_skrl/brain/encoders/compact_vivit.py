"""Compact ViViT encoder for 2-frame video observations (~252 K parameters).

Implements ViViT Model 1 with factored positional embeddings following
Arnab et al. (2021) "ViViT: A Video Vision Transformer". Each of the T=2
input frames is split into P×P patches, producing T × (H/P) × (W/P) tokens
total. Spatial and temporal position embeddings are added separately and then
summed (factored positional encoding) to keep the embedding table small.

Expects a 2-frame FrameStack body wrapper so observations arrive as
(C*T, H, W) = (6, 64, 64) in CHW format. The encoder internally reshapes to
(B, C, T, H, W) before patch extraction via Conv3d.

Parameter budget (64×64 RGB 2-frame, embed_dim=96, depth=3, features_dim=96):
    Tubelet embed   :  ~18 K
    Pos embeddings  :  ~12 K
    3 × Transformer : ~221 K
    Total encoder   : ~252 K
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block shared by ViT and ViViT encoders."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
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


class CompactViViT(HWCFeatureExtractor):
    """Compact ViViT encoder for paired-frame NETT observations.

    Uses tubelet embedding (Conv3d with temporal kernel = 1) so each frame is
    processed independently into spatial patches. The T=2 frames produce
    2 × (H/P)² = 128 spatiotemporal tokens (for 64×64, P=8). A learnable CLS
    token is prepended, and factored spatial + temporal position embeddings are
    added before full self-attention across all tokens.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 96,
        patch_size: int = 8,
        embed_dim: int = 96,
        depth: int = 3,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        num_frames: int = 2,
        **_,
    ) -> None:
        super().__init__(observation_space, features_dim)
        total_channels, height, width = image_channels_hw(observation_space)
        self.num_frames = int(num_frames)
        self.base_channels = total_channels // self.num_frames  # 3 for RGB
        self.patch_size = patch_size

        assert height % patch_size == 0 and width % patch_size == 0
        n_h = height // patch_size
        n_w = width // patch_size
        self.n_spatial = n_h * n_w          # patches per frame
        self.n_tokens = self.num_frames * self.n_spatial  # total spatiotemporal tokens

        # Tubelet embedding: one frame at a time (temporal kernel = 1)
        self.patch_embed = nn.Conv3d(
            self.base_channels, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        # CLS token + factored position embeddings (Arnab et al., Eq. 3)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.n_spatial, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frames, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, features_dim) if embed_dim != features_dim else nn.Identity()

        # Init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._prepare_image(observations)  # (B, C*T, H, W)
        B, CT, H, W = x.shape
        C = self.base_channels
        T = self.num_frames

        # Reshape to volume → (B, C, T, H, W)
        x = x.view(B, C, T, H, W)

        # Tubelet embed → (B, embed_dim, T, n_h, n_w)
        x = self.patch_embed(x)
        _, D, T_out, n_h, n_w = x.shape

        # Flatten spatial dims → (B, T, n_h*n_w, D) → (B, T, n_spatial, D)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T_out, n_h * n_w, D)

        # Factored positional encoding: spatial + temporal (broadcast)
        x = x + self.spatial_pos_embed.unsqueeze(1)    # (B, T, n_spatial, D)
        x = x + self.temporal_pos_embed.unsqueeze(2)   # (B, T, 1, D) → broadcast

        # Flatten to sequence → (B, T*n_spatial, D)
        x = x.reshape(B, T_out * n_h * n_w, D)

        # Prepend CLS token
        cls = (self.cls_token + self.cls_pos_embed).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + T*n_spatial, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)[:, 0]  # CLS token output
        return self.head(x)
