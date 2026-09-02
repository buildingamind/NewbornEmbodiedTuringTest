"""Convolution-free ViViT encoder for 2-frame video observations.

A pure-transformer Video Vision Transformer following Arnab et al. (2021)
"ViViT: A Video Vision Transformer". There are NO convolutions anywhere: patch
embedding is done by unfolding each frame into non-overlapping P×P patches and
projecting the flattened patch vectors with ``nn.Linear``. Self-attention is
performed with the shared pre-norm ``_TransformerBlock``.

Expects a 2-frame FrameStack body wrapper so observations arrive as
(C*T, H, W) = (6, H, W) in CHW format. ``_prepare_image`` yields (B, C*T, H, W).

Frame-fusion sweep (``temporal_mode``)
--------------------------------------
The user's exploration axis is "how multiple frames are encoded". Four
convolution-free strategies are provided (C=3 RGB, T=2 frames,
n_spatial=(H/P)*(W/P)):

    "joint"    Per-frame linear patch tokens (each patch -> Linear(3*P*P, D)).
               T*n_spatial tokens, factored spatial + temporal pos embeds,
               full joint space-time self-attention (ViViT Model 1).
    "early"    Early fusion: at each spatial location concatenate both frames'
               flattened patches (2*3*P*P) and project with one Linear ->
               n_spatial tokens. Spatial pos embed only.
    "factored" Per-frame tokens (like joint) but factored attention: each block
               does spatial self-attention within a frame, then temporal
               self-attention across frames at each location (ViViT Model 3).
    "late"     Each frame encoded independently through the SHARED transformer
               stack (per-frame patch tokens + spatial pos + optional per-frame
               CLS); the two per-frame pooled embeddings are averaged.

Pooling (``pool``)
------------------
    "cls"     prepend a learned CLS token, output = norm(x)[:, 0] then head.
    "mean"    no CLS token, output = norm(x).mean(dim=1) then head.
    "spatial" no CLS token; KEEP the per-patch tokens as a coarse spatial
              feature map instead of collapsing them. The patch tokens are
              resolved to an (n_h x n_w x D) grid (meaned over frames for the
              multi-frame temporal modes), channel-reduced with a Linear, then
              adaptive-avg-pooled (NOT a convolution) to a GxG grid, flattened
              and projected to features_dim. This preserves WHERE information
              for the downstream policy (like the CNN encoders' feature maps).

Defaults reproduce ViViT Model 1: ``temporal_mode="joint", pool="cls"``.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .utils.temporal import validate_framestack_depth
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


def _patchify(frame: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Unfold a (B, C, H, W) frame into flattened non-overlapping patches.

    Returns (B, n_spatial, C*P*P) where patches are ordered row-major
    (n_h, n_w) and each patch vector is laid out as (C, P, P) flattened.
    """
    B, C, H, W = frame.shape
    P = patch_size
    # (B, C, n_h, P, n_w, P)
    x = frame.reshape(B, C, H // P, P, W // P, P)
    # -> (B, n_h, n_w, C, P, P)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    # -> (B, n_spatial, C*P*P)
    return x.reshape(B, (H // P) * (W // P), C * P * P)


class CompactViViT(HWCFeatureExtractor):
    """Convolution-free ViViT encoder for paired-frame NETT observations.

    Linear patch embedding only — no Conv2d/Conv3d anywhere. See the module
    docstring for the four ``temporal_mode`` frame-fusion strategies and the
    two ``pool`` options.

    Args:
        pool: "cls" (prepend CLS token, output = norm(x)[:, 0]), "mean"
            (no CLS token, output = norm(x).mean(dim=1)), or "spatial"
            (no CLS token, keep a coarse GxG spatial feature map; see module
            docstring).
        temporal_mode: "joint", "early", "factored", or "late" (see module
            docstring). Default "joint" (ViViT Model 1).
        spatial_grid: GxG output grid size for ``pool="spatial"`` (default 4).
        spatial_reduce_dim: per-token channel width after the reduce Linear in
            ``pool="spatial"`` (default 16; chosen so all of joint/early/late
            stay under the 700k-param budget at G=4).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        patch_size: int = 16,
        embed_dim: int = 120,
        depth: int = 3,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        num_frames: int = 2,
        pool: str = "cls",
        temporal_mode: str = "joint",
        spatial_grid: int = 4,
        spatial_reduce_dim: int = 16,
        **_,
    ) -> None:
        super().__init__(observation_space, features_dim)
        total_channels, height, width = image_channels_hw(observation_space)
        self.num_frames = validate_framestack_depth(
            total_channels, num_frames, type(self).__name__)
        self.base_channels = total_channels // self.num_frames  # 3 for RGB
        self.total_channels = total_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        if pool not in ("cls", "mean", "spatial"):
            raise ValueError(
                f"pool must be 'cls', 'mean', or 'spatial', got {pool!r}"
            )
        if temporal_mode not in ("joint", "early", "factored", "late"):
            raise ValueError(
                f"temporal_mode must be 'joint', 'early', 'factored', or "
                f"'late', got {temporal_mode!r}"
            )
        self.pool = pool
        self.temporal_mode = temporal_mode
        # Spatial mode never prepends a CLS token (saves params; we keep the
        # per-patch grid intact for a coarse spatial feature map).
        self.use_cls = pool == "cls"
        self.use_spatial = pool == "spatial"

        assert height % patch_size == 0 and width % patch_size == 0
        n_h = height // patch_size
        n_w = width // patch_size
        self.n_h = n_h
        self.n_w = n_w
        self.n_spatial = n_h * n_w          # patches per frame

        C = self.base_channels
        P = patch_size
        patch_dim = C * P * P               # flattened single-frame patch

        # ---- Linear patch embedding --------------------------------------
        if temporal_mode == "early":
            # Concatenate both frames' patches at each location, one projection.
            self.patch_embed = nn.Linear(self.num_frames * patch_dim, embed_dim)
        else:
            # Per-frame patch projection (shared across frames).
            self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # ---- CLS token + factored position embeddings --------------------
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.n_spatial, embed_dim))
        # Temporal pos embed only used by per-frame-token modes (joint/factored).
        if temporal_mode in ("joint", "factored"):
            self.temporal_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_frames, embed_dim)
            )
        else:
            self.temporal_pos_embed = None

        self.blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        # A single lightweight temporal-axis attention block for "factored"
        # mode (shared across depth) keeps the parameter budget small while
        # still mixing information across the T frames at each location.
        if temporal_mode == "factored":
            self.temporal_block = _TransformerBlock(embed_dim, num_heads, mlp_ratio)
        else:
            self.temporal_block = None

        self.norm = nn.LayerNorm(embed_dim)
        if self.use_spatial:
            # Spatial feature-map head: reduce channels per token, adaptive
            # pool the (n_h x n_w) grid to GxG, then a Linear to features_dim.
            self.spatial_grid = int(spatial_grid)
            self.spatial_reduce_dim = int(spatial_reduce_dim)
            self.reduce = nn.Linear(embed_dim, self.spatial_reduce_dim)
            self.head = nn.Linear(
                self.spatial_grid * self.spatial_grid * self.spatial_reduce_dim,
                features_dim,
            )
        else:
            self.head = (
                nn.Linear(embed_dim, features_dim)
                if embed_dim != features_dim
                else nn.Identity()
            )

        # Init
        if self.use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        if self.temporal_pos_embed is not None:
            nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---- helpers ---------------------------------------------------------
    def _frames(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C*T, H, W) -> (B, T, C, H, W)."""
        B, CT, H, W = x.shape
        return x.view(B, self.num_frames, self.base_channels, H, W)

    def _pool_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Apply final norm + pooling to a token sequence (CLS already added)."""
        x = self.norm(x)
        if self.use_cls:
            return x[:, 0]
        return x.mean(dim=1)

    def _spatial_head(self, grid: torch.Tensor) -> torch.Tensor:
        """(B, n_h, n_w, D) grid -> (B, features_dim) spatial feature vector.

        Applies the final norm + per-token channel reduction, adaptive-avg-pools
        the (n_h x n_w) grid to GxG (pooling, not convolution), flattens and
        projects with the head.
        """
        grid = self.norm(grid)                      # (B, n_h, n_w, D)
        grid = self.reduce(grid)                    # (B, n_h, n_w, R)
        # -> (B, R, n_h, n_w) for adaptive pooling over the spatial grid.
        grid = grid.permute(0, 3, 1, 2)
        grid = torch.nn.functional.adaptive_avg_pool2d(
            grid, (self.spatial_grid, self.spatial_grid)
        )                                           # (B, R, G, G)
        flat = grid.reshape(grid.shape[0], -1)      # (B, G*G*R)
        return self.head(flat)

    def _run_blocks(self, tok: torch.Tensor) -> torch.Tensor:
        """Run the shared transformer stack over (B, N, D) tokens (no pool)."""
        for block in self.blocks:
            tok = block(tok)
        return tok

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._prepare_image(observations)  # (B, C*T, H, W)
        B = x.shape[0]
        T = self.num_frames

        if self.temporal_mode == "early":
            frames = self._frames(x)                       # (B, T, C, H, W)
            # Patchify each frame, concat along feature dim per location.
            patches = [
                _patchify(frames[:, t], self.patch_size) for t in range(T)
            ]                                              # T × (B, n_spatial, C*P*P)
            tok = torch.cat(patches, dim=-1)               # (B, n_spatial, T*C*P*P)
            tok = self.patch_embed(tok)                    # (B, n_spatial, D)
            tok = tok + self.spatial_pos_embed
            if self.use_spatial:
                tok = self._run_blocks(tok)                # (B, n_spatial, D)
                grid = tok.reshape(B, self.n_h, self.n_w, self.embed_dim)
                return self._spatial_head(grid)
            return self.head(self._encode_single(tok))

        if self.temporal_mode == "late":
            frames = self._frames(x)                       # (B, T, C, H, W)
            feats = []
            grids = []
            for t in range(T):
                tok = _patchify(frames[:, t], self.patch_size)  # (B, n_spatial, C*P*P)
                tok = self.patch_embed(tok)                # (B, n_spatial, D)
                tok = tok + self.spatial_pos_embed
                if self.use_spatial:
                    enc = self._run_blocks(tok)            # (B, n_spatial, D)
                    grids.append(
                        enc.reshape(B, self.n_h, self.n_w, self.embed_dim)
                    )
                else:
                    feats.append(self._encode_single(tok))  # (B, D)
            if self.use_spatial:
                grid = torch.stack(grids, dim=0).mean(dim=0)  # mean per-frame grid
                return self._spatial_head(grid)
            feat = torch.stack(feats, dim=0).mean(dim=0)   # average per-frame
            return self.head(feat)

        # joint / factored: per-frame patch tokens with factored pos embeds.
        frames = self._frames(x)                           # (B, T, C, H, W)
        per_frame = [
            self.patch_embed(_patchify(frames[:, t], self.patch_size))
            for t in range(T)
        ]                                                  # T × (B, n_spatial, D)
        tok = torch.stack(per_frame, dim=1)                # (B, T, n_spatial, D)
        tok = tok + self.spatial_pos_embed.unsqueeze(1)    # spatial pos
        tok = tok + self.temporal_pos_embed.unsqueeze(2)   # temporal pos

        if self.temporal_mode == "joint":
            seq = tok.reshape(B, T * self.n_spatial, self.embed_dim)
            if self.use_cls:
                cls = (self.cls_token + self.cls_pos_embed).expand(B, -1, -1)
                seq = torch.cat([cls, seq], dim=1)
            for block in self.blocks:
                seq = block(seq)
            if self.use_spatial:
                # No CLS in spatial mode: T*n_spatial tokens stay in order.
                grid = seq.reshape(
                    B, T, self.n_h, self.n_w, self.embed_dim
                ).mean(dim=1)                           # (B, n_h, n_w, D)
                return self._spatial_head(grid)
            return self.head(self._pool_seq(seq))

        # factored attention (ViViT Model 3).
        return self._encode_factored(tok)

    def _encode_single(self, tok: torch.Tensor) -> torch.Tensor:
        """Run the shared stack over (B, n_spatial, D) tokens -> (B, D) pooled."""
        B = tok.shape[0]
        if self.use_cls:
            cls = (self.cls_token + self.cls_pos_embed).expand(B, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
        for block in self.blocks:
            tok = block(tok)
        return self._pool_seq(tok)

    def _encode_factored(self, tok: torch.Tensor) -> torch.Tensor:
        """Factored space-then-time attention. tok: (B, T, n_spatial, D)."""
        B, T, N, D = tok.shape
        for sblock in self.blocks:
            # Spatial attention: tokens within each frame.
            s = tok.reshape(B * T, N, D)
            s = sblock(s)
            tok = s.reshape(B, T, N, D)
            # Temporal attention: across frames at each spatial location
            # (shared temporal block reused at every depth).
            tt = tok.permute(0, 2, 1, 3).reshape(B * N, T, D)
            tt = self.temporal_block(tt)
            tok = tt.reshape(B, N, T, D).permute(0, 2, 1, 3).contiguous()

        if self.use_spatial:
            # tok: (B, T, N, D); resolve N -> (n_h, n_w) and mean over T.
            grid = tok.reshape(B, T, self.n_h, self.n_w, D).mean(dim=1)
            return self._spatial_head(grid)

        seq = tok.reshape(B, T * N, D)
        if self.use_cls:
            cls = (self.cls_token + self.cls_pos_embed).expand(B, -1, -1)
            seq = torch.cat([cls, seq], dim=1)
        # The factored blocks already ran; just norm + pool the final tokens.
        return self.head(self._pool_seq(seq))
