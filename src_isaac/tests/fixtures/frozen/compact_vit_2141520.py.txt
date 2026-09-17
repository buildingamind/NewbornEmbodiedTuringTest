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
from .utils.pool import DeterministicAvgPool2d


class _TokenMixer(nn.Module):
    """Content-INDEPENDENT stand-in for self-attention, for the QK ablation.

    ★ WHY THIS SHAPE. The question is whether a ViT beats a CNN on binding because of
    *content-dependent routing* (QK) or merely because it has a global receptive field, a
    residual stream and LayerNorms. "Delete the attention" cannot answer that -- it removes
    all four at once. This keeps everything except the content-dependence:

        MHSA      : out = W_o ( softmax(QK^T/sqrt(d)) V )   <- weights depend on the INPUT
        uniform   : out = W_o ( mean_over_tokens( V ) )     <- weights are FIXED at 1/N
        mixer     : out = W_o ( M V ),  M learned (N x N)   <- weights LEARNED but input-blind

    `V` and `W_o` are kept in both modes, so the token-value pathway and the output
    projection are unchanged; only the mixing weights lose their dependence on the tokens.
    `mixer` is the stronger control of the two -- it can still learn *which* positions matter,
    just not condition that on what is at them.

    ⚠ PARAMETER COUNT IS NOT MATCHED HERE. Dropping Q and K removes 2*dim^2 + 2*dim per
    block, and `mixer` adds N^2. Match it at the CONFIG level by raising `embed_dim` until
    examples/count_params.py agrees with the ViT arm -- an ablation that also removes a third
    of the parameters proves nothing.

    ⚠ RESULT OF RECORD (SIDE_LOCK_INVESTIGATION.md Phase 12): at n=56, spatially pooled and
    parameter-matched, qk - mixer on binding is +0.053 at p = 0.163 -- NOT significant, and
    not a refutation either (~228 agents/arm would be needed). The one effect surviving
    Bonferroni runs the OTHER way: the ablation is BETTER at `1color`. Do not cite the n=28
    wave (+0.129, p = 0.021); it did not replicate.
    """

    def __init__(self, dim: int, num_tokens: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        if mode == "mixer":
            # Learned but input-independent token->token weights, row-softmaxed so the
            # mixing stays a convex combination exactly as attention's rows are.
            #
            # ★ INITIALISED DIAGONAL-DOMINANT, NOT ZERO (fixed 2026-08-01, v1 was zeros).
            # softmax(zeros) is EXACTLY uniform, which makes every output token identical --
            # i.e. the `uniform` lesion. v1 therefore started every mixer at the degenerate
            # solution and had to break a perfectly symmetric point to escape; over 2000
            # episodes it did not, and the arm failed its positive control (`rest` 0.531,
            # 24/28 agents below 0.75). Starting near the IDENTITY means "each token keeps
            # itself", a benign residual-like state that still lets the rows spread out.
            #
            # The coefficient is 2.0, chosen so the branch also lands near the QK branch's
            # INIT SCALE: at N=65 it gives a diagonal weight of 0.104 against uniform's
            # 1/65 = 0.015 (clearly asymmetric) at 1.4x the QK output RMS. Larger is more
            # asymmetric but overshoots the scale (eye*3.0 -> 2.5x); smaller collapses back
            # toward uniform (eye*0.5 -> diagonal 0.025, essentially the lesion).
            self.mix = nn.Parameter(
                torch.eye(num_tokens) * 2.0 + torch.randn(num_tokens, num_tokens) * 0.02)
        elif mode != "uniform":
            raise ValueError(f"_TokenMixer mode must be 'uniform' or 'mixer', got {mode!r}")

    def reset_parameters(self) -> None:
        """Match ``nn.MultiheadAttention``'s init scale.

        ★ CompactViT._init_weights applies ``trunc_normal(std=0.02)`` to every ``nn.Linear``.
        ``nn.MultiheadAttention`` is hit ASYMMETRICALLY by that pass: its ``out_proj`` is a
        Linear subclass and so gets trunc_normal, while ``in_proj_weight`` is a bare
        Parameter and keeps torch's ``xavier_uniform``. Left alone, this mixer got
        trunc_normal on BOTH and came out **3.2x weaker** than the QK branch at init
        (0.008 vs 0.026 output RMS); xavier on both overshoots to **4.6x**. Mirroring the
        asymmetry -- xavier on the value projection, trunc_normal on the output projection --
        is what actually matches.

        CompactViT calls this AFTER its global pass, so ``v`` is corrected here and ``proj``
        is deliberately left as the global pass set it.
        """
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.zeros_(self.v.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v(x)                                   # (B, N, D)
        if self.mode == "uniform":
            mixed = v.mean(dim=1, keepdim=True).expand_as(v)
        else:
            w = torch.softmax(self.mix, dim=-1)         # (N, N)
            mixed = torch.einsum("ij,bjd->bid", w, v)
        return self.proj(mixed)


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block (MHSA + FFN, residual connections).

    ``attn_mode`` selects the token-mixing operator: ``"qk"`` is the real thing; ``"uniform"``
    and ``"mixer"`` are the ablations described on :class:`_TokenMixer`.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0,
                 num_tokens: int | None = None, attn_mode: str = "qk") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attn_mode == "qk":
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        else:
            if num_tokens is None:
                raise ValueError("num_tokens is required for a non-'qk' attn_mode")
            self.attn = _TokenMixer(dim, num_tokens, attn_mode)
        self.attn_mode = attn_mode
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        if self.attn_mode == "qk":
            attn_out, _ = self.attn(normed, normed, normed)
        else:
            attn_out = self.attn(normed)
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
        pool: str = "cls",
        spatial_grid: int = 4,
        spatial_reduce_dim: int = 16,
        stem: str = "linear",
        attn_mode: str = "qk",
        **_,
    ) -> None:
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)

        assert height % patch_size == 0 and width % patch_size == 0, (
            f"Image size ({height}×{width}) must be divisible by patch_size={patch_size}"
        )
        self._n_h = height // patch_size
        self._n_w = width // patch_size
        num_patches = self._n_h * self._n_w
        self.pool = pool

        if stem == "conv":
            # Conv-STEM patch embed (Xiao et al. 2021, "Early Convolutions Help
            # Transformers See Better"): replace the single strided patch conv with
            # log2(patch_size) stride-2 3x3 convs. Adds CNN locality/optimization
            # stability so a PLAIN ViT (no contrastive aux) can train. Channel
            # schedule ends at embed_dim, keeping params ~= the linear patch embed.
            assert patch_size & (patch_size - 1) == 0, "conv stem needs power-of-2 patch_size"
            nl = int(math.log2(patch_size))
            chs = [channels] + [max(16, embed_dim // (2 ** (nl - 1 - i))) for i in range(nl)]
            layers: list[nn.Module] = []
            for i in range(nl):
                layers.append(nn.Conv2d(chs[i], chs[i + 1], kernel_size=3, stride=2, padding=1))
                if i < nl - 1:
                    layers.append(nn.GELU())
            self.patch_embed = nn.Sequential(*layers)
        else:
            # Patch embedding via a strided convolution (linear projection of patches)
            self.patch_embed = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # num_tokens is patches + the CLS token, matching what the blocks actually see.
        self.attn_mode = attn_mode
        self.blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, num_heads, mlp_ratio,
                               num_tokens=num_patches + 1, attn_mode=attn_mode)
             for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if pool == "spatial":
            # Preserve object LOCATION (which CLS pooling destroys — the cause of
            # the ViViT collapse): reduce each patch token, fold tokens back to the
            # patch grid, adaptive-pool to a small HxW grid, flatten -> Linear. Keeps
            # the spatial layout the policy needs to choose a turn direction.
            self.token_reduce = nn.Linear(embed_dim, spatial_reduce_dim)
            # ★ RECTANGULAR GRIDS ALLOWED (2026-08-12). `spatial_grid` may be an int (square)
            # or an (h, w) pair. The eye is NON-SQUARE and staying that way: at 128x80 with
            # patch 16 the token grid is 5x8, whose only common square divisor is 1 -- i.e. a
            # global average, which is exactly the CLS-pooling collapse this readout exists to
            # avoid. A pair keeps the spatial layout: valid choices divide 5 and 8 separately,
            # e.g. (5, 4) -> 20 cells, the closest analogue to the old square 4x4 = 16.
            g = (int(spatial_grid), int(spatial_grid)) if isinstance(spatial_grid, int) \
                else tuple(int(v) for v in spatial_grid)
            self._grid = g
            # ⚠ MUST NOT be F.adaptive_avg_pool2d. Its CUDA backward
            # (adaptive_avg_pool2d_backward_cuda) has no deterministic implementation, and
            # skrl_patches raises the PPO update to use_deterministic_algorithms(True,
            # warn_only=False) -- so a spatial-pooled ViT dies at the FIRST optimizer step:
            #   RuntimeError: adaptive_avg_pool2d_backward_cuda does not have a
            #   deterministic implementation
            # That killed all 8 processes of the first spatial ablation arm on 2026-08-03.
            # Every other encoder in this package already routes through
            # DeterministicAvgPool2d; this one was the sole holdout. On an evenly-divisible
            # grid (8x8 tokens -> 4x4, the configured case) it is the SAME arithmetic, so
            # this changes no value -- it only makes the backward deterministic.
            #
            # ⚠ FAIL AT CONSTRUCTION, NOT AT THE FIRST FORWARD. The token grid is
            # (_n_h, _n_w) = (height//patch, width//patch); a ragged pool would need the
            # non-deterministic adaptive kernel, so it is rejected here rather than blowing
            # up mid-training. The message names the divisors that actually work.
            if self._n_h % g[0] or self._n_w % g[1]:
                oh = [d for d in range(1, self._n_h + 1) if self._n_h % d == 0]
                ow = [d for d in range(1, self._n_w + 1) if self._n_w % d == 0]
                raise ValueError(
                    f"CompactViT(pool='spatial', spatial_grid={spatial_grid}) does not divide "
                    f"the {self._n_h}x{self._n_w} token grid from {height}x{width} at patch "
                    f"{patch_size}. Valid heights {oh}, valid widths {ow} -- pass a pair, e.g. "
                    f"spatial_grid=({oh[-1]}, {ow[-2] if len(ow) > 1 else ow[-1]}). Falling "
                    f"back to adaptive pooling is NOT an option: its CUDA backward is "
                    f"nondeterministic.")
            self.spatial_pool = DeterministicAvgPool2d(g)
            self.head = nn.Linear(spatial_reduce_dim * g[0] * g[1], features_dim)
        else:
            # Project from transformer dim to features_dim (identity if equal)
            self.head = nn.Linear(embed_dim, features_dim) if embed_dim != features_dim else nn.Identity()

        # Weight init following ViT paper
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        # ★ AFTER the global pass, restore the mixer's value projection to xavier so the
        # ablated branch matches nn.MultiheadAttention's init SCALE. The global pass above
        # hits every nn.Linear with trunc_normal(0.02), which MultiheadAttention only
        # receives on out_proj -- see _TokenMixer.reset_parameters. Left uncorrected the
        # mixer starts 3.2x weaker than QK, which confounded the first ablation arms.
        for m in self.modules():
            if isinstance(m, _TokenMixer):
                m.reset_parameters()

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

        x = self.norm(x)
        if self.pool == "spatial":
            tok = self.token_reduce(x[:, 1:])                 # (B, N, rdim), drop CLS
            r = tok.shape[-1]
            grid = tok.transpose(1, 2).reshape(B, r, self._n_h, self._n_w)
            grid = self.spatial_pool(grid)
            return self.head(grid.flatten(1))
        return self.head(x[:, 0])  # CLS token
