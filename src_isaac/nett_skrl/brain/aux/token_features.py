"""Spatial TOKENS from a ViT trunk, for auxiliary losses that work over patches, not the readout.

Wave 17 (notes/researcher/wave17-object-centric.md, piece P0) puts CLTT, a VideoSAUR-style
affinity target, an ego-motion residual and slots on the ViT's PATCH tokens. Every one of those
methods is defined over spatial positions; each would still run -- and still report a falling
loss -- on a pooled vector reshaped into one "token". So this module REFUSES encoders without a
genuine token path instead of substituting anything, mirroring `spatial_features` in
slot_contrast_aux.py.

⭐ THE FEATURE CACHE (commit 2141520) CANNOT SERVE OR CAPTURE THESE TOKENS, BY CONSTRUCTION.
`models/utils/features.features_forward` is the only writer and the only reader of
`_nett_shared_feature_cache`, and it stores the encoder's POOLED OUTPUT keyed on the input
tensor's identity. `spatial_tokens` calls `encode_tokens_prepared`, which runs the trunk afresh
and never touches that attribute, so (a) a token call is never answered with a cached pooled
vector, and (b) no token tensor -- which owns a graph as large as the pooled one's -- is left
pinned on the module past the step. tests/test_token_features.py asserts both, inside a live
`shared_feature_cache()`. ⚠ The cost is one extra trunk forward per token call; if a later loss
wants to share the RL minibatch's trunk pass it needs its own cache with its own release, not a
key in this one.

⛔ OBJECTIVE CHANGE 2026-09-17 (owner, workspace DECISIONS): `cltt_ref` now excludes each anchor's OWN FRAME from its negatives, so every cltt_ref arm trained before this commit ran a different objective and is NOT comparable to one trained after it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..encoders.base import NETTFeatureExtractor


def _has_token_path(encoder: nn.Module) -> bool:
    impl = getattr(type(encoder), "encode_tokens", None)
    return impl is not None and impl is not NETTFeatureExtractor.encode_tokens


def spatial_tokens(encoder: nn.Module, prepared: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """(B, N, D) post-trunk spatial tokens and their grid (n_h, n_w), from a PREPARED image.

    ``prepared`` is the normalized (B, C*T, H, W) float tensor that `_prepare_image` returns --
    what every memory-drawing aux already builds under ``no_grad`` -- and the trunk runs with
    gradients ON, exactly as `encode_prepared` does for the pooled path.

    ⛔ REFUSES rather than falls back. Supported: CompactViT (CLS dropped, post final norm) and
    UnityViT (SimpleViT has no CLS; post transformer norm). Anything else raises `TypeError`
    naming the encoder, including every CNN: a conv map IS spatial, but `spatial_features`
    already serves it, and silently accepting one here would let a "token" row run on a
    different trunk family than its label says.
    """
    if not _has_token_path(encoder):
        raise TypeError(
            f"{type(encoder).__name__} implements no `encode_tokens`, so there are no spatial "
            f"tokens to return. A token-level objective over the POOLED vector is a different "
            f"method; refusing rather than substituting one. Token paths exist on CompactViT "
            f"and UnityViT."
        )
    tokens, grid = encoder.encode_tokens_prepared(prepared)
    n_h, n_w = (int(g) for g in grid)
    if tokens.dim() != 3 or tokens.shape[1] != n_h * n_w:
        raise ValueError(
            f"{type(encoder).__name__}.encode_tokens returned {tuple(tokens.shape)} with grid "
            f"{(n_h, n_w)}; expected (B, {n_h * n_w}, D). A CLS token left in, or a grid from "
            f"the wrong patch size, would misplace every token in `tokens_as_map`."
        )
    return tokens, (n_h, n_w)


def tokens_as_map(tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """(B, N, D) row-major tokens -> (B, D, n_h, n_w), so code written for conv maps can read them.

    Row-major is the order both supported encoders produce (a Conv2d/Rearrange patch embed
    flattened over (h, w)); this is the same fold CompactViT's spatial readout performs.
    """
    n_h, n_w = (int(g) for g in grid)
    if tokens.dim() != 3 or tokens.shape[1] != n_h * n_w:
        raise ValueError(
            f"tokens {tuple(tokens.shape)} do not fold onto grid {(n_h, n_w)}: need N == "
            f"{n_h * n_w}."
        )
    b, _, d = tokens.shape
    return tokens.transpose(1, 2).reshape(b, d, n_h, n_w)
