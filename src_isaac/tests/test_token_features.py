"""Token adapter for wave-17 aux losses: real tokens or a refusal, never a pooled substitute.

Three claims, each against real encoders:
  1. `spatial_tokens` returns the trunk's own tokens (CompactViT and UnityViT), gradients on.
  2. It REFUSES every encoder without a token path, rather than reshaping the pooled vector.
  3. The shared-encoder feature cache (2141520) neither serves the token path nor captures it.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.aux.token_features import spatial_tokens, tokens_as_map
from nett_skrl.brain.encoders.compact_vit import CompactViT
from nett_skrl.brain.encoders.nature_cnn import NatureCNN
from nett_skrl.brain.encoders.unity_vit import UnityViT
from nett_skrl.brain.models.utils.features import (
    _CACHE_ATTR, clear_feature_cache, features_forward, shared_feature_cache,
)

H, W = 80, 128
SMALL_VIT = dict(features_dim=32, patch_size=16, embed_dim=32, depth=1, num_heads=2)


@pytest.fixture(autouse=True)
def _seeded(monkeypatch):
    monkeypatch.delenv("NETT_DISABLE_FEATURE_CACHE", raising=False)
    monkeypatch.delenv("NETT_DECOUPLE_ENCODER", raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3)
            yield
    finally:
        torch.set_num_threads(threads)


def _space(c=6):
    return gym.spaces.Box(0, 255, shape=(H, W, c), dtype=np.uint8)


def _obs(c=6, b=2):
    return torch.randint(0, 256, (b, H, W, c), dtype=torch.uint8)


# ------------------------------------------------------------------ 1. the tokens are real

@pytest.mark.parametrize("pool", ["cls", "spatial"])
def test_compact_vit_tokens_match_the_encoder_hook_on_the_prepared_image(pool):
    extra = {"spatial_grid": (5, 4)} if pool == "spatial" else {}
    enc = CompactViT(_space(), pool=pool, **SMALL_VIT, **extra)
    x = _obs()
    with torch.no_grad():
        prepared = enc._prepare_image(x)
    tokens, grid = spatial_tokens(enc, prepared)
    assert grid == (5, 8)
    assert tokens.shape == (2, 40, 32)
    assert tokens.requires_grad
    ref, _ = enc.encode_tokens(x)
    assert torch.equal(tokens, ref)
    assert enc._skip_prepare is False, "the prepared-image flag must be restored"


def test_unity_vit_tokens_reproduce_its_forward_bitwise():
    """⛔ UnityViT's hook is a SECOND COPY of the trunk path (the verbatim SimpleViT block must
    not be edited), so it is pinned to `forward` bitwise, output and gradients both."""
    enc = UnityViT(_space(), features_dim=16, patch_size=16, depth=2, heads=2,
                   intermediate_size=32, hidden_size=32)
    x = _obs()
    with torch.no_grad():
        prepared = enc._prepare_image(x)
    tokens, grid = spatial_tokens(enc, prepared)
    assert grid == (5, 8) and tokens.shape == (2, 40, 32)
    via_tokens = enc.model.linear_head(tokens.mean(dim=1))
    via_forward = enc(x)
    assert torch.equal(via_tokens, via_forward)
    w = torch.linspace(-1, 1, via_forward.numel()).view_as(via_forward)
    (via_forward * w).sum().backward()
    g_fwd = {n: p.grad.clone() for n, p in enc.named_parameters()}
    enc.zero_grad(set_to_none=True)
    (via_tokens * w).sum().backward()
    for n, p in enc.named_parameters():
        assert torch.equal(p.grad, g_fwd[n]), n


def test_tokens_as_map_is_row_major_and_matches_the_spatial_readout_fold():
    enc = CompactViT(_space(), pool="spatial", spatial_grid=(5, 4), **SMALL_VIT)
    with torch.no_grad():
        prepared = enc._prepare_image(_obs())
        tokens, grid = spatial_tokens(enc, prepared)
        fmap = tokens_as_map(tokens, grid)
        assert fmap.shape == (2, 32, 5, 8)
        for i in range(5):
            for j in range(8):
                assert torch.equal(fmap[:, :, i, j], tokens[:, i * 8 + j])
        # The same fold CompactViT's own spatial readout applies, so a map consumer reads
        # positions exactly as the policy readout does.
        tok = enc.token_reduce(tokens)
        readout_grid = tok.transpose(1, 2).reshape(2, tok.shape[-1], 5, 8)
        assert torch.equal(readout_grid, tokens_as_map(tok, grid))
        assert torch.equal(enc.head(enc.spatial_pool(readout_grid).flatten(1)),
                           enc.encode_prepared(prepared))


def test_tokens_as_map_refuses_a_grid_that_does_not_fold():
    with pytest.raises(ValueError, match="do not fold"):
        tokens_as_map(torch.zeros(2, 41, 8), (5, 8))     # a CLS token left in


# ------------------------------------------------------------------ 2. refusal, never fallback

class _PooledOnly(nn.Module):
    features_dim = 8

    def encode_prepared(self, x):
        return x.flatten(1)[:, :8]


@pytest.mark.parametrize("make", [
    lambda: NatureCNN(_space(), features_dim=16),
    lambda: _PooledOnly(),
], ids=["NatureCNN", "pooled-only-module"])
def test_encoders_without_a_token_path_are_refused(make):
    enc = make()
    with pytest.raises(TypeError, match="refusing rather than substituting"):
        spatial_tokens(enc, torch.zeros(2, 6, H, W))


def test_a_malformed_token_return_is_refused():
    class _BadViT(CompactViT):
        def encode_tokens(self, observations):
            x = self._trunk(self._prepare_image(observations))
            return x, (self._n_h, self._n_w)          # CLS NOT dropped: N = 41

    enc = _BadViT(_space(), **SMALL_VIT)
    with pytest.raises(ValueError, match="CLS token left in"):
        spatial_tokens(enc, torch.zeros(2, 6, H, W))


# ------------------------------------------------------------------ 3. the feature cache

class _Head(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.encoder = enc
        self.trunk = nn.Linear(enc.features_dim, 4)


def _counting_vit():
    enc = CompactViT(_space(), **SMALL_VIT)
    enc.trunk_calls = 0

    def _hook(*_):
        enc.trunk_calls += 1
    enc.patch_embed.register_forward_hook(_hook)
    return enc


def _live_tensor_attrs(module):
    return [k for k, v in vars(module).items()
            if torch.is_tensor(v) and v.requires_grad
            or isinstance(v, tuple) and any(torch.is_tensor(t) and t.requires_grad for t in v)]


def test_token_path_is_not_served_from_the_pooled_cache():
    """Actor populates the cache; a token call on THE SAME tensor must still run the trunk and
    must leave the cached entry exactly as it was (neither read as tokens nor overwritten)."""
    enc = _counting_vit()
    actor = _Head(enc)
    x = _obs()
    with shared_feature_cache(enc):
        pooled = features_forward(actor, {"observations": x})
        entry = getattr(enc, _CACHE_ATTR)
        assert entry is not None and entry[0] is x and enc.trunk_calls == 1
        with torch.no_grad():
            prepared = enc._prepare_image(x)
        tokens, _ = spatial_tokens(enc, prepared)
        assert enc.trunk_calls == 2, "token path must run the trunk, not hit the cache"
        assert getattr(enc, _CACHE_ATTR) is entry, "token path must not touch the cache entry"
        assert tokens.shape[1] == 40 and pooled.shape[-1] == 4
        # And the RL path still hits after a token call: the token call did not invalidate it.
        features_forward(actor, {"observations": x})
        assert enc.trunk_calls == 2


def test_token_path_never_populates_the_cache_and_pins_nothing():
    enc = _counting_vit()
    x = _obs()
    with shared_feature_cache(enc):
        with torch.no_grad():
            prepared = enc._prepare_image(x)
        tokens, _ = spatial_tokens(enc, prepared)
        assert getattr(enc, _CACHE_ATTR, None) is None
        # Even keyed on the prepared tensor, a later RL call must MISS and run the trunk.
        features_forward(_Head(enc), {"observations": prepared})
        assert enc.trunk_calls == 2
    clear_feature_cache(enc)
    del tokens
    # ⛔ No graph-owning tensor may survive on the module: a token tensor pinned here would
    # re-pin the very memory the cache's clear exists to free.
    assert _live_tensor_attrs(enc) == []
