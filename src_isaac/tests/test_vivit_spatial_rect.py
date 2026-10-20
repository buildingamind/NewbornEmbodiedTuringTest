"""Owner, workspace DECISIONS §173: ViViT (joint fusion) + spatial pooling at a 1x4 and a 2x8 grid.

CompactViViT's spatial head now takes an int G or a (rows, cols) pair, exactly as CompactViT does, and
pools with DeterministicAvgPool2d (refusing a non-dividing grid AT CONSTRUCTION) instead of
F.adaptive_avg_pool2d, whose CUDA backward is nondeterministic under the PPO update's
use_deterministic_algorithms(True). On a dividing grid the two are the same windows, so the values must
not move. Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.compact_vivit import CompactViViT
from nett_skrl.brain.registry import encoder_mapping

_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
LABELS = ("ViViT-SP1x4", "ViViT-SP2x8")
# a small trunk for the structural tests (the label tests build the real VIVIT_CFG)
_SMALL = dict(features_dim=32, patch_size=16, embed_dim=24, depth=1, num_heads=3, num_frames=2,
              pool="spatial", spatial_reduce_dim=4)


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _space(h, w, channels=6):
    return gym.spaces.Box(0, 255, (channels, h, w), np.uint8)


def _build(spec, h, w, channels=6, seed=0):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(h, w, channels), **kwargs)


def _n(module):
    return sum(p.numel() for p in module.parameters())


class _AdaptivePool(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, g):
        return F.adaptive_avg_pool2d(g, self.size)


def _adaptive_reference(enc, x):
    """The pre-§173 head: identical modules, F.adaptive_avg_pool2d in place of the deterministic pool."""
    pool = enc.spatial_pool
    enc.spatial_pool = _AdaptivePool(enc.spatial_grid)
    try:
        return enc(x)
    finally:
        enc.spatial_pool = pool


# ---------------------------------------------------------------- labels

def test_labels_are_plain_vivit_plus_the_spatial_head_only(campaign):
    base = campaign.MODELS["ViViT"]
    assert base["cfg"]["temporal_mode"] == "joint" and base["cfg"]["pool"] == "cls"
    for label, grid in zip(LABELS, ((1, 4), (2, 8))):
        spec = campaign.MODELS[label]
        assert spec["cfg"] == {**base["cfg"], "pool": "spatial", "spatial_grid": grid, "spatial_reduce_dim": 16}
        assert {k: v for k, v in spec.items() if k != "cfg"} == {k: v for k, v in base.items() if k != "cfg"}
        assert "aux" not in spec and spec["framestack"] is True
    # the default does NOT change (§173: test before any default changes)
    assert campaign.VIVIT_CFG["pool"] == "cls"
    params = inspect.signature(CompactViViT).parameters
    assert params["pool"].default == "cls" and params["spatial_grid"].default == 4


def test_label_parameter_counts(campaign):
    M = campaign.MODELS
    nf = campaign._FRAMESTACK_N
    assert nf == 2, "counts below are 2-frame; NETT_FRAMESTACK_N must be unset for this test"
    ch = 3 * nf
    assert _n(_build(M["ViViT"], 80, 128, ch)) == 694_016
    assert _n(_build(M["ViViT-SP1x4"], 80, 128, ch)) == 655_088
    assert _n(_build(M["ViViT"], 160, 256, ch)) == 711_296
    assert _n(_build(M["ViViT-SP1x4"], 160, 256, ch)) == 672_368
    assert _n(_build(M["ViViT-SP2x8"], 160, 256, ch)) == 770_672


def test_divisibility_by_eye(campaign):
    M = campaign.MODELS
    # 128x80 -> 5x8 tokens: 1x4 builds, 2x8 refuses (5 % 2) naming the valid divisors
    assert _build(M["ViViT-SP1x4"], 80, 128).spatial_grid == (1, 4)
    with pytest.raises(ValueError, match=r"5x8 token grid.*Valid rows \[1, 5\], valid cols \[1, 2, 4, 8\]"):
        _build(M["ViViT-SP2x8"], 80, 128)
    # 256x160 -> 10x16: both build
    for label in LABELS:
        enc = _build(M[label], 160, 256)
        assert (enc.n_h, enc.n_w) == (10, 16)
    # 448x280: 280 % 16 = 8, so no patch-16 ViViT builds there at all, before any grid is considered
    for label in ("ViViT", *LABELS):
        with pytest.raises(AssertionError):
            _build(M[label], 280, 448)


# ---------------------------------------------------------------- grid parsing / refusal

@pytest.mark.parametrize("grid, expect", [(4, (4, 4)), (2, (2, 2)), ((1, 4), (1, 4)), ([2, 8], (2, 8)),
                                          ((8, 1), (8, 1))])
def test_int_and_pair_grids(grid, expect):
    enc = CompactViViT(_space(128, 128), spatial_grid=grid, **_SMALL)       # 8x8 tokens
    assert enc.spatial_grid == expect
    assert enc.spatial_pool.output_size == expect
    assert enc.head.in_features == expect[0] * expect[1] * 4


@pytest.mark.parametrize("mode", ["joint", "early", "factored", "late"])
@pytest.mark.parametrize("hw, grid", [((80, 128), (2, 8)), ((80, 128), 4), ((160, 256), 4), ((160, 256), (4, 4)),
                                      ((160, 256), (2, 3))])
def test_non_dividing_grid_refused_at_construction(mode, hw, grid):
    with pytest.raises(ValueError, match="does not divide.*nondeterministic"):
        CompactViViT(_space(*hw), spatial_grid=grid, temporal_mode=mode, **_SMALL)


def test_head_input_is_rows_cols_reduce(campaign):
    for label, (rows, cols) in zip(LABELS, ((1, 4), (2, 8))):
        enc = _build(campaign.MODELS[label], 160, 256)
        assert enc.head.in_features == rows * cols * 16
        assert enc.head.out_features == 512


def test_spatial_head_no_longer_calls_adaptive_pooling():
    src = (Path(__file__).resolve().parents[1] / "nett_skrl" / "brain" / "encoders" / "compact_vivit.py").read_text()
    body = src[src.index("def _spatial_head"):src.index("def _run_blocks")]
    assert "adaptive_avg_pool2d" not in body


# ---------------------------------------------------------------- value equality with the old head

@pytest.mark.parametrize("mode", ["joint", "early", "factored", "late"])
@pytest.mark.parametrize("hw, grid", [((128, 128), 4), ((128, 128), (2, 8)), ((80, 128), (1, 4)),
                                      ((80, 128), (5, 4)), ((160, 256), (2, 8)), ((160, 256), (1, 4))])
def test_dividing_grid_equals_adaptive_pooling(mode, hw, grid):
    torch.manual_seed(0)
    enc = CompactViViT(_space(*hw), spatial_grid=grid, temporal_mode=mode, **_SMALL).eval()
    x = torch.randint(0, 256, (2, 6, *hw), dtype=torch.uint8).float()
    with torch.no_grad():
        # bit-identical on CPU (avg_pool2d visits the same windows as adaptive pooling on a dividing grid)
        assert torch.equal(enc(x), _adaptive_reference(enc, x))


def test_label_equals_adaptive_pooling_at_256x160(campaign):
    for label in LABELS:
        enc = _build(campaign.MODELS[label], 160, 256).eval()
        x = torch.randint(0, 256, (2, 6, 160, 256), dtype=torch.uint8).float()
        with torch.no_grad():
            assert torch.equal(enc(x), _adaptive_reference(enc, x)), label


# ---------------------------------------------------------------- strict determinism, forward + backward

def test_forward_backward_under_deterministic_algorithms(campaign):
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for label in LABELS:
            grads = []
            for _ in range(2):
                enc = _build(campaign.MODELS[label], 160, 256, seed=3)
                x = torch.randint(0, 256, (2, 6, 160, 256), dtype=torch.uint8,
                                  generator=torch.Generator().manual_seed(5)).float()
                out = enc(x)
                assert out.shape == (2, 512) and torch.isfinite(out).all()
                out.square().mean().backward()
                g = torch.cat([p.grad.flatten() for p in enc.parameters() if p.grad is not None])
                assert torch.isfinite(g).all() and g.abs().sum() > 0
                assert enc.reduce.weight.grad is not None and enc.head.weight.grad is not None
                grads.append(g)
            assert torch.equal(grads[0], grads[1]), label
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
