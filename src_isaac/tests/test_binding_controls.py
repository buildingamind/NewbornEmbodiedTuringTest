"""Binding controls (owner 2026-11-02): CNN+GAP (NatureCNN, explicit (1, 3) pool grid) and the Static-Mixer ViT (ViT-Mixer).

Pinned at the 128x80 eye under NETT_INIT_SCHEME=orthogonal (the S3 recipe):
  * CNN+GAP is MODELS["CNN"] plus cfg pool_grid (1, 3) and nothing else; its pool output is the mean of the conv map
    over all rows and each third of the columns; the head input is 512 like every arm; encoder counts for CNN+GAP and CNN.
  * pool_grid refusals: ragged grid, malformed grid, pool_grid without spatial_pool.
  * pool_grid=None leaves NatureCNN's measured grid (3, 6) and its parameter count unchanged.
  * ViT-Mixer at 128x80 mixes 41 tokens with the diagonal-dominant init intact after the fleet's orthogonal init.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.brain.encoders.compact_vit import CompactViT  # noqa: E402
from nett_skrl.brain.encoders.nature_cnn import NatureCNN  # noqa: E402
from nett_skrl.brain.encoders.utils.pool import DeterministicAvgPool2d  # noqa: E402
from nett_skrl.brain.models.utils.init import init_encoder  # noqa: E402

H, W = 80, 128
OBS = gym.spaces.Box(0, 255, (H, W, 3), np.uint8)
ENCODER_COUNT = {"CNN": 773_995, "CNN+GAP": 197_995, "ViT-Mixer": 682_675}


def _n(m):
    return sum(p.numel() for p in m.parameters())


def _cnn(label, **over):
    torch.manual_seed(0)
    return NatureCNN(OBS, **{**ct.MODELS[label]["cfg"], **over})


def test_cnn_gap_is_cnn_plus_pool_grid_only():
    base, gap = ct.MODELS["CNN"], ct.MODELS["CNN+GAP"]
    assert {k: v for k, v in gap.items() if k != "cfg"} == {k: v for k, v in base.items() if k != "cfg"}
    assert gap["cfg"] == {**base["cfg"], "pool_grid": (1, 3)}


def test_cnn_gap_builds_and_counts(monkeypatch):
    monkeypatch.setenv("NETT_INIT_SCHEME", "orthogonal")
    monkeypatch.setenv("NETT_AMP", "off")
    for label, grid in (("CNN", (3, 6)), ("CNN+GAP", (1, 3))):
        e = _cnn(label)
        init_encoder(e, float(np.sqrt(2)))
        assert e.pool_grid == grid
        assert _n(e) == ENCODER_COUNT[label], (label, _n(e))
        y = e(torch.randint(0, 256, (2, H, W, 3), dtype=torch.uint8).float())
        assert y.shape == (2, 512) and torch.isfinite(y).all()
    assert _cnn("CNN+GAP").linear[0].in_features == 75 * 3


def test_cnn_gap_pool_is_row_global_column_thirds(monkeypatch):
    monkeypatch.setenv("NETT_AMP", "off")
    e = _cnn("CNN+GAP").eval()
    x = torch.randint(0, 256, (2, H, W, 3), dtype=torch.uint8).float()
    fmap = e.encode_spatial(x)                                  # (2, 75, 6, 12)
    assert tuple(fmap.shape[1:]) == (75, 6, 12)
    want = torch.stack([fmap[..., 4 * j:4 * (j + 1)].mean(dim=(2, 3)) for j in range(3)], dim=-1)   # (2, 75, 3)
    pools = [m for m in e.cnn if isinstance(m, DeterministicAvgPool2d)]
    assert len(pools) == 1
    out = pools[0](fmap)
    assert tuple(out.shape[1:]) == (75, 1, 3)
    torch.testing.assert_close(out[:, :, 0, :], want)


@pytest.mark.parametrize("grid", [(4, 4), (1, 5), (7, 1)])
def test_pool_grid_ragged_refuses(grid):
    with pytest.raises(ValueError, match="does not divide"):
        _cnn("CNN", pool_grid=grid)


@pytest.mark.parametrize("grid", [(1,), (0, 3), (1.0, 3), "1x3", (1, 3, 1)])
def test_pool_grid_malformed_refuses(grid):
    with pytest.raises(ValueError, match="two positive ints"):
        _cnn("CNN", pool_grid=grid)


def test_pool_grid_needs_spatial_pool():
    with pytest.raises(ValueError, match="needs spatial_pool=True"):
        _cnn("CNN", pool_grid=(1, 3), spatial_pool=False)


def test_pool_grid_list_form_accepted():
    assert _cnn("CNN", pool_grid=[1, 3]).pool_grid == (1, 3)


def test_vit_mixer_at_eye_keeps_diagonal_init(monkeypatch):
    monkeypatch.setenv("NETT_INIT_SCHEME", "orthogonal")
    monkeypatch.setenv("NETT_AMP", "off")
    torch.manual_seed(0)
    e = CompactViT(OBS, **ct.MODELS["ViT-Mixer"]["cfg"])
    init_encoder(e, float(np.sqrt(2)))
    assert _n(e) == ENCODER_COUNT["ViT-Mixer"]
    mixes = [m.mix for m in e.modules() if hasattr(m, "mix") and isinstance(m.mix, torch.nn.Parameter)]
    assert len(mixes) == ct.MODELS["ViT-Mixer"]["cfg"]["depth"]
    for mix in mixes:
        assert tuple(mix.shape) == (41, 41)
        d = mix.detach().diag()
        assert torch.allclose(d, torch.full_like(d, 2.0), atol=0.1)
        off = mix.detach() - torch.diag(d)
        assert off.abs().max() < 0.15
    y = e(torch.randint(0, 256, (2, H, W, 3), dtype=torch.uint8).float())
    assert y.shape == (2, 512) and torch.isfinite(y).all()
