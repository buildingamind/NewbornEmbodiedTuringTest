"""NETT_CONV3D_ORTHO_INIT: opt-in orthogonal init for Conv3d layers.

orthogonal_init matched only (Conv2d, Linear), so Conv3d stems kept PyTorch's default init.
Unset must rebuild every existing model byte-identically; '1' must give the Conv3d stems the
same orthogonal/zero-bias treatment as every other layer.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.encoders.compact_3dcnn import Compact3DCNN
from nett_skrl.brain.encoders.guess_what_moves import GuessWhatMoves
from nett_skrl.brain.models.utils.init import conv3d_ortho_init_enabled, orthogonal_init

# The default eye (128x80), 2-frame stack.
OBS = gym.spaces.Box(low=0, high=255, shape=(6, 80, 128), dtype=np.uint8)
GAIN = math.sqrt(2.0)


def _build(cls, seed=0):
    torch.manual_seed(seed)
    enc = cls(OBS)
    enc.apply(lambda m: orthogonal_init(m, GAIN))
    return enc


def _conv3d(enc):
    layers = [m for m in enc.modules() if isinstance(m, nn.Conv3d)]
    assert layers, "encoder has no Conv3d"
    return layers


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("NETT_CONV3D_ORTHO_INIT", raising=False)
    assert conv3d_ortho_init_enabled() is False
    monkeypatch.setenv("NETT_CONV3D_ORTHO_INIT", "0")
    assert conv3d_ortho_init_enabled() is False


@pytest.mark.parametrize("bad", ["", "yes", "true", "2", "on"])
def test_refuses_anything_but_0_or_1(monkeypatch, bad):
    monkeypatch.setenv("NETT_CONV3D_ORTHO_INIT", bad)
    with pytest.raises(ValueError, match="NETT_CONV3D_ORTHO_INIT"):
        conv3d_ortho_init_enabled()


@pytest.mark.parametrize("cls", [Compact3DCNN, GuessWhatMoves])
def test_unset_is_byte_identical_to_the_old_rule(monkeypatch, cls):
    """Unset: every parameter equals what the pre-knob (Conv2d, Linear)-only rule produced."""
    monkeypatch.delenv("NETT_CONV3D_ORTHO_INIT", raising=False)
    new = _build(cls).state_dict()

    def old_rule(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=GAIN)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    torch.manual_seed(0)
    ref = cls(OBS)
    ref.apply(old_rule)
    ref = ref.state_dict()
    assert new.keys() == ref.keys()
    for k in ref:
        assert torch.equal(new[k], ref[k]), k


@pytest.mark.parametrize("cls", [Compact3DCNN, GuessWhatMoves])
def test_unset_leaves_conv3d_at_pytorch_default(monkeypatch, cls):
    monkeypatch.delenv("NETT_CONV3D_ORTHO_INIT", raising=False)
    for m in _conv3d(_build(cls)):
        bound = 1.0 / math.sqrt(m.weight[0].numel())
        assert float(m.weight.abs().max()) <= bound + 1e-6
        if m.bias is not None:
            assert float(m.bias.abs().max()) > 0.0


@pytest.mark.parametrize("cls", [Compact3DCNN, GuessWhatMoves])
def test_on_makes_conv3d_orthogonal_with_zero_bias(monkeypatch, cls, capsys):
    monkeypatch.setenv("NETT_CONV3D_ORTHO_INIT", "1")
    layers = _conv3d(_build(cls))
    for m in layers:
        w = m.weight.detach().flatten(1)          # (out, fan_in)
        rows, cols = w.shape
        gram = w @ w.T if rows <= cols else w.T @ w
        eye = torch.eye(min(rows, cols)) * GAIN ** 2
        assert torch.allclose(gram, eye, atol=1e-4)
        if m.bias is not None:
            assert torch.count_nonzero(m.bias) == 0
    out = capsys.readouterr().out
    assert out.count("[NETT init] Conv3d orthogonal") == len(layers)


@pytest.mark.parametrize("cls", [Compact3DCNN, GuessWhatMoves])
def test_on_changes_only_conv3d_parameters_in_distribution(monkeypatch, cls):
    """Non-Conv3d layers keep the orthogonal rule; only Conv3d tensors change kind."""
    monkeypatch.setenv("NETT_CONV3D_ORTHO_INIT", "1")
    enc = _build(cls)
    for m in enc.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
            assert torch.count_nonzero(m.bias) == 0
