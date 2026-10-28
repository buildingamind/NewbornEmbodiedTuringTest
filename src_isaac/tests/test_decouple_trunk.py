"""NETT_DECOUPLE_TRUNK (owner 2026-10-28, BASELINE detached SC257 / Mask75; models/utils/features.py).

The RL path detaches at the aux's read point: PPO's gradient reaches the encoder READOUT (NatureCNN.linear; CompactViT
CLS -> head) and not the trunk, the aux's gradient reaches the trunk, the features are unchanged, and the knob is
refused beside NETT_DECOUPLE_ENCODER or on an encoder with no trunk / readout split.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_baseline_aux as tba  # noqa: E402  (encoder + aux builders, the agent's way)
from nett_skrl.brain.models.utils import features as F  # noqa: E402

LABELS = ("CNN-D512P+MASK75", "CNN-D512P+SC257", "ViT-P8-666K+SC257", "ViT-P8-666K+MAE")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in tba._KNOBS + ("NETT_DECOUPLE_TRUNK", "NETT_DECOUPLE_ENCODER", "NETT_MIRROR_CKPT", "NETT_FEATURE_CACHE"):
        monkeypatch.delenv(k, raising=False)


def _readout_prefix(label):
    return "linear." if "CNN" in label else "head."


def _model(enc):
    torch.manual_seed(3)
    return SimpleNamespace(encoder=enc, trunk=nn.Linear(512, 4), observation_space=enc.observation_space)


def _split(enc, label):
    pre = _readout_prefix(label)
    readout = {n: p for n, p in enc.named_parameters() if n.startswith(pre)}
    trunk = {n: p for n, p in enc.named_parameters() if not n.startswith(pre)}
    assert readout and trunk
    return readout, trunk


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("mirrored", [False, True])
def test_ppo_gradient_reaches_the_readout_only_and_values_are_unchanged(label, mirrored, monkeypatch):
    enc = tba._encoder(label)
    model = _model(enc)
    x = tba._batch()
    fwd = F.features_forward_mirrored if mirrored else F.features_forward
    ref = fwd(model, {"observations": x})
    monkeypatch.setenv("NETT_DECOUPLE_TRUNK", "1")
    F.clear_feature_cache(enc)
    out = fwd(model, {"observations": x})
    ref, out = (ref if mirrored else (ref,)), (out if mirrored else (out,))
    for a, b in zip(ref, out):
        assert torch.allclose(a, b, atol=1e-6)
    sum(o.sum() for o in out).backward()
    readout, trunk = _split(enc, label)
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in readout.values()), "readout untrained"
    assert all(p.grad is None for p in trunk.values()), [n for n, p in trunk.items() if p.grad is not None]


@pytest.mark.parametrize("label", ("CNN-D512P+MASK75", "CNN-D512P+SC257", "ViT-P8-666K+SC257"))
def test_aux_still_reaches_the_trunk_under_the_knob(label, monkeypatch):
    monkeypatch.setenv("NETT_DECOUPLE_TRUNK", "1")
    enc, aux = tba._build(label, monkeypatch)
    aux.compute(enc, tba._batch()).backward()
    _, trunk = _split(enc, label)
    missing = [n for n, p in trunk.items() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not missing, f"trunk tensors the aux does not reach: {missing}"


def test_refused_beside_decouple_encoder(monkeypatch):
    enc = tba._encoder("CNN-D512P+SC257")
    monkeypatch.setenv("NETT_DECOUPLE_TRUNK", "1")
    monkeypatch.setenv("NETT_DECOUPLE_ENCODER", "1")
    with pytest.raises(ValueError, match="contradictory"):
        F.features_forward(_model(enc), {"observations": tba._batch()})


def test_refused_on_an_encoder_without_a_readout_split(monkeypatch):
    class _NoSplit(nn.Module):
        observation_space = None

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(1, 512)

        def forward(self, x):
            return self.lin(x.float().mean(dim=(1, 2, 3), keepdim=False)[:, None])

    monkeypatch.setenv("NETT_DECOUPLE_TRUNK", "1")
    with pytest.raises(TypeError, match="forward_trunk_detached"):
        F.features_forward(_model(_NoSplit()), {"observations": tba._batch()})
