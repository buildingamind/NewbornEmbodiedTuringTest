"""NETT_INIT_SCHEME: Kaiming He for every ReLU encoder, DEFAULT 'kaiming' (workspace DECISIONS §179 + add. 1).

Owner, 2026-10-21: "Kaiming init should be the default moving forward. Don't let that affect the compatibility of
the current runs." and "For any encoder that uses ReLU, change their default to Kaiming init". So:
  * 'orthogonal' must rebuild every model byte-identically to the pre-§179 rule, which was
    ``encoder.apply(lambda m: orthogonal_init(m, hidden_gain))`` -- through the real builder, so the call sites and
    their RNG order are pinned, not just the helper;
  * under the default, a ReLU encoder's Conv2d / Linear get Kaiming He normal (fan_in, ReLU) with zero bias; its
    Conv3d keeps PyTorch's default (§176); a non-ReLU encoder, the MLP trunk and the output heads build exactly as
    before;
  * RELU_ENCODERS is audited against what each registered encoder's forward calls, and every encoder_mapping
    entry must be classified;
  * anything else refuses, and NETT_CONV3D_ORTHO_INIT=1 refuses on a ReLU encoder under 'kaiming'.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.brain.brain import Brain  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.models import builder as builder_mod  # noqa: E402
from nett_skrl.brain.models.model_cfg import ModelCfg  # noqa: E402
from nett_skrl.brain.models.utils import backbone as backbone_mod  # noqa: E402
from nett_skrl.brain.models.utils.init import (  # noqa: E402
    RELU_ENCODERS, encoder_owns_init, encoder_uses_relu, init_encoder, init_scheme, kaiming_init, orthogonal_init)
from nett_skrl.brain.registry import algorithm_spec, encoder_mapping  # noqa: E402

H, W = 80, 128
GAIN = math.sqrt(2.0)
RELU_LABELS = ("CNN", "3DCNN", "SimCLR-CLTT", "GuessWhatMoves", "XSP")
OTHER_LABELS = ("ViT", "ViViT")
#: Registered encoders that are NOT ReLU encoders, with the reason. Every encoder_mapping entry must be in
#: RELU_ENCODERS, here, or declare OWNS_INIT.
NON_RELU = {
    "CompactViT": "GELU",
    "UnityViT": "GELU",
    "CompactViViT": "GELU",
    "ViTSlotRecon": "mixed: GELU ViT body + ReLU slot MLP (kept orthogonal; DECISIONS §179 add. 1)",
}


def _space(label):
    ch = 6 if ct.MODELS[label]["framestack"] else 3
    return gym.spaces.Box(low=0, high=255, shape=(ch, H, W), dtype=np.uint8)


def _kw(label):
    kw = EncoderCfg(**ct.MODELS[label]["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    return kw


def _encoder(label, seed=0):
    torch.manual_seed(seed)
    return encoder_mapping[ct.MODELS[label]["encoder"]](_space(label), **_kw(label))


def _build(label, shared, seed=0):
    spec = ct.MODELS[label]
    torch.manual_seed(seed)
    return builder_mod.build_models_for_algorithm(
        algorithm_spec(Brain(encoder=spec["encoder"], algorithm="PPO").algorithm),
        encoder_cls=encoder_mapping[spec["encoder"]], encoder_kwargs=_kw(label), observation_space=_space(label),
        action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32), device=torch.device("cpu"),
        cfg=ModelCfg(shared_encoder=shared))


def _states(models):
    return {f"{name}.{k}": v.clone() for name, m in models.items() for k, v in m.state_dict().items()}


def _pre179(encoder, gain):
    """The literal pre-§179 call at both encoder sites (builder.py shared encoder, backbone.py encoder)."""
    encoder.apply(lambda module: orthogonal_init(module, gain))


def _patch_pre179(monkeypatch):
    monkeypatch.setattr(builder_mod, "init_encoder", _pre179)
    monkeypatch.setattr(backbone_mod, "init_encoder", _pre179)


def _is_orthogonal(w):
    m = w.reshape(w.shape[0], -1)
    g = m @ m.T if m.shape[0] <= m.shape[1] else m.T @ m
    d = torch.diagonal(g)
    return bool(torch.allclose(g, torch.diag(d), atol=1e-4 * float(d.abs().max())))


# ------------------------------------------------------------------------------------------------ 1. the knob

def test_default_is_kaiming_and_both_values_read(monkeypatch):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    assert init_scheme() == "kaiming"
    for v in ("kaiming", "orthogonal"):
        monkeypatch.setenv("NETT_INIT_SCHEME", v)
        assert init_scheme() == v


@pytest.mark.parametrize("bad", ["", "ortho", "Kaiming", "xavier", "default", "1"])
def test_refuses_anything_else(monkeypatch, bad):
    monkeypatch.setenv("NETT_INIT_SCHEME", bad)
    with pytest.raises(ValueError, match="NETT_INIT_SCHEME"):
        init_scheme()


def test_conv3d_ortho_needs_the_orthogonal_scheme_on_a_relu_encoder(monkeypatch):
    monkeypatch.setenv("NETT_CONV3D_ORTHO_INIT", "1")
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    with pytest.raises(ValueError, match="NETT_CONV3D_ORTHO_INIT=1 needs NETT_INIT_SCHEME=orthogonal"):
        init_encoder(_encoder("3DCNN"), GAIN)
    monkeypatch.setenv("NETT_INIT_SCHEME", "orthogonal")
    enc = _encoder("3DCNN")
    init_encoder(enc, GAIN)
    assert all(_is_orthogonal(m.weight) for m in enc.modules() if isinstance(m, nn.Conv3d))


# ---------------------------------------------------- 2. compatibility: the pre-§179 build, bit for bit

@pytest.mark.parametrize("label", RELU_LABELS[:2] + OTHER_LABELS)
@pytest.mark.parametrize("shared", [True, False])
def test_orthogonal_rebuilds_the_pre179_models_bit_for_bit(monkeypatch, label, shared):
    monkeypatch.setenv("NETT_INIT_SCHEME", "orthogonal")
    new = _states(_build(label, shared))
    _patch_pre179(monkeypatch)
    old = _states(_build(label, shared))
    assert new.keys() == old.keys()
    assert all(torch.equal(new[k], old[k]) for k in old), [k for k in old if not torch.equal(new[k], old[k])]


@pytest.mark.parametrize("label", OTHER_LABELS)
@pytest.mark.parametrize("shared", [True, False])
def test_default_leaves_non_relu_encoders_bit_for_bit_unchanged(monkeypatch, label, shared):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    new = _states(_build(label, shared))
    _patch_pre179(monkeypatch)
    old = _states(_build(label, shared))
    assert new.keys() == old.keys() and all(torch.equal(new[k], old[k]) for k in old)


@pytest.mark.parametrize("label", RELU_LABELS[:2])
@pytest.mark.parametrize("shared", [True, False])
def test_default_changes_relu_encoders_only_in_encoder_tensors(monkeypatch, label, shared):
    """Control for test 2: the default build of a ReLU encoder differs from the old one, in encoder.* only --
    and the trunk / heads, built after the encoder, still match (Kaiming and orthogonal_ draw the same count)."""
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    new = _states(_build(label, shared))
    _patch_pre179(monkeypatch)
    old = _states(_build(label, shared))
    diff = [k for k in old if not torch.equal(new[k], old[k])]
    assert diff and all(".encoder." in f".{k}" for k in diff), diff


# --------------------------------------------------------------------------------- 3. 'kaiming' (the default)

@pytest.mark.parametrize("label", RELU_LABELS)
def test_default_is_kaiming_on_a_relu_encoder(monkeypatch, label, capsys):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    monkeypatch.delenv("NETT_CONV3D_ORTHO_INIT", raising=False)
    enc = _encoder(label)
    init_encoder(enc, GAIN)
    assert f"[NETT init] encoder {type(enc).__name__} kaiming (NETT_INIT_SCHEME=kaiming, relu=yes)" in capsys.readouterr().out
    layers = [x for x in enc.modules() if isinstance(x, (nn.Conv2d, nn.Linear))]
    assert layers
    big = 0
    for x in layers:
        if x.bias is not None:
            assert torch.equal(x.bias, torch.zeros_like(x.bias)), x
        if x.weight.numel() >= 4096:          # enough draws for the std to be tight
            big += 1
            assert float(x.weight.std()) == pytest.approx(math.sqrt(2.0 / x.weight[0].numel()), rel=0.1), x
            assert not _is_orthogonal(x.weight), x
    assert big >= 1


@pytest.mark.parametrize("label", OTHER_LABELS)
def test_default_keeps_orthogonal_on_a_non_relu_encoder(monkeypatch, label, capsys):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    enc = _encoder(label)
    init_encoder(enc, GAIN)
    assert f"[NETT init] encoder {type(enc).__name__} orthogonal (NETT_INIT_SCHEME=kaiming, relu=no)" in capsys.readouterr().out
    big = [x for x in enc.modules() if isinstance(x, (nn.Conv2d, nn.Linear)) and x.weight.numel() >= 4096]
    assert big and all(_is_orthogonal(x.weight) for x in big)


def test_kaiming_check_fails_on_the_orthogonal_build(monkeypatch):
    """Positive control for test 3: the std / non-orthogonal checks reject the pre-§179 init of the same encoder."""
    monkeypatch.setenv("NETT_INIT_SCHEME", "orthogonal")
    enc = _encoder("CNN")
    init_encoder(enc, GAIN)
    big = [x for x in enc.modules() if isinstance(x, (nn.Conv2d, nn.Linear)) and x.weight.numel() >= 4096]
    assert big and all(_is_orthogonal(x.weight) for x in big)


def test_kaiming_leaves_conv3d_at_pytorch_default(monkeypatch):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    monkeypatch.delenv("NETT_CONV3D_ORTHO_INIT", raising=False)
    ref, enc = _encoder("3DCNN"), _encoder("3DCNN")
    init_encoder(enc, GAIN)
    c3 = [n for n, x in enc.named_modules() if isinstance(x, nn.Conv3d)]
    assert c3
    sr, se = ref.state_dict(), enc.state_dict()
    for n in c3:
        assert torch.equal(se[f"{n}.weight"], sr[f"{n}.weight"]) and torch.equal(se[f"{n}.bias"], sr[f"{n}.bias"]), n
    assert any(not torch.equal(se[k], sr[k]) for k in sr)          # the Conv2d / Linear layers did change


def test_kaiming_init_matches_the_definition():
    torch.manual_seed(1)
    lin = nn.Linear(512, 256)
    kaiming_init(lin)
    torch.manual_seed(1)
    ref = nn.Linear(512, 256)
    nn.init.kaiming_normal_(ref.weight, mode="fan_in", nonlinearity="relu")
    assert torch.equal(lin.weight, ref.weight) and torch.equal(lin.bias, torch.zeros(256))
    c3 = nn.Conv3d(3, 4, (2, 3, 3))
    before = c3.weight.clone()
    kaiming_init(c3)
    assert torch.equal(c3.weight, before)


# ------------------------------------------------------------------------- 4. the ReLU classification, audited

_ACTS = {F.relu: "relu", torch.relu: "relu", torch.relu_: "relu", F.relu_: "relu", torch.Tensor.relu: "relu",
         torch.Tensor.relu_: "relu", F.gelu: "gelu", F.elu: "elu", F.silu: "silu", F.mish: "mish",
         F.leaky_relu: "leaky_relu", F.hardswish: "hardswish", torch.tanh: "tanh", torch.Tensor.tanh: "tanh",
         torch.sigmoid: "sigmoid", torch.Tensor.sigmoid: "sigmoid", F.softplus: "softplus"}


class _Record(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in _ACTS:
            self.seen.add(_ACTS[func])
        return func(*args, **(kwargs or {}))


def _activations(label):
    enc = _encoder(label)
    rec = _Record()
    ch = _space(label).shape[0]
    with rec, torch.no_grad():
        enc(torch.randint(0, 256, (2, ch * H * W)).float())
    return enc, rec.seen


def test_every_registered_encoder_is_classified():
    for name, cls in encoder_mapping.items():
        names = {c.__name__ for c in cls.__mro__}
        assert getattr(cls, "OWNS_INIT", False) or names & RELU_ENCODERS or cls.__name__ in NON_RELU, (name, cls)
        assert not (names & RELU_ENCODERS and cls.__name__ in NON_RELU), name


@pytest.mark.parametrize("label", RELU_LABELS)
def test_relu_encoders_call_relu_and_no_other_hidden_nonlinearity(label):
    enc, seen = _activations(label)
    assert encoder_uses_relu(enc) and not encoder_owns_init(enc)
    assert "relu" in seen and not seen & {"gelu", "elu", "silu", "mish", "leaky_relu", "hardswish", "tanh"}, seen
    # the one sigmoid allowed is XSP's figure gate
    assert "sigmoid" not in seen or type(enc).__name__ == "XSPEncoder", seen


@pytest.mark.parametrize("label", OTHER_LABELS + ("UnityViT-CLTT-Ref", "ViT-SlotRecon"))
def test_non_relu_encoders_call_something_else(label):
    enc, seen = _activations(label)
    assert not encoder_uses_relu(enc) and type(enc).__name__ in NON_RELU
    assert "gelu" in seen, seen


def test_relu_set_covers_the_unlabelled_cnns():
    """Encoders with no campaign label: built at defaults; the *Pretrained subclasses classify through the MRO."""
    for name in ("small", "Resnet10CNN", "Resnet18CNN", "compact_cnn"):
        enc = encoder_mapping[name](gym.spaces.Box(0, 255, (3, H, W), np.uint8))
        rec = _Record()
        with rec, torch.no_grad():
            enc(torch.randint(0, 256, (2, 3 * H * W)).float())
        assert encoder_uses_relu(enc) and rec.seen == {"relu"}, (name, rec.seen)
    for name in ("compact_cnn_pretrained", "compact_3dcnn_pretrained"):
        assert {c.__name__ for c in encoder_mapping[name].__mro__} & RELU_ENCODERS, name
