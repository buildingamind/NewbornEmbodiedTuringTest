"""NETT_INIT_SCHEME: the fleet's encoder init by layer family, DEFAULT 'family' (workspace DECISIONS §179, add. 1-3).

Owner, 2026-10-21: "Kaiming init should be the default moving forward. Don't let that affect the compatibility of
the current runs." / "For any encoder that uses ReLU, change their default to Kaiming init" / "all transformer-based
encoders should use truncated normal init by default." / "Yes use Kaiming init by default" (Conv3d in a ReLU encoder)
/ "Truncated normal for the transformer-based layers. Kaiming for ReLU layers. Orthogonal for tanh layers." So:
  * 'orthogonal' must rebuild every model byte-identically to the pre-§179 rule, which was
    ``encoder.apply(lambda m: orthogonal_init(m, hidden_gain))`` -- through the real builder, so the call sites and
    their RNG order are pinned, not just the helper;
  * under the default, ReLU layers (Conv2d / Conv3d / Linear) get Kaiming He normal (fan_in, ReLU) with zero bias,
    and transformer layers (Conv2d / Linear / MultiheadAttention in_proj) get trunc_normal(.02) with zero bias;
    LayerNorm, GRUCell and bare Parameters are untouched; the MLP trunk and heads keep orthogonal (tanh included);
  * the families are audited against what each registered encoder's forward calls, every encoder_mapping entry must
    be classified, and a mixed encoder must place every layer under a declared prefix;
  * anything else refuses, and NETT_CONV3D_ORTHO_INIT=1 refuses on a family encoder under 'family'.
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
    RELU_ENCODERS, TRANSFORMER_ENCODERS, TRUNC_NORMAL_STD, encoder_family, encoder_owns_init, init_encoder,
    init_scheme, kaiming_init, layer_families, orthogonal_init, trunc_normal_init)
from nett_skrl.brain.registry import algorithm_spec, encoder_mapping  # noqa: E402

H, W = 80, 128
GAIN = math.sqrt(2.0)
RELU_LABELS = ("CNN", "3DCNN", "SimCLR-CLTT", "GuessWhatMoves", "XSP")
TRANSFORMER_LABELS = ("ViT", "ViViT", "UnityViT-CLTT-Ref")
MIXED_LABELS = ("ViT-SlotRecon",)
BUILD_LABELS = ("CNN", "3DCNN", "ViT", "ViViT", "ViT-SlotRecon")
#: ViT-SlotRecon's layers outside its CompactViT trunk, and the family each must take (DECISIONS §179 add. 3).
SLOT_FAMILIES = {"slot_in.1": "relu", "slot_in.3": "relu", "slot_attention.to_q": "transformer",
                 "slot_attention.to_k": "transformer", "slot_attention.to_v": "transformer",
                 "slot_attention.mlp.0": "relu", "slot_attention.mlp.2": "relu"}


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


def _weights(module):
    """(weight, bias) pairs a family rule sets on this module."""
    if isinstance(module, nn.MultiheadAttention):
        return [(module.in_proj_weight, module.in_proj_bias)]
    return [(module.weight, module.bias)]


def _family_ok(enc, min_numel=1024):
    """{layer name: True if its weights carry its family's init} for every layer with enough draws to judge."""
    out = {}
    for name, fam in layer_families(enc).items():
        module = dict(enc.named_modules())[name]
        for w, b in _weights(module):
            if b is not None and not torch.equal(b, torch.zeros_like(b)):
                out[name] = False
                continue
            if w.numel() < min_numel:
                continue
            want = math.sqrt(2.0 / w[0].numel()) if fam == "relu" else TRUNC_NORMAL_STD
            out[name] = (abs(float(w.std()) / want - 1) < 0.1) and not _is_orthogonal(w)
    return out


# ------------------------------------------------------------------------------------------------ 1. the knob

def test_default_is_family_and_both_values_read(monkeypatch):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    assert init_scheme() == "family"
    for v in ("family", "orthogonal"):
        monkeypatch.setenv("NETT_INIT_SCHEME", v)
        assert init_scheme() == v


def test_the_unpushed_kaiming_name_refuses_with_a_pointer(monkeypatch):
    monkeypatch.setenv("NETT_INIT_SCHEME", "kaiming")
    with pytest.raises(ValueError, match="renamed 'family'"):
        init_scheme()


@pytest.mark.parametrize("bad", ["", "ortho", "Kaiming", "xavier", "default", "1", "trunc_normal", "Family"])
def test_refuses_anything_else(monkeypatch, bad):
    monkeypatch.setenv("NETT_INIT_SCHEME", bad)
    with pytest.raises(ValueError, match="NETT_INIT_SCHEME"):
        init_scheme()


@pytest.mark.parametrize("label", ("3DCNN", "ViT", "ViT-SlotRecon"))
def test_conv3d_ortho_needs_the_orthogonal_scheme(monkeypatch, label):
    monkeypatch.setenv("NETT_CONV3D_ORTHO_INIT", "1")
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    with pytest.raises(ValueError, match="NETT_CONV3D_ORTHO_INIT=1 needs NETT_INIT_SCHEME=orthogonal"):
        init_encoder(_encoder(label), GAIN)
    monkeypatch.setenv("NETT_INIT_SCHEME", "orthogonal")
    enc = _encoder(label)
    init_encoder(enc, GAIN)
    assert all(_is_orthogonal(m.weight) for m in enc.modules() if isinstance(m, nn.Conv3d))


# ---------------------------------------------------- 2. compatibility: the pre-§179 build, bit for bit

@pytest.mark.parametrize("label", BUILD_LABELS)
@pytest.mark.parametrize("shared", [True, False])
def test_orthogonal_rebuilds_the_pre179_models_bit_for_bit(monkeypatch, label, shared):
    monkeypatch.setenv("NETT_INIT_SCHEME", "orthogonal")
    new = _states(_build(label, shared))
    _patch_pre179(monkeypatch)
    old = _states(_build(label, shared))
    assert new.keys() == old.keys()
    assert all(torch.equal(new[k], old[k]) for k in old), [k for k in old if not torch.equal(new[k], old[k])]


@pytest.mark.parametrize("label", BUILD_LABELS)
@pytest.mark.parametrize("shared", [True, False])
def test_default_changes_every_encoder_layer(monkeypatch, label, shared):
    """Control for test 2: under the default every family layer of the encoder differs from the old build. (The
    trunk / heads keep their distribution; their values may move, since the family rules draw a different count
    from the global RNG than orthogonal_ -- Conv3d and MultiheadAttention in_proj are new draws.)"""
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    monkeypatch.delenv("NETT_CONV3D_ORTHO_INIT", raising=False)
    new = _states(_build(label, shared))
    _patch_pre179(monkeypatch)
    old = _states(_build(label, shared))
    assert new.keys() == old.keys()
    layers = layer_families(_encoder(label))
    suffix = {n: ".in_proj_weight" if isinstance(m, nn.MultiheadAttention) else ".weight"
              for n, m in _encoder(label).named_modules() if n in layers}
    for name in layers:
        keys = [k for k in old if k.endswith(f".encoder.{name}{suffix[name]}") or k == f"encoder.{name}{suffix[name]}"]
        assert keys, name
        assert all(not torch.equal(new[k], old[k]) for k in keys), name


# ------------------------------------------------------------------------------ 3. the families (the default)

@pytest.mark.parametrize("label", RELU_LABELS)
def test_relu_encoder_is_kaiming_on_every_layer_conv3d_included(monkeypatch, label, capsys):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    monkeypatch.delenv("NETT_CONV3D_ORTHO_INIT", raising=False)
    enc = _encoder(label)
    init_encoder(enc, GAIN)
    assert f"[NETT init] encoder {type(enc).__name__} kaiming (NETT_INIT_SCHEME=family, family=relu)" \
        in capsys.readouterr().out
    fams = layer_families(enc)
    assert fams and set(fams.values()) == {"relu"}
    assert set(fams) == {n for n, m in enc.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear))}
    ok = _family_ok(enc)
    assert ok and all(ok.values()), ok


@pytest.mark.parametrize("label", ("3DCNN", "GuessWhatMoves"))
def test_conv3d_of_a_relu_encoder_is_kaiming(monkeypatch, label):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    monkeypatch.delenv("NETT_CONV3D_ORTHO_INIT", raising=False)
    enc = _encoder(label)
    init_encoder(enc, GAIN)
    c3 = [m for m in enc.modules() if isinstance(m, nn.Conv3d)]
    assert c3
    for m in c3:
        assert float(m.weight.std()) == pytest.approx(math.sqrt(2.0 / m.weight[0].numel()), rel=0.1)
        assert m.bias is None or torch.equal(m.bias, torch.zeros_like(m.bias))


@pytest.mark.parametrize("label", TRANSFORMER_LABELS)
def test_transformer_encoder_is_trunc_normal_on_every_layer(monkeypatch, label, capsys):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    ref, enc = _encoder(label), _encoder(label)
    init_encoder(enc, GAIN)
    assert f"[NETT init] encoder {type(enc).__name__} trunc_normal (NETT_INIT_SCHEME=family, family=transformer)" \
        in capsys.readouterr().out
    fams = layer_families(enc)
    assert set(fams.values()) == {"transformer"}
    ok = _family_ok(enc)
    assert ok and all(ok.values()), ok
    mha = [m for m in enc.modules() if isinstance(m, nn.MultiheadAttention)]
    if mha:                       # the packed QKV is a transformer layer (add. 3), not left at xavier
        assert all(float(m.in_proj_weight.std()) == pytest.approx(TRUNC_NORMAL_STD, rel=0.1) for m in mha)
    # untouched: LayerNorm and bare Parameters (CLS / position embeddings) keep the constructor's values
    sr, se = ref.state_dict(), enc.state_dict()
    lns = [n for n, m in enc.named_modules() if isinstance(m, nn.LayerNorm)]
    assert lns and all(torch.equal(se[f"{n}.weight"], sr[f"{n}.weight"]) for n in lns)
    owned = {f"{n}.{p}" for n, m in enc.named_modules() if n in fams for p, _ in m.named_parameters(recurse=False)}
    bare = [k for k, _ in enc.named_parameters() if k not in owned and not any(k.startswith(f"{n}.") for n in lns)]
    assert all(torch.equal(se[k], sr[k]) for k in bare), [k for k in bare if not torch.equal(se[k], sr[k])]


def test_vit_slot_recon_is_per_layer(monkeypatch, capsys):
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    ref, enc = _encoder("ViT-SlotRecon"), _encoder("ViT-SlotRecon")
    assert encoder_family(enc) == "mixed"
    init_encoder(enc, GAIN)
    assert "[NETT init] encoder ViTSlotRecon kaiming+trunc_normal (NETT_INIT_SCHEME=family, family=mixed)" \
        in capsys.readouterr().out
    fams = layer_families(enc)
    assert {n: f for n, f in fams.items() if not n.startswith("vit.")} == SLOT_FAMILIES
    assert {f for n, f in fams.items() if n.startswith("vit.")} == {"transformer"}
    for name, fam in fams.items():
        w, b = _weights(dict(enc.named_modules())[name])[0]
        want = math.sqrt(2.0 / w[0].numel()) if fam == "relu" else TRUNC_NORMAL_STD
        assert float(w.std()) == pytest.approx(want, rel=0.2), name      # small layers: 64x64 draws
        assert b is None or torch.equal(b, torch.zeros_like(b)), name
    sr, se = ref.state_dict(), enc.state_dict()
    gru = [k for k in sr if k.startswith("slot_attention.gru.") or k.startswith("slot_attention.slots_")]
    assert gru and all(torch.equal(se[k], sr[k]) for k in gru)          # GRUCell and slot mu / sigma untouched


def test_family_checks_fail_on_the_orthogonal_build(monkeypatch):
    """Positive control for test 3: the family checks reject the pre-§179 init of the same encoders."""
    monkeypatch.setenv("NETT_INIT_SCHEME", "orthogonal")
    for label in ("CNN", "ViT", "ViT-SlotRecon"):
        enc = _encoder(label)
        init_encoder(enc, GAIN)
        ok = _family_ok(enc)
        assert ok and not any(ok.values()), (label, ok)


def test_layer_families_refuses_an_undeclared_layer_or_an_unknown_family():
    class Mixed(nn.Module):
        INIT_FAMILIES = {"a": "relu"}

        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 4)

    with pytest.raises(ValueError, match="under no INIT_FAMILIES prefix"):
        layer_families(Mixed())

    class Unknown(Mixed):
        INIT_FAMILIES = {"a": "relu", "b": "tanh"}

    with pytest.raises(ValueError, match="unknown families"):
        layer_families(Unknown())


def test_rules_match_their_definitions():
    torch.manual_seed(1)
    lin = nn.Linear(512, 256)
    kaiming_init(lin)
    torch.manual_seed(1)
    ref = nn.Linear(512, 256)
    nn.init.kaiming_normal_(ref.weight, mode="fan_in", nonlinearity="relu")
    assert torch.equal(lin.weight, ref.weight) and torch.equal(lin.bias, torch.zeros(256))
    c3 = nn.Conv3d(3, 4, (2, 3, 3))
    kaiming_init(c3)
    assert torch.equal(c3.bias, torch.zeros(4))
    torch.manual_seed(2)
    lin = nn.Linear(512, 256)
    trunc_normal_init(lin)
    torch.manual_seed(2)
    ref = nn.Linear(512, 256)
    nn.init.trunc_normal_(ref.weight, std=0.02)
    assert torch.equal(lin.weight, ref.weight) and torch.equal(lin.bias, torch.zeros(256))
    mha = nn.MultiheadAttention(64, 4, batch_first=True)
    out_before = mha.out_proj.weight.clone()
    trunc_normal_init(mha)               # the packed QKV only; out_proj is reached as its own Linear
    assert torch.equal(mha.out_proj.weight, out_before) and torch.equal(mha.in_proj_bias, torch.zeros(192))
    for m in (nn.LayerNorm(8), nn.GRUCell(8, 8)):
        before = {k: v.clone() for k, v in m.state_dict().items()}
        kaiming_init(m)
        trunc_normal_init(m)
        assert all(torch.equal(m.state_dict()[k], v) for k, v in before.items())


# ------------------------------------------------------------------------- 4. the classification, audited

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


def test_every_registered_encoder_is_classified_once():
    for name, cls in encoder_mapping.items():
        names = {c.__name__ for c in cls.__mro__}
        kinds = [bool(names & RELU_ENCODERS), bool(names & TRANSFORMER_ENCODERS),
                 bool(getattr(cls, "INIT_FAMILIES", None)), bool(getattr(cls, "OWNS_INIT", False))]
        assert sum(kinds) == 1, (name, cls, kinds)


@pytest.mark.parametrize("label", RELU_LABELS)
def test_relu_encoders_call_relu_and_no_other_hidden_nonlinearity(label):
    enc, seen = _activations(label)
    assert encoder_family(enc) == "relu" and not encoder_owns_init(enc)
    assert "relu" in seen and not seen & {"gelu", "elu", "silu", "mish", "leaky_relu", "hardswish", "tanh"}, seen
    # the one sigmoid allowed is XSP's figure gate
    assert "sigmoid" not in seen or type(enc).__name__ == "XSPEncoder", seen


@pytest.mark.parametrize("label", TRANSFORMER_LABELS)
def test_transformer_encoders_call_gelu_and_no_relu(label):
    enc, seen = _activations(label)
    assert encoder_family(enc) == "transformer"
    assert any("Attention" in type(m).__name__ for m in enc.modules())
    assert "gelu" in seen and not seen & {"relu", "tanh"}, seen


def test_the_mixed_encoder_calls_both():
    enc, seen = _activations("ViT-SlotRecon")
    assert encoder_family(enc) == "mixed" and {"relu", "gelu"} <= seen and "tanh" not in seen, seen


def test_relu_set_covers_the_unlabelled_cnns():
    """Encoders with no campaign label: built at defaults; the *Pretrained subclasses classify through the MRO."""
    for name in ("small", "Resnet10CNN", "Resnet18CNN", "compact_cnn"):
        enc = encoder_mapping[name](gym.spaces.Box(0, 255, (3, H, W), np.uint8))
        rec = _Record()
        with rec, torch.no_grad():
            enc(torch.randint(0, 256, (2, 3 * H * W)).float())
        assert encoder_family(enc) == "relu" and rec.seen == {"relu"}, (name, rec.seen)
    for name in ("compact_cnn_pretrained", "compact_3dcnn_pretrained"):
        assert {c.__name__ for c in encoder_mapping[name].__mro__} & RELU_ENCODERS, name


def test_the_trunk_keeps_orthogonal_for_tanh(monkeypatch):
    """'Orthogonal for tanh layers': the MLP trunk at activation 'tanh' is orthogonal under the default."""
    monkeypatch.delenv("NETT_INIT_SCHEME", raising=False)
    spec = ct.MODELS["CNN"]
    torch.manual_seed(0)
    models = builder_mod.build_models_for_algorithm(
        algorithm_spec(Brain(encoder=spec["encoder"], algorithm="PPO").algorithm),
        encoder_cls=encoder_mapping[spec["encoder"]], encoder_kwargs=_kw("CNN"), observation_space=_space("CNN"),
        action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32), device=torch.device("cpu"),
        cfg=ModelCfg(shared_encoder=False, activation="tanh"))
    for m in models.values():
        assert any(isinstance(x, nn.Tanh) for x in m.trunk.modules())
        lin = [x for x in m.trunk.modules() if isinstance(x, nn.Linear)]
        assert lin and all(_is_orthogonal(x.weight) for x in lin)
