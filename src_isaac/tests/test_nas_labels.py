"""NAS labels (workspace DECISIONS §80 item 2): NAS-CNN, NAS-3DCNN, NAS-ViT, NAS-ViViT.

Each label takes its encoder cfg from NETT_NAS_CFG (a JSON object) and resolves OUTSIDE MODELS.
Pinned here:
  (a) each family's BASE config is its registered label's model: same encoder, same framestack,
      the same cfg once the constructor's defaults are filled in, the same parameter count and the
      same weights at one seed, built on CPU at the campaign eye (128x80, 3 channels per frame);
  (b) every schema rule refuses, and so does the env contract (missing for a NAS label, present for
      a registered one, malformed JSON);
  (c) MODELS is untouched. ⚠ ASSERTED STRUCTURALLY, NOT AGAINST A FROZEN COPY: a literal snapshot of
      ~95 entries would fail on every legitimate label added after it and so stop being read. What
      makes "unchanged" true is that NAS labels never enter MODELS and a registered label resolves to
      the MODELS object itself -- both asserted below. (At 1ac01fa the MODELS repr was compared
      against the pin tree's once, by hand: equal, ordered.)
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the default eye
NAS = ("NAS-CNN", "NAS-3DCNN", "NAS-ViT", "NAS-ViViT")


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    monkeypatch.delenv("NETT_NAS_CFG", raising=False)
    import campaign_train

    return campaign_train


def _nas(campaign, label, **kv):
    """`label`'s base config with `kv` applied (a value of ... deletes the key)."""
    cfg = {**campaign.NAS_LABELS[label]["base"], **kv}
    return {k: v for k, v in cfg.items() if v is not ...}


def _spec(campaign, label, **kv):
    return campaign.nas_spec(label, json.dumps(_nas(campaign, label, **kv)))


def _kwargs(spec):
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    return kw


def _resolved(spec):
    """The cfg the constructor actually runs with: its signature defaults, then the spec's kwargs."""
    sig = inspect.signature(encoder_mapping[spec["encoder"]].__init__)
    defaults = {n: p.default for n, p in sig.parameters.items()
                if p.default is not inspect.Parameter.empty}
    return {**defaults, **_kwargs(spec)}


def _build(campaign, spec, seed=0):
    channels = 3 * (campaign._FRAMESTACK_N if spec["framestack"] else 1)
    space = gym.spaces.Box(low=0, high=255, shape=(channels, H, W), dtype=np.uint8)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](space, **_kwargs(spec))


def _params(enc):
    return sum(p.numel() for p in enc.parameters())


# ---------------------------------------------------------------- (c) MODELS is untouched

def test_nas_labels_resolve_outside_models(campaign):
    assert set(campaign.NAS_LABELS) == set(NAS)
    assert not [k for k in campaign.MODELS if k.startswith("NAS-")]
    for label in NAS:
        spec = _spec(campaign, label)
        assert set(spec) == {"encoder", "cfg", "framestack"}, "a NAS spec carries no aux/seg/pre"
        assert campaign.segmentation_wrappers(spec) == (["framestack"] if spec["framestack"] else [])


def test_a_registered_label_resolves_to_its_models_entry_itself(campaign):
    for model in campaign.MODELS:
        assert campaign.resolve_spec(model) is campaign.MODELS[model]


# ---------------------------------------------------------------- (a) bases reproduce registered labels

# (NAS label, overrides of its base, the registered label that config must BE)
PAIRS = [
    ("NAS-CNN",   {},                                          "CNN2F"),
    ("NAS-CNN",   {"framestack": False},                       "CNN"),
    ("NAS-3DCNN", {},                                          "3DCNN"),
    ("NAS-3DCNN", {"stem_dim": 160, "mid_dim": 640},           "3DCNN-2M-conv"),
    ("NAS-3DCNN", {"stem_kernel_hw": 5},                       "3DCNN-Sp5"),
    ("NAS-ViT",   {},                                          "ViT2F"),
    ("NAS-ViT",   {"framestack": False},                       "ViT"),
    ("NAS-ViT",   {"embed_dim": 256},                          "ViT2F-2M"),
    ("NAS-ViT",   {"embed_dim": 132, "pool": "spatial", "spatial_grid": [5, 4],
                   "framestack": False},                       "ViT-Sp-Rect"),
    ("NAS-ViViT", {},                                          "ViViT"),
]


@pytest.mark.parametrize("label,kv,registered", PAIRS, ids=[f"{p[0]}={p[2]}" for p in PAIRS])
def test_nas_config_reproduces_the_registered_label(campaign, label, kv, registered):
    nas, reg = _spec(campaign, label, **kv), campaign.MODELS[registered]
    assert nas["encoder"] == reg["encoder"]
    assert nas["framestack"] is reg["framestack"]
    assert _resolved(nas) == _resolved(reg)
    a, b = _build(campaign, nas), _build(campaign, reg)
    assert _params(a) == _params(b)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_base_cfg_adds_only_constructor_defaults(campaign):
    """The base's keys beyond the registered cfg (stem_dim, spatial_pool, ...) must be the
    constructor's own defaults, or "base == registered" holds only by the cfg dict's luck."""
    for label, fam in campaign.NAS_LABELS.items():
        nas, reg = _spec(campaign, label), campaign.MODELS[fam["base_label"]]
        sig = inspect.signature(encoder_mapping[nas["encoder"]].__init__).parameters
        for k in set(nas["cfg"]) - set(reg["cfg"]):
            assert nas["cfg"][k] == sig[k].default, (label, k)


def test_cnn_framestack_is_cnn2f_construction_not_input_frames(campaign):
    """CNN2F has NO input_frames key: nature_cnn takes the whole stack. NAS-CNN must match."""
    assert "input_frames" not in campaign.MODELS["CNN2F"]["cfg"]
    assert "input_frames" not in _spec(campaign, "NAS-CNN")["cfg"]
    assert _build(campaign, _spec(campaign, "NAS-CNN")).cnn[0].in_channels == 3 * campaign._FRAMESTACK_N


@pytest.mark.parametrize("label,kv,count", [
    ("NAS-3DCNN", {}, 695_981),
    ("NAS-3DCNN", {"stem_dim": 160, "mid_dim": 640}, 2_005_933),
    ("NAS-ViT", {"embed_dim": 256}, 2_117_632),
])
def test_known_encoder_counts_at_128x80_two_frames(campaign, label, kv, count):
    assert campaign._FRAMESTACK_N == 2
    assert _params(_build(campaign, _spec(campaign, label, **kv))) == count


def test_temporal_families_take_num_frames_from_framestack_n(campaign):
    assert _spec(campaign, "NAS-3DCNN")["cfg"]["num_frames"] == campaign._FRAMESTACK_N
    assert _spec(campaign, "NAS-ViViT")["cfg"]["num_frames"] == campaign._FRAMESTACK_N


def test_vit_takes_no_dropout(campaign, monkeypatch):
    """NETT_VIT_DROPOUT belongs to the two U35 dropout labels; ViT/ViT2F do not take it."""
    for kv in ({}, {"pool": "spatial", "spatial_grid": [5, 4]}):
        cfg = _spec(campaign, "NAS-ViT", **kv)["cfg"]
        assert "dropout" not in cfg and "dropout_scope" not in cfg


def test_spatial_grid_reaches_the_encoder_as_a_tuple(campaign):
    cfg = _spec(campaign, "NAS-ViT", pool="spatial", spatial_grid=[5, 4])["cfg"]
    assert cfg["spatial_grid"] == (5, 4) and type(cfg["spatial_grid"]) is tuple
    assert cfg["spatial_reduce_dim"] == 16


# ---------------------------------------------------------------- (b) refusals

REFUSE = [
    # unknown / missing keys
    ("NAS-CNN",   {"dropout": 0.1},                        "unknown"),
    ("NAS-CNN",   {"input_frames": 2},                     "unknown"),
    ("NAS-CNN",   {"conv_dim": ...},                       "missing"),
    ("NAS-CNN",   {"framestack": ...},                     "missing"),
    ("NAS-3DCNN", {"stem_kernel_hw": ...},                 "missing"),
    ("NAS-3DCNN", {"num_frames": 2},                       "unknown"),
    ("NAS-ViT",   {"mlp_ratio": ...},                      "missing"),
    ("NAS-ViT",   {"temporal_mode": "joint"},              "unknown"),
    ("NAS-ViT",   {"spatial_grid": [5, 4]},                "unknown"),     # with pool cls
    ("NAS-ViT",   {"pool": "spatial"},                     "missing"),     # no spatial_grid
    ("NAS-ViViT", {"stem": "linear"},                      "unknown"),
    ("NAS-ViViT", {"pool": ...},                           "missing"),
    # wrong JSON types
    ("NAS-CNN",   {"features_dim": "512"},                 "features_dim"),
    ("NAS-CNN",   {"conv_dim": 75.0},                      "conv_dim"),
    ("NAS-CNN",   {"conv_dim": True},                      "conv_dim"),
    ("NAS-CNN",   {"spatial_pool": 1},                     "spatial_pool"),
    ("NAS-CNN",   {"framestack": "true"},                  "framestack"),
    ("NAS-3DCNN", {"stem_dim": 32.0},                      "stem_dim"),
    ("NAS-ViT",   {"mlp_ratio": 2},                        "mlp_ratio"),
    ("NAS-ViT",   {"num_heads": 4.0},                      "num_heads"),
    ("NAS-ViT",   {"pool": None},                          "pool"),
    ("NAS-ViT",   {"pool": "spatial", "spatial_grid": "5,4"}, "spatial_grid"),
    ("NAS-ViViT", {"depth": [3]},                          "depth"),
    # out of range / outside the choice set
    ("NAS-CNN",   {"features_dim": 1024},                  "features_dim"),
    ("NAS-CNN",   {"conv_dim": 15},                        "conv_dim"),
    ("NAS-CNN",   {"conv_dim": 257},                       "conv_dim"),
    ("NAS-3DCNN", {"features_dim": 128},                   "features_dim"),
    ("NAS-3DCNN", {"stem_dim": 15},                        "stem_dim"),
    ("NAS-3DCNN", {"stem_dim": 161},                       "stem_dim"),
    ("NAS-3DCNN", {"mid_dim": 31},                         "mid_dim"),
    ("NAS-3DCNN", {"mid_dim": 641},                        "mid_dim"),
    ("NAS-3DCNN", {"conv_dim": 300},                       "conv_dim"),
    ("NAS-3DCNN", {"stem_kernel_hw": 1},                   "stem_kernel_hw"),
    ("NAS-ViT",   {"features_dim": 256},                   "features_dim"),
    ("NAS-ViT",   {"patch_size": 4},                       "patch_size"),
    ("NAS-ViT",   {"num_heads": 5, "embed_dim": 160},      "num_heads"),
    ("NAS-ViT",   {"depth": 0},                            "depth"),
    ("NAS-ViT",   {"depth": 9},                            "depth"),
    ("NAS-ViT",   {"mlp_ratio": 0.5},                      "mlp_ratio"),
    ("NAS-ViT",   {"mlp_ratio": 4.5},                      "mlp_ratio"),
    ("NAS-ViT",   {"pool": "mean"},                        "pool"),
    ("NAS-ViT",   {"stem": "patch"},                       "stem"),
    ("NAS-ViT",   {"pool": "spatial", "spatial_grid": [4, 4]}, "spatial_grid"),
    ("NAS-ViT",   {"pool": "spatial", "spatial_grid": [5.0, 4.0]}, "spatial_grid"),
    ("NAS-ViViT", {"temporal_mode": "divided"},            "temporal_mode"),
    ("NAS-ViViT", {"features_dim": 256},                   "features_dim"),
    # embed_dim against num_heads
    ("NAS-ViT",   {"embed_dim": 146},                      "not divisible"),
    ("NAS-ViViT", {"embed_dim": 145},                      "not divisible"),
    ("NAS-ViT",   {"embed_dim": 520, "num_heads": 8},      "16..64"),
    ("NAS-ViT",   {"embed_dim": 30, "num_heads": 2},       "16..64"),
    ("NAS-ViViT", {"embed_dim": 390, "num_heads": 6},      "16..64"),
    # family rules
    ("NAS-ViViT", {"pool": "spatial"},                     "adaptive pooling"),
    ("NAS-ViViT", {"pool": "spatial", "spatial_grid": [5, 4]}, "adaptive pooling"),
    ("NAS-3DCNN", {"framestack": False},                   "framestack"),
    ("NAS-ViViT", {"framestack": False},                   "framestack"),
]


@pytest.mark.parametrize("label,kv,match", REFUSE,
                         ids=[f"{r[0]}:{sorted(r[1])}:{r[2]}" for r in REFUSE])
def test_schema_refuses(campaign, label, kv, match):
    with pytest.raises(ValueError, match=match):
        _spec(campaign, label, **kv)


@pytest.mark.parametrize("raw,match", [
    ("{", "not valid JSON"),
    ("", "not valid JSON"),
    ("[1, 2]", "must be a JSON object"),
    ("null", "must be a JSON object"),
    ('"{}"', "must be a JSON object"),
    ('{"conv_dim": 75, "conv_dim": 80}', "repeats key"),
])
def test_malformed_json_refuses(campaign, raw, match):
    with pytest.raises(ValueError, match=match):
        campaign.nas_spec("NAS-CNN", raw)


def test_nan_mlp_ratio_refuses(campaign):
    raw = json.dumps(_nas(campaign, "NAS-ViT")).replace('"mlp_ratio": 2.0', '"mlp_ratio": NaN')
    assert "NaN" in raw
    with pytest.raises(ValueError, match="mlp_ratio"):
        campaign.nas_spec("NAS-ViT", raw)


def test_nas_label_without_nas_cfg_refuses(campaign, monkeypatch):
    monkeypatch.delenv("NETT_NAS_CFG", raising=False)
    for label in NAS:
        with pytest.raises(ValueError, match="unset"):
            campaign.resolve_spec(label)


@pytest.mark.parametrize("raw", ['{"features_dim": 512}', ""])
def test_nas_cfg_beside_a_registered_label_refuses(campaign, monkeypatch, raw):
    monkeypatch.setenv("NETT_NAS_CFG", raw)
    for model in ("CNN", "CNN2F", "3DCNN", "ViT2F", "ViViT"):
        with pytest.raises(ValueError, match="not a NAS label"):
            campaign.resolve_spec(model)


def test_resolve_spec_reads_the_env(campaign, monkeypatch):
    monkeypatch.setenv("NETT_NAS_CFG", json.dumps(_nas(campaign, "NAS-3DCNN", conv_dim=26)))
    spec = campaign.resolve_spec("NAS-3DCNN")
    assert spec["cfg"] == {**campaign.MODELS["3DCNN"]["cfg"], "conv_dim": 26,
                           "stem_dim": 32, "mid_dim": 64, "stem_kernel_hw": 3}
    assert _resolved(spec) == _resolved(campaign.MODELS["3DCNN-250K"])


@pytest.mark.parametrize("model,raw", [
    ("NAS-CNN", None), ("CNN2F", '{"features_dim": 512}'), ("NAS-ViT", '{"pool": "cls"}')])
def test_main_refuses_before_anything_runs(campaign, monkeypatch, model, raw):
    """The refusal is in main()'s path, before Kit: resolve_spec is the line after the label check."""
    monkeypatch.setenv("NETT_MODEL", model)
    monkeypatch.setenv("NETT_EXPERIMENT", "binding")
    if raw is None:
        monkeypatch.delenv("NETT_NAS_CFG", raising=False)
    else:
        monkeypatch.setenv("NETT_NAS_CFG", raw)
    with pytest.raises(ValueError):
        campaign.main()


def test_main_rejects_an_unknown_label(campaign, monkeypatch):
    monkeypatch.setenv("NETT_MODEL", "NAS-Mixer")
    assert campaign.main() == 2


def test_canonical_cfg_is_one_sorted_token(campaign):
    raw = '{"framestack": true, "spatial_pool": true, "conv_dim": 75, "features_dim": 512}'
    assert campaign.nas_canonical(raw) == \
        '{"conv_dim":75,"features_dim":512,"framestack":true,"spatial_pool":true}'


# ---------------------------------------------------------------- the build-time line

def _build_models(campaign, spec):
    from dataclasses import dataclass

    from nett_skrl.brain.models import ModelCfg
    from nett_skrl.brain.models.builder import build_models_for_algorithm

    @dataclass(frozen=True)
    class _Spec:
        actor_type = "gaussian"
        critic_type = "value"
        model_keys = ("policy", "value")

    channels = 3 * (campaign._FRAMESTACK_N if spec["framestack"] else 1)
    obs = gym.spaces.Box(low=0, high=255, shape=(channels, H, W), dtype=np.uint8)
    act = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    torch.manual_seed(0)
    return build_models_for_algorithm(
        _Spec(), encoder_cls=encoder_mapping[spec["encoder"]], encoder_kwargs=_kwargs(spec),
        observation_space=obs, action_space=act, device="cpu",
        cfg=ModelCfg(shared_encoder=True, hidden_sizes=[]))


def test_builder_prints_the_nas_line_with_the_exact_encoder_count(campaign, monkeypatch, capsys):
    raw = json.dumps(_nas(campaign, "NAS-3DCNN", stem_dim=160, mid_dim=640))
    monkeypatch.setenv("NETT_MODEL", "NAS-3DCNN")
    monkeypatch.setenv("NETT_NAS_CFG", raw)
    models = _build_models(campaign, campaign.resolve_spec("NAS-3DCNN"))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[NETT NAS]")]
    assert lines == [f"[NETT NAS] label=NAS-3DCNN cfg={campaign.nas_canonical(raw)} "
                     "encoder_params=2005933"]
    assert _params(models["policy"].encoder) == 2_005_933
    assert models["policy"].encoder is models["value"].encoder


def test_builder_is_silent_without_nas_cfg(campaign, monkeypatch, capsys):
    monkeypatch.delenv("NETT_NAS_CFG", raising=False)
    monkeypatch.setenv("NETT_MODEL", "3DCNN")
    _build_models(campaign, campaign.MODELS["3DCNN"])
    assert "[NETT NAS]" not in capsys.readouterr().out
