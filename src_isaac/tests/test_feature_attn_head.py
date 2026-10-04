"""Feature-token attention head (brain/models/utils/feature_attn.py; owner request 2026-10-04).

Unset must be byte-identical to the MLP trunk; set, the head must tokenize every feature
coordinate (or group) with its own embedding and attend across them.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.models import GaussianActor, ModelCfg, ValueCritic
from nett_skrl.brain.models.utils.feature_attn import (
    FEATURE_ATTN_DEFAULTS,
    FeatureTokenAttention,
    feature_attn_trunk,
    resolve_feature_attn,
)
from nett_skrl.brain.models.utils.init import init_output, orthogonal_init
from nett_skrl.brain.models.utils.mlp import mlp_trunk

F = 16


class _TinyEncoder(nn.Module):
    def __init__(self, observation_space, **kw):
        super().__init__()
        self.features_dim = F
        self.net = nn.Linear(int(np.prod(observation_space.shape)), F)

    def forward(self, x):
        return self.net(x.float().flatten(1))


_OBS = gym.spaces.Box(low=0, high=255, shape=(4,), dtype=np.float32)
_ACT = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _actor(cfg, cls=GaussianActor):
    return cls(encoder_cls=_TinyEncoder, encoder_kwargs={}, observation_space=_OBS,
               action_space=_ACT, device="cpu", cfg=cfg)


# ── unset = unchanged ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("hidden", [[], [8], [8, 8, 8, 8]])
def test_unset_builds_the_old_mlp_trunk_byte_for_byte(hidden):
    """The pre-change _build_backbone, inlined: encoder, orthogonal, mlp_trunk, orthogonal, head."""
    cfg = ModelCfg(hidden_sizes=list(hidden))
    assert cfg.feature_attn is None
    torch.manual_seed(7)
    got = _actor(cfg)
    torch.manual_seed(7)
    enc = _TinyEncoder(_OBS)
    enc.apply(lambda m: orthogonal_init(m, cfg.hidden_gain))
    trunk, last = mlp_trunk(F, list(hidden), cfg.activation)
    trunk.apply(lambda m: orthogonal_init(m, cfg.hidden_gain))
    head = nn.Linear(last, 2)
    init_output(head, cfg)
    want = {**{f"encoder.{k}": v for k, v in enc.state_dict().items()},
            **{f"trunk.{k}": v for k, v in trunk.state_dict().items()},
            **{f"mean_layer.{k}": v for k, v in head.state_dict().items()}}
    sd = {k: v for k, v in got.state_dict().items() if k != "log_std"}
    assert sd.keys() == want.keys()
    for k in want:
        assert torch.equal(sd[k], want[k]), k
    assert isinstance(got.trunk, nn.Sequential)


def test_campaign_driver_adds_no_model_key_when_unset():
    """No `feature_attn` key at all in the model cfg or campaign_timing.json when the env is unset."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py").read_text()
    assert '**({"feature_attn": feature_attn} if feature_attn is not None else {}),' in src
    assert src.count('if feature_attn is not None else {})') == 2


# ── the head itself ───────────────────────────────────────────────────────────
def test_resolve_fills_defaults_and_refuses_unknowns():
    assert resolve_feature_attn({}) == FEATURE_ATTN_DEFAULTS
    assert resolve_feature_attn({"group": 8})["group"] == 8
    with pytest.raises(ValueError, match="unknown key"):
        resolve_feature_attn({"depth": 2})
    with pytest.raises(ValueError, match="positive integer"):
        resolve_feature_attn({"heads": 0})
    with pytest.raises(ValueError, match="positive integer"):
        resolve_feature_attn({"dim": 2.5})
    with pytest.raises(ValueError, match="not divisible by heads"):
        resolve_feature_attn({"dim": 10, "heads": 4})


@pytest.mark.parametrize("group,tokens", [(1, 16), (4, 4), (16, 1)])
def test_token_count_and_output_width(group, tokens):
    m, last = feature_attn_trunk(F, {"dim": 8, "heads": 2, "group": group})
    assert m.num_tokens == tokens and last == 8
    out = m(torch.randn(5, F))
    assert out.shape == (5, 8) and torch.isfinite(out).all()


def test_group_must_divide_features():
    with pytest.raises(ValueError, match="not divisible by group"):
        feature_attn_trunk(F, {"group": 3})


def test_each_position_has_its_own_embedding():
    """Token i depends on coordinate i alone, through W_i, b_i (no shared projection)."""
    m = FeatureTokenAttention(F, dim=8, heads=2, blocks=1, group=1, mlp_ratio=2.0)
    f = torch.zeros(1, F)
    f[0, 3] = 1.0
    t = m.tokens(f)
    assert torch.allclose(t[0, 3], m.tok_weight[3, 0] + m.tok_bias[3])
    for i in range(F):
        if i != 3:
            assert torch.allclose(t[0, i], m.tok_bias[i])


def test_attention_mixes_positions():
    """With attention, perturbing one coordinate moves the pooled output through every token."""
    torch.manual_seed(0)
    m = FeatureTokenAttention(F, dim=8, heads=2, blocks=1, group=1, mlp_ratio=2.0)
    f = torch.randn(1, F, requires_grad=True)
    m(f).sum().backward()
    assert (f.grad.abs() > 0).all()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_init_is_the_modules_own_not_orthogonal():
    torch.manual_seed(1)
    a = _actor(ModelCfg(hidden_sizes=[], feature_attn={"dim": 8, "heads": 2}))
    torch.manual_seed(1)
    enc = _TinyEncoder(_OBS)               # consume the encoder's draws in the same order
    enc.apply(lambda m: orthogonal_init(m, ModelCfg().hidden_gain))
    ref = FeatureTokenAttention(F, dim=8, heads=2, blocks=1, group=1, mlp_ratio=2.0)
    for (k, v), (k2, v2) in zip(a.trunk.state_dict().items(), ref.state_dict().items()):
        assert k == k2
        assert torch.equal(v, v2), k


# ── through the actor and critic ──────────────────────────────────────────────
@pytest.mark.parametrize("cls", [GaussianActor, ValueCritic])
def test_actor_and_critic_take_the_attention_trunk(cls, capsys):
    m = _actor(ModelCfg(hidden_sizes=[], feature_attn={"dim": 8, "heads": 2, "group": 2}), cls=cls)
    assert isinstance(m.trunk, FeatureTokenAttention)
    assert "[NETT head] feature_attn tokens=8 group=2 dim=8 blocks=1" in capsys.readouterr().out
    out, _ = m.compute({"observations": torch.rand(3, 4)})
    assert out.shape[0] == 3 and torch.isfinite(out).all()


def test_hidden_sizes_with_attention_refuses():
    with pytest.raises(ValueError, match="hidden_sizes must be"):
        _actor(ModelCfg(hidden_sizes=[8], feature_attn={}))


def test_schema_accepts_the_key_and_rejects_typos():
    import jsonschema
    from nett_skrl.nett import _load_schema
    s = _load_schema()["properties"]["brain"]["properties"]["model"]
    jsonschema.validate({"feature_attn": None}, s)
    jsonschema.validate({"feature_attn": resolve_feature_attn({})}, s)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"feature_attn": {"depth": 1}}, s)


# ── NETT_FEATURE_ATTN (campaign driver) ───────────────────────────────────────
def _campaign(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train
    return campaign_train


@pytest.mark.parametrize("val,want", [
    (None, None), ("", None),
    ("1", FEATURE_ATTN_DEFAULTS),
    ("group=8", {**FEATURE_ATTN_DEFAULTS, "group": 8}),
    ("dim=32, heads=2,blocks=2,mlp_ratio=4", {**FEATURE_ATTN_DEFAULTS, "dim": 32, "heads": 2, "blocks": 2, "mlp_ratio": 4.0}),
])
def test_feature_attn_from_env(monkeypatch, val, want):
    ct = _campaign(monkeypatch)
    if val is None:
        monkeypatch.delenv("NETT_FEATURE_ATTN", raising=False)
    else:
        monkeypatch.setenv("NETT_FEATURE_ATTN", val)
    assert ct.feature_attn_from_env([]) == want


@pytest.mark.parametrize("val,msg", [
    ("yes", "key=value"), ("dim", "key=value"), ("dim=", "key=value"), ("dim=x", "not a number"),
    ("dim=8,dim=8", "given twice"), ("depth=2", "unknown key"), ("heads=0", "positive integer"),
])
def test_feature_attn_from_env_refuses(monkeypatch, val, msg):
    ct = _campaign(monkeypatch)
    monkeypatch.setenv("NETT_FEATURE_ATTN", val)
    with pytest.raises(ValueError, match=msg):
        ct.feature_attn_from_env([])


def test_feature_attn_with_hidden_sizes_refuses(monkeypatch):
    ct = _campaign(monkeypatch)
    monkeypatch.setenv("NETT_FEATURE_ATTN", "1")
    with pytest.raises(ValueError, match="unset NETT_HIDDEN_SIZES"):
        ct.feature_attn_from_env([256])
