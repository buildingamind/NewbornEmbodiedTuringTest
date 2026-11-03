"""BL2 EMBDIV labels (owner 2026-11-03: "Both losses. Both λ. Both BL2 backbones on both Base and Video").

  <trunk>+UNIF / <trunk>+COS2 = MODELS[<trunk>] + aux embdiv_unif / embdiv_cos2, aux_weight 0.01 (lambda by
  NETT_AUX_WEIGHT_OVERRIDE), for the four BL2 trunks: Base CNN-D512P / ViT-P8-666K, Video 3DCNN / ViViT-P8-T2.

Pinned per label at 128x80: the spec is the BL2 twin plus the aux alone (encoder parameter count = the twin's); the
term builds on the trunk, its loss is the form's loss on the encoder output the heads read (the whole frame stack on
the two video trunks), finite, and its gradient reaches every encoder tensor with no parameters of its own; the
label builds through Brain -> build_agents as an AuxLossPPO at 0.01 and at the override; the per-update collapse
diagnostics are finite on every trunk.
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
sys.path.insert(0, str(_SRC / "tests"))

import campaign_train as ct  # noqa: E402
from nett_skrl.brain.aux import embdiv_aux as ed  # noqa: E402
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402
import test_embdiv_aux as base  # noqa: E402  (its autouse knob-cleaning fixture is re-exported below)

_clean_knobs = base._clean_knobs
H, W = 80, 128
TRUNKS = {"CNN-D512P": "NatureCNN", "ViT-P8-666K": "CompactViT", "3DCNN": "Compact3DCNN", "ViViT-P8-T2": "CompactViViT"}
FORMS = {"UNIF": ("embdiv_unif", "uniformity"), "COS2": ("embdiv_cos2", "cos2")}
LABELS = {f"{t}+{f}": (t, *FORMS[f]) for t in TRUNKS for f in FORMS}


def _channels(spec):
    return 3 * ct._FRAMESTACK_N if spec["framestack"] else 3


def _space(spec):
    return gym.spaces.Box(low=0, high=255, shape=(_channels(spec), H, W), dtype=np.uint8)


def _encoder(label, seed=0):
    spec = ct.MODELS[label]
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(spec), **kwargs)


def _batch(spec, n=8, seed=1):
    return torch.randint(0, 256, (n, _channels(spec), H, W), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(seed))


def test_eight_labels_exist():
    assert len(LABELS) == 8 and all(lab in ct.MODELS for lab in LABELS)


@pytest.mark.parametrize("label", LABELS)
def test_spec_is_the_bl2_twin_plus_the_aux_alone(label):
    trunk, key, form = LABELS[label]
    assert ct.MODELS[label] == {**ct.MODELS[trunk], "aux": key, "aux_weight": 0.01}
    assert ed.KEY_FORM[key] == form
    twin = sum(p.numel() for p in _encoder(trunk).parameters())
    assert sum(p.numel() for p in _encoder(label).parameters()) == twin
    assert type(_encoder(label)).__name__ == TRUNKS[trunk]
    print(f"{label}: encoder params {twin}")


@pytest.mark.parametrize("label", LABELS)
def test_term_on_the_trunk_reads_the_heads_input_and_reaches_every_encoder_tensor(label):
    trunk, key, form = LABELS[label]
    spec = ct.MODELS[label]
    enc = _encoder(label)
    term = AUX_LOSSES[key](enc)
    assert isinstance(term, ed.EmbDivTerm) and term.form == form
    assert list(term.parameters()) == []
    obs = _batch(spec, 8)
    with torch.no_grad():
        z = enc(obs)
        assert z.shape == (8, enc.features_dim if hasattr(enc, "features_dim") else z.shape[1])
        want = term.loss(z)
        got = term.compute(enc, obs)
    assert torch.allclose(got, want, atol=1e-6)
    enc.zero_grad(set_to_none=True)
    loss = term.compute(enc, obs)
    assert loss.dim() == 0 and torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0
               or not torch.isfinite(p.grad).all()]
    assert not missing, f"{label}: encoder tensors the term does not reach: {missing}"
    d = ed.embedding_diagnostics(z)
    assert all(np.isfinite(d[k]) for k in ed.DIAG_METRICS[:6]), d


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("override,want", [(None, 0.01), ("0.002", 0.002), ("0.05", 0.05)])
def test_label_builds_through_build_agents_at_its_weight_and_the_override(monkeypatch, label, override, want):
    from nett_skrl.brain.aux.ppo_aux import AuxLossPPO
    if override is not None:
        monkeypatch.setenv("NETT_AUX_WEIGHT_OVERRIDE", override)
    spec = ct.MODELS[label]
    base._export_like_main(monkeypatch, spec)

    class _Env(base._Env):
        observation_space = _space(spec)
    agent = base._production_agent(spec, env_cls=_Env)
    assert isinstance(agent, AuxLossPPO) and agent._aux_kind == spec["aux"]
    assert agent._aux_weight == pytest.approx(want)
    assert isinstance(agent._aux, ed.EmbDivTerm) and agent._aux.form == LABELS[label][2]
    assert agent._aux_encoder is agent.policy.encoder
    assert agent.optimizer.param_groups[-1]["params"] == []
