"""NETT_ENC_LR_SCALE (nett_skrl/brain/aux/enc_lr.py): the encoder + aux head in ONE param group at lr x scale.

LRP2 (workspace notes/researcher2/lrp2): add. 327's "2x LR" doubled the encoder step and the heads' step together;
this knob separates them. Production path throughout: Brain -> build_agents -> AuxLossPPO(cltt_ref), shared encoder.
"""
from __future__ import annotations

import pytest
import torch

from nett_skrl.brain.aux import enc_lr
from nett_skrl.brain.aux import ppo_aux
from nett_skrl.brain.aux.enc_lr import ENC_LR_KEY, encoder_parameters

from test_vit_fp16 import _agent, _rollout_and_update

BASE = 1e-3   # _agent's algorithm_cfg learning_rate


def _ids(params):
    return [id(p) for p in params]


def _head_ids(agent):
    return _ids(agent._aux.head.parameters())


def _all_model_ids(agent):
    seen, out = set(), []
    for m in (agent.policy, agent.value):
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(id(p))
    return out


# ── unset: the param groups are exactly today's ──────────────────────────────────────────────────

def test_unset_groups_are_unchanged(monkeypatch):
    monkeypatch.delenv("NETT_ENC_LR_SCALE", raising=False)
    agent = _agent(monkeypatch)
    groups = agent.optimizer.param_groups
    assert len(groups) == 2
    assert _ids(groups[0]["params"]) == _all_model_ids(agent)      # skrl's chain, deduped
    assert _ids(groups[1]["params"]) == _head_ids(agent)           # the aux head group, added last
    assert [g["lr"] for g in groups] == [BASE, BASE]
    assert agent._enc_lr_scale is None


@pytest.mark.parametrize("empty", ["", "  "])
def test_empty_value_is_unset(monkeypatch, empty):
    monkeypatch.setenv("NETT_ENC_LR_SCALE", empty)
    assert enc_lr.enc_lr_scale() is None


# ── set: an exact partition at two lrs ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("scale", [0.5, 2.0, 4.0])
def test_set_partitions_exactly(monkeypatch, scale, capsys):
    monkeypatch.setenv("NETT_ENC_LR_SCALE", str(scale))
    agent = _agent(monkeypatch)
    assert agent.policy.encoder is agent.value.encoder               # the shared encoder, counted once below
    enc = _ids(encoder_parameters(agent))
    assert len(enc) == len(set(enc)) == len(list(agent.policy.encoder.parameters()))
    groups = agent.optimizer.param_groups
    assert len(groups) == 2
    g0, g1 = _ids(groups[0]["params"]), _ids(groups[1]["params"])
    assert g1 == _head_ids(agent) + enc
    assert g0 == [i for i in _all_model_ids(agent) if i not in set(enc)]
    flat = g0 + g1
    assert len(flat) == len(set(flat))                                # none duplicated
    assert set(flat) == set(_all_model_ids(agent)) | set(_head_ids(agent))   # none lost
    assert groups[0]["lr"] == BASE and groups[1]["lr"] == pytest.approx(BASE * scale)
    out = capsys.readouterr().out
    assert f"[NETT lr] groups heads={BASE:.6g} encoder+aux={BASE * scale:.6g}" in out


def test_scale_one_trains_identically_to_unset(monkeypatch):
    """Scale 1 moves tensors between groups and changes nothing else: same Adam math per tensor, same clip set."""
    monkeypatch.delenv("NETT_ENC_LR_SCALE", raising=False)
    a = _agent(monkeypatch, seed=3)
    _rollout_and_update(a)
    monkeypatch.setenv("NETT_ENC_LR_SCALE", "1")
    b = _agent(monkeypatch, seed=3)
    _rollout_and_update(b)
    sa, sb = a.policy.state_dict(), b.policy.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    for pa, pb in zip(a._aux.head.parameters(), b._aux.head.parameters()):
        assert torch.equal(pa, pb)


def test_scale_moves_only_the_encoder_and_head_step(monkeypatch):
    """At scale 2 the heads' first step equals the unset run's; the encoder's does not."""
    monkeypatch.delenv("NETT_ENC_LR_SCALE", raising=False)
    a = _agent(monkeypatch, seed=5)
    p0 = {k: v.clone() for k, v in a.policy.state_dict().items()}
    _rollout_and_update(a)
    monkeypatch.setenv("NETT_ENC_LR_SCALE", "2")
    b = _agent(monkeypatch, seed=5)
    assert all(torch.equal(p0[k], v) for k, v in b.policy.state_dict().items())
    _rollout_and_update(b)
    sa, sb = a.policy.state_dict(), b.policy.state_dict()
    enc_keys = [k for k in sa if k.startswith("encoder.")]
    assert enc_keys and any(not torch.equal(sa[k], sb[k]) for k in enc_keys)
    moved_a = sum(float((sa[k] - p0[k]).abs().sum()) for k in enc_keys)
    moved_b = sum(float((sb[k] - p0[k]).abs().sum()) for k in enc_keys)
    assert moved_b > 1.5 * moved_a                                    # Adam: step ~ lr, so ~2x


def test_each_group_anneals_from_its_own_base_and_is_logged(monkeypatch):
    monkeypatch.setenv("NETT_ENC_LR_SCALE", "4")
    monkeypatch.setenv("NETT_DIAG_LR_ANNEAL", "linear")
    agent = _agent(monkeypatch)
    _rollout_and_update(agent)                                        # update(timestep=16, timesteps=100)
    g = agent.optimizer.param_groups
    assert g[0]["lr"] == pytest.approx(BASE * 0.84)
    assert g[1]["lr"] == pytest.approx(BASE * 4 * 0.84)
    assert agent.tracking_data[ENC_LR_KEY][-1] == pytest.approx(BASE * 4 * 0.84)


def test_unset_logs_no_encoder_lr(monkeypatch):
    monkeypatch.delenv("NETT_ENC_LR_SCALE", raising=False)
    agent = _agent(monkeypatch)
    _rollout_and_update(agent)
    assert ENC_LR_KEY not in agent.tracking_data


# ── refusals ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "x2"])
def test_bad_values_refused(monkeypatch, bad):
    monkeypatch.setenv("NETT_ENC_LR_SCALE", bad)
    with pytest.raises(ValueError, match="NETT_ENC_LR_SCALE"):
        enc_lr.enc_lr_scale()


def test_no_aux_agent_refused(monkeypatch):
    monkeypatch.delenv("NETT_AUX_LOSS", raising=False)
    monkeypatch.setenv("NETT_ENC_LR_SCALE", "2")
    with pytest.raises(ValueError, match="applied only by AuxLossPPO"):
        _agent(monkeypatch, aux=None)


def test_resumed_optimizer_refused(monkeypatch):
    monkeypatch.setenv("NETT_ENC_LR_SCALE", "2")
    agent = _agent(monkeypatch)
    p = agent.optimizer.param_groups[0]["params"][0]
    agent.optimizer.state[p] = {"step": torch.tensor(3.0)}            # what agent.load leaves behind
    with pytest.raises(ValueError, match="resumed optimizer"):
        _rollout_and_update(agent)


def test_frozen_encoder_refused(monkeypatch):
    monkeypatch.setenv("NETT_ENC_LR_SCALE", "2")
    monkeypatch.setenv("NETT_FREEZE_ENCODER", "1")
    with pytest.raises(ValueError, match="no trainable parameter"):
        _agent(monkeypatch)


def test_duplicated_encoder_refused(monkeypatch):
    monkeypatch.setenv("NETT_ENC_LR_SCALE", "2")
    monkeypatch.setenv("NETT_SHARED_ENCODER_DUP", "1")
    with pytest.raises(ValueError, match="duplicated encoder"):
        _agent(monkeypatch)


def test_term_owning_its_optimizer_refused(monkeypatch):
    real = ppo_aux.AUX_LOSSES["cltt_ref"]

    def owning(encoder):
        term = real(encoder)
        term.owns_optimizer = True
        return term
    monkeypatch.setitem(ppo_aux.AUX_LOSSES, "cltt_ref", owning)
    monkeypatch.setenv("NETT_ENC_LR_SCALE", "2")
    with pytest.raises(ValueError, match="owns its optimizer"):
        _agent(monkeypatch)
