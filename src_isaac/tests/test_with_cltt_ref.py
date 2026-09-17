"""`WithCLTTRef`: cltt_ref + one new term, and every parameter of both reaches the optimizer.

⛔ THE FAILURE THIS EXISTS FOR IS SILENT. AuxLossPPO optimizes `aux.head.parameters()` and
nothing else. A composite that exposed only one head would still train the encoder through
both losses and still log both -- while the other head's parameters stayed at their init
forever. So the claims are asserted through a REAL AuxLossPPO constructor path (PPO.__init__
stubbed with a real torch optimizer) and a real CompactViT, not against the module in isolation.
"""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.aux import ppo_aux
from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss
from nett_skrl.brain.aux.knobs import _env_flag_strict, _env_positive_float
from nett_skrl.brain.aux.ppo_aux import AuxLossPPO, PPO, track_transit_mask
from nett_skrl.brain.aux.token_features import spatial_tokens
from nett_skrl.brain.aux.with_cltt_ref import WithCLTTRef
from nett_skrl.brain.encoders.compact_vit import CompactViT

H, W, C = 80, 128, 6


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NETT_AUX_CLTT_REF_WEIGHT", "NETT_AUX_CLTT_REF_OFFSETS", "NETT_AUX_CLTT_REF_DIAG",
                 "NETT_AUX_CLTT_REF_TEMP", "NETT_AUX_CHANNELS_PER_FRAME",
                 "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_STRICT", "NETT_AUX_ALLOW_ZERO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            yield
    finally:
        torch.set_num_threads(threads)


class _Memory:
    def __init__(self, t_max=12, n_env=2):
        self.tensors = {
            "observations": torch.randint(0, 256, (t_max, n_env, H, W, C), dtype=torch.uint8),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "actions": torch.randn(t_max, n_env, 2),
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


class _Gate(nn.Module):
    """A bare Parameter inside the head, reachable only through `head` -- the case a composite
    that dropped a head would silently leave untrained."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1))


class _TokenTerm(nn.Module):
    """Minimal stand-in for a wave-17 term: memory-drawn tokens -> a trainable head -> a loss."""

    needs_memory = True

    def __init__(self, encoder):
        super().__init__()
        d = encoder.embed_dim_for_test
        self.head = nn.ModuleDict({"mlp": nn.Sequential(nn.Linear(d, 8), nn.ReLU(),
                                                        nn.Linear(8, 1)),
                                   "gate": _Gate()})
        self._memory = None
        self.last_scalars = {}
        self.last_window_turn = 0.25

    def attach_memory(self, memory):
        self._memory = memory

    def compute(self, encoder, observations):
        raw = self._memory.tensors["observations"][0:4, 0]
        with torch.no_grad():
            prepared = encoder._prepare_image(raw)
        tokens, _ = spatial_tokens(encoder, prepared)
        self.last_scalars = {"tokens": float(tokens.shape[1])}
        return (self.head["mlp"](tokens) * self.head["gate"].p).pow(2).mean()


def _encoder():
    enc = CompactViT(gym.spaces.Box(0, 255, shape=(H, W, C), dtype=np.uint8),
                     features_dim=32, patch_size=16, embed_dim=32, depth=1, num_heads=2)
    enc.embed_dim_for_test = 32
    return enc


def _agent(monkeypatch, encoder, memory, build):
    """AuxLossPPO's REAL constructor, with PPO.__init__ replaced by a real torch optimizer over
    the encoder -- the group the policy owns in a real agent."""

    def init_ppo(self, *args, **kwargs):
        self.memory = memory
        self.policy = SimpleNamespace(encoder=encoder)
        self.optimizer = torch.optim.Adam([{"params": list(encoder.parameters())}], lr=1e-3)

    monkeypatch.setattr(PPO, "__init__", init_ppo)
    monkeypatch.setitem(ppo_aux.AUX_LOSSES, "test_with_cltt_ref", build)
    return AuxLossPPO(aux_loss="test_with_cltt_ref", aux_weight=1.0)


def test_every_parameter_of_both_terms_gets_a_gradient_and_one_optimizer_slot(monkeypatch):
    enc, mem = _encoder(), _Memory()
    agent = _agent(monkeypatch, enc, mem, lambda e: WithCLTTRef(e, _TokenTerm(e), "tok"))
    aux = agent._aux
    assert aux.cltt_ref._memory is mem and aux.term._memory is mem

    ids = [id(p) for g in agent.optimizer.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter appears in two optimizer slots"
    ref_params = list(aux.cltt_ref.head.parameters())
    term_params = list(aux.term.head.parameters())
    expected = {id(p) for p in [*enc.parameters(), *ref_params, *term_params]}
    assert set(ids) == expected
    # The stand-in's extra Parameter is reachable ONLY through `head`, so this is the case a
    # head-less composite would drop.
    assert id(aux.term.head["gate"].p) in set(ids)

    loss = agent._aux_weight * aux.compute(enc, None)
    loss.backward()
    for label, params in (("cltt_ref head", ref_params), ("term head", term_params)):
        for p in params:
            assert p.grad is not None and p.grad.abs().sum() > 0, label
    # Both terms reach the trunk: the encoder gets gradient from each addend separately.
    enc.zero_grad(set_to_none=True)
    aux.term.compute(enc, None).backward()
    assert enc.patch_embed.weight.grad.abs().sum() > 0


def test_loss_is_weighted_ref_plus_term_and_scalars_are_prefixed(monkeypatch):
    monkeypatch.setenv("NETT_AUX_CLTT_REF_WEIGHT", "0.5")
    enc, mem = _encoder(), _Memory()
    comp = WithCLTTRef(enc, _TokenTerm(enc), "tok")
    comp.attach_memory(mem)
    torch.manual_seed(1)
    total = comp.compute(enc, None)
    s = comp.last_scalars
    assert float(total) == pytest.approx(0.5 * s["cltt_ref_loss"] + s["tok_loss"], rel=1e-6)
    assert s["cltt_ref_weight"] == 0.5
    assert s["tok_tokens"] == 40.0
    assert {"cltt_ref_B", "cltt_ref_t_max", "cltt_ref_chance"} <= set(s)
    assert all("/" not in k and " " not in k for k in s)
    assert comp.last_window_turn == 0.25


def test_the_cltt_ref_addend_is_the_control_objective_unchanged(monkeypatch):
    """At the default weight the reference addend must equal what the CONTROL row computes from
    the same state: same class, same knobs, same RNG position, same head weights."""
    enc, mem = _encoder(), _Memory()
    comp = WithCLTTRef(enc, _TokenTerm(enc), "tok")
    comp.attach_memory(mem)
    control = CLTTReferenceAuxLoss(enc)
    control.head.load_state_dict(comp.cltt_ref.head.state_dict())
    control.attach_memory(mem)
    torch.manual_seed(3)
    control_loss = control.compute(enc, None)
    torch.manual_seed(3)
    comp.compute(enc, None)
    assert comp.cltt_ref_weight == 1.0
    assert comp.last_scalars["cltt_ref_loss"] == float(control_loss)


def test_transit_telemetry_is_forwarded_to_the_reader(monkeypatch):
    enc, mem = _encoder(), _Memory()
    comp = WithCLTTRef(enc, _TokenTerm(enc), "tok")
    comp.attach_memory(mem)
    comp.compute(enc, None)
    tracked = {}
    track_transit_mask(SimpleNamespace(track_data=tracked.__setitem__),
                       comp.last_window_turn, comp.last_window_turn, 1)
    assert tracked == {"Loss / Aux transit state": 0.0, "Loss / Aux transit |turn|": 0.25}


class _BareTerm(nn.Module):
    needs_memory = False

    def __init__(self, head):
        super().__init__()
        self.head = head

    def compute(self, encoder, observations):
        return torch.zeros(())


def test_refuses_a_head_holding_encoder_parameters():
    enc = _encoder()
    with pytest.raises(ValueError, match="ENCODER parameters"):
        WithCLTTRef(enc, _BareTerm(nn.ModuleList([enc.patch_embed])), "bare")


def test_refuses_a_head_shared_with_cltt_ref(monkeypatch):
    enc = _encoder()
    shared = nn.Linear(4, 4)
    real_init = CLTTReferenceAuxLoss.__init__

    def init(self, encoder):
        real_init(self, encoder)
        self.head = shared
    monkeypatch.setattr(CLTTReferenceAuxLoss, "__init__", init)
    with pytest.raises(ValueError, match="shares parameters"):
        WithCLTTRef(enc, _BareTerm(shared), "bare")


def test_refuses_an_empty_head_and_bad_names():
    enc = _encoder()
    with pytest.raises(ValueError, match="no parameters"):
        WithCLTTRef(enc, _BareTerm(nn.Identity()), "bare")
    for name in ("", "Tok", "cltt_ref_extra", "a/b", "a b"):
        with pytest.raises(ValueError, match="snake_case"):
            WithCLTTRef(enc, _BareTerm(nn.Linear(2, 2)), name)


def test_memory_is_forwarded_only_to_terms_that_need_it():
    enc, mem = _encoder(), _Memory()
    comp = WithCLTTRef(enc, _BareTerm(nn.Linear(2, 2)), "bare")
    comp.attach_memory(mem)
    assert comp.cltt_ref._memory is mem
    assert not hasattr(comp.term, "_memory")


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "one", ""])
def test_weight_knob_refuses_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("NETT_AUX_CLTT_REF_WEIGHT", raw)
    with pytest.raises(ValueError, match="finite number > 0"):
        WithCLTTRef(_encoder(), _BareTerm(nn.Linear(2, 2)), "bare")


def test_strict_flag_accepts_the_fleet_spellings_and_raises_on_anything_else(monkeypatch):
    monkeypatch.delenv("NETT_TEST_FLAG", raising=False)
    assert _env_flag_strict("NETT_TEST_FLAG") is False
    assert _env_flag_strict("NETT_TEST_FLAG", default=True) is True
    for on in ("1", "true", "YES", " on "):
        monkeypatch.setenv("NETT_TEST_FLAG", on)
        assert _env_flag_strict("NETT_TEST_FLAG") is True, on
    for off in ("", "0", "False", "no", "OFF"):
        monkeypatch.setenv("NETT_TEST_FLAG", off)
        assert _env_flag_strict("NETT_TEST_FLAG", default=True) is False, off
    for bad in ("ture", "2", "enabled", "y"):
        monkeypatch.setenv("NETT_TEST_FLAG", bad)
        with pytest.raises(ValueError, match="not a boolean"):
            _env_flag_strict("NETT_TEST_FLAG")


def test_positive_float_default_and_parse(monkeypatch):
    monkeypatch.delenv("NETT_TEST_W", raising=False)
    assert _env_positive_float("NETT_TEST_W", 1.0) == 1.0
    monkeypatch.setenv("NETT_TEST_W", " 2.5 ")
    assert _env_positive_float("NETT_TEST_W", 1.0) == 2.5
