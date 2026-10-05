"""The Unity-recipe schedules reach an aux-loss agent (AuxLossPPO.update), and the factory guard
refuses every recipe knob the chosen agent class would not apply (owner request 2026-10-05:
ViT-CLTT-Ref on the Unity recipe).

Before this change AuxLossPPO applied none of them: NETT_ADAM_EPS / NETT_DIAG_LR_ANNEAL were
refused, and NETT_DIAG_ENT_START was silently skipped (it was not on the guard's list).
"""

from __future__ import annotations

import inspect

import pytest
import torch

from nett_skrl.brain import agent_factory
from nett_skrl.brain.agent_factory import build_agents
from nett_skrl.brain.aux import AuxLossPPO
from nett_skrl.brain.brain import Brain
from nett_skrl.brain.ppo_metrics import MetricsPPO

from test_brain import _FakeSkrlEnv, _tiny_algorithm_cfg

_KNOBS = ("NETT_ADAM_EPS", "NETT_ADV_NORM", "NETT_DIAG_LR_ANNEAL", "NETT_DIAG_ENT_START")


class _Built(Exception):
    """Raised in place of construction: reaching it means the guard let the build through."""


@pytest.fixture
def clean_env(monkeypatch):
    for k in _KNOBS + ("NETT_AUX_LOSS", "NETT_AUX_WEIGHT"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _build(monkeypatch, *, aux, algorithm="PPO"):
    """Run build_agents up to the agent constructor; return the class it would build."""
    picked = {}
    if aux:
        monkeypatch.setattr(agent_factory, "_aux_loss_settings", lambda: ("simclr", 1.0))

        class _StubAux(AuxLossPPO):
            def __init__(self, *a, **k):
                picked["cls"] = AuxLossPPO
                raise _Built
        monkeypatch.setattr(agent_factory, "AuxLossPPO", _StubAux)
    else:
        import nett_skrl.brain.ppo_metrics as pm

        class _StubM(MetricsPPO):
            def __init__(self, *a, **k):
                picked["cls"] = MetricsPPO
                raise _Built
        monkeypatch.setattr(pm, "MetricsPPO", _StubM)
    brain = Brain(algorithm=algorithm, algorithm_cfg=_tiny_algorithm_cfg(algorithm))
    with pytest.raises(_Built):
        build_agents(brain, _FakeSkrlEnv(), torch.device("cpu"))
    return picked["cls"]


# ── AuxLossPPO accepts the schedules ──────────────────────────────────────────
@pytest.mark.parametrize("env", [
    {"NETT_ADAM_EPS": "1e-8"},
    {"NETT_DIAG_LR_ANNEAL": "linear"},
    {"NETT_DIAG_ENT_START": "0.15"},
    {"NETT_ADV_NORM": "rollout"},
    {"NETT_ADAM_EPS": "1e-8", "NETT_DIAG_LR_ANNEAL": "linear", "NETT_DIAG_ENT_START": "0.15",
     "NETT_ADV_NORM": "rollout"},                                   # the UnityRecipe env's four
])
def test_aux_agent_builds_with_the_recipe_schedules(clean_env, env):
    for k, v in env.items():
        clean_env.setenv(k, v)
    assert _build(clean_env, aux=True) is AuxLossPPO


def test_aux_agent_refuses_minibatch_advantage_norm(clean_env):
    clean_env.setenv("NETT_ADV_NORM", "minibatch")
    with pytest.raises(ValueError, match=r"\['NETT_ADV_NORM'\] are not applied by"):
        _build(clean_env, aux=True)


# ── MetricsPPO unchanged; other agents refuse every knob, ENT_START included ──
def test_metrics_agent_still_takes_all_four(clean_env):
    for k, v in (("NETT_ADAM_EPS", "1e-8"), ("NETT_DIAG_LR_ANNEAL", "linear"),
                 ("NETT_DIAG_ENT_START", "0.15"), ("NETT_ADV_NORM", "minibatch")):
        clean_env.setenv(k, v)
    assert _build(clean_env, aux=False) is MetricsPPO


@pytest.mark.parametrize("knob,val", [("NETT_ADAM_EPS", "1e-8"), ("NETT_DIAG_LR_ANNEAL", "linear"),
                                      ("NETT_DIAG_ENT_START", "0.15"), ("NETT_ADV_NORM", "minibatch")])
def test_non_ppo_agent_refuses_each_knob(clean_env, knob, val):
    clean_env.setenv(knob, val)
    brain = Brain(algorithm="A2C", algorithm_cfg=_tiny_algorithm_cfg("A2C"))
    with pytest.raises(ValueError, match=rf"\['{knob}'\] are not applied by"):
        build_agents(brain, _FakeSkrlEnv(), torch.device("cpu"))


def test_non_ppo_agent_builds_with_no_knob(clean_env):
    brain = Brain(algorithm="A2C", algorithm_cfg=_tiny_algorithm_cfg("A2C"))
    assert build_agents(brain, _FakeSkrlEnv(), torch.device("cpu"))


# ── AuxLossPPO.update applies them first, on both paths ──────────────────────
def test_aux_update_applies_schedules_before_anything_else():
    src = inspect.getsource(AuxLossPPO.update)
    body = src[src.index("def update"):]
    call = body.index("apply_diag_schedules(self, timestep=timestep, timesteps=timesteps)")
    assert call < body.index("if self._aux is None")                 # disabled path too
    assert call < body.index("compute_gae")
    assert "minibatch_advantage_norm" not in body                    # refused, not applied


def test_schedules_mutate_the_fields_aux_update_reads():
    """AuxLossPPO's loop reads cfg.entropy_loss_scale and steps self.optimizer, the two things
    apply_diag_schedules writes, so the schedule is live there and not a dead write."""
    src = inspect.getsource(AuxLossPPO.update)
    assert "self.cfg.entropy_loss_scale" in src
    assert "self.scaler.step(self.optimizer)" in src
