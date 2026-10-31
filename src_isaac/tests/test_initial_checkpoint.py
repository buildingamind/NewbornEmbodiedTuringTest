"""NETT_CHECKPOINT_INITIAL: agent_0.pt before training (owner 2026-10-30, checkpoints at 0/25/50/75%)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nett_skrl.recording.checkpoints import (
    brain_checkpoint_dir,
    initial_checkpoint_enabled,
    load_latest_checkpoints,
    pick_checkpoint,
)
from nett_skrl.recording.run_recorder import RunRecorder


class _Agent:
    """skrl's whole-agent save/load contract: a dict of state dicts."""

    def __init__(self, exp_dir, seed):
        torch.manual_seed(seed)
        self.experiment_dir = str(exp_dir)
        self.policy = torch.nn.Linear(4, 2)
        self.loaded = None

    def save(self, path):
        torch.save({"policy": self.policy.state_dict()}, path)

    def load(self, path):
        self.loaded = path
        self.policy.load_state_dict(torch.load(path)["policy"])


def _agents(tmp_path, n=2):
    return [_Agent(brain_checkpoint_dir(tmp_path, b).parent, seed=b) for b in range(1, n + 1)]


def _cfg():
    return SimpleNamespace(hparams_dir=None)


def test_knob_parsing(monkeypatch):
    monkeypatch.delenv("NETT_CHECKPOINT_INITIAL", raising=False)
    assert initial_checkpoint_enabled() is False
    monkeypatch.setenv("NETT_CHECKPOINT_INITIAL", "0")
    assert initial_checkpoint_enabled() is False
    monkeypatch.setenv("NETT_CHECKPOINT_INITIAL", "1")
    assert initial_checkpoint_enabled() is True
    monkeypatch.setenv("NETT_CHECKPOINT_INITIAL", "yes")
    with pytest.raises(ValueError, match="NETT_CHECKPOINT_INITIAL"):
        initial_checkpoint_enabled()


def test_unset_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("NETT_CHECKPOINT_INITIAL", raising=False)
    agents = _agents(tmp_path)
    RunRecorder(agents, num_envs=2).before_train(_cfg())
    assert not list(tmp_path.rglob("*.pt"))


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("NETT_CHECKPOINT_INITIAL", "1")
    agents = _agents(tmp_path)
    RunRecorder(agents, num_envs=2).before_train(_cfg(), dry_run=True)
    assert not list(tmp_path.rglob("*.pt"))


def test_writes_agent_0_per_brain_and_round_trips(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NETT_CHECKPOINT_INITIAL", "1")
    agents = _agents(tmp_path)
    before = [a.policy.weight.detach().clone() for a in agents]
    RunRecorder(agents, num_envs=2).before_train(_cfg())
    assert "[NETT checkpoints] initial agent_0.pt written for 2/2 brain(s) before training" in capsys.readouterr().out
    for b, (agent, w0) in enumerate(zip(agents, before), start=1):
        path = brain_checkpoint_dir(tmp_path, b) / "agent_0.pt"
        assert path.is_file()
        with torch.no_grad():
            agent.policy.weight.add_(1.0)   # "training" moves the weights
        agent.load(str(path))
        assert torch.equal(agent.policy.weight, w0)
    # brains are distinct: brain 2's file is not brain 1's
    w1 = torch.load(brain_checkpoint_dir(tmp_path, 1) / "agent_0.pt")["policy"]["weight"]
    w2 = torch.load(brain_checkpoint_dir(tmp_path, 2) / "agent_0.pt")["policy"]["weight"]
    assert not torch.equal(w1, w2)


def test_agent_0_is_never_a_fallback(tmp_path):
    d = brain_checkpoint_dir(tmp_path, 1)
    d.mkdir(parents=True)
    (d / "agent_0.pt").touch()
    # a train that died before its first periodic save: only agent_0 -> nothing to pick
    assert pick_checkpoint(d) is None
    (d / "agent_500.pt").touch()
    assert pick_checkpoint(d) == d / "agent_500.pt"
    (d / "final_agent.pt").touch()
    assert pick_checkpoint(d) == d / "final_agent.pt"


def test_only_agent_0_refuses_test_unless_named(tmp_path, monkeypatch):
    from nett_skrl.recording.checkpoints import MissingCheckpointError

    d = brain_checkpoint_dir(tmp_path, 1)
    d.mkdir(parents=True)
    (d / "agent_0.pt").touch()
    agent = SimpleNamespace(loaded=None, load=lambda p: setattr(agent, "loaded", p))
    cfg = SimpleNamespace(path=str(tmp_path), dry_run=False)
    monkeypatch.delenv("NETT_TEST_CHECKPOINT", raising=False)
    with pytest.raises(MissingCheckpointError):
        load_latest_checkpoints([agent], cfg, context="test")
    assert agent.loaded is None
    monkeypatch.setenv("NETT_TEST_CHECKPOINT", "agent_0.pt")   # explicit: the 0% policy
    load_latest_checkpoints([agent], cfg, context="test")
    assert agent.loaded == str(d / "agent_0.pt")
