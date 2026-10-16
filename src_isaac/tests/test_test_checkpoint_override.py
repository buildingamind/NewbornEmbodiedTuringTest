"""NETT_TEST_CHECKPOINT: TEST loads a named checkpoint (owner 2026-10-16, checkpoints at 1/2 and 3/4)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from nett_skrl.recording.checkpoints import (
    MissingCheckpointError,
    brain_checkpoint_dir,
    load_latest_checkpoints,
    resolve_test_checkpoint,
)


class _Agent:
    def __init__(self):
        self.loaded = None

    def load(self, path):
        self.loaded = path


def _run(tmp_path, names, n=2):
    for b in range(1, n + 1):
        d = brain_checkpoint_dir(tmp_path, b)
        d.mkdir(parents=True)
        for name in names:
            (d / name).touch()
    return SimpleNamespace(path=str(tmp_path), dry_run=False), [_Agent() for _ in range(n)]


def test_unset_keeps_the_final_model(tmp_path, monkeypatch):
    monkeypatch.delenv("NETT_TEST_CHECKPOINT", raising=False)
    cfg, agents = _run(tmp_path, ["agent_1900.pt", "agent_3800.pt", "final_agent.pt"])
    load_latest_checkpoints(agents, cfg, context="test")
    assert all(a.loaded.endswith("final_agent.pt") for a in agents)


def test_named_checkpoint_is_loaded_by_every_brain(tmp_path, monkeypatch):
    monkeypatch.setenv("NETT_TEST_CHECKPOINT", "agent_3800.pt")
    cfg, agents = _run(tmp_path, ["agent_1900.pt", "agent_3800.pt", "final_agent.pt"])
    load_latest_checkpoints(agents, cfg, context="test")
    # positive control: the default above picked final_agent.pt from the same files
    assert [a.loaded for a in agents] == [str(brain_checkpoint_dir(tmp_path, b) / "agent_3800.pt") for b in (1, 2)]


def test_absent_named_checkpoint_refuses_rather_than_falling_back(tmp_path, monkeypatch):
    monkeypatch.setenv("NETT_TEST_CHECKPOINT", "agent_5700.pt")
    cfg, agents = _run(tmp_path, ["agent_3800.pt", "final_agent.pt"])
    with pytest.raises(MissingCheckpointError, match="agent_5700.pt"):
        load_latest_checkpoints(agents, cfg, context="test")
    assert all(a.loaded is None for a in agents)
    cfg.dry_run = True   # a dry run still refuses a NAMED file: no silent substitute
    with pytest.raises(MissingCheckpointError):
        load_latest_checkpoints(agents, cfg, context="test")


def test_train_resume_ignores_the_knob(tmp_path, monkeypatch):
    monkeypatch.setenv("NETT_TEST_CHECKPOINT", "agent_3800.pt")
    cfg, agents = _run(tmp_path, ["agent_3800.pt", "final_agent.pt"])
    load_latest_checkpoints(agents, cfg, context="train")
    assert all(a.loaded.endswith("final_agent.pt") for a in agents)


@pytest.mark.parametrize("bad", ["../final_agent.pt", "agent_3800", "agent_x.pt", "final_agent.ckpt", "/abs/agent_1.pt"])
def test_malformed_names_raise(monkeypatch, bad):
    monkeypatch.setenv("NETT_TEST_CHECKPOINT", bad)
    with pytest.raises(ValueError):
        resolve_test_checkpoint()


@pytest.mark.parametrize("good", ["final_agent.pt", "best_agent.pt", "agent_3800.pt"])
def test_wellformed_names_pass(monkeypatch, good):
    monkeypatch.setenv("NETT_TEST_CHECKPOINT", good)
    assert resolve_test_checkpoint() == good
