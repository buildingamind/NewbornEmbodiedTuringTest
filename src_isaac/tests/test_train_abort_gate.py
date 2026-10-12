"""TEST never runs on fresh weights; a train abort fails the arm (DECISIONS §90.3).

FINDINGS §4dh.44h.143: ten arms on three nodes were tested and published as untrained
policies. Two links let that through, and each has known answers here:

1. ``load_latest_checkpoints`` logged "using fresh weights" at INFO and carried on.
2. ``_run_mode_subprocess`` read any -6/-9/-11/134/139 from TRAIN as "Isaac teardown,
   outputs intact", so TEST was spawned after a train that never finished.

The I567 shape (12/12 brains, no checkpoint at all) and the L35-L37 shape (training
stopped part-way, no final checkpoint) are both reproduced.
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace

import pytest

from fault_injection import StubProc, stub_spawn_context
from nett_skrl.recording import (
    MissingCheckpointError,
    brains_missing_final_checkpoint,
    load_latest_checkpoints,
)
from nett_skrl.runtime import task_runner


def _touch(path, *, age_s: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pt")
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))
    return path


def _final(root, brain_id, **kw):
    return _touch(root / "wandb_runs" / f"brain_{brain_id}" / "checkpoints" / "final_agent.pt", **kw)


class _Agent:
    def __init__(self, fail: bool = False):
        self.loaded = None
        self.fail = fail

    def load(self, path):
        if self.fail:
            raise ValueError("corrupt")
        self.loaded = path


def _cfg(root, *, dry_run=False, num_brains=3):
    return SimpleNamespace(path=root, dry_run=dry_run, num_brains=num_brains,
                           name="run", condition="Object1", device=0,
                           logger=logging.getLogger("test.train_abort"))


# --- brains_missing_final_checkpoint -------------------------------------------------

def test_missing_final_known_answers(tmp_path):
    assert brains_missing_final_checkpoint(tmp_path, 3) == [1, 2, 3]
    _final(tmp_path, 2)
    assert brains_missing_final_checkpoint(tmp_path, 3) == [1, 3]
    _final(tmp_path, 1)
    _final(tmp_path, 3)
    assert brains_missing_final_checkpoint(tmp_path, 3) == []


def test_missing_final_ignores_snapshots_and_stale_finals(tmp_path):
    # A periodic agent_N.pt is loadable but is NOT evidence training finished.
    _touch(tmp_path / "wandb_runs" / "brain_1" / "checkpoints" / "agent_50000.pt")
    _final(tmp_path, 2, age_s=3600)          # left by an earlier chunk
    _final(tmp_path, 3)
    since = time.time() - 600
    assert brains_missing_final_checkpoint(tmp_path, 3, since=since) == [1, 2]
    assert brains_missing_final_checkpoint(tmp_path, 3) == [1]


# --- load_latest_checkpoints ---------------------------------------------------------

def test_test_refuses_when_any_brain_has_no_checkpoint(tmp_path):
    _final(tmp_path, 1)
    _final(tmp_path, 3)
    agents = [_Agent(), _Agent(), _Agent()]
    with pytest.raises(MissingCheckpointError) as exc:
        load_latest_checkpoints(agents, _cfg(tmp_path))
    assert "1/3" in str(exc.value) and "brain_2:" in str(exc.value)
    assert "brain_1:" not in str(exc.value) and "brain_3:" not in str(exc.value)


def test_test_refuses_the_i567_shape_naming_every_brain(tmp_path):
    agents = [_Agent() for _ in range(12)]
    with pytest.raises(MissingCheckpointError) as exc:
        load_latest_checkpoints(agents, _cfg(tmp_path, num_brains=12))
    assert "12/12" in str(exc.value)
    assert all(f"brain_{i}:" in str(exc.value) for i in range(1, 13))


def test_test_refuses_a_failed_load(tmp_path):
    for i in (1, 2):
        _final(tmp_path, i)
    with pytest.raises(MissingCheckpointError) as exc:
        load_latest_checkpoints([_Agent(), _Agent(fail=True)], _cfg(tmp_path))
    assert "brain_2: load of" in str(exc.value)


def test_test_loads_when_every_brain_is_covered(tmp_path):
    _final(tmp_path, 1)
    _touch(tmp_path / "wandb_runs" / "brain_2" / "checkpoints" / "agent_100.pt")
    agents = [_Agent(), _Agent()]
    load_latest_checkpoints(agents, _cfg(tmp_path))
    assert agents[0].loaded.endswith("final_agent.pt")
    assert agents[1].loaded.endswith("agent_100.pt")


def test_train_and_dry_run_keep_the_fresh_weight_fallback(tmp_path):
    agents = [_Agent(), _Agent()]
    load_latest_checkpoints(agents, _cfg(tmp_path), context="train")
    load_latest_checkpoints(agents, _cfg(tmp_path, dry_run=True))
    assert [a.loaded for a in agents] == [None, None]


# --- _run_mode_subprocess: a tolerated exit from TRAIN --------------------------------

class _NoopReaper:
    def __init__(self) -> None:
        self.reaped = []

    from contextlib import contextmanager as _cm

    @_cm
    def launch_scope(self):
        yield

    def adopt(self, pid):
        return None

    def reap(self, reason):
        self.reaped.append(reason)


def _spawn(monkeypatch, tmp_path, mode, exitcode, *, dry_run=False, before_join=None):
    monkeypatch.setattr(task_runner, "TaskReaper", lambda **kw: _NoopReaper())

    def _join(*a, **k):
        if before_join:
            before_join()                      # what the child wrote while it ran
        return "exited"

    monkeypatch.setattr(task_runner, "join_with_reap", _join)
    stub_spawn_context(monkeypatch, StubProc(exitcode=exitcode))
    task_runner._spawn_mode_subprocess(SimpleNamespace(config=_cfg(tmp_path, dry_run=dry_run)), mode)


@pytest.mark.parametrize("code", [-6, -9, -11, 134, 139])
def test_train_abort_with_no_checkpoint_fails_the_arm(tmp_path, monkeypatch, code):
    with pytest.raises(task_runner.TrainAbortedError) as exc:
        _spawn(monkeypatch, tmp_path, "train", code)
    assert "3/3" in str(exc.value)


def test_train_abort_part_way_fails_even_with_snapshots(tmp_path, monkeypatch):
    # The L35-L37 shape: stopped at 88-90%, periodic snapshots but no final save.
    def _write():
        for i in (1, 2, 3):
            _touch(tmp_path / "wandb_runs" / f"brain_{i}" / "checkpoints" / "agent_50000.pt")
    with pytest.raises(task_runner.TrainAbortedError):
        _spawn(monkeypatch, tmp_path, "train", -6, before_join=_write)


def test_train_abort_with_a_stale_final_from_an_earlier_chunk_fails(tmp_path, monkeypatch):
    for i in (1, 2, 3):
        _final(tmp_path, i, age_s=7200)
    with pytest.raises(task_runner.TrainAbortedError):
        _spawn(monkeypatch, tmp_path, "train", -11)


def test_train_abort_one_brain_short_fails_and_names_it(tmp_path, monkeypatch):
    with pytest.raises(task_runner.TrainAbortedError) as exc:
        _spawn(monkeypatch, tmp_path, "train", -11,
               before_join=lambda: [_final(tmp_path, i) for i in (1, 3)])
    assert "1/3" in str(exc.value) and "brain_2" in str(exc.value)


def test_teardown_crash_after_a_finished_train_still_passes(tmp_path, monkeypatch):
    _spawn(monkeypatch, tmp_path, "train", -11,
           before_join=lambda: [_final(tmp_path, i) for i in (1, 2, 3)])


def test_tolerated_exit_from_test_and_dry_run_is_not_judged_here(tmp_path, monkeypatch):
    _spawn(monkeypatch, tmp_path, "test", -11)
    _spawn(monkeypatch, tmp_path, "train", -11, dry_run=True)


def test_train_abort_is_not_a_wave_continue_casualty():
    # _task_waiter continues past ReapedTaskError only; this must fail fast.
    from nett_skrl.runtime import reap

    assert issubclass(task_runner.TrainAbortedError, RuntimeError)
    assert not issubclass(task_runner.TrainAbortedError, reap.ReapedTaskError)
