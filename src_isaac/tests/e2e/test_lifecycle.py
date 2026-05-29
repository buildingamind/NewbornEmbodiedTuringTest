"""Subprocess-lifecycle regression tests.

The subprocess-per-mode refactor in ``task_runner`` is load-bearing for
two reasons: it bypasses Kit's fragile teardown SIGSEGV, and it keeps the
parent worker free of Isaac imports so a second mode's ``SimulationContext``
can be created cleanly. These tests lock in both guarantees without
spinning up real Isaac Sim — the four tests run in ~1s combined.

Marked ``e2e_isaac`` so they live alongside the real-Isaac tests; they
also depend on the NETT-skrl + nett_isaac modules being importable.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from pathlib import Path

import pytest

from .conftest import DESIGN_SHEET_MINIMAL, MEDIA_ROOT


pytestmark = pytest.mark.e2e_isaac


# ---------------------------------------------------------------------------
# 1) Parent worker stays Isaac-free
# ---------------------------------------------------------------------------


def _minimal_task(tmp_path: Path):
    """Build a Task that the parent can construct without importing Isaac."""
    from nett_skrl.brain import Brain
    from nett_skrl.environment import Environment
    from nett_skrl.runtime.task import Task

    env = Environment(
        design_sheet=str(DESIGN_SHEET_MINIMAL),
        media_root=str(MEDIA_ROOT),
        conditions=["Object1"],
        headless=True,
        binocular_vision=True,
        input_resolution=64,
        reward_types=["closeness"],
    )
    brain = Brain(
        algorithm="PPO",
        encoder="small",
        batch_size=32,
        buffer_size=64,
        wandb={"mode": "disabled"},
    )
    return Task(
        brain=brain,
        wrappers=(),
        env=env,
        condition="Object1",
        output_dir=tmp_path,
        modes=["train"],
        episodes={"train": 2},
        memory=6,
        num_brains=1,
    )


def test_run_task_does_not_import_isaac_modules_in_parent(monkeypatch, tmp_path):
    """``run_task`` in the parent must not pull in ``isaaclab*`` / ``omni*``.

    The actual Isaac work happens inside a spawn()ed subprocess; the parent
    only orchestrates. We neuter the spawn step to keep the test fast and
    assert that no Isaac module name appears in ``sys.modules`` afterwards.
    """
    from nett_skrl.runtime import task_runner

    monkeypatch.setattr(
        task_runner,
        "_spawn_mode_subprocess",
        lambda task, mode, **overrides: None,
    )

    before = set(sys.modules)
    task = _minimal_task(tmp_path)

    task_runner.run_task(task)

    new = set(sys.modules) - before
    isaac_new = {m for m in new if m.startswith(("isaaclab", "omni"))}
    assert not isaac_new, f"parent imported Isaac modules: {sorted(isaac_new)}"


# ---------------------------------------------------------------------------
# 2) Bypass Kit teardown via late-registered os._exit atexit hook
# ---------------------------------------------------------------------------


def test_exit_worker_cleanly_registers_os_exit_atexit(monkeypatch):
    """``_exit_worker_cleanly`` must register ``os._exit(0)`` as an atexit.

    Python runs atexit hooks LIFO; registering ``os._exit`` last means it
    fires first, bypassing whatever fragile teardown Kit registered earlier
    in the same process. If a future refactor stops registering it, Isaac
    Sim's teardown SIGSEGV will surface as a worker-level error again.
    """
    from nett_skrl.runtime import task_runner

    calls: list[tuple] = []
    monkeypatch.setattr(atexit, "register", lambda *args, **kw: calls.append((args, kw)))

    task_runner._exit_worker_cleanly(logging.getLogger("test_lifecycle"))

    matches = [args for args, _ in calls if args and args[0] is os._exit and args[1:] == (0,)]
    assert matches, f"expected atexit.register(os._exit, 0); got {calls!r}"


def test_exit_worker_cleanly_finishes_wandb_runs_before_exit(monkeypatch):
    """All wandb runs must be ``finish()``ed before the worker exits.

    With wandb's reinit='create_new', N per-brain runs are alive simultaneously
    inside one worker; if they are not finished before ``os._exit(0)``, the
    online-mode upload is truncated. The loop in ``_exit_worker_cleanly`` is
    the only place this finish happens — this test makes sure it does.
    """
    import wandb as wandb_mod
    from nett_skrl.runtime import task_runner

    state = {"finished": 0}

    def fake_finish() -> None:
        state["finished"] += 1
        if state["finished"] >= 3:
            wandb_mod.run = None

    monkeypatch.setattr(wandb_mod, "run", object(), raising=False)
    monkeypatch.setattr(wandb_mod, "finish", fake_finish)
    # Do not actually os._exit during the test process.
    monkeypatch.setattr(atexit, "register", lambda *a, **kw: None)

    task_runner._exit_worker_cleanly(logging.getLogger("test_lifecycle"))

    assert state["finished"] == 3, f"wandb.finish called {state['finished']} times; expected 3"


# ---------------------------------------------------------------------------
# 3) Subprocess exit handling
# ---------------------------------------------------------------------------


class _FakeProcess:
    """Mock of ``multiprocessing.Process`` that exits with a fixed code."""

    def __init__(self, target=None, args=(), daemon=False, exitcode: int = 7) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon
        self.exitcode = exitcode

    def start(self) -> None:
        pass

    def join(self) -> None:
        pass


def test_spawn_mode_subprocess_warns_on_teardown_signal(monkeypatch, caplog, tmp_path):
    """Known Isaac teardown signals warn so flushed artifacts remain usable."""
    from nett_skrl.runtime import task_runner

    class _FakeCtx:
        @staticmethod
        def Process(target=None, args=(), daemon=False, **_kw):
            return _FakeProcess(target=target, args=args, daemon=daemon, exitcode=-11)

    monkeypatch.setattr("multiprocessing.get_context", lambda mode: _FakeCtx())

    task = _minimal_task(tmp_path)

    with caplog.at_level(logging.WARNING):
        task_runner._spawn_mode_subprocess(task, "train")

    assert any("exited with code -11" in rec.message for rec in caplog.records), \
        f"no warning logged about exit code -11; got: {[r.message for r in caplog.records]}"


def test_spawn_mode_subprocess_raises_on_python_failure(monkeypatch, tmp_path):
    """Ordinary Python failures must fail the run instead of looking successful."""
    from nett_skrl.runtime import task_runner

    class _FakeCtx:
        @staticmethod
        def Process(target=None, args=(), daemon=False, **_kw):
            return _FakeProcess(target=target, args=args, daemon=daemon, exitcode=1)

    monkeypatch.setattr("multiprocessing.get_context", lambda mode: _FakeCtx())

    task = _minimal_task(tmp_path)
    with pytest.raises(RuntimeError, match="failed with exit code 1"):
        task_runner._spawn_mode_subprocess(task, "train")
