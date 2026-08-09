"""A failing mode, and a wedged teardown, must END the worker -- not hang the wave.

TWO GAPS, both structural, both closed here.

1. ``_exit_worker_cleanly`` -- whose entire job is ``atexit.register(os._exit, 0)``,
   bypassing Kit's fragile teardown -- was reached ONLY on success. Any exception fell
   through into Kit's REAL teardown, which is where this stack hangs, and the parent's
   join is unbounded (``join_with_reap(absolute_timeout=None)`` and ``NETT_REAP_TIMEOUT``
   defaults to 0). A failing mode therefore hung the whole wave instead of failing it,
   and lost its artifacts on the way: ``_finalize_env_artifacts`` was on the success path
   only. It also never produced exit 77, so ``run_nett_tolerating_stalls`` -- which
   retries ``StallError`` alone -- could not see it either.

2. The post-stepping window could not be guarded by the ordinary watchdog EVEN IF LEFT
   ARMED. ``stall_guard``'s watchdog is a daemon thread, which CPython stops scheduling
   during ``Py_FinalizeEx``, and its SIGALRM backstop is armed inside ``_trigger`` -- i.e.
   only after a detection that by then cannot happen. Measured: guard armed with a 300s
   grace, child still ``R`` at 130% CPU ten minutes later, no trigger.
   ⚠ Raising ``NETT_STALL_TIMEOUT_S`` cannot fix this. A longer grace does not make a
   stopped thread run. The timer must be armed on ENTRY to the window.

No Kit and no GPU here: these are the plumbing, exercised directly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from nett_skrl.runtime import stall_guard, task_runner
from nett_skrl.runtime.reap import (
    is_device_lost_exit,
    is_stall_exit,
    is_teardown_wedge_exit,
    is_vram_oom_exit,
    teardown_wedge_exit_code,
)

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- 1. the error path hard-exits, non-zero, after salvaging artifacts ------


class _Boom(RuntimeError):
    pass


def test_error_path_hard_exits_nonzero_and_never_returns(monkeypatch):
    """It must call ``os._exit(1)``: returning would re-enter Kit's teardown."""
    calls: dict = {}
    monkeypatch.setattr(task_runner.os, "_exit", lambda code: calls.setdefault("code", code))
    monkeypatch.setattr(task_runner, "_finalize_env_artifacts",
                        lambda loaded, logger: calls.setdefault("finalized", loaded))
    monkeypatch.setattr(task_runner.crash_guard, "disarm",
                        lambda: calls.setdefault("crash_disarmed", True))
    monkeypatch.setattr(task_runner.stall_guard, "disarm",
                        lambda: calls.setdefault("stall_disarmed", True))

    sentinel = object()

    def boom(task, mode, overrides, state):
        state["loaded"] = sentinel
        raise _Boom("mode blew up")

    monkeypatch.setattr(task_runner, "_run_single_mode_body", boom)
    monkeypatch.setattr(task_runner, "_in_spawned_worker", lambda: True)
    task_runner._run_single_mode(_FakeTask(), "train")

    assert calls["code"] == 1, (
        "os._exit(0) would report a FAILED mode as a success; the code must be non-zero"
    )
    assert calls["finalized"] is sentinel, (
        "artifacts must be salvaged on the error path -- they were lost on every failure"
    )
    assert calls["crash_disarmed"] and calls["stall_disarmed"], (
        "a guard left armed during the exit can fire and mislabel the cause"
    )


def test_error_path_exit_code_is_not_mistaken_for_any_special_cause():
    """1 must stay outside the tolerated set AND outside 75/76/77/78."""
    assert not task_runner._is_tolerated_isaac_teardown_exit(1)
    assert not is_device_lost_exit(1)
    assert not is_vram_oom_exit(1)
    assert not is_stall_exit(1)
    assert not is_teardown_wedge_exit(1)


def test_error_path_survives_a_missing_env_and_a_broken_logger(monkeypatch):
    """Salvage is best effort: nothing in the abort path may raise."""
    codes: list[int] = []
    monkeypatch.setattr(task_runner.os, "_exit", codes.append)

    def boom(task, mode, overrides, state):
        raise _Boom("failed before embed(), so state has no env")

    def bad_finalize(loaded, logger):
        raise RuntimeError("finalize itself is broken")

    monkeypatch.setattr(task_runner, "_run_single_mode_body", boom)
    monkeypatch.setattr(task_runner, "_finalize_env_artifacts", bad_finalize)
    monkeypatch.setattr(task_runner, "_in_spawned_worker", lambda: True)
    task_runner._run_single_mode(_FakeTask(logger=None), "test")
    assert codes == [1]


class _FakeTask:
    def __init__(self, logger=...):
        import logging as _logging

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.logger = (
            _logging.getLogger("test.fake") if logger is ... else logger
        )


def test_in_process_caller_gets_the_exception_instead_of_a_dead_interpreter():
    """⚠ os._exit in the MAIN process would take the caller -- pytest -- down with it.

    One red test would then vanish the whole suite with no report. The hard exit exists
    to dodge Kit's teardown in a disposable child; called in-process there is no child,
    so raising is both safer and more informative.
    """
    import multiprocessing as mp

    assert mp.current_process().name == "MainProcess"
    assert task_runner._in_spawned_worker() is False

    def boom(task, mode, overrides, state):
        raise _Boom("in-process failure")

    import unittest.mock as _mock

    with _mock.patch.object(task_runner, "_run_single_mode_body", boom):
        with pytest.raises(_Boom):
            task_runner._run_single_mode(_FakeTask(), "train")


# --- 2. the teardown backstop ----------------------------------------------


def test_teardown_backstop_arms_the_kernel_timer_on_ENTRY(monkeypatch):
    """The timer must be armed when the window OPENS, not after a detection.

    ⚠ This is the property, not an implementation detail: the detector that would
    otherwise arm it (a daemon watchdog thread) is exactly what stops running here.
    """
    armed: list = []
    monkeypatch.setattr(signal, "setitimer",
                        lambda which, seconds: armed.append((which, seconds)))
    monkeypatch.setattr(stall_guard, "_teardown_backstop_armed", False)
    monkeypatch.setenv("NETT_TEARDOWN_BUDGET_S", "120")
    monkeypatch.setenv("NETT_TEARDOWN_KERNEL_GRACE_S", "30")

    assert stall_guard.arm_teardown_backstop() is True
    assert armed == [(signal.ITIMER_REAL, 150.0)], (
        "kernel deadline must be the thread budget PLUS a grace, so the thread -- which "
        "names the cause -- gets first refusal"
    )
    # Idempotent: a second call must not stack a second timer or a second thread.
    assert stall_guard.arm_teardown_backstop() is True
    assert len(armed) == 1


def test_teardown_backstop_is_armed_by_the_clean_exit_path(monkeypatch):
    """Wiring test. If this regresses, the window silently reopens."""
    order: list[str] = []
    monkeypatch.setattr(stall_guard, "arm_teardown_backstop",
                        lambda *a, **k: order.append("arm") or True)
    monkeypatch.setattr(task_runner.crash_guard, "disarm", lambda: order.append("crash"))
    monkeypatch.setattr(task_runner.stall_guard, "disarm", lambda: order.append("stall"))
    monkeypatch.setattr(task_runner.atexit, "register", lambda *a, **k: order.append("atexit"))
    import logging

    task_runner._exit_worker_cleanly(logging.getLogger("test.exit"))
    assert order[0] == "arm", (
        f"the backstop must be armed BEFORE the disarms open the window; got {order}"
    )
    assert "atexit" in order


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux signals")
def test_teardown_backstop_really_exits_a_wedged_teardown():
    """End to end, in a real process: a main thread stuck in a C call still dies.

    ``time.sleep`` releases the GIL, which is the measured signature of this hang
    (``R`` at ~130% CPU inside native Kit code). The watchdog thread must reclaim it
    with the attributable exit code rather than leaving it to the kernel timer.
    """
    prog = textwrap.dedent(
        f"""
        import os, sys, time
        sys.path.insert(0, {SRC!r})
        os.environ["NETT_TEARDOWN_BUDGET_S"] = "3"
        os.environ["NETT_TEARDOWN_KERNEL_GRACE_S"] = "60"
        from nett_skrl.runtime import stall_guard
        stall_guard.arm_teardown_backstop()
        time.sleep(300)              # stands in for a wedged Kit shutdown
        """
    )
    started = time.time()
    proc = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                          text=True, timeout=90)
    elapsed = time.time() - started
    assert proc.returncode == teardown_wedge_exit_code(), proc.stderr[-2000:]
    assert elapsed < 45, f"took {elapsed:.0f}s; the thread deadline did not fire first"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux signals")
def test_kernel_timer_bounds_it_even_with_the_watchdog_thread_gone():
    """The un-blockable half. Kill the thread's ability to run; the kernel still fires.

    Simulated by disabling the thread deadline outright (a huge budget) while leaving a
    short kernel grace -- the situation once CPython stops scheduling daemon threads.
    Yields -14, which the parent reads conservatively as DEVICE_LOST: bounded, and
    accepted as the price of an escape nothing in userspace can block.
    """
    prog = textwrap.dedent(
        f"""
        import os, sys, time
        sys.path.insert(0, {SRC!r})
        os.environ["NETT_TEARDOWN_BUDGET_S"] = "600"
        os.environ["NETT_TEARDOWN_KERNEL_GRACE_S"] = "-597"   # kernel fires at 3s
        from nett_skrl.runtime import stall_guard
        stall_guard.arm_teardown_backstop()
        time.sleep(300)
        """
    )
    proc = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                          text=True, timeout=90)
    assert proc.returncode == -signal.SIGALRM, proc.stderr[-2000:]


# --- 3. the parent's reading of exit 78 -------------------------------------


def test_teardown_wedge_is_a_success_with_a_reap_not_a_casualty(monkeypatch):
    """The mode's work COMPLETED; only shutdown hung. Reap, warn, continue."""
    reaped: list[str] = []

    class _Reaper:
        task_key = "t"

        def reap(self, reason):
            reaped.append(reason)

    class _Proc:
        exitcode = teardown_wedge_exit_code()
        pid = 1

    monkeypatch.setattr(task_runner, "join_with_reap", lambda *a, **k: "exited")
    monkeypatch.setattr(task_runner, "visible_device_scope",
                        lambda d: __import__("contextlib").nullcontext())

    import multiprocessing as mp

    class _Ctx:
        def Process(self, **kw):
            return _Proc()

    monkeypatch.setattr(mp, "get_context", lambda kind: _Ctx())
    _Proc.start = lambda self: None

    task = _FakeTask()
    task.config.device = 0
    task.config.name = "run"
    task.config.condition = "cond"
    reaper = _Reaper()
    monkeypatch.setattr(reaper, "launch_scope",
                        lambda: __import__("contextlib").nullcontext(), raising=False)
    monkeypatch.setattr(reaper, "adopt", lambda pid: None, raising=False)

    # No exception: a teardown wedge must NOT fail the mode.
    task_runner._run_mode_subprocess(task, "test", reaper, {})
    assert reaped == ["teardown-wedge-exit"], (
        "the GPU is still held by whatever the os._exit'd child spawned; reap is required"
    )


def test_teardown_code_is_distinct_from_every_other_cause():
    """78 must not collide with 75 (device-lost) / 76 (OOM) / 77 (stall)."""
    code = teardown_wedge_exit_code()
    assert code == 78
    assert not is_device_lost_exit(code)
    assert not is_vram_oom_exit(code)
    assert not is_stall_exit(code)
    assert is_teardown_wedge_exit(code)
    # ⚠ and the ambiguous SIGALRM codes must NOT be claimed by it: they are shared with
    # DEVICE_LOST and keep the conservative casualty reading.
    assert not is_teardown_wedge_exit(-14)
    assert not is_teardown_wedge_exit(142)
