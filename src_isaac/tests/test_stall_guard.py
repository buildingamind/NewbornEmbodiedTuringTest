"""stall_guard: bounds the Kit render-pump hang that crash_guard cannot see.

The hang under test (measured 2026-07-30): Kit wedges inside
``SimulationContext.render`` -> ``self._app.update()``, GPU at 2% while the process burns
136% CPU, emitting NO ``DEVICE_LOST`` and no crash signature -- so ``crash_guard`` never
fires and, with ``NETT_REAP_TIMEOUT`` disabled by default, the parent's join is unbounded.

These tests are Isaac-free: the guard's signal is our own step counter, deliberately, so
it is testable without booting Kit. The real hard-exit is exercised in a SUBPROCESS
(``os._exit`` would take the test runner down with it).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from nett_skrl.runtime import stall_guard
from nett_skrl.runtime.reap import (
    ReapedTaskError,
    StallError,
    is_device_lost_exit,
    is_stall_exit,
    stall_exit_code,
)


@pytest.fixture(autouse=True)
def _disarm_after():
    yield
    stall_guard.disarm()


# --- exit-code contract ----------------------------------------------------


def test_stall_exit_code_is_distinct_from_device_lost_and_oom():
    """The three causes want different responses, so they must stay separable."""
    assert stall_exit_code() == 77
    assert is_stall_exit(77)
    assert not is_stall_exit(75)   # DEVICE_LOST
    assert not is_stall_exit(76)   # VRAM OOM
    assert not is_stall_exit(0)
    assert not is_stall_exit(None)


def test_sigalrm_backstop_codes_stay_with_device_lost():
    """-14/142 are ambiguous (both guards arm ITIMER_REAL), so the conservative
    DEVICE_LOST reading must keep them rather than the newer stall path claiming them."""
    for code in (-14, 142):
        assert is_device_lost_exit(code)
        assert not is_stall_exit(code)


def test_stall_error_is_a_reaped_task_error():
    """This is what makes a wave CONTINUE past a stalled cell instead of dying: the
    waiter catches ReapedTaskError. With 2-4 of 8 cells hanging, failing the wave on one
    would make multi-cell waves unusable."""
    assert issubclass(StallError, ReapedTaskError)


def test_exit_code_is_env_overridable(monkeypatch):
    monkeypatch.setenv("NETT_STALL_EXIT_CODE", "91")
    assert stall_exit_code() == 91
    assert is_stall_exit(91)
    assert not is_stall_exit(77)


# --- arm / disarm / progress ----------------------------------------------


def test_disabled_by_env_does_not_arm(monkeypatch):
    monkeypatch.setenv("NETT_STALL_GUARD", "0")
    assert stall_guard.arm() is False


def test_default_is_on(monkeypatch):
    monkeypatch.delenv("NETT_STALL_GUARD", raising=False)
    # A huge budget so the watchdog can never fire during the test.
    monkeypatch.setenv("NETT_STALL_STARTUP_GRACE_S", "100000")
    monkeypatch.setenv("NETT_STALL_TIMEOUT_S", "100000")
    assert stall_guard.arm() is True


def test_arm_is_idempotent(monkeypatch):
    monkeypatch.setenv("NETT_STALL_STARTUP_GRACE_S", "100000")
    monkeypatch.setenv("NETT_STALL_TIMEOUT_S", "100000")
    assert stall_guard.arm() is True
    assert stall_guard.arm() is True


def test_arm_resets_the_step_counter(monkeypatch):
    """A re-arm must start from a clean clock, not an ancient last-progress stamp."""
    monkeypatch.setenv("NETT_STALL_STARTUP_GRACE_S", "100000")
    monkeypatch.setenv("NETT_STALL_TIMEOUT_S", "100000")
    stall_guard.arm()
    stall_guard.note_progress(5)
    assert stall_guard.steps_seen() == 5
    stall_guard.disarm()
    stall_guard.arm()
    assert stall_guard.steps_seen() == 0


def test_note_progress_is_safe_when_disarmed():
    """It runs on the hot path and deliberately does not check whether the guard is
    armed, so it must be harmless either way."""
    stall_guard.disarm()
    stall_guard.note_progress()
    stall_guard.note_progress(3)
    assert stall_guard.steps_seen() >= 4


# --- the hard-exit path, in a subprocess ----------------------------------


_CHILD = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, {src!r})
    from nett_skrl.runtime import stall_guard
    stall_guard.arm()
    {body}
    # If the guard did not fire we exit 0, and the test fails on the exit code.
    time.sleep({linger})
    os._exit(0)
    """
)


def _run_child(body: str, env: dict, linger: float = 8.0, timeout: float = 60.0):
    src = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = _CHILD.format(src=src, body=body, linger=linger)
    full = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, "-c", code], env=full, capture_output=True, text=True,
        timeout=timeout,
    )


def test_hard_exits_with_the_stall_code_when_no_progress_ever_arrives():
    """The 'wedged before the first step' shape -- seen at 7/12500 and 31/12500."""
    out = _run_child(
        body="pass",  # never calls note_progress
        env={
            "NETT_STALL_GUARD": "1",
            "NETT_STALL_STARTUP_GRACE_S": "1",
            "NETT_STALL_TIMEOUT_S": "1",
            "NETT_STALL_POLL_S": "1",
        },
    )
    assert out.returncode == 77, (out.returncode, out.stdout[-2000:], out.stderr[-2000:])


def test_hard_exits_mid_run_after_progress_stops():
    """The 'wedged mid-run' shape -- steps flow, then stop. This is the case that
    matters most: a naive guard armed once at startup would already have fired."""
    out = _run_child(
        body=(
            "for _ in range(5):\n"
            "    stall_guard.note_progress()\n"
            "    time.sleep(0.2)\n"
        ),
        env={
            "NETT_STALL_GUARD": "1",
            "NETT_STALL_STARTUP_GRACE_S": "60",
            "NETT_STALL_TIMEOUT_S": "1",
            "NETT_STALL_POLL_S": "1",
        },
    )
    assert out.returncode == 77, (out.returncode, out.stderr[-2000:])


def test_does_not_fire_while_progress_continues():
    """A healthy run must survive. Steps here keep arriving for well past the timeout,
    which is the false-positive this guard must not have -- killing a healthy cell
    mid-PPO-update would be worse than the hang it protects against."""
    out = _run_child(
        body=(
            "for _ in range(25):\n"
            "    stall_guard.note_progress()\n"
            "    time.sleep(0.2)\n"
        ),
        env={
            "NETT_STALL_GUARD": "1",
            "NETT_STALL_STARTUP_GRACE_S": "60",
            "NETT_STALL_TIMEOUT_S": "2",
            "NETT_STALL_POLL_S": "1",
        },
        linger=0.0,
    )
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])


def test_disarm_stops_the_watchdog():
    """A finished run stops stepping and then spends real time in analysis/teardown; a
    still-armed watchdog would eventually read that as a stall."""
    out = _run_child(
        body=(
            "stall_guard.note_progress()\n"
            "stall_guard.disarm()\n"
        ),
        env={
            "NETT_STALL_GUARD": "1",
            "NETT_STALL_STARTUP_GRACE_S": "1",
            "NETT_STALL_TIMEOUT_S": "1",
            "NETT_STALL_POLL_S": "1",
        },
        linger=6.0,
    )
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])


def test_disabled_guard_never_exits():
    out = _run_child(
        body="pass",
        env={
            "NETT_STALL_GUARD": "0",
            "NETT_STALL_STARTUP_GRACE_S": "1",
            "NETT_STALL_TIMEOUT_S": "1",
            "NETT_STALL_POLL_S": "1",
        },
        linger=5.0,
    )
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])


def test_startup_grace_is_separate_from_the_steady_state_timeout():
    """Kit boot + scene build + first render takes minutes under contention, so the
    pre-first-step budget must be its own (larger) number. With a long grace and a tiny
    steady-state timeout, a child that never reports progress must SURVIVE."""
    out = _run_child(
        body="pass",
        env={
            "NETT_STALL_GUARD": "1",
            "NETT_STALL_STARTUP_GRACE_S": "60",
            "NETT_STALL_TIMEOUT_S": "1",
            "NETT_STALL_POLL_S": "1",
        },
        linger=5.0,
    )
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])
