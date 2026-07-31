"""PDEATHSIG: a task subprocess must die when its parent is killed from OUTSIDE.

The gap this closes: every ``reaper.reap()`` call site is on a path the parent CHOOSES
(vram-oom / stall / device-lost / absolute-timeout), and there is no SIGTERM handler, so
an externally killed parent strands its spawn child at PPID=1 holding GPU memory.
Measured 2026-07-30: 29 such orphans up to 8h old (52GB RAM), plus three Kit orphans
holding ~31GB of VRAM.

These tests use SIGKILL on the parent deliberately -- that is the case a SIGTERM handler
structurally cannot cover, and the whole reason this is enforced by the kernel.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from nett_skrl.runtime.pdeathsig import arm

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only"
)

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_parent(child_arms: bool) -> tuple[int, int]:
    """Start parent -> spawn child; return (parent_pid, child_pid). Child loops forever."""
    prog = textwrap.dedent(
        f"""
        import multiprocessing as mp, os, sys, time
        sys.path.insert(0, {SRC!r})

        def worker():
            if {child_arms!r}:
                from nett_skrl.runtime.pdeathsig import arm
                arm()
            while True:
                time.sleep(0.2)

        if __name__ == "__main__":
            mp.set_start_method("spawn", force=True)
            p = mp.Process(target=worker, daemon=False)
            p.start()
            print(f"{{os.getpid()}} {{p.pid}}", flush=True)
            time.sleep(120)
        """
    )
    # MUST be a real file, not `python -c`: the spawn child re-imports __main__ to
    # unpickle the target, and a -c program has no importable __main__.
    import tempfile

    fh = tempfile.NamedTemporaryFile("w", suffix="_pdeath.py", delete=False)
    fh.write(prog)
    fh.close()
    proc = subprocess.Popen(
        [sys.executable, fh.name], stdout=subprocess.PIPE, text=True
    )
    line = proc.stdout.readline().strip()
    ppid, cpid = (int(x) for x in line.split())
    return ppid, cpid


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_gone(pid: int, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


def test_child_WITHOUT_pdeathsig_is_stranded():
    """Characterises the leak. If this ever fails, orphaning was fixed elsewhere."""
    ppid, cpid = _run_parent(child_arms=False)
    try:
        os.kill(ppid, signal.SIGKILL)
        time.sleep(2.0)
        assert _alive(cpid), "expected the child to survive — that IS the bug"
        assert os.getppid() != cpid
    finally:
        for p in (cpid, ppid):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass


def test_child_WITH_pdeathsig_dies_with_a_SIGKILLED_parent():
    """The fix. SIGKILL on the parent is deliberate: no handler could cover it."""
    ppid, cpid = _run_parent(child_arms=True)
    try:
        os.kill(ppid, signal.SIGKILL)
        assert _wait_gone(cpid, timeout=15.0), (
            "child outlived its SIGKILLed parent — PDEATHSIG did not take"
        )
    finally:
        for p in (cpid, ppid):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass


def test_arm_is_safe_to_call_and_reports():
    """Must never raise: a missing safety net cannot be allowed to break a run."""
    assert arm() in (True, False)
