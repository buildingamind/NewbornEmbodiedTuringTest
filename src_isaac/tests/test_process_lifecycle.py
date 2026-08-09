"""A run killed from OUTSIDE must not strand a GPU-holding process at PPID=1.

THE BUG. ``pdeathsig`` is armed in the mode subprocess -- the Isaac/Kit process -- so it
fires when that process's parent, a ``ProcessPoolExecutor`` worker, dies. Nothing killed
the pool worker when the DRIVER died. Measured 2026-08-09 against a GPU-free stand-in
with the real ``Executor``/``TaskReaper``/``pdeathsig``:

    kill -TERM driver -> pool worker AND Isaac child alive at PPID=1
    kill -KILL driver -> pool worker AND Isaac child alive at PPID=1
    kill -INT  driver -> driver ALSO alive; it hangs in Executor.__exit__ waiting for a
                         worker that is blocked in p.join()

Historically that path produced 29 orphans up to 8h old holding 52 GB of RAM, plus three
Kit orphans holding ~31 GB of VRAM -- and because admission control reads *physical* free
VRAM, one orphan is enough to make a later run raise JobTooBigError on an empty GPU.

⚠ THE TEST KILLS THE WRAPPER, NOT THE IMMEDIATE PARENT. That distinction is the whole
point: NEXT_STEPS records that a test which kills the immediate parent passes against the
broken code, because ``pdeathsig`` handles exactly that case. The layer these tests
exercise is the one ABOVE the process that arms it.

No GPU is required. The Isaac payload is a ``time.sleep`` loop; every process-lifecycle
mechanism under test (the pool, the guards, the reaper, the handlers) is the real one.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest

from nett_skrl.runtime import lifecycle

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="/proc + PR_SET_PDEATHSIG are Linux-only"
)

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# driver -> Executor pool worker (run_task) -> mode subprocess (stands in for Isaac).
_TREE = '''
import multiprocessing as mp, os, sys, time
sys.path.insert(0, {src!r})

from nett_skrl.runtime import pdeathsig
from nett_skrl.runtime.executor import Executor
from nett_skrl.runtime.lifecycle import (
    cleanup_on_signal, driver_cleanup, register_reaper, unregister_reaper,
)
from nett_skrl.runtime.reap import TaskReaper

GUARDED = {guarded!r}


def mode_subprocess():
    """Stands in for task_runner._run_single_mode: the process that holds the GPU."""
    pdeathsig.arm()
    while True:
        time.sleep(0.2)


def run_task(_):
    """Stands in for task_runner._spawn_mode_subprocess, running in a pool worker."""
    ctx = mp.get_context("spawn")
    reaper = TaskReaper(task_key="test/lifecycle/train")
    if GUARDED:
        register_reaper(reaper)
    try:
        with reaper.launch_scope():
            p = ctx.Process(target=mode_subprocess, daemon=False)
            p.start()
        reaper.adopt(p.pid)
        print(f"MODE {{p.pid}}", flush=True)
        p.join()
    finally:
        if GUARDED:
            unregister_reaper(reaper)


def make_executor():
    if GUARDED:
        return Executor(verbose=True, max_tasks=1)
    # The pre-fix pool, verbatim: a stdlib spawn ProcessPoolExecutor with no initializer,
    # so its workers arm nothing. Using nett's Executor here would silently import the
    # fix into the control arm and make the regression test pass vacuously.
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"))


def body():
    with make_executor() as ex:
        fut = ex.submit(run_task, 0)
        print(f"DRIVER {{os.getpid()}}", flush=True)
        fut.result()


if __name__ == "__main__":
    if GUARDED:
        with cleanup_on_signal(driver_cleanup, name="test-driver"):
            body()
    else:
        body()
'''


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return fh.read().decode("utf-8", "replace").rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _wait_gone(pids, timeout: float) -> list[int]:
    end = time.time() + timeout
    while time.time() < end:
        left = [p for p in pids if _alive(p)]
        if not left:
            return []
        time.sleep(0.2)
    return [p for p in pids if _alive(p)]


def _pool_worker_of(driver: int, mode: int) -> int:
    """The pool worker is the mode subprocess's parent, i.e. the driver's child."""
    with open(f"/proc/{mode}/stat", "rb") as fh:
        raw = fh.read().decode("utf-8", "replace")
    ppid = int(raw[raw.rindex(")") + 2:].split()[1])
    assert ppid != driver, "the mode subprocess must NOT be a direct child of the driver"
    return ppid


class _Tree:
    """Start the three-layer tree; expose its pids; guarantee cleanup."""

    def __init__(self, guarded: bool) -> None:
        fh = tempfile.NamedTemporaryFile("w", suffix="_lifetree.py", delete=False)
        # A real file, not `python -c`: the spawn child re-imports __main__ to unpickle
        # its target, and a -c program has no importable __main__.
        fh.write(textwrap.dedent(_TREE).format(src=SRC, guarded=guarded))
        fh.close()
        self.path = fh.name
        # stderr to a FILE, not a pipe: the handlers' log lines are the observable for
        # the ownership test, and a second pipe nobody drains deadlocks the child once
        # its buffer fills. A file also survives the process being killed mid-write.
        self.err_path = self.path + ".err"
        self._err = open(self.err_path, "w+")
        # start_new_session so the test runner's own signals can never reach the tree,
        # and so a leaked member is still findable by session id.
        self.proc = subprocess.Popen(
            [sys.executable, self.path], stdout=subprocess.PIPE, text=True,
            stderr=self._err, start_new_session=True,
        )
        seen: dict[str, int] = {}
        while len(seen) < 2:
            line = self.proc.stdout.readline()
            if not line:
                raise AssertionError("tree died before reporting its pids")
            key, _, value = line.strip().partition(" ")
            if key in ("DRIVER", "MODE"):
                seen[key] = int(value)
        self.driver = seen["DRIVER"]
        self.mode = seen["MODE"]
        self.worker = _pool_worker_of(self.driver, self.mode)

    @property
    def pids(self) -> list[int]:
        return [self.driver, self.worker, self.mode]

    def stderr(self) -> str:
        """Everything the tree has logged so far."""
        try:
            with open(self.err_path, errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    def close(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in self.pids:
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
            time.sleep(0.5)
        try:
            self.proc.stdout.close()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            self._err.close()
        except Exception:
            pass
        for path in (self.path, self.err_path):
            try:
                os.unlink(path)
            except OSError:
                pass


@pytest.fixture
def tree(request):
    made: list[_Tree] = []

    def factory(guarded: bool) -> _Tree:
        t = _Tree(guarded)
        made.append(t)
        return t

    yield factory
    for t in made:
        t.close()


# --- the regression --------------------------------------------------------


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGKILL])
def test_killing_the_driver_strands_nothing(tree, sig):
    """THE regression. Kill the WRAPPER; assert zero survivors anywhere in the tree.

    All three signals, because each fails differently without the fix: TERM and KILL
    orphan the worker and the Isaac child, INT additionally hangs the driver.

    ⚠ THE SIGKILL CASE PROVES LESS THAN IT LOOKS, AND CANNOT BE MADE TO PROVE MORE HERE.
    ``_Tree``'s stand-in for the mode subprocess is a ``time.sleep`` loop with the DEFAULT
    SIGTERM disposition, so it dies the instant PDEATHSIG's TERM arrives. Real Kit
    installs its own handler and WEDGES: measured 2026-08-09, ``kill -9 <driver>`` reaped
    the pool worker but left the Isaac child at 160% CPU holding 2.7 GB until
    ``stall_guard`` reclaimed it at 612 s (exit 77) -- bounded, not zero. A GPU-free
    stand-in structurally cannot reproduce that, which is exactly why this green must not
    be quoted as "``kill -9`` is covered". TERM and INT genuinely are, end to end, and
    those are the paths the acceptance bar names. See ``executor._worker_init``.
    """
    t = tree(guarded=True)
    assert all(_alive(p) for p in t.pids), "tree did not come up"
    os.kill(t.driver, sig)
    survivors = _wait_gone(t.pids, timeout=45.0)
    assert survivors == [], (
        f"{signal.Signals(sig).name} on the driver left survivors {survivors} "
        f"(driver={t.driver} pool_worker={t.worker} mode={t.mode})"
    )


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_unguarded_tree_strands_the_gpu_process(tree, sig):
    """Characterises the leak, so the regression above cannot pass vacuously.

    If this ever fails, the orphaning was fixed somewhere else and the test above stopped
    proving anything -- delete this one deliberately rather than letting it rot.
    """
    t = tree(guarded=False)
    os.kill(t.driver, sig)
    time.sleep(3.0)
    assert _alive(t.worker), "expected the pool worker to be stranded -- that IS the bug"
    assert _alive(t.mode), "expected the Isaac stand-in to be stranded -- that IS the bug"


def test_pool_worker_reaps_with_ownership_evidence(tree):
    """The worker must use the TaskReaper, not merely pass the signal along.

    A wedged Kit ignores SIGTERM; only the reaper has the (pid, start_ticks) + env-token
    evidence needed to escalate safely. The log line is the observable.

    ⚠ THE EARLIER VERSION ASSERTED ONLY ``not _alive(t.mode)``, which the preceding test
    already covers -- so it duplicated that one and checked NOTHING about ownership, the
    single property it is named for. It could not do better while the tree's stderr was
    inherited by the test runner; ``_Tree`` now captures it to a file, so the log line is
    reachable and this test can assert the mechanism rather than the outcome.
    """
    t = tree(guarded=True)
    os.kill(t.driver, signal.SIGTERM)
    assert _wait_gone(t.pids, timeout=45.0) == []
    assert not _alive(t.mode)

    err = t.stderr()
    # The pool worker must have run its REAPER, not merely let the signal propagate --
    # that is the difference that matters against a real Kit, which ignores a bare TERM.
    assert "lifecycle" in err and str(t.mode) in err, (
        "expected the pool worker's cleanup to name the mode subprocess it reaped, as "
        f"evidence it went through the TaskReaper rather than relying on signal "
        f"propagation. mode={t.mode}\n--- tree stderr ---\n{err[-3000:]}"
    )


# --- unit-level behaviour of the module ------------------------------------


def test_descendants_reaches_a_grandchild_and_skips_the_resource_tracker():
    """A two-level `pgrep -P` walk never reaches Isaac; this walk must.

    The resource_tracker must be walked THROUGH but never returned: it installs SIG_IGN
    for INT and TERM, so including it burned the whole grace window and then reported a
    false survivor on every clean shutdown.
    """
    import multiprocessing as mp

    prog = textwrap.dedent(
        f"""
        import subprocess, sys, time
        sys.path.insert(0, {SRC!r})
        kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print(kid.pid, flush=True)
        time.sleep(120)
        """
    )
    fh = tempfile.NamedTemporaryFile("w", suffix="_desc.py", delete=False)
    fh.write(prog)
    fh.close()
    proc = subprocess.Popen([sys.executable, fh.name], stdout=subprocess.PIPE, text=True)
    grandchild = int(proc.stdout.readline().strip())
    # Force a resource_tracker into existence as a direct child of THIS process.
    mp.get_context("spawn").Semaphore(0)
    try:
        pids = [i.pid for i in lifecycle.descendants()]
        assert proc.pid in pids
        assert grandchild in pids, "the walk stopped at the first level"
        from nett_skrl.runtime.reap import _cmdline
        assert not any("resource_tracker" in _cmdline(p) for p in pids)
    finally:
        for p in (grandchild, proc.pid):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
        proc.stdout.close()
        proc.wait(timeout=5)
        os.unlink(fh.name)


def test_cleanup_runs_exactly_once_and_reports_survivors(tmp_path):
    """Idempotent under a repeated signal, and honest about what it could not reclaim.

    ⚠ THIS RUNS IN A SUBPROCESS ON PURPOSE. The earlier version of this test never
    invoked the handler at all -- it asserted only that cleanup does NOT run on a clean
    exit, so its `return [424242]` was dead code and NEITHER named property (exactly
    once, reports survivors) was tested. It could not invoke it in-process because the
    handler deliberately restores the default disposition and re-raises, which would
    kill the test runner. A child process is the way to get the real path.

    The re-entrant signal is the point: cleanup sends itself a SECOND SIGTERM while it
    is still running, which is exactly what an impatient operator does.
    """
    marker = tmp_path / "calls.txt"
    prog = textwrap.dedent(
        f"""
        import os, signal, sys, time
        sys.path.insert(0, {SRC!r})
        import logging; logging.basicConfig(level=logging.INFO)
        from nett_skrl.runtime import lifecycle

        def cleanup(reason):
            with open({str(marker)!r}, "a") as fh:
                fh.write(reason + chr(10))
                fh.flush()
            # Re-entrant: a second signal WHILE cleanup is in flight must not re-run it.
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.5)
            return [424242]          # a pid we pretend we could not reclaim

        with lifecycle.cleanup_on_signal(cleanup, name="unit-test"):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)
        """
    )
    proc = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                          text=True, timeout=60)

    assert proc.returncode == -signal.SIGTERM, (
        f"the handler must re-raise with the DEFAULT disposition so the exit status "
        f"stays 128+SIGTERM; got {proc.returncode}\n{proc.stderr[-2000:]}"
    )
    reasons = marker.read_text().split() if marker.exists() else []
    assert len(reasons) == 1, (
        f"cleanup must run EXACTLY ONCE even under a repeated signal; ran {len(reasons)} "
        f"time(s): {reasons}"
    )
    assert "424242" in proc.stderr, (
        "a pid the cleanup could not reclaim must be REPORTED, not silently dropped -- "
        f"that is the difference between a bounded leak and an invisible one\n"
        f"{proc.stderr[-2000:]}"
    )


def test_cleanup_is_not_installed_off_the_main_thread():
    """`signal.signal` raises off-thread; a missing net must never break a run."""
    import threading

    errors: list[BaseException] = []

    def body():
        try:
            with lifecycle.cleanup_on_signal(lambda r: [], name="off-thread"):
                pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th = threading.Thread(target=body)
    th.start()
    th.join(timeout=10)
    assert errors == []


def test_reap_registered_survives_a_raising_reaper():
    """Cleanup must never raise out of a signal handler."""

    class Boom:
        task_key = "boom"

        def reap(self, reason):
            raise RuntimeError("nope")

    lifecycle.register_reaper(Boom())          # type: ignore[arg-type]
    try:
        assert lifecycle.reap_registered("unit-test") == []
    finally:
        for r in list(lifecycle._reapers):
            lifecycle.unregister_reaper(r)
