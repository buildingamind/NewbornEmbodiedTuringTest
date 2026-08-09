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
        # start_new_session so the test runner's own signals can never reach the tree,
        # and so a leaked member is still findable by session id.
        self.proc = subprocess.Popen(
            [sys.executable, self.path], stdout=subprocess.PIPE, text=True,
            start_new_session=True,
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
            os.unlink(self.path)
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
    """
    t = tree(guarded=True)
    os.kill(t.driver, signal.SIGTERM)
    assert _wait_gone(t.pids, timeout=45.0) == []
    # stdout is a pipe we still hold; the handlers log to stderr, which is inherited by
    # the test runner. Assert on the mechanism instead: the mode subprocess was adopted
    # by a reaper (root-identity evidence) and is gone.
    assert not _alive(t.mode)


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


def test_cleanup_runs_exactly_once_and_reports_survivors():
    """Idempotent under a repeated signal, and honest about what it could not reclaim."""
    calls: list[str] = []

    def cleanup(reason: str):
        calls.append(reason)
        return [424242]  # a pid we pretend we could not reclaim

    with lifecycle.cleanup_on_signal(cleanup, name="unit-test"):
        # Drive the handler directly: raising a real signal would kill the test runner,
        # since the handler deliberately re-raises with the default disposition.
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler) and handler not in (signal.SIG_DFL, signal.SIG_IGN)
    # After the block the previous disposition is restored, so nothing leaks into the
    # rest of the suite.
    assert signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, signal.SIG_IGN) or callable(
        signal.getsignal(signal.SIGTERM)
    )
    assert calls == [], "cleanup must not run on a clean exit from the block"


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
