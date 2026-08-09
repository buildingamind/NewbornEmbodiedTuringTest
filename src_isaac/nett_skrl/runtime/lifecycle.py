"""Signal-driven cleanup -- the half of orphan prevention PDEATHSIG cannot do.

WHY THIS EXISTS
---------------
``reap.py`` cleans up thoroughly, but every ``reaper.reap()`` call site is on a path the
process CHOOSES: the VRAM-OOM exit, the stall exit, the device-lost exit, and
``join_with_reap``'s absolute timeout. Nothing runs when the process is killed from
OUTSIDE -- the shell ``timeout`` that bounds every wave, a ``pkill``, an operator, Ctrl-C.
``pdeathsig`` closed the SIGKILL case for the ONE layer that arms it. This module closes
the SIGINT/SIGTERM case, for every layer.

THE TREE, MEASURED (2026-08-09, GPU-free stand-in reproducing the real shape)::

    driver          python -m nett / NETT.run()          <- no handler  (this module)
     |-- resource_tracker
     `-- pool worker  ProcessPoolExecutor, runs run_task <- no handler, no pdeathsig
          `-- mode subprocess  spawn, Isaac/Kit, GPU     <- pdeathsig armed

``kill -TERM <driver>`` left the pool worker AND the mode subprocess alive at PPID=1;
``kill -KILL <driver>`` did the same; ``kill -INT <driver>`` left ALL THREE alive -- the
driver itself hung, because ``ProcessPoolExecutor.__exit__`` waits for a worker that is
blocked in ``p.join()``. PDEATHSIG on the mode subprocess never fired in any of the three,
because its parent -- the pool worker -- was still alive. That is the whole bug: the
UNPROTECTED layer is the pool worker, not the Isaac process.

⚠ THIS IS NOT THE RETRACTED "arm every spawn descendant" PLAN. That plan was built on the
premise that Isaac lives one level BELOW the process that arms pdeathsig; it does not (see
NEXT_STEPS 2026-07-31, item 5 -- the mode subprocess IS the Isaac process). The gap is one
level ABOVE: the pool worker, which arms nothing and which nothing terminates.

WHAT THIS MODULE PROVIDES
-------------------------
* :func:`cleanup_on_signal` -- a context manager that installs SIGINT/SIGTERM handlers (and
  an ``atexit`` net) around a block, runs a caller-supplied cleanup exactly once, reports
  what it could not reclaim, and then re-raises the signal with the default disposition so
  the exit status stays the conventional 128+signo.
* :func:`register_reaper` / :func:`reap_registered` -- so a signalled process can drive the
  EXISTING, ownership-checked :class:`~nett_skrl.runtime.reap.TaskReaper` rather than
  inventing a second kill path.
* :func:`terminate_descendants` -- a bounded, TERM-only walk of this process's own
  descendants, for the driver, which owns no reaper.

⚠ TERM-ONLY, AND IT REPORTS SURVIVORS RATHER THAN CLAIMING SUCCESS. A render-pump-wedged
Kit process ignores SIGINT and SIGTERM and outlives a shell ``timeout`` (measured: 8h into
a 900s cap). Escalation to SIGKILL is left to ``TaskReaper.reap``, which has the ownership
evidence to do it safely; this module never sends SIGKILL itself. Repeated ``-9`` of wedged
Kit boots has previously put this host's driver into a CUDA bad state needing a root reset,
so "report the survivor" is the correct answer here, not "kill it harder".
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
import time
import weakref
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Sequence

from .reap import (
    ProcessIdentity,
    TaskReaper,
    _all_pids,
    _cmdline,
    _is_shared_infrastructure,
    _read_stat,
)

_LOG = logging.getLogger("nett.lifecycle")

#: SIGTERM -> report grace for :func:`terminate_descendants` (seconds). Generous on
#: purpose: a signalled pool worker runs a full ``TaskReaper.reap`` (bounded by
#: ~2 * ``NETT_REAP_TERM_GRACE`` = 20s) before it dies, and the driver must not declare
#: survivors while that is still in progress. Polls, so the healthy case returns at once.
TERM_GRACE = float(os.environ.get("NETT_LIFECYCLE_TERM_GRACE", "25"))
#: Poll interval while waiting for the tree to drain (seconds).
POLL = float(os.environ.get("NETT_LIFECYCLE_POLL", "0.25"))
#: Escape hatch: 1 restores the old handler-free behaviour.
DISABLED = os.environ.get("NETT_LIFECYCLE_DISABLE", "0") == "1"

_lock = threading.Lock()
_reapers: "weakref.WeakSet[TaskReaper]" = weakref.WeakSet()


# --- reaper registry -------------------------------------------------------


def register_reaper(reaper: TaskReaper) -> None:
    """Make *reaper* reachable from a signal handler in THIS process.

    Held weakly: a task that finishes normally drops its reaper and this registry must
    never be the thing that keeps it alive.
    """
    with _lock:
        _reapers.add(reaper)


def unregister_reaper(reaper: TaskReaper) -> None:
    with _lock:
        _reapers.discard(reaper)


def reap_registered(reason: str, logger: Optional[logging.Logger] = None) -> list[int]:
    """``reap()`` every registered reaper. Returns the pids it could NOT reclaim.

    Reuses the existing ownership evidence (env token / root identity / lineage
    snapshot), so a signalled process can never kill another user's -- or another
    task's -- work.
    """
    log = logger or _LOG
    survivors: list[int] = []
    with _lock:
        pending = list(_reapers)
    for reaper in pending:
        try:
            report = reaper.reap(reason)
            survivors.extend(report.survivors)
        except Exception:  # noqa: BLE001 - cleanup must never raise out of a handler
            log.exception("lifecycle: reap(%s) failed for %s", reason, reaper.task_key)
    return survivors


# --- descendant walk (for the driver, which owns no reaper) ----------------


def descendants(pid: Optional[int] = None) -> list[ProcessIdentity]:
    """Every live descendant of *pid* (default: this process), deepest LAST.

    A full ``/proc`` sweep rather than a two-level ``pgrep -P`` walk: the Isaac process
    is a THIRD-level descendant of the driver, and a shallow walk never reaches it. That
    mistake has been made twice in this project.

    ⚠ ``multiprocessing``'s resource_tracker is WALKED THROUGH BUT NOT RETURNED. It
    installs ``SIG_IGN`` for both SIGINT and SIGTERM on purpose, so signalling it can
    never do anything -- it exits on its own once the last writer to its pipe closes.
    Including it cost the entire grace window and then reported it as a "survivor":
    measured 25s of pointless waiting plus a false alarm on every clean shutdown.
    """
    root = os.getpid() if pid is None else pid
    children: dict[int, list[int]] = {}
    for candidate in _all_pids():
        stat = _read_stat(candidate)
        if stat is not None and stat[2] != "Z":
            children.setdefault(stat[0], []).append(candidate)

    out: list[ProcessIdentity] = []
    seen: set[int] = set()
    frontier = [root]
    while frontier:
        nxt: list[int] = []
        for parent in frontier:
            for child in sorted(children.get(parent, ())):
                if child in seen:
                    continue
                seen.add(child)
                ident = ProcessIdentity.of(child)
                if ident is not None and not _is_shared_infrastructure(child):
                    out.append(ident)
                nxt.append(child)   # still walk THROUGH it: it may have children
        frontier = nxt
    return out


def terminate_descendants(
    reason: str,
    grace: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> list[int]:
    """SIGTERM this process's whole descendant tree; return the pids still alive.

    ⚠ NO SIGKILL. See the module docstring. The point of the bounded wait is to give a
    signalled pool worker time to run its own ``TaskReaper.reap`` -- which DOES escalate,
    with ownership evidence -- and then to REPORT anything that outlived even that.
    """
    log = logger or _LOG
    if DISABLED:
        log.warning("lifecycle: NETT_LIFECYCLE_DISABLE=1; not terminating descendants")
        return []
    window = TERM_GRACE if grace is None else grace
    # Snapshot before signalling: identities are (pid, start_ticks), so a pid recycled
    # onto an unrelated process during the grace can never be mistaken for a survivor.
    targets = descendants()
    if not targets:
        return []

    log.warning(
        "lifecycle: %s -- terminating %d descendant process(es): %s",
        reason, len(targets), [i.pid for i in targets],
    )
    # Deepest last from `descendants`, so signal in reverse: children before parents.
    # Not load-bearing for correctness (every one of them is signalled either way), but
    # it stops a parent from noticing a dead child and reporting a spurious failure
    # before it is itself told to stop.
    for ident in reversed(targets):
        if not ident.alive():
            continue
        try:
            os.kill(ident.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            log.warning("lifecycle: not permitted to signal pid %d", ident.pid)

    deadline = time.monotonic() + max(0.0, window)
    remaining = list(targets)
    while remaining and time.monotonic() < deadline:
        remaining = [i for i in remaining if i.alive()]
        if not remaining:
            break
        time.sleep(POLL)
    remaining = [i for i in remaining if i.alive()]

    for ident in remaining:
        log.error(
            "lifecycle: SURVIVOR pid=%d after SIGTERM + %.0fs -- NOT killed (a wedged "
            "Kit ignores TERM, and -9 on this host has bricked the driver before). "
            "cmd=%.160s",
            ident.pid, window, _cmdline(ident.pid),
        )
    return [i.pid for i in remaining]


# --- the handler ------------------------------------------------------------


@contextmanager
def cleanup_on_signal(
    cleanup: Callable[[str], Sequence[int]],
    name: str,
    logger: Optional[logging.Logger] = None,
    signals: Sequence[int] = (signal.SIGINT, signal.SIGTERM),
) -> Iterator[None]:
    """Run *cleanup* when this process is signalled, then die with the right status.

    *cleanup* takes a reason string and returns the pids it could not reclaim. It runs
    AT MOST ONCE per process, from the handler or from ``atexit``, whichever fires first.

    On a signal the handler restores the default disposition and re-raises to itself, so
    the process still exits 128+signo -- a wave driver that greps exit codes keeps
    working, and a Ctrl-C still reads as a Ctrl-C.

    Handlers are installed only from the MAIN THREAD (``signal.signal`` raises anywhere
    else) and only when this is not a nested install; both are no-ops rather than errors,
    because a cleanup net that refuses to load is worse than one that is absent.
    """
    log = logger or _LOG
    if DISABLED:
        yield
        return

    done = threading.Event()

    def _run(reason: str) -> None:
        if done.is_set():
            return
        done.set()
        try:
            survivors = list(cleanup(reason) or ())
        except Exception:  # noqa: BLE001 - never raise out of a handler
            log.exception("lifecycle[%s]: cleanup(%s) raised", name, reason)
            return
        if survivors:
            log.error(
                "lifecycle[%s]: %d process(es) SURVIVED cleanup (%s): %s -- check "
                "`nvidia-smi --query-compute-apps=pid,used_memory` before the next run",
                name, len(survivors), reason, survivors,
            )
        else:
            log.info("lifecycle[%s]: cleanup(%s) reclaimed everything", name, reason)

    installed: dict[int, object] = {}

    def _handler(signo: int, _frame) -> None:
        signame = signal.Signals(signo).name
        log.warning("lifecycle[%s]: received %s -- cleaning up before exit", name, signame)
        _run(f"signal-{signame}")
        # Re-raise with the default disposition so the exit status is 128+signo.
        try:
            signal.signal(signo, signal.SIG_DFL)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            pass
        os.kill(os.getpid(), signo)

    if threading.current_thread() is threading.main_thread():
        for signo in signals:
            try:
                previous = signal.getsignal(signo)
                signal.signal(signo, _handler)
                installed[signo] = previous
            except (ValueError, OSError, RuntimeError):
                log.debug("lifecycle[%s]: cannot install handler for %s",
                          name, signo, exc_info=True)
    else:
        log.debug("lifecycle[%s]: not the main thread; signal handlers not installed", name)

    # The atexit net covers `sys.exit`, an unhandled exception, and any path that unwinds
    # without passing through the handler. It CANNOT cover os._exit or SIGKILL -- those
    # are pdeathsig's job, one layer down.
    exit_hook = lambda: _run("interpreter-exit")  # noqa: E731
    atexit.register(exit_hook)
    try:
        yield
    finally:
        atexit.unregister(exit_hook)
        for signo, previous in installed.items():
            try:
                signal.signal(signo, previous)  # type: ignore[arg-type]
            except (ValueError, OSError, TypeError):
                pass


def driver_cleanup(reason: str) -> list[int]:
    """Cleanup for the top-level driver: reap what it owns, then TERM the rest of the tree.

    The driver holds no ``TaskReaper`` of its own in the normal wave path -- its tasks run
    in pool workers -- but it DOES spawn Kit directly for tasklist validation, and that
    child is registered. Reap those first, then sweep whatever is left.
    """
    survivors = reap_registered(reason)
    survivors.extend(terminate_descendants(reason))
    # A reaped pid can also show up in the descendant sweep; report each once.
    return sorted(set(survivors))


def worker_cleanup(reason: str) -> list[int]:
    """Cleanup for a pool worker: drive the registered ``TaskReaper`` for its live task."""
    return reap_registered(reason)
