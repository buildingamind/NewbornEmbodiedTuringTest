"""Thin ``ProcessPoolExecutor`` wrapper that mutes child stdout and cannot orphan."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

#: Keeps each pool worker's signal handlers installed for the worker's whole life.
_WORKER_CLEANUP_STACK: "contextlib.ExitStack | None" = None


def _mute_stdout() -> None:
    """Silence a worker's stdout.

    MODULE-LEVEL ON PURPOSE. ``spawn`` pickles the initializer to send it to each worker,
    and a closure defined inside ``__init__`` is not picklable -- "Can't pickle local
    object 'Executor.__init__.<locals>.mute'". It worked as a closure only because the pool
    used to fork, which inherits the function object instead of sending it.
    """
    sys.stdout = open(os.devnull, "w")


def _worker_init(mute: bool) -> None:
    """Pool-worker entry hook: make this worker impossible to orphan, then mute it.

    ★ THE POOL WORKER WAS THE UNPROTECTED LAYER (measured 2026-08-09). The tree is
    ``driver -> pool worker -> mode subprocess (Isaac/Kit, GPU)``. ``pdeathsig`` is armed
    in the MODE SUBPROCESS, so it fires when the POOL WORKER dies -- but nothing killed
    the pool worker when the DRIVER died. Killing the driver with TERM, INT or KILL left
    the pool worker AND the Isaac process alive at PPID=1, every time.

    Two nets, deliberately different in kind:

    * ``pdeathsig.arm()`` -- kernel-enforced, so it reaches THIS WORKER even under
      ``kill -9 <driver>``, which no handler can. When the driver dies this worker gets
      SIGTERM, and the handler below turns that into a real reap.

      ⚠ THAT IS NOT "``kill -9`` IS COVERED", AND AN EARLIER VERSION OF THIS COMMENT SAID
      IT WAS. Measured 2026-08-09 against real Kit, not the stand-in: ``kill -9 <driver>``
      does reap this worker, but the Isaac child then IGNORED the reaper's SIGTERM and
      wedged at 160% CPU holding 2.7 GB. ``stall_guard`` reclaimed it at 612 s (exit 77),
      so the leak is BOUNDED AT ~10 MIN, not zero. And ``kill -9`` on the shell ``timeout``
      WRAPPER leaves the whole tree running indefinitely: nothing arms PDEATHSIG in the
      driver itself. Closing that would mean arming it in ``NETT.run()``, which would also
      kill legitimately detached runs (``nohup ... &`` then logout), so it is deliberately
      not done -- prefer TERM/INT on the wrapper, which ARE covered end to end.
      ⚠ ``tests/test_process_lifecycle.py``'s SIGKILL case cannot see any of this: its
      stand-in for the mode subprocess is a ``time.sleep`` loop with the DEFAULT SIGTERM
      disposition, so it dies instantly where Kit wedges. Do not read that green as
      evidence about Kit.
    * ``cleanup_on_signal(worker_cleanup)`` -- on SIGINT/SIGTERM, drive the ``TaskReaper``
      this worker registered for its in-flight task (``task_runner._spawn_mode_subprocess``)
      so the Isaac child is terminated WITH ownership evidence and its VRAM released,
      instead of merely inheriting a SIGTERM it may ignore.

    ⚠ NOT the retracted "arm every spawn descendant" plan -- that one aimed BELOW the
    process that already arms pdeathsig, and there is nothing there. This aims ABOVE it.

    The context manager is entered and never exited: this hook returns into
    ``_process_worker``'s loop, and the handlers must outlive it for the whole life of the
    worker. ``ExitStack`` without ``close()`` is that, stated explicitly.
    """
    global _WORKER_CLEANUP_STACK

    from . import pdeathsig
    from .lifecycle import cleanup_on_signal, worker_cleanup

    pdeathsig.arm()
    stack = contextlib.ExitStack()
    stack.enter_context(
        cleanup_on_signal(worker_cleanup, name=f"pool-worker:{os.getpid()}")
    )
    # Parked on the module so it is not garbage-collected -- __exit__ restores the default
    # handlers, and running it here would undo the guard the instant this frame returns.
    _WORKER_CLEANUP_STACK = stack
    if mute:
        _mute_stdout()


class Executor(ProcessPoolExecutor):
    """Process pool sized to the work, not to the machine.

    ``max_workers`` used to be ``os.cpu_count()``. On Python 3.11
    ProcessPoolExecutor spawns the FULL complement on the first ``submit`` (it
    does not grow lazily), so a run with a single task forked **64 worker
    processes** -- measured: 65 procs/cell, which is what the standing
    "~65 helper procs per cell" note was actually seeing. An 8-way wave forked
    ~520 of them.

    They are cheap-ish but not free: fork storms at startup, 64 PIDs of
    scheduler/reaper bookkeeping per cell, and 64 more processes to orphan when
    a cell is killed. (They are NOT a memory problem -- measured PSS for the
    whole 65-proc tree was ~961 MB, i.e. copy-on-write shared; the ~60 GB summed
    RSS is double-counting.)

    Args:
        verbose: When False, mute child stdout.
        max_tasks: Upper bound on tasks submitted concurrently, if the caller can
            PROVE it. The pool is sized to ``max_tasks + 1`` -- the +1 keeps a
            worker free for the blocking ``validate_tasklist`` / dry-run
            ``future_wait`` submits, which would otherwise stall behind
            long-running training tasks. ``None`` means "bound unknown": keep the
            historical ``os.cpu_count()`` size rather than guess, because a pool
            smaller than the task count would silently cap real concurrency (VRAM
            can admit many tasks across several GPUs) and slow big sweeps down.
    """

    def __init__(self, verbose: bool, max_tasks: int | None = None) -> None:
        cores = os.cpu_count() or 1
        workers = cores if max_tasks is None else min(max(1, int(max_tasks)) + 1, cores)
        # ★ FIX B (2026-07-28): SPAWN, not the Linux default fork.
        #
        # Workers here call `validate_tasklist` -> `Environment.load` -> `AppLauncher`, i.e.
        # they boot Kit and touch CUDA. Doing that in a FORKED child is unsafe in general
        # (CUDA contexts do not survive fork) and was concretely fatal: the fork inherits
        # the parent's argv, and under `python -m pytest` Kit died on it --
        # "Ill formed parameter: -m" then a segfault, which killed a POOL WORKER and so
        # broke the whole pool (BrokenProcessPool), failing every task rather than one.
        #
        # `_spawn_mode_subprocess` already chose spawn deliberately for exactly this
        # reason; the pool was the remaining fork path, and it was the one running Kit
        # first. Aligning them removes the asymmetry rather than papering over it.
        #
        # COST, accepted: spawn re-imports the module in each worker and requires picklable
        # arguments. Task objects already cross a spawn boundary in
        # `_spawn_mode_subprocess`, so they qualify. Worker startup is slower, which is
        # noise next to a Kit boot.
        # The initializer is ALWAYS set now, not only when muting: it is what arms this
        # worker's orphan guards (see _worker_init). Passing None when verbose=True used
        # to mean a verbose run had no guards at all.
        super().__init__(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(not verbose,),
            mp_context=mp.get_context("spawn"),
        )

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        super().__exit__(exc_type, exc_val, exc_tb)
        return False
