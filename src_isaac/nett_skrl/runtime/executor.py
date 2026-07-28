"""Thin ``ProcessPoolExecutor`` wrapper that optionally mutes child stdout."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

def _mute_stdout() -> None:
    """Silence a worker's stdout.

    MODULE-LEVEL ON PURPOSE. ``spawn`` pickles the initializer to send it to each worker,
    and a closure defined inside ``__init__`` is not picklable -- "Can't pickle local
    object 'Executor.__init__.<locals>.mute'". It worked as a closure only because the pool
    used to fork, which inherits the function object instead of sending it.
    """
    sys.stdout = open(os.devnull, "w")


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
        super().__init__(
            max_workers=workers,
            initializer=None if verbose else _mute_stdout,
            mp_context=mp.get_context("spawn"),
        )

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        super().__exit__(exc_type, exc_val, exc_tb)
        return False
