"""Thin ``ProcessPoolExecutor`` wrapper that optionally mutes child stdout."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

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
        def mute() -> None:
            sys.stdout = open(os.devnull, "w")

        cores = os.cpu_count() or 1
        workers = cores if max_tasks is None else min(max(1, int(max_tasks)) + 1, cores)
        super().__init__(
            max_workers=workers,
            initializer=None if verbose else mute,
        )

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        super().__exit__(exc_type, exc_val, exc_tb)
        return False
