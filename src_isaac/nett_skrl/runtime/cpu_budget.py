"""Per-cell CPU budget: one knob, every thread pool.

A NETT "cell" is one Isaac process (one brain x one condition). Nothing inside a
cell can discover how many *other* cells share the host: a wave launcher (e.g.
.nett_perf/fast/wave.sh) starts N independent `diag_knowngood`/NETT processes,
one per GPU, each with num_brains=1. So a cell's Executor sees a single task and
CANNOT infer the wave width -- the budget has to be told to it, or defaulted.

Left to themselves, three separate pools each size themselves to the WHOLE host
and collectively oversubscribe it:

  1. Kit/carb  -- Isaac's SimulationApp sets carb.tasking threadCount to
     min(os.cpu_count(), limit_cpu_threads=32) -> 32 threads/cell on a 64-core
     box, ~15-17% CPU each (~5 cores), regardless of wave width.
  2. torch     -- defaults intra-op AND inter-op to os.cpu_count() -> 64/cell.
  3. OpenMP/BLAS -- OMP_NUM_THREADS unset -> one thread per core, per cell.

Measured on this 64-core host (ne16, wheeled, video on, res128): one SOLO
training cell draws ~11.6 cores, so an 8-way wave asks for ~93 cores of 64 ->
run-queue thrash (the reported "load ~162, 4.9 it/s vs 14 solo"). The env-step
benchmark alone draws only ~5.4 cores/cell and does NOT collapse -- the extra
~6 cores are the PPO update + decode, i.e. the torch/OMP pools above.

So: pick ONE number (cores this cell may use) and drive every pool from it.
Wave launchers should set NETT_KIT_THREADS = cores / concurrent_cells.

MEASURED LADDER (64 cores, 8x A10, ne16 wheeled res128, 60 ep x 200 steps,
aggregate train it/s over all cells; "stock" = none of these pools pinned):

    rung                              cells   it/s/cell   aggregate   vs stock
    8-way stock                        8/8       3.96        31.6       1.00x
    8-way  NETT_KIT_THREADS=8          8/8      14.79       116.3       3.68x
    16-way NETT_KIT_THREADS=4 (2/GPU) 16/16     12.08       200.6       6.35x
    24-way NETT_KIT_THREADS=2 (3/GPU) 21/24     12.02       254.4       8.05x

Once the pools are pinned a cell draws ~3 cores instead of ~11.6, so the box has
room for MORE cells than GPUs: packing 2 cells/GPU buys +72% aggregate for only
-18% per-cell. 16-way is the recommended production point -- it completed 16/16.

24-way is faster still but only completed 21/24: the other 3 HUNG in Kit init,
all at the same line ("[omni.kvdb.plugin] Disabling key-value database because
another kit instance is running"). That is a shared-Kit-cache race between
simultaneously starting instances, not CPU or VRAM exhaustion (VRAM peaked at
~11 GB of 24, and the host had 1007 GB RAM free). Staggering the launches ~5s
apart cut it from 13/24 hung to 3/24 but did not cure it. Fixing it properly
(per-cell Kit cache/kvdb dir) is what unlocks 3+ cells/GPU.
"""

from __future__ import annotations

import os

from ..environment.physx_strategy import _env_int, usable_cpu_threads

#: Cores per cell when nothing says otherwise. Chosen for the 8-way wave this
#: host is provisioned for (64 cores / 8 cells). It is also FASTER than Isaac's
#: 32 solo (343.1 -> 360.0 env-steps/s at ~41% less CPU), so defaulting low
#: costs nothing when a cell runs alone -- there is no wave width at which 32
#: wins. Override with NETT_KIT_THREADS.
DEFAULT_CELL_THREADS = 8

#: Name kept as-is: it shipped (and is tested) as the Kit-thread knob before it
#: also came to govern torch/OMP. One budget, one knob.
ENV_VAR = "NETT_KIT_THREADS"


def cell_cpu_threads() -> int:
    """Cores this cell may use, clamped to [1, cores actually schedulable].

    Respects cpuset affinity via ``usable_cpu_threads`` so a cgroup-limited or
    taskset-pinned cell cannot ask for more than it can run on.
    """
    return max(1, min(_env_int(ENV_VAR, DEFAULT_CELL_THREADS), usable_cpu_threads()))


def kit_thread_args(num_threads: int, existing: str = "") -> str:
    """Build the ``kit_args`` string that pins Kit's CPU thread pools.

    Composes with any ``existing`` kit_args (space-separated, Kit CLI syntax) by
    appending after -- later flags win when Kit re-parses argv, so this still
    overrides an equivalent flag placed earlier in ``existing``.

    `limit_cpu_threads` (SimulationApp's own knob) is NOT usable here: it is
    absent from ``AppLauncher._SIM_APP_CFG_TYPES``, so passing it as a kwarg is
    silently dropped. kit_args is the only path that reaches Kit.

    MEASURED LIMITATION: only the carb.tasking pool actually shrinks (32 -> 8,
    verified per-cell via /proc). The omni.tbb.globalcontrol flag is accepted but
    does NOT shrink the tbb.worker pool on Isaac Sim 5.1 (it stays ~31) -- that
    plugin reads its setting before our kit_args CLI arg lands. It is kept
    because it is the documented pairing and is harmless (tbb.workers sample at
    ~0% CPU), but the measured win comes entirely from carb.tasking.
    """
    threads = max(1, int(num_threads))
    parts = [existing] if existing else []
    parts.append(f"--/plugins/carb.tasking.plugin/threadCount={threads}")
    parts.append(f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={threads}")
    return " ".join(parts)


def apply_torch_thread_limits(num_threads: int | None = None) -> int:
    """Pin torch/OpenMP to this cell's budget. Call in the CHILD, before torch work.

    torch reads OMP_NUM_THREADS at import and sizes its intra-op pool from
    os.cpu_count() otherwise, so an 8-way wave would run 8 x 64 = 512 intra-op
    threads over 64 cores during the PPO update. Setting the env vars covers any
    BLAS/OpenMP library loaded later; set_num_threads covers an already-imported
    torch. Both are needed.

    Returns the applied thread count. Idempotent.
    """
    threads = cell_cpu_threads() if num_threads is None else max(1, int(num_threads))
    # Must be set before a BLAS/OpenMP runtime is first initialized to bind.
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    try:
        import torch
    except ImportError:  # torch-free callers (tests, validation)
        return threads
    torch.set_num_threads(threads)
    # Inter-op ALSO defaults to cpu_count, but torch forbids changing it once the
    # parallel region has started -- best-effort, and harmless if already fixed.
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    return threads
