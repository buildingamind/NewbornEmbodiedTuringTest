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
aggregate train it/s with ALL cells concurrently active; "stock" = nothing pinned):

    rung                                active   it/s/cell   aggregate   vs stock
    8-way stock                          8/8        3.96        31.6       1.00x
    8-way  NETT_KIT_THREADS=8            8/8       14.79       116.3       3.68x
    16-way NETT_KIT_THREADS=4 (2/GPU)   16/16      12.08       200.6       6.35x  <- knee
    20-way NETT_KIT_THREADS=3           20/20       9.94       200.3       6.34x
    24-way NETT_KIT_THREADS=2 (3/GPU)   24/24       8.71       209.0       6.61x  *

Once the pools are pinned a cell draws ~3 cores instead of ~11.6, so the box fits
more cells than it has GPUs: 2 cells/GPU buys +72% aggregate for -18% per-cell.
**Past 16 the box is saturated** -- 20 and 24 cells buy nothing (aggregate flat at
~200-209 while per-cell falls), so 16-way is the production point.

* 24-way only reached 24/24 with a per-cell Kit cache dir (NETT_KIT_CACHE_ID); with the
shared cache 3/24 hung in Kit init on a shared-cache race. That knob was REMOVED as
dead weight -- aggregate is flat past 16, so there is no reason to run 24 and pay for
it. If you ever need >16 cells/host, see git log "per-cell Kit cache dirs" (B@c43fd46).

Do not be fooled by a staggered launch: a 24-way run with a 5s stagger reported
12.02 it/s/cell / 254.4 aggregate, but that is an ARTIFACT -- the stagger spread
the launches over 115s so the cells never fully overlapped, and 3 of the 24 hung
and never competed at all. Measure with every cell concurrently active or the
number is fiction.
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


#: Floor under the inductor compile pool, deliberately ABOVE the cell budget when
#: the budget is 1. compile_threads=1 disables async compile entirely and costs
#: +24% cold-compile wall (92.9s vs 74.6s, measured below), while those workers
#: spend their lives blocked rather than competing for cores -- so this is the one
#: pool that does NOT follow the budget all the way down.
MIN_COMPILE_THREADS = 2


def apply_torch_thread_limits(num_threads: int | None = None) -> int:
    """Pin torch/OpenMP/inductor to this cell's budget. Call in the CHILD, before
    torch work.

    torch reads OMP_NUM_THREADS at import and sizes its intra-op pool from
    os.cpu_count() otherwise, so an 8-way wave would run 8 x 64 = 512 intra-op
    threads over 64 cores during the PPO update. Setting the env vars covers any
    BLAS/OpenMP library loaded later; set_num_threads covers an already-imported
    torch. Both are needed.

    INDUCTOR is a FOURTH pool that none of the above reaches:
    ``torch._inductor.config.compile_threads`` has its own default of
    ``min(32, cpu_count)`` = 32 here, and torch spawns one compile_worker
    subprocess that forks that many workers -> **33 descendant processes per
    cell** while the reward ``torch.compile`` runs (see nett_isaac.rewards).

    MEASURED 2026-07-30 (cold caches, the six reward graphs, A10):

        compile_threads   cold compile   procs/cell   steady-state
              32 (torch)     74.6-77.8s       33      0.295-0.304 ms/call
               8            75.2s              9      0.300
               4            75.3s              5      0.308
               1            92.9s              0      0.310

    4 / 8 / 32 are indistinguishable: the six graphs compile SEQUENTIALLY and
    each has too few kernels to feed 32 workers, so the pool has nothing to chew
    on. There is no wave width -- and no solo case -- at which 32 wins, which is
    why this can simply follow the budget.

    Scope, so nobody over-values this: the pool is only spawned on a COLD
    inductor cache (after a torch upgrade or a rewards.py edit). Warm, the same
    six graphs take 3.9s and spawn **1** process, so this is a startup-burst
    tidy-up, not a throughput lever. It is also NOT a stall cause -- the Kit
    render-pump wedge reproduces with NETT_REWARD_COMPILE=0 and no pool at all.

    ⚠ THE ENV VAR ALONE IS A NO-OP HERE. ``decide_compile_threads()`` runs at
    ``import torch`` (module-level in ``torch._inductor.config``), and by the time
    the spawn child reaches this function torch is ALREADY imported -- unpickling
    the Task pulls it in. Measured: setting only TORCHINDUCTOR_COMPILE_THREADS
    late leaves ``compile_threads`` at 32 and still forks 33 processes; assigning
    the config attribute gives 5. So we do BOTH, exactly like OMP_NUM_THREADS +
    ``set_num_threads``: the env var binds a torch imported later, the attribute
    covers the one already loaded. Same silent-drop class as ``limit_cpu_threads``
    in :func:`kit_thread_args` -- if you touch this, re-check the process count,
    do not assume the setting took.

    ``setdefault`` preserves torch's own precedence: an explicit
    TORCHINDUCTOR_COMPILE_THREADS still wins (=1 is the documented way to make
    pdb usable), and the attribute write honours it too.

    Returns the applied thread count. Idempotent.
    """
    threads = cell_cpu_threads() if num_threads is None else max(1, int(num_threads))
    # Must be set before a BLAS/OpenMP runtime is first initialized to bind.
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    # Read by decide_compile_threads() at torch import: binds only a torch
    # imported AFTER this point. The attribute write below covers the usual case.
    compile_threads = max(MIN_COMPILE_THREADS, threads)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(compile_threads))
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
    # torch was already imported (the normal case in a spawn child), so the env
    # var above did nothing -- assign the resolved value. Honour an explicit
    # override rather than the budget, matching torch's own precedence.
    try:
        from torch._inductor import config as _inductor_config

        _inductor_config.compile_threads = int(
            os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", compile_threads)
        )
    except Exception:  # noqa: BLE001 - a torch without inductor is not fatal here
        pass
    return threads
