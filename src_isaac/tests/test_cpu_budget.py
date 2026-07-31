"""Per-cell CPU budget + process-pool sizing.

Covers the wave-scale oversubscription fixes: every cell used to size Kit's
carb.tasking pool (32), torch's intra-op pool (os.cpu_count()=64) and the
ProcessPoolExecutor (os.cpu_count()=64 workers, spawned eagerly on first submit)
to the whole host, regardless of how many cells shared it. Measured on a 64-core
box: an 8-way train wave ran 528 processes at load ~162 and 3.96 it/s/cell vs
13.68 solo; pinning all three to a per-cell budget restored ~14.8 it/s/cell.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from unittest import mock

import pytest

from nett_skrl.runtime.cpu_budget import (
    DEFAULT_CELL_THREADS,
    ENV_VAR,
    MIN_COMPILE_THREADS,
    apply_torch_thread_limits,
    cell_cpu_threads,
    kit_thread_args,
)
from nett_skrl.runtime.executor import Executor


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)


# --- budget ---------------------------------------------------------------


def test_default_is_not_the_whole_host():
    """The bug was every pool sizing itself to os.cpu_count()."""
    assert cell_cpu_threads() == DEFAULT_CELL_THREADS
    assert DEFAULT_CELL_THREADS < (os.cpu_count() or 1)


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "3")
    assert cell_cpu_threads() == 3


def test_clamped_to_at_least_one(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "0")
    assert cell_cpu_threads() == 1


def test_never_exceeds_schedulable_cpus(monkeypatch):
    """A cpuset/taskset-pinned cell must not ask for more than it can run on."""
    monkeypatch.setenv(ENV_VAR, "9999")
    monkeypatch.setattr("nett_skrl.runtime.cpu_budget.usable_cpu_threads", lambda: 4)
    assert cell_cpu_threads() == 4


def test_garbage_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "not-a-number")
    assert cell_cpu_threads() == DEFAULT_CELL_THREADS


# --- kit args -------------------------------------------------------------


def test_kit_args_pin_both_pools():
    args = kit_thread_args(8).split()
    assert "--/plugins/carb.tasking.plugin/threadCount=8" in args
    assert "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8" in args


def test_kit_args_append_after_existing_so_last_wins():
    args = kit_thread_args(4, existing="--/some/flag=1").split()
    assert args[0] == "--/some/flag=1"
    assert args.index("--/plugins/carb.tasking.plugin/threadCount=4") > 0


def test_kit_args_clamp():
    assert "threadCount=1" in kit_thread_args(0)


# --- torch ----------------------------------------------------------------


def test_apply_torch_thread_limits_sets_env_and_torch(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    fake = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"torch": fake}):
        applied = apply_torch_thread_limits(6)
    assert applied == 6
    assert os.environ["OMP_NUM_THREADS"] == "6"
    assert os.environ["MKL_NUM_THREADS"] == "6"
    fake.set_num_threads.assert_called_once_with(6)


def test_apply_torch_thread_limits_survives_fixed_interop(monkeypatch):
    """torch forbids set_num_interop_threads once the pool started; not fatal."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    fake = mock.MagicMock()
    fake.set_num_interop_threads.side_effect = RuntimeError("already started")
    with mock.patch.dict("sys.modules", {"torch": fake}):
        assert apply_torch_thread_limits(2) == 2
    fake.set_num_threads.assert_called_once_with(2)


def test_apply_torch_thread_limits_defaults_to_budget(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "5")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with mock.patch.dict("sys.modules", {"torch": mock.MagicMock()}):
        assert apply_torch_thread_limits() == 5


# --- inductor compile pool ------------------------------------------------
#
# The FOURTH host-sized pool: torch._inductor.config.compile_threads defaults to
# min(32, cpu_count) = 32, and torch forks a compile_worker per thread -> 33
# descendant processes per cell during the reward torch.compile. Measured cold on
# an A10: 4/8/32 threads all compile the six reward graphs in 74.6-75.3s (the
# graphs compile sequentially and have too few kernels to feed 32 workers), so
# following the budget is free; only compile_threads=1 is slower (92.9s).


def _fake_torch():
    """A mocked torch whose ``_inductor.config`` is importable.

    ``from torch._inductor import config`` goes through the real import machinery,
    which cannot walk a MagicMock's __path__ -- so the submodules must be in
    sys.modules for the assignment under test to be reachable at all.
    """
    torch = mock.MagicMock()
    inductor = mock.MagicMock()
    config = mock.MagicMock()
    config.compile_threads = 32  # torch's own default on this host
    inductor.config = config
    torch._inductor = inductor
    return {
        "torch": torch,
        "torch._inductor": inductor,
        "torch._inductor.config": config,
    }


def test_compile_pool_follows_the_budget(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    mods = _fake_torch()
    with mock.patch.dict("sys.modules", mods):
        apply_torch_thread_limits(4)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "4"
    assert mods["torch._inductor.config"].compile_threads == 4


def test_compile_pool_is_set_on_an_ALREADY_IMPORTED_torch(monkeypatch):
    """REGRESSION: the env var alone is a NO-OP at this call site.

    decide_compile_threads() runs at ``import torch``, and the spawn child has
    already imported torch (unpickling the Task pulls it in) by the time the
    budget is applied. Measured: env-var-only left compile_threads at 32 and
    still forked 33 processes; assigning the attribute gave 5. If this test is
    ever "simplified" down to the env var, the pin silently stops working --
    the same silent-drop class as ``limit_cpu_threads`` in kit_thread_args.
    """
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    mods = _fake_torch()
    with mock.patch.dict("sys.modules", mods):
        apply_torch_thread_limits(6)
    assert mods["torch._inductor.config"].compile_threads == 6, (
        "compile_threads must be assigned on the loaded torch, not only exported"
    )


def test_compile_pool_never_drops_to_one(monkeypatch):
    """compile_threads=1 disables async compile: +24% cold wall (92.9s vs 74.6s).

    Deliberately ABOVE the budget at budget=1 -- the one pool that does not
    follow it all the way down, because those workers block rather than compete.
    """
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    mods = _fake_torch()
    with mock.patch.dict("sys.modules", mods):
        applied = apply_torch_thread_limits(1)
    assert applied == 1, "the cell budget itself is still 1"
    assert MIN_COMPILE_THREADS == 2
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "2"
    assert mods["torch._inductor.config"].compile_threads == 2


def test_explicit_compile_threads_override_wins(monkeypatch):
    """torch documents =1 as the way to make pdb usable; do not stomp it."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    mods = _fake_torch()
    with mock.patch.dict("sys.modules", mods):
        apply_torch_thread_limits(8)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert mods["torch._inductor.config"].compile_threads == 1


# --- executor pool sizing -------------------------------------------------


def test_pool_sized_to_tasks_plus_one():
    """+1 keeps a worker free for the blocking validate/dry-run submits."""
    with mock.patch.object(ProcessPoolExecutor, "__init__", return_value=None) as init:
        Executor(verbose=True, max_tasks=3)
    assert init.call_args.kwargs["max_workers"] == 4


def test_unknown_bound_keeps_historical_size():
    """A bound we cannot prove must not shrink the pool below the task count."""
    with mock.patch.object(ProcessPoolExecutor, "__init__", return_value=None) as init:
        Executor(verbose=True, max_tasks=None)
    assert init.call_args.kwargs["max_workers"] == os.cpu_count()


def test_pool_never_exceeds_cpu_count():
    with mock.patch.object(ProcessPoolExecutor, "__init__", return_value=None) as init:
        Executor(verbose=True, max_tasks=10_000)
    assert init.call_args.kwargs["max_workers"] == os.cpu_count()


def test_pool_is_actually_small_for_one_task():
    """Regression: 1 task used to fork 64 workers (Py3.11 spawns eagerly)."""
    ex = Executor(verbose=True, max_tasks=1)
    try:
        assert ex._max_workers == 2
    finally:
        ex.shutdown(wait=True)


# --- NETT's task bound ----------------------------------------------------


def _nett(configs):
    from nett_skrl.nett import NETT

    obj = NETT.__new__(NETT)  # skip schema validation; only _max_concurrent_tasks
    obj.configs = configs
    return obj


def test_task_bound_counts_brains_times_conditions():
    n = _nett([{"num_brains": 3, "environment": {"conditions": ["A", "B"]}}])
    assert n._max_concurrent_tasks() == 6


def test_task_bound_sums_across_configs():
    n = _nett([
        {"num_brains": 1, "environment": {"conditions": ["A"]}},
        {"num_brains": 2, "environment": {"conditions": ["A", "B"]}},
    ])
    assert n._max_concurrent_tasks() == 5


def test_task_bound_reads_conditions_from_design_sheet(tmp_path):
    sheet = tmp_path / "d.csv"
    sheet.write_text(
        "ImprintCondition,Phase,TestCondition,TargetVideo,NonTargetVideo,LeftMonitor,RightMonitor\n"
        "Object1,Train,,a.mp4,b.mp4,a.mp4,b.mp4\n"
        "Object1,Test,Rest,a.mp4,b.mp4,a.mp4,b.mp4\n"
        "Object2,Train,,b.mp4,a.mp4,b.mp4,a.mp4\n"
        "Object2,Test,Rest,b.mp4,a.mp4,b.mp4,a.mp4\n"
    )
    n = _nett([{"num_brains": 2, "environment": {"design_sheet": str(sheet)}}])
    assert n._max_concurrent_tasks() == 4  # 2 brains x 2 conditions


def test_task_bound_unknown_when_unresolvable():
    """An experiment bundle (no explicit conditions, no sheet) -> don't guess."""
    n = _nett([{"num_brains": 1, "environment": {"experiment": "/some/bundle.zip"}}])
    assert n._max_concurrent_tasks() is None


def test_task_bound_unknown_when_sheet_unreadable():
    n = _nett([{"num_brains": 1, "environment": {"design_sheet": "/no/such.csv"}}])
    assert n._max_concurrent_tasks() is None


# --- checkpointing must not chunk training ---------------------------------
#
# `checkpoint_freq` used to be folded into _training_boundaries, so setting it split a
# 500-episode run into 33 subprocesses AND copied final_agent.pt over each agent_{step}.pt
# -- yielding 33 byte-identical files (1 MD5 across all of them). Saving the model is a
# side effect of training; skrl writes agent_{timestep}.pt from inside its loop.


def test_checkpoint_freq_does_not_chunk_training():
    """REGRESSION: only eval_freq may create training boundaries."""
    from nett_skrl.runtime.task_runner import _training_boundaries

    # eval disabled -> exactly one boundary, i.e. ONE continuous training subprocess
    assert _training_boundaries(250_000, eval_freq=10_000_000) == [250_000]


def test_eval_freq_still_chunks_because_eval_must_pause_training():
    from nett_skrl.runtime.task_runner import _training_boundaries

    assert _training_boundaries(200_000, eval_freq=50_000) == [
        50_000, 100_000, 150_000, 200_000
    ]


def test_training_boundaries_takes_no_checkpoint_argument():
    """The coupling is gone at the signature level, not just in the body."""
    import inspect
    from nett_skrl.runtime.task_runner import _training_boundaries

    assert "checkpoint_freq" not in inspect.signature(_training_boundaries).parameters
