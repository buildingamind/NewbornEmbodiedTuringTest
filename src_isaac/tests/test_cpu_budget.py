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
