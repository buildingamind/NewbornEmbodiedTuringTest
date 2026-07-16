"""`max_parallel_envs: "auto"` and the square-grid snap on the TRAIN planner.

The load-bearing property, same as the test-phase selector: a resolved num_envs must
tile into a SQUARE camera grid (Isaac Sim #488) and be a multiple of num_brains. The
VRAM model is free to pick any count that fits -- the snap is what keeps its answer
legal. A bad pick does not crash, it silently distorts every rendered frame, so it is
pinned here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nett_skrl.nett import NETT
from nett_skrl.runtime.parallel_envs import (
    capped_num_envs,
    grow_num_envs,
    is_valid_num_envs,
    largest_valid_num_envs,
    select_test_num_envs,
    smallest_valid_num_envs,
    snap_num_envs,
)


# --- the shared predicate ------------------------------------------------


@pytest.mark.parametrize(
    "n, nb, valid",
    [
        (16, 1, True),   # 4x4
        (80, 1, True),   # 9x9, one empty tile -- empties are harmless
        (20, 1, False),  # 5x4
        (6, 3, False),   # 3x2, though a clean multiple of 3
        (9, 3, True),    # 3x3
        (2, 2, False),   # 2x1 -- the worst distortion case
        (4, 2, True),    # 2x2
        (8, 3, False),   # 3x3 but not a multiple of 3 -> breaks brain scoping
    ],
)
def test_is_valid_num_envs(n, nb, valid):
    assert is_valid_num_envs(n, nb) is valid


def test_snap_prefers_down():
    # 20 (5x4) -> 16 (4x4). Down, because the request is a VRAM ceiling.
    assert snap_num_envs(20, 1) == (16, False)
    assert snap_num_envs(81, 1) == (81, False)
    assert snap_num_envs(79, 1) == (79, False)  # 9x9, in-band


def test_snap_goes_up_only_when_nothing_below_qualifies():
    # num_brains=6: multiples are 6 (3x2), 12 (4x3), 18 (5x4), 24 (5x5) -- so a
    # ceiling of 12 has no legal count at or below it and must go UP to 24.
    n, went_up = snap_num_envs(12, 6)
    assert (n, went_up) == (24, True)
    assert is_valid_num_envs(n, 6)


def test_smallest_valid_is_not_always_num_brains():
    assert smallest_valid_num_envs(1) == 1
    assert smallest_valid_num_envs(2) == 4   # 2 tiles 2x1
    assert smallest_valid_num_envs(3) == 3   # 2x2
    assert smallest_valid_num_envs(6) == 24


def test_largest_valid_returns_none_when_nothing_fits():
    assert largest_valid_num_envs(5, 6) is None


# --- the VRAM model's snap ------------------------------------------------


def _planner(free_gb: float, per_env_gb: float, fixed_gb: float):
    """A NETT whose dry run reports fixed + n*per_env, so max_fit is exact."""
    nett = object.__new__(NETT)
    nett.logger = logging.getLogger("test")
    nett.devices = [0]
    nett.free_device_memory = {0: free_gb * 1024**3}
    nett.memory_manager = SimpleNamespace(
        get_most_free_gpu=lambda devices: (0, free_gb * 1024**3),
        get_free_memory=lambda d: free_gb * 1024**3,
    )
    plans = []
    probed_modes = []

    class _Body:
        def adjust_to_agent(self, env, **kwargs):
            plans.append(kwargs["num_envs"])
            env.num_brains = kwargs["num_brains"]
            env.num_envs = kwargs["num_envs"]

    def _estimate(self, brain, body, env, output_dir, mode="train"):
        probed_modes.append(mode)
        return (fixed_gb + env.num_envs * per_env_gb) * 1024**3

    nett._estimate_task_memory_via_dry_run = _estimate.__get__(nett, NETT)
    nett.probed_modes = probed_modes
    return nett, _Body(), plans


def _resolve(nett, body, *, num_brains, num_envs, env_cap, tmp_path):
    brain = SimpleNamespace(envs_per_brain=max(1, num_envs // num_brains))
    env = SimpleNamespace(conditions=["Object1"], num_brains=num_brains, num_envs=num_envs)
    return NETT._resolve_task_memory_and_envs(
        nett, "auto", brain, body, env, tmp_path,
        num_brains=num_brains, num_envs=num_envs, steps_per_episode=200,
        env_cap=env_cap,
    )


def test_auto_snaps_a_distorted_fit_down_to_square(tmp_path):
    """The regression this fixes: free VRAM fits ~20 envs, and the old planner
    snapped only to a num_brains multiple -- returning 20, a 5x4 grid, and training
    every frame through a distorted fisheye."""
    # fixed=1GB, per_env=1GB, free=26.25GB -> budget 21GB -> max_fit 20.
    nett, body, _ = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    _memory, num_envs = _resolve(
        nett, body, num_brains=1, num_envs=64, env_cap=None, tmp_path=tmp_path
    )
    assert num_envs == 16
    assert is_valid_num_envs(num_envs, 1)


def test_auto_is_bounded_by_measured_vram_not_the_recipe(tmp_path):
    """"auto" means the ceiling comes from measurement: a tiny GPU gets a small count
    even though the recipe asked for 64."""
    nett, body, _ = _planner(free_gb=6.25, per_env_gb=1.0, fixed_gb=1.0)
    _memory, num_envs = _resolve(
        nett, body, num_brains=1, num_envs=64, env_cap=None, tmp_path=tmp_path
    )
    assert num_envs == 4  # fits 4, and 4 is 2x2


def test_explicit_cap_still_binds_under_auto_memory(tmp_path):
    """An integer max_parallel_envs is an upper bound the VRAM model may not exceed,
    even when far more would fit."""
    nett, body, _ = _planner(free_gb=1000.0, per_env_gb=1.0, fixed_gb=1.0)
    _memory, num_envs = _resolve(
        nett, body, num_brains=1, num_envs=16, env_cap=16, tmp_path=tmp_path
    )
    assert num_envs == 16


def test_auto_respects_num_brains(tmp_path):
    nett, body, _ = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    _memory, num_envs = _resolve(
        nett, body, num_brains=4, num_envs=64, env_cap=None, tmp_path=tmp_path
    )
    assert num_envs == 16  # 20 fits but is 5x4; 16 is 4x4 and divisible by 4
    assert is_valid_num_envs(num_envs, 4)


def _search(nett, body, *, num_brains, seed, max_envs, tmp_path, budget=None):
    """Drive the verified search directly. ``max_envs=None`` is the TEST-phase call
    (grow freely); a value is the TRAIN call (never exceed the recipe)."""
    brain = SimpleNamespace(envs_per_brain=max(1, seed // num_brains))
    env = SimpleNamespace(conditions=["Object1"], num_brains=num_brains, num_envs=seed)
    return NETT._search_max_envs_verified(
        nett, brain, body, env, tmp_path,
        num_brains=num_brains, seed_envs=seed, steps_per_episode=200,
        max_envs=max_envs, budget=budget,
    )


def test_train_auto_never_grows_above_the_recipe(tmp_path):
    """num_envs is load-bearing for learning: envs_per_brain is derived from
    rollouts//steps_per_episode, so growing it rewrites the PPO batch composition.
    Auto's job on the train path is to VERIFY the recipe's count, not to redefine the
    experiment -- even though 16 more envs would fit here."""
    nett, body, _ = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    _memory, num_envs = _resolve(
        nett, body, num_brains=1, num_envs=4, env_cap=None, tmp_path=tmp_path
    )
    assert num_envs == 4  # budget fits 16, but the recipe asked for 4


def test_test_search_grows_upward_when_the_seed_fits(tmp_path):
    """The TEST-phase path: no recipe to honour, so the search doubles up. 4 -> 8 -> 16
    fit (5/9/17GB of a 21GB budget); 32 would need 33GB.

    32 is never PROBED: after 8 and 16 the measured slope proves it cannot fit, and the
    reported count is still the largest that was actually measured. Every plan here is a
    real measurement -- nothing is derived except the decision to stop."""
    nett, body, plans = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    memory, num_envs = _search(
        nett, body, num_brains=1, seed=4, max_envs=None, tmp_path=tmp_path
    )
    assert num_envs == 16
    assert plans[:3] == [4, 8, 16]
    assert 32 not in plans
    assert memory / 1024**3 == pytest.approx(17.0)


def test_grow_escapes_the_sparse_low_end():
    """Valid counts are sparse at the bottom, so 2x-and-snap-down stalls: from 1 it
    targets 2 (a 2x1 grid, invalid) and snaps back to 1. Growth must still advance."""
    assert grow_num_envs(1, 1) == 3   # 2 is invalid; 3 tiles 2x2
    assert grow_num_envs(3, 1) == 4
    assert grow_num_envs(4, 1) == 8
    assert grow_num_envs(64, 1) == 121  # 128 snaps down to 121 (11x11)
    for n in (1, 3, 4, 8, 16, 32, 64, 121, 242):
        assert grow_num_envs(n, 1) > n


def test_test_search_grows_from_a_seed_of_one(tmp_path):
    """Regression from a REAL run: the recipe implied envs_per_brain=1 (rollouts=16 //
    steps=50), the search seeded at 1, reported "num_envs=1 verified" and stopped --
    because growth was 2x-then-snap-down, and 2 (a 2x1 grid) snapped back to 1."""
    nett, body, plans = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    _memory, num_envs = _search(
        nett, body, num_brains=1, seed=1, max_envs=None, tmp_path=tmp_path
    )
    assert num_envs == 16  # budget 21GB, mem=1+n -> 16 fits at 17GB, 32 does not
    assert plans[:5] == [1, 3, 4, 8, 16]


def test_auto_does_not_extrapolate_into_a_superlinear_wall(tmp_path):
    """Regression for the 2916-env OOM.

    VRAM is superlinear in num_envs (measured: 2.48GB @2 envs -> 13.48GB @256 -> OOM
    @400). The old 2-point linear fit probed 2 and 4, saw a ~6MB/env slope, and
    "fitted" ~2900 envs into 23.5GB; the run died in vkAllocateMemory. This fake
    reproduces that curve, so a planner that extrapolates fails this test.
    """
    fixed, free_gb = 2.47, 23.5

    def consumed_gb(n):  # ~the measured curve: fixed + 0.002 * n**1.5
        return fixed + 0.002 * (n ** 1.5)

    budget = free_gb * NETT._VRAM_SAFETY
    nett, body, plans = _planner(free_gb=free_gb, per_env_gb=0, fixed_gb=0)

    def _estimate(self, brain, body_, env, output_dir, mode="train"):
        return consumed_gb(env.num_envs) * 1024**3

    nett._estimate_task_memory_via_dry_run = _estimate.__get__(nett, NETT)

    memory, num_envs = _search(
        nett, body, num_brains=1, seed=16, max_envs=None, tmp_path=tmp_path
    )

    # A linear fit at 2/4 envs would claim >1000 envs fit. The truth is ~400.
    assert num_envs < 400
    assert is_valid_num_envs(num_envs, 1)
    # Whatever it picked, that count's MEASURED cost must fit the budget -- the
    # property the old estimator violated.
    assert consumed_gb(num_envs) <= budget
    assert memory / 1024**3 == pytest.approx(consumed_gb(num_envs))
    # ...and it must not be timid: the next step up is what busted the budget.
    assert consumed_gb(snap_num_envs(2 * num_envs, 1)[0]) > budget


def test_auto_returns_none_when_nothing_fits(tmp_path):
    """A GPU too small for even the smallest valid count hands off to the caller's
    descending scan rather than inventing a number."""
    nett, body, _ = _planner(free_gb=0.001, per_env_gb=1.0, fixed_gb=100.0)
    brain = SimpleNamespace(envs_per_brain=4)
    env = SimpleNamespace(conditions=["Object1"], num_brains=1, num_envs=4)
    assert NETT._search_max_envs_verified(
        nett, brain, body, env, tmp_path,
        num_brains=1, seed_envs=4, steps_per_episode=200,
    ) is None


def test_test_ceiling_divides_the_budget_across_co_scheduled_tasks(tmp_path):
    """Tasks are packed per GPU by declared memory. A test ceiling measured against
    ALL of free VRAM would be a lie the moment two tasks land on one GPU and both
    enter test -- each would try to allocate the whole card."""
    nett, body, _ = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    brain = SimpleNamespace(envs_per_brain=4)
    env = SimpleNamespace(
        conditions=["a", "b"], num_brains=1, num_envs=4,
        iterations_per_test_episode={"a": 52, "b": 52},
    )

    def _ceiling(num_tasks):
        return NETT._resolve_test_env_ceiling(
            nett, brain, body, env, tmp_path, num_brains=1, train_envs=4,
            steps_per_episode=200, episodes={"test": 20}, num_tasks=num_tasks,
        )

    alone, _mem = _ceiling(1)      # whole GPU: budget 21GB -> 16 envs (17GB)
    shared, _mem2 = _ceiling(2)    # 2 tasks / 1 GPU: budget 10.5GB -> 8 envs (9GB)
    assert alone == 16
    assert shared == 8


def test_each_phase_is_probed_in_its_own_mode(tmp_path):
    """Train and test have different footprints -- test allocates no optimizer and no
    rollout buffer -- so each search must measure its own phase. Probing test with a
    train dry run would be both wrong (high) and slow (a whole rollout per rung)."""
    nett, body, _ = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    _resolve(nett, body, num_brains=1, num_envs=4, env_cap=None, tmp_path=tmp_path)
    assert set(nett.probed_modes) == {"train"}

    nett2, body2, _ = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    brain = SimpleNamespace(envs_per_brain=4)
    env = SimpleNamespace(
        conditions=["a"], num_brains=1, num_envs=4,
        iterations_per_test_episode={"a": 52},
    )
    NETT._resolve_test_env_ceiling(
        nett2, brain, body2, env, tmp_path, num_brains=1, train_envs=4,
        steps_per_episode=200, episodes={"test": 20}, num_tasks=1,
    )
    assert set(nett2.probed_modes) == {"test"}


def test_no_test_phase_means_no_test_search(tmp_path):
    """Nothing to widen, so it must not burn dry runs measuring a phase that never
    runs."""
    nett, body, plans = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    brain = SimpleNamespace(envs_per_brain=4)
    env = SimpleNamespace(conditions=["a"], num_brains=1, num_envs=4)
    assert NETT._resolve_test_env_ceiling(
        nett, brain, body, env, tmp_path, num_brains=1, train_envs=4,
        steps_per_episode=200, episodes={"train": 100}, num_tasks=1,
    ) is None
    assert plans == []


# --- the probe is bounded -------------------------------------------------


def test_dry_run_probe_is_bounded_and_real_runs_are_not(tmp_path, monkeypatch):
    """Regression: a 484-env probe hit a Vulkan OOM and then HUNG, pinning 24GB with
    no reap, because crash_guard's bounded exit keys on DEVICE_LOST and an allocation
    OOM is not one. A probe knows its own budget, so it gets an absolute cap; a real
    run's duration is unbounded by design and must NOT."""
    import nett_skrl.nett as nett_module
    from nett_skrl.runtime.task import Task, TaskConfig

    nett = object.__new__(NETT)
    nett.logger = logging.getLogger("test")
    nett.devices = [0]
    nett.free_device_memory = {0: 1}
    nett.memory_manager = SimpleNamespace(
        get_most_free_gpu=lambda d: (0, 1), get_free_memory=lambda d: 1
    )
    seen = {}

    class _Fut:
        def result(self):
            return None

    def _submit(fn, task):
        seen["timeout"] = task.config.dry_run_timeout
        seen["modes"] = list(task.config.modes)
        (task.config.path).mkdir(parents=True, exist_ok=True)
        (task.config.path / "mem.txt").write_text("0")
        return _Fut()

    nett.executor = SimpleNamespace(submit=_submit)
    monkeypatch.setattr(nett_module, "future_wait", lambda *a, **k: None)

    brain = SimpleNamespace(envs_per_brain=1)
    body = SimpleNamespace(adjust_to_agent=lambda env, **kw: None)
    env = SimpleNamespace(conditions=["Object1"], num_brains=1, num_envs=4)

    NETT._estimate_task_memory_via_dry_run(nett, brain, body, env, tmp_path, mode="test")
    assert seen["timeout"] == nett_module._dry_run_timeout_for("test")
    assert seen["timeout"] > 0
    assert seen["modes"] == ["test"]  # the probe's mode is the caller's

    NETT._estimate_task_memory_via_dry_run(nett, brain, body, env, tmp_path, mode="train")
    assert seen["timeout"] == nett_module._dry_run_timeout_for("train")
    # A train probe runs a whole rollout; a test probe runs a few steps. One cap
    # for both would either tax the search or false-fail the slow one.
    assert nett_module._dry_run_timeout_for("train") > nett_module._dry_run_timeout_for("test")

    # A task that is not a dry run carries no cap.
    real = TaskConfig("c", tmp_path, ["train"])
    assert real.dry_run_timeout is None


def test_set_dry_run_carries_the_timeout(tmp_path):
    from nett_skrl.runtime.task import Task

    task = Task(
        SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
        "c", tmp_path, ["train"],
    )
    assert task.config.dry_run_timeout is None
    task.set_dry_run(True, timeout=42.0)
    assert task.config.dry_run is True and task.config.dry_run_timeout == 42.0


# --- config.yaml records the RESOLVED value -------------------------------


def test_auto_envs_requires_auto_memory():
    """Without a dry run there is nothing to derive the ceiling from, so this must
    fail loudly rather than quietly fall back to the recipe's count."""
    nett = object.__new__(NETT)
    nett.logger = logging.getLogger("test")
    nett.output_path = Path("/tmp")
    with pytest.raises(ValueError, match="requires task_memory"):
        NETT.single_run(
            nett,
            name="x",
            environment={},
            episodes={"train": 1},
            task_memory=8.0,
            max_parallel_envs="auto",
        )


def test_capped_num_envs_passes_auto_through():
    """capped_num_envs must not try to int("auto"); it returns the recipe's preferred
    count as the probe baseline and leaves resolution to the VRAM model."""
    assert capped_num_envs(
        num_brains=2, preferred_envs_per_brain=8, max_parallel_envs="auto"
    ) == 16


def test_config_snapshot_records_resolved_integers(tmp_path):
    nett = object.__new__(NETT)
    params = {"task_memory": "auto", "max_parallel_envs": "auto"}
    NETT._write_config_snapshot(tmp_path, params)
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == params

    params["max_parallel_envs"] = 16
    params["task_memory"] = 8.25
    NETT._write_config_snapshot(tmp_path, params)
    written = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert written["max_parallel_envs"] == 16
    assert written["task_memory"] == 8.25
    assert "auto" not in written.values()


def test_recorded_test_num_envs_matches_the_worker_selector(tmp_path):
    """The orchestrator records what the worker will independently compute. If these
    two ever disagree the recorded config stops being the reproducibility contract."""
    nett = object.__new__(NETT)
    nett.logger = logging.getLogger("test")
    env = SimpleNamespace(
        conditions=["ship", "fork"],
        iterations_per_test_episode={"ship": 52, "fork": 13},
    )
    recorded = NETT._resolved_test_num_envs(
        nett, env, num_brains=1, ceiling=64, episodes={"test": 20}
    )
    for condition, rows in env.iterations_per_test_episode.items():
        expected, _ = select_test_num_envs(min(64, rows * 20), 1, rows * 20)
        assert recorded[condition] == expected
        assert is_valid_num_envs(recorded[condition], 1)


def test_single_run_resolves_auto_end_to_end(monkeypatch, tmp_path):
    """The whole wiring: "auto" in, integers out. build_tasks must never see the
    string -- the test selector does min(max_envs, total) on it and would TypeError."""
    from concurrent.futures import Future

    import nett_skrl.nett as nett_module

    class _Brain:
        envs_per_brain = 64
        iterations_per_test_episode = {"Object1": 13}

        def __init__(self, **kwargs):
            pass

        def env_reward_types(self):
            return ()

        def calc_iterations(self, *args):
            pass

    class _Body:
        def adjust_to_agent(self, env, **kwargs):
            env.num_brains = kwargs["num_brains"]
            env.num_envs = kwargs["num_envs"]

    class _Environment:
        conditions = ["Object1"]
        iterations_per_test_episode = {"Object1": 13}
        num_brains = 1
        num_envs = 1

        def __init__(self, **kwargs):
            pass

    class _Executor:
        def submit(self, *args, **kwargs):
            fut = Future()
            fut.set_result(None)
            return fut

    built = {}
    monkeypatch.delenv("NETT_TEST_ENVS", raising=False)
    monkeypatch.setattr(nett_module, "Brain", _Brain)
    monkeypatch.setattr(nett_module, "Body", _Body)
    monkeypatch.setattr(nett_module, "Environment", _Environment)
    monkeypatch.setattr(nett_module, "future_wait", lambda *a, **k: None)
    monkeypatch.setattr(
        nett_module, "build_tasks",
        lambda *a, **k: built.update(kwargs=k, args=a) or ["task"],
    )

    # fixed=1GB, per_env=1GB, free=26.25GB -> fits 20 -> must snap to 16 (4x4).
    nett, body, _ = _planner(free_gb=26.25, per_env_gb=1.0, fixed_gb=1.0)
    monkeypatch.setattr(nett_module, "Body", lambda **kw: body)
    nett.output_path = tmp_path
    nett.executor = _Executor()
    nett._assign_task = lambda task: None

    NETT.single_run(
        nett,
        name="run",
        environment={"design_sheet": "design.csv"},
        episodes={"train": 10, "test": 20},
        task_memory="auto",
        max_parallel_envs="auto",
    )

    assert built["kwargs"]["max_parallel_envs"] == 16
    assert isinstance(built["kwargs"]["max_parallel_envs"], int)

    written = yaml.safe_load((tmp_path / "run" / "config.yaml").read_text())
    assert written["max_parallel_envs"] == 16
    assert written["task_memory"] == pytest.approx(17.0)  # fixed 1 + 16 envs
    # 13 rows x 20 episodes = 260 test episodes; ceiling 16 -> 13 (4x4) divides 260.
    assert written["resolved_test_num_envs"] == {"Object1": 13}


def test_no_test_episodes_records_nothing(tmp_path):
    nett = object.__new__(NETT)
    env = SimpleNamespace(conditions=["ship"], iterations_per_test_episode={"ship": 52})
    assert NETT._resolved_test_num_envs(
        nett, env, num_brains=1, ceiling=16, episodes={"train": 100}
    ) is None


def test_spawn_passes_the_cap_to_join_with_reap(monkeypatch, tmp_path):
    """The bound is only real if it reaches join_with_reap. This is the wiring that
    was missing: the reap module has supported an absolute cap all along, but nothing
    passed one, and NETT_REAP_TIMEOUT defaults to 0 (disabled) -- so a wedged probe
    joined forever."""
    import nett_skrl.runtime.task_runner as tr
    from nett_skrl.runtime.task import Task

    seen = {}

    class _Proc:
        exitcode = 0
        pid = 1234

        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

    class _Reaper:
        logger = logging.getLogger("test")

        def __init__(self, **kw):
            pass

        def launch_scope(self):
            from contextlib import nullcontext

            return nullcontext()

        def adopt(self, pid):
            pass

    monkeypatch.setattr(tr, "TaskReaper", _Reaper)
    monkeypatch.setattr(
        tr, "join_with_reap",
        lambda p, r, **kw: seen.update(kw) or "exited",
    )
    monkeypatch.setattr(
        "multiprocessing.get_context",
        lambda m: SimpleNamespace(Process=lambda **kw: _Proc()),
    )

    task = Task(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
                "c", tmp_path, ["train"])
    task.set_device(0)

    tr._spawn_mode_subprocess(task, "train")
    assert seen["absolute_timeout"] is None  # real run: unbounded, as before

    task.set_dry_run(True, timeout=123.0)
    tr._spawn_mode_subprocess(task, "train")
    assert seen["absolute_timeout"] == 123.0


def test_dry_run_timeout_env_override_applies_to_both_modes(monkeypatch):
    import nett_skrl.nett as nett_module

    monkeypatch.delenv("NETT_DRY_RUN_TIMEOUT", raising=False)
    assert nett_module._dry_run_timeout_for("test") == 300.0
    assert nett_module._dry_run_timeout_for("train") == 900.0
    # Read at call time, so an operator override works without reimporting.
    monkeypatch.setenv("NETT_DRY_RUN_TIMEOUT", "42")
    assert nett_module._dry_run_timeout_for("test") == 42.0
    assert nett_module._dry_run_timeout_for("train") == 42.0


# --- do not probe the rung that would OOM ---------------------------------


def test_min_consumed_is_a_lower_bound_on_a_convex_curve():
    """The direction is the whole point: a line through two points of a CONVEX curve
    stays UNDER it, so extending it under-estimates cost. (The same arithmetic used as
    an upper bound on CAPACITY is the refuted model that claimed 2916 envs fit.)"""
    from nett_skrl.nett import _min_consumed

    # The measured curve (GB at n envs), convex throughout.
    curve = {16: 2.62, 36: 2.85, 64: 3.47, 144: 6.23, 256: 13.48}
    lo, hi = (64, curve[64]), (144, curve[144])
    bound = _min_consumed(lo, hi, 256)
    assert bound <= curve[256], "must never over-estimate a convex curve"
    # And it is informative, not a trivial floor.
    assert bound > curve[144]


def test_min_consumed_is_exact_on_a_straight_line():
    from nett_skrl.nett import _min_consumed

    assert _min_consumed((10, 10.0), (20, 20.0), 40) == pytest.approx(40.0)


def test_cannot_fit_only_claims_what_it_can_prove():
    from nett_skrl.nett import _cannot_fit

    lo, hi = (121, 5.53), (242, 13.03)  # measured, run 8
    # 484 needs >= 13.03 + 0.062*242 = ~28GB: provably over an 18.58GB budget.
    assert _cannot_fit(lo, hi, 484, 18.58) is True
    # With a huge budget the same rung is not provably too big, so it must be probed.
    assert _cannot_fit(lo, hi, 484, 100.0) is False


def test_search_skips_the_rung_it_can_prove_will_not_fit(tmp_path):
    """Regression: the search overshoots to find the ceiling, and that last rung is
    the one that OOMs -- costing the whole probe timeout and pinning the GPU until the
    reap, because a scene-build OOM wedges rather than raising."""
    fixed = 2.0

    def consumed_gb(n):  # convex, like the real thing
        return fixed + 0.002 * (n ** 1.5)

    nett, body, plans = _planner(free_gb=23.5, per_env_gb=0, fixed_gb=0)

    def _estimate(self, brain, body_, env, output_dir, mode="train"):
        return consumed_gb(env.num_envs) * 1024**3

    nett._estimate_task_memory_via_dry_run = _estimate.__get__(nett, NETT)

    _memory, num_envs = _search(
        nett, body, num_brains=1, seed=16, max_envs=None, tmp_path=tmp_path
    )
    budget = 23.5 * NETT._VRAM_SAFETY

    # Every count it actually ran must have fit: it never paid for an OOM.
    for n in plans:
        assert consumed_gb(n) <= budget, f"probed {n}, which does not fit"
    # And the answer is unchanged -- the next rung up genuinely does not fit.
    assert consumed_gb(snap_num_envs(2 * num_envs, 1)[0]) > budget
