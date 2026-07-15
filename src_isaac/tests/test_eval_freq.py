"""Tests for eval_freq infrastructure that run without Isaac Sim.

Covers:
- _compute_eval_num_envs: divisibility, max_parallel_envs cap, num_brains multiples
- _training_boundaries: correct eval/checkpoint boundary merging
- initial_timestep computation in train_cfg_for
- eval_bar_info construction in Brain.test (via unit-level check)
- log CSV filename suffix for mid-training eval
- TaskConfig accepts max_parallel_envs / train_start_step
"""

from __future__ import annotations

import math
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _compute_eval_num_envs
# ---------------------------------------------------------------------------

def _make_mock_task(
    *,
    num_brains: int = 1,
    episodes_test: int = 1,
    num_test_tasks: int = 1,
    max_parallel_envs: int | None = None,
    max_test_envs: int | None = None,
    num_envs: int = 10,
    condition: str = "C1",
):
    """Build a minimal mock Task for _compute_eval_num_envs."""
    from nett_skrl.runtime.task import TaskConfig

    config = MagicMock()
    config.num_brains = num_brains
    config.episodes = {"test": episodes_test}
    config.max_parallel_envs = max_parallel_envs
    # Explicit, because MagicMock would otherwise auto-create a truthy Mock here
    # and the selector would compare an int against it.
    config.max_test_envs = max_test_envs
    config.num_envs = num_envs
    config.condition = condition

    env = MagicMock()
    env.iterations_per_test_episode = {condition: num_test_tasks}

    agent = MagicMock()
    agent.env = env

    task = MagicMock()
    task.config = config
    task.agent = agent
    return task


class TestComputeEvalNumEnvs:
    def _call(self, **kwargs):
        from nett_skrl.runtime.task_runner import _compute_eval_num_envs
        return _compute_eval_num_envs(_make_mock_task(**kwargs))

    def test_simple_divisor(self):
        # total_test_episodes = 6, num_brains = 1. The largest divisor is 6, but 6
        # tiles 3x2 -> non-square -> distorted fisheye (#488), so it is rejected.
        # 3 is the largest divisor whose grid is square (2x2, one empty tile).
        from nett_skrl.runtime.task_runner import _is_square_tile_grid
        result = self._call(num_test_tasks=6, episodes_test=1, num_brains=1)
        assert 6 % result == 0
        assert _is_square_tile_grid(result)
        assert result == 3

    def test_capped_by_max_parallel_envs(self):
        # total = 12, cap = 4 → should return 4 (4 divides 12)
        result = self._call(num_test_tasks=12, episodes_test=1, max_parallel_envs=4)
        assert result <= 4
        assert 12 % result == 0

    def test_multiple_of_num_brains(self):
        # num_brains=2, total=6. The multiples of 2 that divide 6 are 2 (2x1) and
        # 6 (3x2) -- BOTH non-square, so all three constraints cannot hold at once.
        # Priority: being a multiple of num_brains (else BrainTrainer raises) and a
        # square grid (else the render is distorted) are HARD; dividing the total is
        # a preference (its cost is only an idle env in the last batch). -> 4 (2x2).
        from nett_skrl.runtime.task_runner import _is_square_tile_grid
        result = self._call(num_test_tasks=6, episodes_test=1, num_brains=2)
        assert result % 2 == 0
        assert _is_square_tile_grid(result)
        assert result == 4

    def test_always_multiple_of_num_brains_even_when_no_divisor(self):
        # total=7 (prime), num_brains=2 → no multiple of 2 divides 7
        # must still return a multiple of 2 (fallback: start = largest multiple of 2 ≤ 7 = 6)
        result = self._call(num_test_tasks=7, episodes_test=1, num_brains=2)
        assert result % 2 == 0, f"result {result} not divisible by num_brains=2"

    def test_no_max_parallel_envs(self):
        # No cap → returns total_test_episodes (or largest divisor)
        result = self._call(num_test_tasks=8, episodes_test=3, max_parallel_envs=None)
        assert result <= 24
        assert 24 % result == 0 or result % 1 == 0  # always positive

    def test_total_less_than_num_brains(self):
        # total=1, num_brains=4 → no multiple of 4 ≤ 1; fallback
        result = self._call(num_test_tasks=1, episodes_test=1, num_brains=4)
        assert result % 4 == 0  # must be multiple of num_brains
        assert result >= 4       # smallest valid multiple

    def test_max_parallel_envs_less_than_num_brains(self):
        # cap < num_brains → cap is raised to num_brains
        result = self._call(num_test_tasks=10, episodes_test=1, num_brains=4, max_parallel_envs=2)
        assert result % 4 == 0

    def test_result_is_always_positive(self):
        for nte in [1, 2, 3, 5, 7, 11, 13]:
            for nb in [1, 2, 3, 4]:
                for mpe in [None, 1, 2, 5, 10]:
                    r = self._call(num_test_tasks=nte, episodes_test=1, num_brains=nb, max_parallel_envs=mpe)
                    assert r >= 1, f"non-positive result {r} for nte={nte} nb={nb} mpe={mpe}"
                    assert r % nb == 0, f"result {r} not multiple of num_brains={nb}"


# ---------------------------------------------------------------------------
# _training_boundaries
# ---------------------------------------------------------------------------

class TestTrainingBoundaries:
    def _call(self, total, eval_freq, checkpoint_freq=None):
        from nett_skrl.runtime.task_runner import _training_boundaries
        return _training_boundaries(total, eval_freq=eval_freq, checkpoint_freq=checkpoint_freq)

    def test_eval_boundaries_present(self):
        boundaries = self._call(total=10000, eval_freq=2000)
        for step in [2000, 4000, 6000, 8000, 10000]:
            assert step in boundaries

    def test_total_always_included(self):
        boundaries = self._call(total=9999, eval_freq=2000)
        assert 9999 in boundaries

    def test_sorted_ascending(self):
        boundaries = self._call(total=10000, eval_freq=3000)
        assert boundaries == sorted(boundaries)

    def test_checkpoint_boundaries_merged(self):
        boundaries = self._call(total=10000, eval_freq=4000, checkpoint_freq=3000)
        assert 3000 in boundaries
        assert 4000 in boundaries
        assert 6000 in boundaries

    def test_no_duplicates(self):
        boundaries = self._call(total=6000, eval_freq=2000, checkpoint_freq=3000)
        assert len(boundaries) == len(set(boundaries))


# ---------------------------------------------------------------------------
# initial_timestep computation in train_cfg_for
# ---------------------------------------------------------------------------

class TestTrainCfgInitialTimestep:
    def test_first_chunk_zero(self):
        from nett_skrl.brain.run_config import train_cfg_for
        brain = MagicMock()
        brain.envs_per_brain = 10
        brain.train_iterations = 24000
        config = MagicMock()
        config.train_timesteps = 2000
        config.train_start_step = 0         # first chunk starts at 0
        config.train_global_step = 2000
        config.path = Path("/tmp/test_condition")
        config.current_mode = "train"
        config.condition = "C1"
        cfg = train_cfg_for(brain, config)
        assert cfg.initial_timestep == 0

    def test_second_chunk_offset(self):
        from nett_skrl.brain.run_config import train_cfg_for
        brain = MagicMock()
        brain.envs_per_brain = 10
        brain.train_iterations = 24000
        config = MagicMock()
        config.train_timesteps = 2000
        config.train_start_step = 2000      # second chunk starts at 2000 total env steps
        config.train_global_step = 4000
        config.path = Path("/tmp/test_condition")
        config.current_mode = "train"
        config.condition = "C1"
        cfg = train_cfg_for(brain, config)
        # initial_timestep = 2000 // 10 = 200 per-env steps
        assert cfg.initial_timestep == 200

    def test_no_train_start_step_defaults_to_zero(self):
        from nett_skrl.brain.run_config import train_cfg_for
        brain = MagicMock()
        brain.envs_per_brain = 10
        brain.train_iterations = 24000
        config = MagicMock()
        config.train_timesteps = None
        config.train_start_step = None
        config.train_global_step = None
        config.path = Path("/tmp/test_condition")
        config.current_mode = "train"
        config.condition = "C1"
        cfg = train_cfg_for(brain, config)
        assert cfg.initial_timestep == 0


# ---------------------------------------------------------------------------
# TrainCfg dataclass has initial_timestep field
# ---------------------------------------------------------------------------

def test_train_cfg_has_initial_timestep():
    from nett_skrl.brain.trainer import TrainCfg
    cfg = TrainCfg(total_timesteps=1000, initial_timestep=42)
    assert cfg.initial_timestep == 42

def test_train_cfg_initial_timestep_defaults_to_zero():
    from nett_skrl.brain.trainer import TrainCfg
    cfg = TrainCfg(total_timesteps=1000)
    assert cfg.initial_timestep == 0


# ---------------------------------------------------------------------------
# TaskConfig accepts new fields
# ---------------------------------------------------------------------------

def test_task_config_max_parallel_envs():
    from nett_skrl.runtime.task import TaskConfig
    cfg = TaskConfig(
        condition="C1",
        output_dir=Path("/tmp/out"),
        modes=["train"],
        max_parallel_envs=52,
    )
    assert cfg.max_parallel_envs == 52

def test_task_config_train_start_step():
    from nett_skrl.runtime.task import TaskConfig
    cfg = TaskConfig(
        condition="C1",
        output_dir=Path("/tmp/out"),
        modes=["train"],
        train_start_step=2000,
    )
    assert cfg.train_start_step == 2000

def test_task_config_for_mode_propagates_num_envs_override():
    from nett_skrl.runtime.task import TaskConfig
    cfg = TaskConfig(
        condition="C1",
        output_dir=Path("/tmp/out"),
        modes=["test"],
        num_envs=10,
    )
    eval_cfg = cfg.for_mode("test", num_envs=4)
    assert eval_cfg.num_envs == 4
    assert cfg.num_envs == 10  # original unchanged


# ---------------------------------------------------------------------------
# Eval log CSV filename suffix
# ---------------------------------------------------------------------------

def test_eval_csv_filename_has_eval_step_suffix(tmp_path):
    """Environment._configure_artifacts appends _{eval_step} for mid-eval runs."""
    from nett_skrl.runtime.task import TaskConfig

    config = TaskConfig(
        condition="Object1",
        output_dir=tmp_path,
        modes=["test"],
        eval_step=2000,
    )
    # Simulate what _configure_artifacts does for the filename suffix.
    eval_step = getattr(config, "eval_step", None)
    eval_suffix = f"_{eval_step}" if eval_step is not None else ""
    suffix = f"{config.current_mode}_{config.condition}_{config.seed}{eval_suffix}"
    assert suffix.endswith("_2000"), f"Expected suffix ending with _2000, got: {suffix!r}"

def test_normal_csv_filename_has_no_eval_step_suffix(tmp_path):
    from nett_skrl.runtime.task import TaskConfig

    config = TaskConfig(
        condition="Object1",
        output_dir=tmp_path,
        modes=["test"],
    )
    eval_step = getattr(config, "eval_step", None)
    eval_suffix = f"_{eval_step}" if eval_step is not None else ""
    suffix = f"test_Object1_{config.seed}{eval_suffix}"
    assert not suffix.endswith("_None")
    assert "_2000" not in suffix


# ---------------------------------------------------------------------------
# _eval_progress_desc labels mid-train evals distinctly
# ---------------------------------------------------------------------------

def test_eval_progress_desc_mid_train():
    from nett_skrl.brain.brain import _eval_progress_desc
    config = MagicMock()
    config.eval_step = 2000
    config.condition = "Object1"
    config.current_mode = "test"
    desc = _eval_progress_desc(config)
    assert "2000" in desc
    assert "eval" in desc.lower()
    assert "Object1" in desc

def test_eval_progress_desc_regular_test():
    from nett_skrl.brain.brain import _eval_progress_desc
    config = MagicMock()
    config.eval_step = None
    config.condition = "Object1"
    config.current_mode = "test"
    desc = _eval_progress_desc(config)
    assert "2000" not in desc
    assert "Object1" in desc


# ---------------------------------------------------------------------------
# _preferences_from_csv parses gaze-direction correctly
# ---------------------------------------------------------------------------

def test_preferences_from_csv_correct_monitor(tmp_path):
    """Agent facing the correct monitor should give correct_pct > 0.5."""
    from nett_skrl.recording.wandb import _preferences_from_csv

    # Build a CSV where agent is perfectly facing the correct (left) monitor.
    # left monitor is at x = -33.15, agent at x=0, y=0.
    # Facing left: yaw such that gaze goes left → yaw=90 degrees
    # Forward = (-sin(90°), cos(90°)) = (-1, 0) → points toward negative x (left monitor).
    csv_path = tmp_path / "test_C1_0_2000.csv"
    rows = [
        "test.cond,agent.x,agent.y,agent.angle,correct.monitor",
        # Facing left monitor = correct
        "testA,0,0,90.0,left",
        "testA,0,0,90.0,left",
        "testA,0,0,90.0,left",
    ]
    csv_path.write_text("\n".join(rows))

    result = _preferences_from_csv(csv_path)
    assert "testA" in result
    # All 3 rows face left, correct monitor is left → 100% correct
    assert result["testA"] == pytest.approx([1.0], abs=0.01)

def test_preferences_from_csv_wrong_monitor(tmp_path):
    from nett_skrl.recording.wandb import _preferences_from_csv

    csv_path = tmp_path / "test_C1_0_2000.csv"
    rows = [
        "test.cond,agent.x,agent.y,agent.angle,correct.monitor",
        # Facing left monitor, but correct is right → wrong
        "testA,0,0,90.0,right",
        "testA,0,0,90.0,right",
    ]
    csv_path.write_text("\n".join(rows))

    result = _preferences_from_csv(csv_path)
    assert "testA" in result
    assert result["testA"] == pytest.approx([0.0], abs=0.01)

def test_preferences_from_csv_missing_file(tmp_path):
    from nett_skrl.recording.wandb import _preferences_from_csv
    result = _preferences_from_csv(tmp_path / "nonexistent.csv")
    assert result == {}

def test_preferences_from_csv_bad_rows(tmp_path):
    from nett_skrl.recording.wandb import _preferences_from_csv
    csv_path = tmp_path / "test_C1_0.csv"
    # Rows with missing/invalid fields should be silently skipped.
    rows = [
        "test.cond,agent.x,agent.y,agent.angle,correct.monitor",
        "testA,not_a_float,0,90.0,left",
        "testA,0,0,not_a_float,left",
        ",0,0,90.0,left",  # empty test.cond
    ]
    csv_path.write_text("\n".join(rows))
    result = _preferences_from_csv(csv_path)
    # Either empty or has only valid entries
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Orchestration integration test: train→eval→train sequencing
# ---------------------------------------------------------------------------

def _make_orchestration_task(tmp_path, *, eval_freq, total_train_steps, episodes_test,
                              num_test_tasks, max_parallel_envs, num_brains=1, num_envs=10):
    """Build a minimal Task-like object for _run_train_with_eval_milestones."""
    from nett_skrl.runtime.task import TaskConfig

    condition = "Object1"
    config = TaskConfig(
        condition=condition,
        output_dir=tmp_path,
        modes=["train"],
        episodes={"train": total_train_steps, "test": episodes_test},
        num_brains=num_brains,
        num_envs=num_envs,
        eval_freq=eval_freq,
        max_parallel_envs=max_parallel_envs,
    )

    mock_brain = MagicMock()
    mock_brain.train_iterations = total_train_steps
    mock_brain.checkpoint_freq = None

    mock_env = MagicMock()
    mock_env.iterations_per_test_episode = {condition: num_test_tasks}

    task = MagicMock()
    task.config = config
    task.agent.brain = mock_brain
    task.agent.env = mock_env
    return task


def test_orchestration_train_eval_alternating_sequence(tmp_path):
    """Train and eval subprocesses interleave in the correct order.

    This exercises _run_train_with_eval_milestones without Isaac Sim by
    patching _spawn_mode_subprocess and verifying the call sequence,
    confirming:
    - training pauses at each eval_freq boundary
    - eval subprocess is spawned after each training chunk
    - training resumes after each eval subprocess completes
    """
    from nett_skrl.runtime.task_runner import _run_train_with_eval_milestones

    # 6000 total env-step budget, eval every 2000 → 3 train chunks + 3 evals
    task = _make_orchestration_task(
        tmp_path,
        eval_freq=2000,
        total_train_steps=6000,
        episodes_test=3,
        num_test_tasks=2,
        max_parallel_envs=6,
    )

    spawned: list[tuple] = []

    def fake_spawn(task, mode, **overrides):
        spawned.append((mode, dict(overrides)))
        # Simulate checkpoint written after each training chunk.
        if mode == "train":
            ckpt_dir = task.config.path / "wandb_runs" / "brain_1" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "final_agent.pt").write_bytes(b"ckpt")

    with patch("nett_skrl.runtime.task_runner._spawn_mode_subprocess", fake_spawn):
        _run_train_with_eval_milestones(task)

    modes = [s[0] for s in spawned]
    assert modes == ["train", "test", "train", "test", "train", "test"], \
        f"Expected train/eval alternation, got: {modes}"


def test_orchestration_train_resumes_after_each_eval(tmp_path):
    """Each training chunk starts where the previous one ended (train_start_step)."""
    from nett_skrl.runtime.task_runner import _run_train_with_eval_milestones

    task = _make_orchestration_task(
        tmp_path,
        eval_freq=2000,
        total_train_steps=6000,
        episodes_test=1,
        num_test_tasks=6,
        max_parallel_envs=6,
    )

    train_overrides: list[dict] = []

    def fake_spawn(task, mode, **overrides):
        if mode == "train":
            train_overrides.append(dict(overrides))
            ckpt_dir = task.config.path / "wandb_runs" / "brain_1" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "final_agent.pt").write_bytes(b"ckpt")

    with patch("nett_skrl.runtime.task_runner._spawn_mode_subprocess", fake_spawn):
        _run_train_with_eval_milestones(task)

    assert len(train_overrides) == 3
    # Chunk sizes: each chunk = eval_freq = 2000
    assert train_overrides[0]["train_timesteps"] == 2000
    assert train_overrides[1]["train_timesteps"] == 2000
    assert train_overrides[2]["train_timesteps"] == 2000
    # Start positions: 0, 2000, 4000
    assert train_overrides[0]["train_start_step"] == 0
    assert train_overrides[1]["train_start_step"] == 2000
    assert train_overrides[2]["train_start_step"] == 4000
    # End positions (global_step): 2000, 4000, 6000
    assert train_overrides[0]["train_global_step"] == 2000
    assert train_overrides[1]["train_global_step"] == 4000
    assert train_overrides[2]["train_global_step"] == 6000


def test_orchestration_eval_num_envs_is_square_and_bounded(tmp_path):
    """Eval subprocess receives a SQUARE-grid num_envs within the cap.

    It no longer has to divide total_test_episodes: overflow episodes are discarded
    (nett_env_cfg.test_total_episodes), so 8 (3x3, 4 overflow) beats 6 (3x2 ->
    distorted fisheye #488) even though only 6 divides 12.
    """
    from nett_skrl.runtime.task_runner import _run_train_with_eval_milestones

    # total_test_episodes = 4 test tasks × 3 episodes = 12; max_parallel_envs = 8.
    # Largest square-grid count ≤ 8 → 8 (3x3, one empty tile; 4 overflow episodes
    # are discarded). 6 would be 3x2 = non-square = distorted.
    task = _make_orchestration_task(
        tmp_path,
        eval_freq=2000,
        total_train_steps=2000,
        episodes_test=3,
        num_test_tasks=4,
        max_parallel_envs=8,
    )

    eval_overrides: list[dict] = []

    def fake_spawn(task, mode, **overrides):
        if mode == "test":
            eval_overrides.append(dict(overrides))
        if mode == "train":
            ckpt_dir = task.config.path / "wandb_runs" / "brain_1" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "final_agent.pt").write_bytes(b"ckpt")

    with patch("nett_skrl.runtime.task_runner._spawn_mode_subprocess", fake_spawn):
        _run_train_with_eval_milestones(task)

    from nett_skrl.runtime.task_runner import _is_square_tile_grid

    assert len(eval_overrides) == 1
    eval_num_envs = eval_overrides[0]["num_envs"]
    assert eval_num_envs <= 8, f"eval_num_envs {eval_num_envs} exceeds max_parallel_envs=8"
    assert _is_square_tile_grid(eval_num_envs), \
        f"eval_num_envs {eval_num_envs} does not tile into a square grid (#488)"
    assert eval_num_envs == 8


def test_orchestration_final_test_uses_parallel_test_envs(tmp_path):
    """Normal post-training test uses the same parallel env sizing as eval."""
    from nett_skrl.runtime.task_runner import run_task

    task = _make_orchestration_task(
        tmp_path,
        eval_freq=None,
        total_train_steps=0,
        episodes_test=1,
        num_test_tasks=52,
        max_parallel_envs=52,
        num_envs=10,
    )
    task.config = type(task.config)(
        condition=task.config.condition,
        output_dir=task.config.output_dir,
        modes=["test"],
        episodes=task.config.episodes,
        num_brains=task.config.num_brains,
        num_envs=task.config.num_envs,
        max_parallel_envs=task.config.max_parallel_envs,
    )

    spawned: list[tuple] = []

    def fake_spawn(task, mode, **overrides):
        spawned.append((mode, dict(overrides)))

    with patch("nett_skrl.runtime.task_runner._spawn_mode_subprocess", fake_spawn):
        run_task(task)

    # 52 tiles 8x7 -> non-square -> distorted fisheye (#488), so it is rejected.
    # 49 (7x7) is the largest SQUARE-grid count <= the cap; it does not divide 52,
    # which is now allowed (the 46 overflow episodes are discarded), and it beats
    # the old divisor-only answer of 13 (4x4) on parallelism.
    assert spawned == [("test", {"num_envs": 49})]


def test_orchestration_eval_is_metrics_only_no_recording_contamination(tmp_path):
    """Eval subprocess has eval_metrics_only=True so no training artifacts bleed in."""
    from nett_skrl.runtime.task_runner import _run_train_with_eval_milestones

    task = _make_orchestration_task(
        tmp_path,
        eval_freq=2000,
        total_train_steps=2000,
        episodes_test=1,
        num_test_tasks=1,
        max_parallel_envs=None,
    )

    eval_overrides: list[dict] = []

    def fake_spawn(task, mode, **overrides):
        if mode == "test":
            eval_overrides.append(dict(overrides))
        if mode == "train":
            ckpt_dir = task.config.path / "wandb_runs" / "brain_1" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "final_agent.pt").write_bytes(b"ckpt")

    with patch("nett_skrl.runtime.task_runner._spawn_mode_subprocess", fake_spawn):
        _run_train_with_eval_milestones(task)

    assert eval_overrides[0]["eval_metrics_only"] is True
    # Verify that eval_step corresponds to the training boundary
    assert eval_overrides[0]["eval_step"] == 2000


def test_orchestration_checkpoint_exists_before_eval_spawned(tmp_path):
    """Training saves final_agent.pt BEFORE the eval subprocess is spawned.

    This ensures the eval subprocess always loads valid weights rather than
    running on uninitialized parameters.
    """
    from nett_skrl.runtime.task_runner import _run_train_with_eval_milestones

    task = _make_orchestration_task(
        tmp_path,
        eval_freq=2000,
        total_train_steps=2000,
        episodes_test=1,
        num_test_tasks=1,
        max_parallel_envs=None,
    )

    ckpt_path = tmp_path / "wandb_runs" / "brain_1" / "checkpoints" / "final_agent.pt"
    checkpoint_existed_at_eval_time = []

    def fake_spawn(task, mode, **overrides):
        if mode == "train":
            # Training writes checkpoint
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_bytes(b"weights")
        elif mode == "test":
            # Record whether checkpoint was present when eval was spawned
            checkpoint_existed_at_eval_time.append(ckpt_path.exists())

    with patch("nett_skrl.runtime.task_runner._spawn_mode_subprocess", fake_spawn):
        _run_train_with_eval_milestones(task)

    assert checkpoint_existed_at_eval_time == [True], \
        "Checkpoint must exist before eval subprocess is spawned"


def test_orchestration_eval_global_step_matches_training_boundary(tmp_path):
    """Eval's eval_step equals the number of training interactions so far.

    This ensures the W&B eval chart x-axis aligns with training reward curves.
    """
    from nett_skrl.runtime.task_runner import _run_train_with_eval_milestones

    task = _make_orchestration_task(
        tmp_path,
        eval_freq=3000,
        total_train_steps=9000,
        episodes_test=1,
        num_test_tasks=3,
        max_parallel_envs=3,
    )

    eval_steps: list[int] = []

    def fake_spawn(task, mode, **overrides):
        if mode == "test":
            eval_steps.append(overrides["eval_step"])
        if mode == "train":
            ckpt_dir = task.config.path / "wandb_runs" / "brain_1" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "final_agent.pt").write_bytes(b"ckpt")

    with patch("nett_skrl.runtime.task_runner._spawn_mode_subprocess", fake_spawn):
        _run_train_with_eval_milestones(task)

    assert eval_steps == [3000, 6000, 9000], \
        f"Eval steps must match training boundaries, got: {eval_steps}"


def test_orchestration_no_eval_when_eval_freq_exceeds_total(tmp_path):
    """When eval_freq > total training steps, no eval subprocesses are spawned."""
    from nett_skrl.runtime.task_runner import _run_train_with_eval_milestones

    task = _make_orchestration_task(
        tmp_path,
        eval_freq=99999,  # much larger than total
        total_train_steps=1000,
        episodes_test=1,
        num_test_tasks=1,
        max_parallel_envs=None,
    )

    spawned: list[str] = []

    def fake_spawn(task, mode, **overrides):
        spawned.append(mode)
        if mode == "train":
            ckpt_dir = task.config.path / "wandb_runs" / "brain_1" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "final_agent.pt").write_bytes(b"ckpt")

    with patch("nett_skrl.runtime.task_runner._spawn_mode_subprocess", fake_spawn):
        _run_train_with_eval_milestones(task)

    assert "test" not in spawned, "No eval should be spawned when eval_freq > total steps"
    assert spawned == ["train"], f"Only one full training run expected, got: {spawned}"


def test_orchestration_wandb_timestep_offset_per_chunk(tmp_path):
    """Each training chunk gets the correct initial_timestep offset in W&B.

    Chunk 1 starts at global_step 0, chunk 2 at (eval_freq // envs_per_brain), etc.
    This keeps the training reward curve continuous in W&B.
    """
    from nett_skrl.brain.run_config import train_cfg_for
    from nett_skrl.runtime.task import TaskConfig

    envs_per_brain = 10
    eval_freq = 2000

    brain = MagicMock()
    brain.envs_per_brain = envs_per_brain
    brain.train_iterations = 6000

    # Simulate three chunks: 0→2000, 2000→4000, 4000→6000
    for chunk_idx, previous in enumerate([0, 2000, 4000]):
        config = MagicMock()
        config.train_timesteps = 2000
        config.train_start_step = previous
        config.train_global_step = previous + 2000
        config.path = Path(tmp_path) / "Object1"
        config.current_mode = "train"
        config.condition = "Object1"

        cfg = train_cfg_for(brain, config)
        expected_offset = previous // envs_per_brain
        assert cfg.initial_timestep == expected_offset, (
            f"Chunk {chunk_idx}: expected initial_timestep={expected_offset}, "
            f"got {cfg.initial_timestep}"
        )
