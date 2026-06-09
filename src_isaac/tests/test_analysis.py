"""Unit tests for the native analysis helpers.

These exercise the physical chamber-thirds preference metric, CSV aggregation,
and merge tree behavior on synthetic data — no Isaac Sim / tfevents involvement.

The preference metric measures physical proximity to the correct monitor:

    correct_pct = steps in correct outer third / steps in either outer third

where the chamber spans [-half_x, +half_x] and the outer thirds are
    left  outer third : agent_x < -half_x / 3
    right outer third : agent_x >  half_x / 3
    middle (excluded) : |agent_x| <= half_x / 3

See ``in_correct_chamber_third()`` in ``nett_skrl.analysis.api`` for the
authoritative implementation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from nett_skrl.analysis import (
    DEFAULT_CHAMBER_HALF_X,
    analyze,
    in_correct_chamber_third,
    looking_at_monitor,
    log_analysis_to_wandb,
    merge,
    normalize_isaac_output,
    test_viz,
    train_viz,
)

# ---------------------------------------------------------------------------
# Physical chamber-thirds metric — unit tests
# ---------------------------------------------------------------------------

HALF_X = DEFAULT_CHAMBER_HALF_X   # 33.15
THRESH = HALF_X / 3               # ~11.05


class TestInCorrectChamberThird:
    """Tests for in_correct_chamber_third()."""

    # -- Basic threshold behaviour -------------------------------------------

    def test_far_left_correct_left(self):
        # Agent at -20 (past threshold) with correct monitor = left → in outer, correct
        in_outer, in_correct = in_correct_chamber_third(-20.0, "left", HALF_X)
        assert in_outer is True
        assert in_correct is True

    def test_far_right_correct_right(self):
        in_outer, in_correct = in_correct_chamber_third(20.0, "right", HALF_X)
        assert in_outer is True
        assert in_correct is True

    def test_far_left_correct_right(self):
        # Agent at left side but correct monitor is right → in outer, not correct
        in_outer, in_correct = in_correct_chamber_third(-20.0, "right", HALF_X)
        assert in_outer is True
        assert in_correct is False

    def test_far_right_correct_left(self):
        in_outer, in_correct = in_correct_chamber_third(20.0, "left", HALF_X)
        assert in_outer is True
        assert in_correct is False

    # -- Middle-third exclusion ----------------------------------------------

    def test_centre_is_excluded(self):
        # x=0 is in the middle third — excluded from both denominator and numerator
        in_outer, in_correct = in_correct_chamber_third(0.0, "left", HALF_X)
        assert in_outer is False
        assert in_correct is False

    def test_just_inside_middle_third_excluded(self):
        # x = THRESH - ε  → still in middle
        x = THRESH - 0.001
        in_outer, _ = in_correct_chamber_third(x, "right", HALF_X)
        assert in_outer is False

        in_outer, _ = in_correct_chamber_third(-x, "left", HALF_X)
        assert in_outer is False

    def test_exactly_at_threshold_not_outer(self):
        # x = THRESH exactly is NOT past the threshold (strict >)
        in_outer, _ = in_correct_chamber_third(THRESH, "right", HALF_X)
        assert in_outer is False

        in_outer, _ = in_correct_chamber_third(-THRESH, "left", HALF_X)
        assert in_outer is False

    def test_just_past_threshold_is_outer(self):
        # x = THRESH + ε  → outer
        x = THRESH + 0.001
        in_outer, _ = in_correct_chamber_third(x, "right", HALF_X)
        assert in_outer is True

        in_outer, _ = in_correct_chamber_third(-x, "left", HALF_X)
        assert in_outer is True

    # -- Near-wall positions -------------------------------------------------

    def test_at_left_wall(self):
        in_outer, in_correct = in_correct_chamber_third(-HALF_X, "left", HALF_X)
        assert in_outer is True
        assert in_correct is True

    def test_at_right_wall(self):
        in_outer, in_correct = in_correct_chamber_third(HALF_X, "right", HALF_X)
        assert in_outer is True
        assert in_correct is True

    # -- Invalid / NaN inputs ------------------------------------------------

    def test_nan_position_returns_false_false(self):
        in_outer, in_correct = in_correct_chamber_third(math.nan, "left", HALF_X)
        assert in_outer is False
        assert in_correct is False

    def test_inf_position_is_excluded(self):
        # ±inf is not a finite position — treated the same as NaN (excluded)
        in_outer, in_correct = in_correct_chamber_third(math.inf, "right", HALF_X)
        assert in_outer is False
        assert in_correct is False

        in_outer, in_correct = in_correct_chamber_third(-math.inf, "left", HALF_X)
        assert in_outer is False
        assert in_correct is False

    # -- Unknown correct_monitor ---------------------------------------------

    def test_unknown_correct_monitor_never_correct(self):
        in_outer, in_correct = in_correct_chamber_third(-20.0, "neither", HALF_X)
        assert in_outer is True      # still in outer third
        assert in_correct is False   # but not correct


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

_HEADER = [
    "env_id", "episode", "step", "agent.x", "agent.z", "agent.angle",
    "head.flexion", "head.lateral", "left.monitor", "right.monitor",
    "correct.monitor", "experiment.phase", "imprint.cond", "test.cond",
]


def _write_test_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    env_id: int,
    test_cond: str,
    agent_x: float,
    correct: str,
    agent_z: float = 0.0,
    yaw: float = 0.0,
) -> dict:
    """Build one test-CSV row with the given agent_x and correct monitor."""
    base = {h: "" for h in _HEADER}
    base.update({
        "env_id": str(env_id), "episode": "0", "step": "0",
        "agent.x": str(agent_x), "agent.z": str(agent_z), "agent.angle": str(yaw),
        "head.flexion": "0.0", "head.lateral": "0.0",
        "left.monitor": "L.mov", "right.monitor": "R.mov",
        "correct.monitor": correct,
        "experiment.phase": "test", "imprint.cond": "Object1",
        "test.cond": test_cond,
    })
    return base


# ---------------------------------------------------------------------------
# test_viz integration tests — now using position-based metric
# ---------------------------------------------------------------------------


class TestTestVizPositionMetric:
    """Integration tests that verify correct_pct uses the chamber-thirds metric."""

    def test_all_steps_in_correct_third_gives_100pct(self, tmp_path):
        """Agent spends all outer-third steps in the correct (left) outer third."""
        run = tmp_path / "run"
        cond_dir = run / "Object1"
        # x = -20: left outer third, correct=left → 100% preference
        rows = [_row(0, "rest", agent_x=-20.0, correct="left") for _ in range(5)]
        _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

        out = test_viz(run, tmp_path / "out")
        with (out / "test_preferences.csv").open() as f:
            records = list(csv.DictReader(f))
        assert len(records) == 1
        assert pytest.approx(float(records[0]["correct_pct"]), abs=1e-6) == 1.0
        assert int(records[0]["n_steps"]) == 5  # all 5 steps counted

    def test_all_steps_in_wrong_third_gives_0pct(self, tmp_path):
        """Agent spends all outer-third steps in the wrong (right) outer third."""
        run = tmp_path / "run"
        cond_dir = run / "Object1"
        rows = [_row(0, "1color", agent_x=20.0, correct="left") for _ in range(8)]
        _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

        out = test_viz(run, tmp_path / "out")
        with (out / "test_preferences.csv").open() as f:
            records = list(csv.DictReader(f))
        assert len(records) == 1
        assert pytest.approx(float(records[0]["correct_pct"]), abs=1e-6) == 0.0

    def test_mixed_correct_and_wrong_outer_third_steps(self, tmp_path):
        """3 correct + 1 wrong outer-third steps → 75%; 4 middle steps excluded."""
        run = tmp_path / "run"
        cond_dir = run / "Object1"
        rows = (
            [_row(0, "2color", agent_x=-20.0, correct="left")] * 3  # correct outer
            + [_row(0, "2color", agent_x=20.0,  correct="left")] * 1  # wrong outer
            + [_row(0, "2color", agent_x=0.0,   correct="left")] * 4  # middle → excluded
        )
        _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

        out = test_viz(run, tmp_path / "out")
        with (out / "test_preferences.csv").open() as f:
            records = list(csv.DictReader(f))
        assert len(records) == 1
        rec = records[0]
        # Only 4 outer-third steps contribute to n_steps
        assert int(rec["n_steps"]) == 4
        assert pytest.approx(float(rec["correct_pct"]), abs=1e-6) == 3 / 4

    def test_all_middle_steps_defaults_to_chance(self, tmp_path):
        """All steps in the middle third → outer_count=0 → default 0.5."""
        run = tmp_path / "run"
        cond_dir = run / "Object1"
        rows = [_row(0, "rest", agent_x=0.0, correct="left") for _ in range(5)]
        _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

        out = test_viz(run, tmp_path / "out")
        with (out / "test_preferences.csv").open() as f:
            records = list(csv.DictReader(f))
        assert len(records) == 1
        assert pytest.approx(float(records[0]["correct_pct"]), abs=1e-6) == 0.5
        assert int(records[0]["n_steps"]) == 0  # no outer-third steps

    def test_nan_positions_excluded_from_count(self, tmp_path):
        """NaN position steps are excluded; remaining outer-third steps scored."""
        run = tmp_path / "run"
        cond_dir = run / "Object1"
        rows = (
            [_row(0, "1color", agent_x=float("nan"), correct="right")] * 3  # excluded
            + [_row(0, "1color", agent_x=20.0, correct="right")] * 2          # outer correct
        )
        _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

        out = test_viz(run, tmp_path / "out")
        with (out / "test_preferences.csv").open() as f:
            records = list(csv.DictReader(f))
        assert int(records[0]["n_steps"]) == 2
        assert pytest.approx(float(records[0]["correct_pct"]), abs=1e-6) == 1.0

    def test_counterbalanced_left_and_right_trials_scored_independently(self, tmp_path):
        """Counterbalanced design: half left-correct, half right-correct."""
        run = tmp_path / "run"
        cond_dir = run / "Object1"
        # Env 0: correct=left, agent in left third → 100% correct
        # Env 1: correct=right, agent in left third → 0% correct
        rows = (
            [_row(0, "1color", agent_x=-20.0, correct="left")]  * 5
            + [_row(1, "1color", agent_x=-20.0, correct="right")] * 5
        )
        _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

        out = test_viz(run, tmp_path / "out")
        with (out / "test_preferences.csv").open() as f:
            records = sorted(csv.DictReader(f), key=lambda r: r["brain_env_id"])
        assert len(records) == 2
        assert pytest.approx(float(records[0]["correct_pct"]), abs=1e-6) == 1.0
        assert pytest.approx(float(records[1]["correct_pct"]), abs=1e-6) == 0.0

    def test_separates_brains_and_conditions(self, tmp_path):
        """Multiple envs × multiple test conditions → one row each."""
        run = tmp_path / "run"
        cond_dir = run / "Object1"
        rows = [
            _row(0, "rest",   agent_x=-20.0, correct="left"),   # env0 rest: correct
            _row(0, "1color", agent_x=20.0,  correct="left"),   # env0 1color: wrong
            _row(1, "rest",   agent_x=20.0,  correct="left"),   # env1 rest: wrong
            _row(1, "1color", agent_x=-20.0, correct="left"),   # env1 1color: correct
        ]
        _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

        out = test_viz(run, tmp_path / "out")
        with (out / "test_preferences.csv").open() as f:
            records = sorted(
                csv.DictReader(f),
                key=lambda r: (r["test_condition"], r["brain_env_id"]),
            )
        assert len(records) == 4
        by_key = {
            (r["test_condition"], r["brain_env_id"]): float(r["correct_pct"])
            for r in records
        }
        assert by_key[("rest", "0")] == 1.0
        assert by_key[("rest", "1")] == 0.0
        assert by_key[("1color", "0")] == 0.0
        assert by_key[("1color", "1")] == 1.0

    def test_handles_missing_logs(self, tmp_path):
        """Empty run directory → empty CSV header, no crash."""
        out = test_viz(tmp_path / "empty_run", tmp_path / "out")
        with (out / "test_preferences.csv").open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["imprint", "test_condition", "brain_env_id", "n_steps", "correct_pct"]
        assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# Legacy gaze metric — kept for backward-compat, not used in scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_x,agent_z,yaw_deg,expected",
    [
        (0.0, 0.0, 90.0, "left"),
        (0.0, 0.0, 270.0, "right"),
        (-10.0, 0.0, 90.0, "left"),
        (10.0, 0.0, 270.0, "right"),
    ],
)
def test_looking_at_monitor_cardinal_directions(agent_x, agent_z, yaw_deg, expected):
    assert looking_at_monitor(agent_x, agent_z, yaw_deg, DEFAULT_CHAMBER_HALF_X) == expected


# ---------------------------------------------------------------------------
# train_viz: skip when tensorboard isn't present or no tfevents
# ---------------------------------------------------------------------------


def test_train_viz_handles_empty_run(tmp_path):
    out = train_viz(tmp_path / "empty_run", tmp_path / "out")
    with (out / "train_rewards.csv").open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["condition", "brain", "step", "reward"]
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# analyze: top-level integration
# ---------------------------------------------------------------------------


def test_analyze_produces_summary_with_test_metrics(tmp_path):
    run = tmp_path / "run"
    cond_dir = run / "Object1"
    # 10 steps in left outer third, correct=left → 100%
    rows = [_row(0, "rest", agent_x=-20.0, correct="left") for _ in range(10)]
    _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

    out = analyze(run, tmp_path / "analysis")
    summary = json.loads((out / "summary.json").read_text())
    assert "test" in summary and "Object1" in summary["test"]
    rest = summary["test"]["Object1"]["rest"]
    assert rest["correct_pct_mean"] == 1.0
    assert rest["n_brains"] == 1


def test_analyze_scores_chance_when_only_middle_steps(tmp_path):
    """All-middle-third steps → correct_pct=0.5 (chance default)."""
    run = tmp_path / "run"
    cond_dir = run / "Object1"
    rows = [_row(0, "1color", agent_x=0.0, correct="left") for _ in range(10)]
    _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

    out = analyze(run, tmp_path / "analysis")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["test"]["Object1"]["1color"]["correct_pct_mean"] == pytest.approx(0.5)


def test_log_analysis_to_wandb_tolerates_python_tags_in_config(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(
        "name: tagged\n"
        "body:\n"
        "  wrappers:\n"
        "    - !!python/name:nett_skrl.body.wrappers.framestack.FrameStack ''\n"
        "brain:\n"
        "  wandb:\n"
        "    mode: disabled\n"
    )
    log_analysis_to_wandb(run)


# ---------------------------------------------------------------------------
# merge: concat CSVs across runs and re-aggregate
# ---------------------------------------------------------------------------


def test_merge_concatenates_test_preferences(tmp_path):
    run_a = tmp_path / "run_a"
    _write_test_csv(
        run_a / "Object1" / "logs" / "test_Object1_0.csv",
        [_row(0, "rest", agent_x=-20.0, correct="left") for _ in range(5)],
    )
    run_b = tmp_path / "run_b"
    _write_test_csv(
        run_b / "Object1" / "logs" / "test_Object1_0.csv",
        [_row(0, "rest", agent_x=20.0, correct="left") for _ in range(5)],
    )
    out_a = analyze(run_a, tmp_path / "analysis_a")
    out_b = analyze(run_b, tmp_path / "analysis_b")

    merged = merge([out_a, out_b], tmp_path / "merged")
    pref_csvs = sorted(merged.rglob("test_preferences.csv"))
    assert pref_csvs, "no test_preferences.csv survived the merge copy"


# ---------------------------------------------------------------------------
# Legacy normalization
# ---------------------------------------------------------------------------


def test_normalize_isaac_output_copies_expected_layout(tmp_path):
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "train.csv").write_text("x\n")
    ego = run / "recordings" / "egocentric" / "train" / "env_000000"
    ego.mkdir(parents=True)
    (ego / "000000.png").write_bytes(b"png")
    chamber = run / "recordings" / "chamber" / "train" / "env_000000"
    chamber.mkdir(parents=True)
    (chamber / "000000.png").write_bytes(b"png")

    out = normalize_isaac_output(run)
    assert (out / "logs" / "train.csv").exists()
    assert (out / "recordings" / "agent" / "train" / "env_000000" / "000000.png").exists()
    assert (out / "recordings" / "chamber" / "train" / "env_000000" / "000000.png").exists()
