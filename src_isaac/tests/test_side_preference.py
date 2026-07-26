"""Tests for the side-lock diagnostics that sit alongside ``correct_pct``.

WHY THESE EXIST. ``correct_pct`` is

    steps in correct outer third / steps in EITHER outer third

which cannot distinguish two completely different policies:

    a brain that always walks to the SAME wall regardless of the target
        -> scores 1.0 on the episodes whose target is that wall, 0.0 on the
           rest, and averages to ~0.5
    a brain that wanders
        -> also ~0.5

Both read as "chance". Splitting the score by WHICH SIDE held the target
separates them, and ``side_preference = (R - L) / (R + L)`` detects the first
case without any target bookkeeping at all. Measured on the 2026-07-24 emissive
sweep: side-locked brains sat at side_preference = +/-1.00, genuine learners at
-0.21..-0.02.

The third case -- a brain that never leaves the centre third -- has an EMPTY
DENOMINATOR. ``correct_pct`` reports its 0.5 chance default there (intentional,
and kept), but the new metrics must report UNDEFINED so a fabricated value is
not averaged in as though it were a measurement.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from nett_skrl.analysis import (
    DEFAULT_CHAMBER_HALF_X,
    LEARN_MIN_PER_SIDE,
    SIDE_LOCK_MIN_GAP,
    analyze,
    chamber_third,
    in_correct_chamber_third,
    side_bias_verdict,
    side_preference,
    test_viz,
)

HALF_X = DEFAULT_CHAMBER_HALF_X
THIRD = HALF_X / 3.0  # 11.05

_HEADER = [
    "env_id", "episode", "step", "agent.x", "agent.y", "agent.angle",
    "head.flexion", "head.lateral", "left.monitor", "right.monitor",
    "correct.monitor", "experiment.phase", "imprint.cond", "test.cond",
]


def _row(env_id: int, agent_x: float, correct: str, test_cond: str = "rest") -> dict:
    base = {h: "" for h in _HEADER}
    base.update({
        "env_id": str(env_id), "episode": "0", "step": "0",
        "agent.x": str(agent_x), "agent.y": "0.0", "agent.angle": "0.0",
        "head.flexion": "0.0", "head.lateral": "0.0",
        "left.monitor": "L.mov", "right.monitor": "R.mov",
        "correct.monitor": correct,
        "experiment.phase": "test", "imprint.cond": "Object1",
        "test.cond": test_cond,
    })
    return base


def _write_run(run: Path, rows: list[dict]) -> Path:
    path = run / "Object1" / "logs" / "test_Object1_0.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return run


def _records(out_dir: Path) -> list[dict]:
    with (out_dir / "test_preferences.csv").open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# chamber_third — the single geometry definition
# ---------------------------------------------------------------------------


class TestChamberThird:
    def test_classifies_the_three_regions(self):
        assert chamber_third(-20.0, HALF_X) == "left"
        assert chamber_third(20.0, HALF_X) == "right"
        assert chamber_third(0.0, HALF_X) == "middle"

    def test_boundary_is_exclusive_on_both_sides(self):
        # Exactly on the threshold counts as middle (strict inequality), so the
        # two outer thirds cannot both claim the same step.
        assert chamber_third(-THIRD, HALF_X) == "middle"
        assert chamber_third(THIRD, HALF_X) == "middle"
        assert chamber_third(-THIRD - 1e-9, HALF_X) == "left"
        assert chamber_third(THIRD + 1e-9, HALF_X) == "right"

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_is_middle_so_it_biases_neither_side(self, bad):
        assert chamber_third(bad, HALF_X) == "middle"

    def test_agrees_with_in_correct_chamber_third(self):
        """The target-relative helper must be a view on the same geometry."""
        for x in (-30.0, -THIRD, 0.0, THIRD, 30.0, math.nan):
            third = chamber_third(x, HALF_X)
            for correct in ("left", "right"):
                in_outer, in_correct = in_correct_chamber_third(x, correct, HALF_X)
                assert in_outer is (third != "middle")
                assert in_correct is (third == correct)


# ---------------------------------------------------------------------------
# side_preference / side_bias_verdict — pure functions
# ---------------------------------------------------------------------------


class TestSidePreference:
    def test_one_wall_only_saturates(self):
        assert side_preference(100, 0) == -1.0
        assert side_preference(0, 100) == 1.0

    def test_balanced_occupancy_is_zero(self):
        assert side_preference(50, 50) == 0.0

    def test_empty_denominator_is_undefined_not_zero(self):
        # A brain that never entered either outer third expressed NO preference.
        # Reporting 0.0 would make it indistinguishable from a brain that split
        # its time evenly, so the value must be absent.
        assert side_preference(0, 0) is None


class TestSideBiasVerdict:
    def test_high_on_both_sides_is_learning(self):
        assert side_bias_verdict(1.0, 1.0) == "LEARN"
        assert side_bias_verdict(LEARN_MIN_PER_SIDE, LEARN_MIN_PER_SIDE) == "LEARN"

    def test_one_wall_always_is_side_lock(self):
        assert side_bias_verdict(1.0, 0.0) == "SIDE-LOCK"
        assert side_bias_verdict(0.0, 1.0) == "SIDE-LOCK"
        assert side_bias_verdict(SIDE_LOCK_MIN_GAP, 0.0) == "SIDE-LOCK"

    def test_middling_on_both_sides_is_chance(self):
        assert side_bias_verdict(0.5, 0.5) == "chance"
        assert side_bias_verdict(0.6, 0.4) == "chance"

    def test_missing_a_side_yields_no_verdict(self):
        assert side_bias_verdict(None, 1.0) == "n/a"
        assert side_bias_verdict(1.0, None) == "n/a"
        assert side_bias_verdict(None, None) == "n/a"

    def test_learning_check_precedes_side_lock_check(self):
        # 1.0 vs 0.7 clears LEARN on both sides; the 0.3 gap must not matter.
        assert side_bias_verdict(1.0, 0.7) == "LEARN"


# ---------------------------------------------------------------------------
# End-to-end through test_viz on synthetic logs
# ---------------------------------------------------------------------------


class TestSideLockedBrain:
    """Always walks to the LEFT wall; targets alternate. The case correct_pct misses."""

    @pytest.fixture
    def record(self, tmp_path) -> dict:
        rows = [
            _row(0, agent_x=-20.0, correct="left" if i % 2 == 0 else "right")
            for i in range(20)
        ]
        run = _write_run(tmp_path / "run", rows)
        records = _records(test_viz(run, tmp_path / "out"))
        assert len(records) == 1
        return records[0]

    def test_correct_pct_alone_looks_like_chance(self, record):
        # This is the failure mode the diagnostics exist to catch.
        assert float(record["correct_pct"]) == pytest.approx(0.5)

    def test_side_preference_saturates_toward_the_occupied_wall(self, record):
        assert float(record["side_preference"]) == pytest.approx(-1.0)

    def test_per_target_split_is_all_or_nothing(self, record):
        assert float(record["pct_target_left"]) == pytest.approx(1.0)
        assert float(record["pct_target_right"]) == pytest.approx(0.0)

    def test_verdict_is_side_lock(self, record):
        assert record["verdict"] == "SIDE-LOCK"


class TestLearningBrain:
    """Approaches whichever wall holds the target."""

    @pytest.fixture
    def record(self, tmp_path) -> dict:
        rows = []
        for i in range(20):
            correct = "left" if i % 2 == 0 else "right"
            rows.append(_row(0, agent_x=-20.0 if correct == "left" else 20.0,
                             correct=correct))
        run = _write_run(tmp_path / "run", rows)
        records = _records(test_viz(run, tmp_path / "out"))
        assert len(records) == 1
        return records[0]

    def test_correct_pct_is_perfect(self, record):
        assert float(record["correct_pct"]) == pytest.approx(1.0)

    def test_side_preference_is_balanced(self, record):
        # A learner on a counterbalanced target sequence visits both walls
        # equally, so the target-agnostic occupancy carries no bias.
        assert float(record["side_preference"]) == pytest.approx(0.0)

    def test_verdict_is_learn(self, record):
        assert float(record["pct_target_left"]) == pytest.approx(1.0)
        assert float(record["pct_target_right"]) == pytest.approx(1.0)
        assert record["verdict"] == "LEARN"


class TestWanderingBrain:
    """Visits both walls but ignores the target: chance, not side-lock."""

    def test_verdict_is_chance(self, tmp_path):
        rows = []
        for i in range(20):
            # Position cycles independently of the target.
            rows.append(_row(0, agent_x=-20.0 if i % 2 == 0 else 20.0,
                             correct="left" if i % 4 < 2 else "right"))
        run = _write_run(tmp_path / "run", rows)
        rec = _records(test_viz(run, tmp_path / "out"))[0]
        assert float(rec["correct_pct"]) == pytest.approx(0.5)
        assert float(rec["side_preference"]) == pytest.approx(0.0)
        assert rec["verdict"] == "chance"


class TestImmobileBrain:
    """Never leaves the centre third — every ratio has an empty denominator."""

    @pytest.fixture
    def record(self, tmp_path) -> dict:
        rows = [_row(0, agent_x=0.0, correct="left" if i % 2 == 0 else "right")
                for i in range(20)]
        run = _write_run(tmp_path / "run", rows)
        return _records(test_viz(run, tmp_path / "out"))[0]

    def test_correct_pct_keeps_its_chance_default(self, record):
        # Deliberate: 0.5 keeps downstream aggregation stable, and 0.0 would be
        # read as a novel-stimulus preference. n_steps == 0 is how a consumer
        # tells the default apart from a measurement.
        assert float(record["correct_pct"]) == pytest.approx(0.5)
        assert int(record["n_steps"]) == 0

    def test_new_metrics_are_blank_rather_than_defaulted(self, record):
        assert record["side_preference"] == ""
        assert record["pct_target_left"] == ""
        assert record["pct_target_right"] == ""

    def test_verdict_is_not_applicable(self, record):
        assert record["verdict"] == "n/a"


class TestOneSidedTargets:
    """If every episode had the same target side, no verdict is possible."""

    def test_missing_target_side_yields_na(self, tmp_path):
        rows = [_row(0, agent_x=-20.0, correct="left") for _ in range(10)]
        run = _write_run(tmp_path / "run", rows)
        rec = _records(test_viz(run, tmp_path / "out"))[0]
        assert float(rec["correct_pct"]) == pytest.approx(1.0)
        assert float(rec["side_preference"]) == pytest.approx(-1.0)
        assert rec["pct_target_right"] == ""
        # 1.0 on the only side that occurred is NOT evidence of learning: a
        # side-locked brain scores exactly the same on such a sequence.
        assert rec["verdict"] == "n/a"


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def _summary(tmp_path: Path, rows: list[dict]) -> dict:
    run = _write_run(tmp_path / "run", rows)
    out = analyze(run, tmp_path / "analysis")
    return json.loads((out / "summary.json").read_text())["test"]["Object1"]["rest"]


class TestSummaryAggregation:
    def test_learn_fraction_separates_learners_from_side_lock(self, tmp_path):
        rows = []
        for i in range(20):
            correct = "left" if i % 2 == 0 else "right"
            # brain 0 learns, brain 1 parks on the left wall.
            rows.append(_row(0, -20.0 if correct == "left" else 20.0, correct))
            rows.append(_row(1, -20.0, correct))
        entry = _summary(tmp_path, rows)

        # Both brains average to the same headline number...
        assert entry["correct_pct_mean"] == pytest.approx(0.75)
        # ...but only one of them learned.
        assert entry["verdict_counts"] == {
            "LEARN": 1, "SIDE-LOCK": 1, "chance": 0, "n/a": 0
        }
        assert entry["learn_fraction"] == pytest.approx(0.5)

    def test_immobile_brains_are_counted_and_excluded_from_the_defined_mean(
        self, tmp_path
    ):
        rows = []
        for i in range(20):
            correct = "left" if i % 2 == 0 else "right"
            rows.append(_row(0, -20.0 if correct == "left" else 20.0, correct))
            rows.append(_row(1, 0.0, correct))  # immobile
        entry = _summary(tmp_path, rows)

        assert entry["n_brains"] == 2
        assert entry["n_brains_defined"] == 1
        assert entry["n_brains_immobile"] == 1
        # The historical headline still averages the 0.5 fallback in...
        assert entry["correct_pct_mean"] == pytest.approx(0.75)
        # ...the defined-only mean describes the brain that actually moved.
        assert entry["correct_pct_mean_defined"] == pytest.approx(1.0)
        assert entry["verdict_counts"]["n/a"] == 1

    def test_side_preference_mean_skips_undefined_brains(self, tmp_path):
        rows = []
        for i in range(20):
            correct = "left" if i % 2 == 0 else "right"
            rows.append(_row(0, -20.0, correct))  # side-locked left
            rows.append(_row(1, 0.0, correct))    # immobile: no value at all
        entry = _summary(tmp_path, rows)

        # Mean over the ONE brain with a defined value, not a 0.0 for the other.
        assert entry["side_preference_mean"] == pytest.approx(-1.0)
        assert entry["side_preference_abs_mean"] == pytest.approx(1.0)
