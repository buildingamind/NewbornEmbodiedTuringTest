"""Unit tests for the native analysis helpers.

These exercise the gaze-direction math, CSV aggregation, and merge tree
behavior on synthetic data — no Isaac Sim / tfevents involvement.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from nett_skrl.analysis import (
    DEFAULT_CHAMBER_HALF_X,
    analyze,
    looking_at_monitor,
    merge,
    normalize_isaac_output,
    test_viz,
    train_viz,
)


# ---------------------------------------------------------------------------
# Gaze-direction math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_x,agent_z,yaw_deg,expected",
    [
        (0.0, 0.0, 90.0, "left"),     # forward = (-1, 0) → left wall
        (0.0, 0.0, 270.0, "right"),   # forward = (+1, 0) → right wall
        (-10.0, 0.0, 90.0, "left"),   # offset, still facing left
        (10.0, 0.0, 270.0, "right"),  # offset, still facing right
    ],
)
def test_looking_at_monitor_cardinal_directions(agent_x, agent_z, yaw_deg, expected):
    assert looking_at_monitor(agent_x, agent_z, yaw_deg, DEFAULT_CHAMBER_HALF_X) == expected


def test_looking_at_monitor_ties_break_toward_right():
    # Pointing directly forward (yaw=0) — both monitors equidistant; the dot
    # products tie and the implementation falls through to "right" via the
    # strict-greater comparison.
    assert looking_at_monitor(0.0, 0.0, 0.0, DEFAULT_CHAMBER_HALF_X) == "right"


# ---------------------------------------------------------------------------
# test_viz: synthetic LogChannel CSVs
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


def _row(env_id: int, test_cond: str, yaw: float, correct: str) -> dict:
    base = {h: "" for h in _HEADER}
    base.update({
        "env_id": str(env_id), "episode": "0", "step": "0",
        "agent.x": "0.0", "agent.z": "0.0", "agent.angle": str(yaw),
        "head.flexion": "0.0", "head.lateral": "0.0",
        "left.monitor": "L.mov", "right.monitor": "R.mov",
        "correct.monitor": correct,
        "experiment.phase": "test", "imprint.cond": "Object1",
        "test.cond": test_cond,
    })
    return base


def test_test_viz_correctly_scores_gaze_against_target(tmp_path):
    run = tmp_path / "run"
    cond_dir = run / "Object1"
    # 5 steps of "looking left" (yaw=90), all correct=left → 100%.
    rows = [_row(0, "rest", yaw=90.0, correct="left") for _ in range(5)]
    # 2 steps of "looking right" while correct=left → wrong on those.
    rows += [_row(0, "rest", yaw=270.0, correct="left") for _ in range(2)]
    _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

    out = test_viz(run, tmp_path / "out")
    with (out / "test_preferences.csv").open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == 1
    record = records[0]
    assert record["imprint"] == "Object1"
    assert record["test_condition"] == "rest"
    assert record["brain_env_id"] == "0"
    assert int(record["n_steps"]) == 7
    assert pytest.approx(float(record["correct_pct"]), abs=1e-6) == 5 / 7


def test_test_viz_separates_brains_and_conditions(tmp_path):
    run = tmp_path / "run"
    cond_dir = run / "Object1"
    rows = [
        _row(0, "rest", yaw=90.0, correct="left"),    # brain 0, rest: correct
        _row(0, "1color", yaw=270.0, correct="left"), # brain 0, 1color: wrong
        _row(1, "rest", yaw=270.0, correct="left"),   # brain 1, rest: wrong
        _row(1, "1color", yaw=90.0, correct="left"),  # brain 1, 1color: correct
    ]
    _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

    out = test_viz(run, tmp_path / "out")
    with (out / "test_preferences.csv").open() as f:
        records = sorted(csv.DictReader(f), key=lambda r: (r["test_condition"], r["brain_env_id"]))
    # 2 test conds × 2 brains = 4 rows.
    assert len(records) == 4
    by_key = {(r["test_condition"], r["brain_env_id"]): float(r["correct_pct"]) for r in records}
    assert by_key[("rest", "0")] == 1.0
    assert by_key[("rest", "1")] == 0.0
    assert by_key[("1color", "0")] == 0.0
    assert by_key[("1color", "1")] == 1.0


def test_test_viz_handles_missing_logs(tmp_path):
    """Empty run → empty CSV header, no crash."""
    out = test_viz(tmp_path / "empty_run", tmp_path / "out")
    with (out / "test_preferences.csv").open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["imprint", "test_condition", "brain_env_id", "n_steps", "correct_pct"]
    assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# train_viz: skip when tensorboard isn't present or no tfevents
# ---------------------------------------------------------------------------


def test_train_viz_handles_empty_run(tmp_path):
    """No conditions → empty rewards CSV header, no crash."""
    out = train_viz(tmp_path / "empty_run", tmp_path / "out")
    with (out / "train_rewards.csv").open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["condition", "brain", "step", "reward"]
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# analyze: top-level integration over the synthetic CSV path
# ---------------------------------------------------------------------------


def test_analyze_produces_summary_with_test_metrics(tmp_path):
    run = tmp_path / "run"
    cond_dir = run / "Object1"
    rows = [_row(0, "rest", yaw=90.0, correct="left") for _ in range(10)]
    _write_test_csv(cond_dir / "logs" / "test_Object1_0.csv", rows)

    out = analyze(run, tmp_path / "analysis")
    summary = json.loads((out / "summary.json").read_text())
    assert "test" in summary and "Object1" in summary["test"]
    rest = summary["test"]["Object1"]["rest"]
    assert rest["correct_pct_mean"] == 1.0
    assert rest["n_brains"] == 1


# ---------------------------------------------------------------------------
# merge: concat CSVs across runs and re-aggregate
# ---------------------------------------------------------------------------


def test_merge_concatenates_test_preferences(tmp_path):
    # Build two analysis trees with non-overlapping test CSVs.
    run_a = tmp_path / "run_a"
    _write_test_csv(
        run_a / "Object1" / "logs" / "test_Object1_0.csv",
        [_row(0, "rest", yaw=90.0, correct="left") for _ in range(5)],
    )
    run_b = tmp_path / "run_b"
    _write_test_csv(
        run_b / "Object1" / "logs" / "test_Object1_0.csv",
        [_row(0, "rest", yaw=270.0, correct="left") for _ in range(5)],
    )
    out_a = analyze(run_a, tmp_path / "analysis_a")
    out_b = analyze(run_b, tmp_path / "analysis_b")

    merged = merge([out_a, out_b], tmp_path / "merged")
    # Both source preference CSVs should be present somewhere in the tree.
    pref_csvs = sorted(merged.rglob("test_preferences.csv"))
    assert pref_csvs, "no test_preferences.csv survived the merge copy"


# ---------------------------------------------------------------------------
# Legacy normalization test kept from the previous suite
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
