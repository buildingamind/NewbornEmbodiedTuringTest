"""End-to-end test of ``analyze()`` against real Isaac output.

Runs one small train+test smoke, then exercises ``train_viz`` (against the
real tfevents skrl writes) and ``test_viz`` (against the real LogChannel
test CSV). Confirms PNGs land, the summary JSON has both train + test
sections, and the merge helper concatenates two analysis trees without
losing rows.
"""

from __future__ import annotations

import csv
import json

import pytest


pytestmark = pytest.mark.e2e_isaac


def test_analyze_pipeline_against_real_run(run_nett, e2e_smoke_cfg, e2e_output_dir):
    output_dir = run_nett(e2e_smoke_cfg)

    from nett_skrl import analyze, merge

    analysis_dir = analyze(output_dir, e2e_output_dir / "analysis")
    summary = json.loads((analysis_dir / "summary.json").read_text())

    # Train side: at least one brain × condition got reward scalars from tfevents.
    assert "train" in summary and summary["train"], f"empty train summary: {summary}"
    assert "Object1" in summary["train"]
    assert summary["train"]["Object1"], "no per-brain train metrics for Object1"

    # Test side: one record per (imprint, test_condition); the minimal sheet
    # has Object1 with test conds ``rest`` and ``1color``.
    assert "test" in summary and summary["test"]
    assert "Object1" in summary["test"]
    assert {"rest", "1color"}.issubset(summary["test"]["Object1"].keys())

    # CSVs landed and have rows beyond the header.
    train_csv = analysis_dir / "train" / "train_rewards.csv"
    test_csv = analysis_dir / "test" / "test_preferences.csv"
    for path in (train_csv, test_csv):
        assert path.exists(), f"missing {path}"
        with path.open() as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 2, f"{path.name} only has header"

    # PNGs landed for both viz layers.
    assert list((analysis_dir / "train").glob("train_reward_Object1.png")), \
        "no per-condition train reward PNG"
    assert list((analysis_dir / "test").glob("test_preference_Object1.png")), \
        "no per-imprint test preference PNG"

    # Merge against itself — should produce a tree that contains both CSVs.
    merged = merge([analysis_dir], e2e_output_dir / "merged")
    assert list(merged.rglob("train_rewards.csv")), "merge dropped train CSV"
    assert list(merged.rglob("test_preferences.csv")), "merge dropped test CSV"
