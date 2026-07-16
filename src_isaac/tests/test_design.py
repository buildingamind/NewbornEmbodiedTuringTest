"""Tests for the Isaac Lab design-sheet parsing helpers."""

from __future__ import annotations

import os

from pathlib import Path

import pytest

from nett_skrl.environment.design import (
    get_experiment_design,
    validate_conditions,
)


_BINDING_CSV = Path(
    os.environ.get(
        "NETT_BINDING_CSV",
        "/path/to/NewbornEmbodiedTuringTest_Private/isaac_lab/assets/design_sheets/binding.csv",
    )
)


def _has_binding() -> bool:
    return _BINDING_CSV.exists()


@pytest.mark.skipif(not _has_binding(), reason="binding.csv not available")
def test_binding_design_has_two_imprint_conditions():
    design = get_experiment_design(_BINDING_CSV)
    assert set(design) == {"Object1", "Object2"}


@pytest.mark.skipif(not _has_binding(), reason="binding.csv not available")
def test_binding_design_test_row_counts_match():
    design = get_experiment_design(_BINDING_CSV)
    # Both conditions have the same number of test rows in binding.csv.
    assert design["Object1"] == design["Object2"] > 0


def test_validate_conditions_rejects_unknown(tmp_path):
    design = {"A": 1, "B": 2}
    with pytest.raises(ValueError, match="not in design sheet"):
        validate_conditions(design, ["A", "Nope"])


def test_validate_conditions_none_returns_all():
    design = {"A": 1, "B": 2}
    assert sorted(validate_conditions(design, None)) == ["A", "B"]


def test_validate_conditions_subset_passes():
    design = {"A": 1, "B": 2}
    assert validate_conditions(design, ["A"]) == ["A"]


def test_get_experiment_design_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_experiment_design(tmp_path / "nope.csv")


def test_get_experiment_design_minimal_csv(tmp_path):
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "ImprintCondition,Phase,TestCondition,TargetVideo,NonTargetVideo,LeftMonitor,RightMonitor\n"
        "A,Train,,a.mp4,b.mp4,a.mp4,b.mp4\n"
        "A,Test,t1,a.mp4,b.mp4,a.mp4,b.mp4\n"
        "B,Test,t1,a.mp4,b.mp4,a.mp4,b.mp4\n"
    )
    design = get_experiment_design(csv)
    assert design == {"A": 1, "B": 1}
