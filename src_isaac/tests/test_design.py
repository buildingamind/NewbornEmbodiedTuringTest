"""Tests for the Isaac Lab design-sheet parsing helpers."""

from __future__ import annotations

import os

from pathlib import Path

import pytest

from _repo_paths import WORKSPACE

from nett_skrl.environment.design import (
    get_experiment_design,
    validate_conditions,
)


#: The real binding design sheet. ⚠ THIS DEFAULT USED TO BE THE LITERAL PLACEHOLDER
#: ``/path/to/NewbornEmbodiedTuringTest_Private/isaac_lab/assets/design_sheets/binding.csv``
#: — a path that can never exist, so both tests below skipped ALWAYS and on every host,
#: reporting the honest-sounding "binding.csv not available" while never once running
#: (fixed 2026-07-27; they pass against the real sheet). ``/path/to/...`` is this project's
#: convention for DOCS, where a reader substitutes their own path. It must never be the
#: default in code that decides whether a test runs.
#:
#: Resolution mirrors ``_repo_paths``: an env override, else a workspace-relative default.
#: The sheet lives with the stimulus videos, NOT inside either repository — it is
#: experiment data, and neither repo ships one.
_BINDING_CSV = Path(
    os.environ.get(
        "NETT_BINDING_CSV",
        str(WORKSPACE / "videos" / "binding" / "DesignSheet_Binding.csv"),
    )
)


def _has_binding() -> bool:
    return _BINDING_CSV.exists()


#: Name the path that was actually looked for. "binding.csv not available" gave a reader
#: no way to tell a missing file from a wrong default — which is how the bug above hid.
_NO_BINDING = f"design sheet not found at {_BINDING_CSV} (set $NETT_BINDING_CSV)"


@pytest.mark.skipif(not _has_binding(), reason=_NO_BINDING)
def test_binding_design_has_two_imprint_conditions():
    design = get_experiment_design(_BINDING_CSV)
    assert set(design) == {"Object1", "Object2"}


@pytest.mark.skipif(not _has_binding(), reason=_NO_BINDING)
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
