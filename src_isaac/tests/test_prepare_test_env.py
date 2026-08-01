"""``scripts/prepare_test_env.py`` — the skip-vs-block policy.

The script decides whether a missing prerequisite blocks a push. Getting that wrong is
expensive in both directions: block too eagerly and a developer without the stimulus
library can never push at all; block too little and the e2e tree burns forty minutes of
GPU to reach a failure that was knowable in a second. The rule it implements is that the
answer must MATCH THE e2e TREE'S OWN SKIP GATE — block exactly when the tests would fail,
stay quiet when they would skip — so these tests pin the three cases apart.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

SRC_ISAAC = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "_prepare_test_env", SRC_ISAAC / "scripts" / "prepare_test_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load_script()


def _fake_conftest(sheet: Path, media_root: Path):
    """The attribute surface `prepare_stimuli` reads off the real e2e conftest."""
    from e2e.conftest import _sheet_clips

    return types.SimpleNamespace(
        DESIGN_SHEET_MINIMAL=sheet,
        MEDIA_ROOT=media_root,
        DESIGN_SHEET_FULL=Path("/nonexistent/full.csv"),
        MEDIA_ROOT_FULL=Path("/nonexistent/videos"),
        MINIMAL_SMOKE_CFG={"environment": {"input_resolution": 64}},
        _sheet_clips=_sheet_clips,
    )


HEADER = "ImprintCondition,Phase,TestCondition,TargetVideo,NonTargetVideo,LeftMonitor,RightMonitor\n"


@pytest.fixture
def sheet(tmp_path) -> Path:
    p = tmp_path / "sheet.csv"
    p.write_text(HEADER + "Object1,Training,,A.mov,B.mov,A.mov,B.mov\n")
    return p


def test_absent_media_root_is_not_a_problem(sheet, tmp_path):
    """The tree SKIPS on a missing root, so blocking the push would strand a developer."""
    assert prep.prepare_stimuli(_fake_conftest(sheet, tmp_path / "nope"), True) == []


def test_absent_sheet_is_not_a_problem(tmp_path):
    """Same reasoning: no sheet, no run."""
    cf = _fake_conftest(tmp_path / "nope.csv", tmp_path)
    assert prep.prepare_stimuli(cf, True) == []


def test_incomplete_media_root_is_a_problem(sheet, tmp_path):
    """★ The case this script exists for: the tree RUNS and FAILS, so block early.

    A root holding some of the sheet's clips is the shape that used to render blank
    monitors and pass.
    """
    root = tmp_path / "partial"
    root.mkdir()
    (root / "A.mov").write_bytes(b"")

    problems = prep.prepare_stimuli(_fake_conftest(sheet, root), True)

    assert len(problems) == 1
    assert "B.mov" in problems[0] and str(root) in problems[0]


def test_missing_built_asset_names_its_builder(tmp_path):
    """A missing versioned asset means a broken checkout — say which builder makes it."""
    problems = prep.check_built_assets(tmp_path)

    assert len(problems) == len(prep.BUILT_ASSETS)
    assert any("build_chick.py" in p for p in problems)
    assert any("build_chamber.py" in p for p in problems)


def test_built_assets_pass_when_present(tmp_path):
    for rel, _ in prep.BUILT_ASSETS:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    assert prep.check_built_assets(tmp_path) == []


def test_the_real_repo_a_has_its_built_assets():
    """The live checkout, not a fixture — this is the "did the copy bring the USD?" check."""
    from _repo_paths import repo_a_root

    root = repo_a_root()
    if not root.is_dir():
        pytest.skip(f"repoA checkout not found at {root}")
    assert prep.check_built_assets(root) == []
