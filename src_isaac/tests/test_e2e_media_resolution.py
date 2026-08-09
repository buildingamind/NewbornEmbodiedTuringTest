"""The e2e tree's media-root resolution, tested without Isaac.

WHY THIS IS A UNIT TEST OF TEST CODE. The resolution in ``tests/e2e/conftest.py`` decides
which stimulus directory 23 GPU tests run against, and it got that wrong twice — once by
pinning a path that never existed (whole tree auto-skipped), then by pinning repoA's
vendored fixture pair on the belief that it covered ``binding_minimal.csv``, which names a
third clip repoA does not ship. The second miss stayed invisible for as long as missing
media was a warning: the tests passed against BLANK MONITORS. Nothing in the e2e tree can
catch that, because the e2e tree is exactly what stops running when it is wrong.
"""

from __future__ import annotations

from e2e.conftest import _resolve_media_root, _sheet_clips

HEADER = "ImprintCondition,Phase,TestCondition,TargetVideo,NonTargetVideo,LeftMonitor,RightMonitor\n"


def _sheet(tmp_path, *rows: str):
    p = tmp_path / "sheet.csv"
    p.write_text(HEADER + "".join(r + "\n" for r in rows))
    return p


def _root(tmp_path, name: str, *clips: str):
    d = tmp_path / name
    d.mkdir()
    for c in clips:
        (d / c).write_bytes(b"")
    return d


def test_sheet_clips_reads_the_monitor_columns(tmp_path):
    """Clips come from LeftMonitor/RightMonitor (cols 6-7), not the Target columns."""
    sheet = _sheet(
        tmp_path,
        "Object1,Training,,O1_imprint.mov,White.mov,O1_imprint.mov,White.mov",
        "Object1,Test,1color,O1_imprint.mov,O1_1Ca_1.mov,O1_imprint.mov,O1_1Ca_1.mov",
    )
    assert _sheet_clips(sheet) == {"O1_imprint.mov", "White.mov", "O1_1Ca_1.mov"}


def test_sheet_clips_tolerates_quotes_and_a_trailing_comma(tmp_path):
    """Real sheets have both: DesignSheet_Binding.csv ends every row with a comma."""
    sheet = _sheet(tmp_path, 'Object1,Test,rest," A.mov ",B.mov,"A.mov",B.mov,')
    assert _sheet_clips(sheet) == {"A.mov", "B.mov"}


def test_sheet_clips_is_empty_when_unreadable(tmp_path):
    """A missing sheet must not raise — the caller falls back rather than fail collection."""
    assert _sheet_clips(tmp_path / "nope.csv") == set()


def test_resolves_to_the_root_that_holds_every_clip(tmp_path):
    """★ THE REGRESSION. A root holding SOME of the sheet's clips is not the right root.

    This is the exact shape of the bug: the vendored pair covers the training row and
    misses the test row, so the run trains fine and dies (or, before repoA 400683cd, goes
    blank) the moment the test phase starts.
    """
    sheet = _sheet(
        tmp_path,
        "Object1,Training,,O1_imprint.mov,White.mov,O1_imprint.mov,White.mov",
        "Object1,Test,1color,O1_imprint.mov,O1_1Ca_1.mov,O1_imprint.mov,O1_1Ca_1.mov",
    )
    partial = _root(tmp_path, "vendored", "O1_imprint.mov", "White.mov")
    complete = _root(tmp_path, "library", "O1_imprint.mov", "White.mov", "O1_1Ca_1.mov")

    assert _resolve_media_root(sheet, (partial, complete)) == complete


def test_prefers_the_earlier_candidate_when_both_suffice(tmp_path):
    """Order still means preference — vendored clips win when they are enough."""
    sheet = _sheet(tmp_path, "Object1,Training,,A.mov,B.mov,A.mov,B.mov")
    first = _root(tmp_path, "vendored", "A.mov", "B.mov")
    second = _root(tmp_path, "library", "A.mov", "B.mov")

    assert _resolve_media_root(sheet, (first, second)) == first


def test_falls_back_to_the_first_candidate_when_none_suffice(tmp_path):
    """No root satisfies the sheet: name a real directory so the skip/error is honest."""
    sheet = _sheet(tmp_path, "Object1,Training,,A.mov,B.mov,A.mov,B.mov")
    first = _root(tmp_path, "vendored")
    second = _root(tmp_path, "library", "A.mov")

    assert _resolve_media_root(sheet, (first, second)) == first


# ---------------------------------------------------------------------------
# The silent-fallback detector, and finding the library from a git worktree.
#
# ⚠ THE FALLBACK ABOVE IS ONLY SAFE IF SOMETHING NOTICES IT. Nothing did: the tree
# collected, trained for minutes, and then died in the TEST phase inside
# ``NETTEnv.__init__`` on the first clip repoA does not ship — and until repoA learned
# to abandon a half-built env, that constructor failure HUNG the worker instead of
# failing it (measured 2026-08-09, from a worktree checkout).
# ---------------------------------------------------------------------------

import os

from e2e.conftest import _media_incomplete, _stimulus_library_candidates


def test_media_incomplete_names_the_missing_clips(tmp_path, monkeypatch):
    monkeypatch.delenv("NETT_MEDIA_ROOT", raising=False)
    sheet = _sheet(tmp_path, "Object1,Test,1color,A.mov,B.mov,A.mov,B.mov")
    root = _root(tmp_path, "vendored", "A.mov")

    reason = _media_incomplete(sheet, root)
    assert reason and "B.mov" in reason and str(root) in reason
    assert "NETT_MEDIA_ROOT" in reason, "the skip reason must say how to fix it"


def test_media_incomplete_is_silent_when_the_root_suffices(tmp_path, monkeypatch):
    monkeypatch.delenv("NETT_MEDIA_ROOT", raising=False)
    sheet = _sheet(tmp_path, "Object1,Test,1color,A.mov,B.mov,A.mov,B.mov")
    root = _root(tmp_path, "library", "A.mov", "B.mov")

    assert _media_incomplete(sheet, root) is None


def test_media_incomplete_never_second_guesses_an_explicit_override(tmp_path, monkeypatch):
    """``NETT_MEDIA_ROOT`` is documented as honoured verbatim; a skip would overrule it."""
    monkeypatch.setenv("NETT_MEDIA_ROOT", str(tmp_path / "wherever"))
    sheet = _sheet(tmp_path, "Object1,Test,1color,A.mov,B.mov,A.mov,B.mov")
    root = _root(tmp_path, "empty")

    assert _media_incomplete(sheet, root) is None


def test_stimulus_library_is_found_from_this_checkout_whatever_its_depth():
    """The library must resolve from a worktree too, not only two levels up.

    ``WORKSPACE`` is a fixed guess of ``src_isaac/../..``. In a worktree
    (``<workspace>/wt-<name>/NewbornEmbodiedTuringTest/src_isaac``) that lands one level
    short and the guess misses entirely, which is how the vendored-fixture fallback got
    silently selected. Skips where no library exists at all — that is a host without the
    stimulus data, not a resolution bug.
    """
    import pytest

    candidates = _stimulus_library_candidates()
    if not candidates:
        pytest.skip("no videos/binding stimulus library above this checkout")
    assert all(c.is_dir() for c in candidates)
    assert len(set(candidates)) == len(candidates), "candidates must be de-duplicated"
