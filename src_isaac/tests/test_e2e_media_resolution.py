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
