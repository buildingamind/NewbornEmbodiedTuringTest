"""A retest must not leave an analysis that describes the PREVIOUS run.

THE BUG THIS PINS. ``examples/campaign_retest.py`` re-ran the test phase, cleared and
rewrote every ``test_*.csv``, and then returned -- without calling ``analyze``. So
``analysis/summary.json`` still described the run BEFORE the retest.

The cost was nearly a wrong scientific conclusion. On 2026-07-29 a retest was run with
stochastic action selection to test whether the deterministic evaluation was suppressing a
weak preference. Reading ``summary.json`` afterwards showed results identical to the
deterministic baseline to three decimal places in all eight test conditions, which reads
as a clean "the readout was not the cause". Only the implausibility of EXACT agreement --
given a trained policy with an action std of 0.84 -- prompted a check of the file mtimes:
the CSV had been rewritten at 10:30 and the summary was from 05:07. The real comparison,
computed from the new CSV, did differ.

Same failure class as a stale ``golden.json``: an artifact that silently describes a
different run than the one just executed. These are source-text tests because the script
is a driver -- running it needs Isaac, a GPU and a completed run to replay.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_RETEST = Path(__file__).resolve().parents[1] / "examples" / "campaign_retest.py"


def _main_source() -> str:
    tree = ast.parse(_RETEST.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return ast.unparse(fn)


def test_retest_reruns_the_analysis():
    """Rewriting the test CSVs without re-analysing leaves a stale summary."""
    src = _main_source()
    assert "analyze(" in src, (
        "campaign_retest.main must call analyze() after the run; otherwise it rewrites "
        "test_*.csv and leaves analysis/summary.json describing the previous run")


def test_retest_refuses_a_summary_older_than_its_data():
    """A belt-and-braces guard, because the failure is silent and reads as a result.

    If the analysis somehow does not take effect, the script must say so rather than
    exit 0 and let the next reader trust the file.
    """
    src = _main_source()
    assert "st_mtime" in src and "summary" in src, (
        "campaign_retest.main must verify analysis/summary.json is newer than the "
        "test_*.csv files it summarises, and fail loudly if it is not")


def test_retest_moves_old_test_csvs_aside_without_deleting_them():
    """The previous test CSVs must be moved out of the glob, never deleted.

    A retest used to ``unlink()`` every ``test_*.csv``. On 2026-09-05 that destroyed a
    COMPLETE, valid test CSV from an arm that had wedged in teardown after finishing its
    test loop; the "recovery" was the data loss. The fresh run still must not collide on
    env_id/episode, so the old files are renamed with a ``.superseded_<stamp>`` suffix,
    which no longer matches ``test_*.csv``.
    """
    src = _main_source()
    assert "unlink()" not in src, (
        "campaign_retest must not delete previous test CSVs; rename them instead")
    assert "test_*.csv" in src and ".rename(" in src and "superseded_" in src, (
        "campaign_retest must move previous test_*.csv files aside by rename with a "
        "superseded_<stamp> suffix before replaying")


def test_retest_records_the_eval_action_mode():
    """Provenance: a retest's numbers are only interpretable alongside this flag.

    ``NETT_EVAL_STOCHASTIC`` changes whether evaluation takes the policy mean or a
    sample, which changes what the resulting scores MEAN. Recording it in the timing
    JSON keeps a retest self-describing.

    ⚠ IT MUST GO THROUGH ``eval_stochastic_enabled()``, not a second ``os.environ.get``
    with its own default (tightened 2026-08-12, when the default flipped 0 -> 1). This
    line previously hardcoded ``"0"``; left alone it would have recorded the exact
    OPPOSITE of the protocol the retest actually ran under -- a self-describing file
    confidently describing the wrong thing, which is worse than not recording it at all.
    """
    src = _main_source()
    assert "eval_stochastic_enabled()" in src, (
        "campaign_retest must record the evaluation action mode with its results")
    assert 'os.environ.get("NETT_EVAL_STOCHASTIC"' not in src, (
        "campaign_retest must not re-parse NETT_EVAL_STOCHASTIC with its own default -- "
        "it would drift from the accessor the trainer actually reads")


@pytest.mark.parametrize("needle", ["analyze(", "st_mtime"])
def test_guards_are_inside_main_not_merely_imported(needle):
    """Guard against the checks being deleted from main while an import lingers."""
    assert needle in _main_source(), f"{needle} must live inside main()"
