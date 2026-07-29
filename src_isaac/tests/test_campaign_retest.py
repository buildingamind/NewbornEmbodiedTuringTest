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


def test_retest_clears_old_test_csvs_before_replaying():
    """The pre-existing behaviour that made the stale summary so easy to miss."""
    src = _main_source()
    assert "unlink()" in src and "test_*.csv" in src, (
        "campaign_retest must clear the previous test CSVs before replaying")


def test_retest_records_the_eval_action_mode():
    """Provenance: a retest's numbers are only interpretable alongside this flag.

    ``NETT_EVAL_STOCHASTIC`` changes whether evaluation takes the policy mean or a
    sample, which changes what the resulting scores MEAN. Recording it in the timing
    JSON keeps a retest self-describing.
    """
    src = _main_source()
    assert "NETT_EVAL_STOCHASTIC" in src, (
        "campaign_retest must record the evaluation action mode with its results")


@pytest.mark.parametrize("needle", ["analyze(", "st_mtime"])
def test_guards_are_inside_main_not_merely_imported(needle):
    """Guard against the checks being deleted from main while an import lingers."""
    assert needle in _main_source(), f"{needle} must live inside main()"
