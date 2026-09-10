"""The e2e tree's auto-skip must not hide a misconfigured invocation as a missing one.

⛔ MEASURED 2026-09-10: `pytest tests/e2e -m e2e_isaac -n 3` returned **21 skipped in 3.09s,
exit code 0**, on a host where every one of those tests passes. `OMNI_KIT_ACCEPT_EULA` was
unset, so Kit's import-time bootstrap blocked on the interactive licence prompt, pytest's
capture broke the stdin read, and the conftest's prerequisite probe recorded it as
"isaacsim import failed" -- which is the same line a machine with no Isaac Sim produces.

The skip is correct (a developer without Isaac must still run the unit suite) and the exit
code is pytest's, not ours. What is fixable is that the reason named the symptom instead of
the one-line fix, so a green run that executed nothing was indistinguishable from a green
run that executed everything unless you counted the tests.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

_spec = importlib.util.spec_from_file_location("e2e_conftest_probe",
                                               REPO / "tests" / "e2e" / "conftest.py")
_cft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cft)

EULA_ERROR = SystemExit(
    "Unable to bootstrap inner kit kernel: pytest: reading from stdin while output is "
    "captured!  Consider using `-s`.")


def test_the_eula_skip_names_the_variable_that_fixes_it(monkeypatch):
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    text = _cft._eula_hint(EULA_ERROR)
    assert "OMNI_KIT_ACCEPT_EULA" in text
    assert "SKIPS ITSELF GREEN" in text, "the reader must be told the run proved nothing"


def test_the_original_error_is_preserved_not_replaced(monkeypatch):
    """A hint that swallows the real message sends the next reader to the wrong place."""
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    assert str(EULA_ERROR) in _cft._eula_hint(EULA_ERROR)


def test_a_genuine_absence_gets_no_eula_hint(monkeypatch):
    """A machine without Isaac Sim must not be told to accept a licence it cannot use."""
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    err = ImportError("No module named 'isaacsim'")
    assert _cft._eula_hint(err) == str(err)


def test_no_hint_once_the_variable_is_set(monkeypatch):
    """With the licence accepted, a bootstrap failure is a REAL failure, and pointing at
    the env var would send the reader to a box that is already ticked."""
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    err = SystemExit("Unable to bootstrap inner kit kernel: something else entirely")
    assert _cft._eula_hint(err) == str(err)


def test_every_known_signature_of_this_failure_is_matched(monkeypatch):
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    for sig in ("Do you accept the EULA? (Yes/No)",
                "reading from stdin while output is captured",
                "Unable to bootstrap inner kit kernel"):
        assert "OMNI_KIT_ACCEPT_EULA is UNSET" in _cft._eula_hint(RuntimeError(sig)), sig
