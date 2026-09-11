"""The licence preflight, tested at the CLASS rather than at one entry point.

⛔ WHY THIS FILE EXISTS. The check lived as a pasted `if` in `gate_a_resume.py`, with a
comment recording that it had already cost two launches. It then cost a third on
2026-09-11 in `capture_observations.py`, which did not have it. A lesson written into
one callsite protects one callsite. These tests assert the rule AND that every
Kit-booting entry point in the tree reaches it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nett_skrl.runtime.kit_preflight import (  # noqa: E402
    eula_blocks_boot, require_eula_or_explain,
)


class _Stdin:
    def __init__(self, tty): self._tty = tty
    def isatty(self): return self._tty


@pytest.mark.parametrize("env,tty,blocked", [
    ({}, False, True),                                 # nohup, unset -> the failure
    ({}, True, False),                                 # a human can answer the prompt
    ({"OMNI_KIT_ACCEPT_EULA": "YES"}, False, False),   # the fix
    ({"OMNI_KIT_ACCEPT_EULA": "YES"}, True, False),
])
def test_the_rule(env, tty, blocked):
    assert eula_blocks_boot(env, _Stdin(tty)) is blocked


def test_an_unanswerable_stdin_blocks_rather_than_crashing():
    """⚠ A closed or replaced stdin cannot answer a prompt either, and an exception from
    `isatty()` must not be read as 'a tty is available'."""
    class Broken:
        def isatty(self): raise ValueError("I/O operation on closed file")
    assert eula_blocks_boot({}, Broken()) is True


def test_the_message_names_the_variable_and_both_surfaced_errors():
    import io
    buf = io.StringIO()
    assert require_eula_or_explain({}, _Stdin(False), buf) is False
    text = buf.getvalue()
    assert "OMNI_KIT_ACCEPT_EULA=YES" in text, "must name the fix, not just the problem"
    # Both spellings the user actually sees, so a search for either finds this.
    assert "EOF when reading a line" in text
    assert "exit code 1" in text
    buf2 = io.StringIO()
    assert require_eula_or_explain({"OMNI_KIT_ACCEPT_EULA": "1"}, _Stdin(False), buf2) is True
    assert buf2.getvalue() == "", "no noise on the happy path"


def test_every_kit_booting_entry_point_reaches_the_preflight():
    """⛔ THE TEST THAT WOULD HAVE CAUGHT THE THIRD OCCURRENCE. A driver that boots Kit
    and does not consult the preflight -- directly or via Environment.load -- is the
    defect, not a style difference."""
    boots = sorted(
        p for p in (ROOT / "examples").glob("*.py")
        if "body.embed" in p.read_text() or "AppLauncher" in p.read_text()
    )
    assert boots, "the glob found no Kit-booting driver; the probe is broken, not the tree"
    missing = [p.name for p in boots
               if "kit_preflight" not in p.read_text()
               and "OMNI_KIT_ACCEPT_EULA" not in p.read_text()]
    assert not missing, (
        f"these drivers boot Kit with no licence preflight: {missing}. Call "
        f"nett_skrl.runtime.kit_preflight.require_eula_or_explain() before the boot.")


def test_the_library_chokepoint_still_holds_the_guard():
    """Entry points outside examples/ exist too (tests, ad-hoc scripts). Environment.load
    is what they all reach, so the guard must survive there even if a driver forgets."""
    src = (ROOT / "nett_skrl" / "environment" / "environment.py").read_text()
    load = src[src.index("def load(self, config"):]
    guard = load.index("eula_blocks_boot")
    assert guard < load.index("AppLauncher("), "the guard must precede the boot, not follow it"
