"""docs/env_vars.md must stay complete, because its VALUE IS THE ABSENCE CHECK.

⛔ THE DEFECT THIS REPLACES. `campaign_train.py`'s docstring carries a "Select via env:" list
of 21 variables. The code reads 143. Nothing marked that list as partial, so it read as an
index and was not one: a name missing from it might be undocumented or might not exist, and
those are opposite facts. A queue row naming a `NETT_*` that does not exist launches, trains,
scores, and files under the experimental label while running the control -- no raise, no
warning, no missing column. That happened: a row was written `env: {NETT_BODY_WRAPPERS: lumnorm}`
and that variable is read nowhere.

⇒ An index is only worth consulting if absence from it MEANS something. These tests are what
make absence mean something; without them the file becomes partial again on the next commit
that adds a knob, and silently resumes lying.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "gen_env_index.py"
DOC = ROOT / "docs" / "env_vars.md"

_spec = importlib.util.spec_from_file_location("_gen_env_index", GEN)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_the_index_is_not_stale():
    """Regenerate and compare. The message names the fix so nobody has to guess it."""
    assert DOC.exists(), f"{DOC} is missing -- run: python scripts/gen_env_index.py"
    assert DOC.read_text() == gen.render(gen.scan()), (
        "docs/env_vars.md is stale. A partial index is worse than none, because absence from "
        "it stops meaning anything. Run: python scripts/gen_env_index.py")


def test_the_checker_actually_fails_on_a_stale_file(tmp_path, monkeypatch):
    """⛔ RUN THE CHECK AGAINST A BROKEN INPUT. A --check that cannot fail is not a check, and
    it would pass identically on the day the index goes wrong."""
    fake = tmp_path / "env_vars.md"
    fake.write_text("# deliberately wrong\n")
    monkeypatch.setattr(gen, "OUT", fake)
    monkeypatch.setattr(sys, "argv", ["gen_env_index.py", "--check"])
    assert gen.main() == 1, "--check returned 0 on a file that does not match the code"


def test_there_is_something_to_index():
    """Establish the n: an empty scan would make every assertion here vacuous."""
    found = gen.scan()
    assert len(found) > 100, f"expected >100 NETT_* variables, found {len(found)}"


def test_every_variable_campaign_train_advertises_is_really_read():
    """The other direction: a docstring naming a knob the code dropped is equally misleading,
    and it fails *safe* only by accident -- a reader sets it and nothing happens."""
    doc = (ROOT / "examples" / "campaign_train.py").read_text().split('"""')[1]
    # A name may be cited as a COUNTER-EXAMPLE -- the docstring warns about a variable that
    # does not exist, which is the whole point of the warning. Such a citation must say so
    # inline, so the exemption is visible at the mention rather than hidden in this test.
    # ⛔ THE TRAILING \b IS LOAD-BEARING. Without it the regex BACKTRACKS: given
    # "NETT_BODY_WRAPPERS (DOES NOT EXIST)" the greedy match fails the lookahead, so the engine
    # retries one character shorter -- NETT_BODY_WRAPPER -- whose next character is "S", which
    # is not the marker, so the lookahead SUCCEEDS and a name that does not exist is reported
    # under a name that also does not exist. A negative lookahead on a variable-length token is
    # only as strong as the boundary that stops it shrinking.
    advertised = {m.group(1)
                  for m in re.finditer(r"\b(NETT_[A-Z0-9_]+)\b(?!\s*\(DOES NOT EXIST\))", doc)}
    read = set(gen.scan())
    orphans = advertised - read
    assert not orphans, (
        f"campaign_train's docstring advertises {sorted(orphans)}, which the code never reads")


def test_the_index_records_that_NETT_AUX_BATCH_has_more_than_one_default():
    """⚠ A REAL HAZARD THE INDEX SURFACED ON ITS FIRST RUN, pinned so it cannot be tidied away.

    NETT_AUX_BATCH is read at 13 sites with FOUR different literal defaults ("0", "256", "512",
    and a parameter). So "the aux batch" is not one number: an arm that does not set it gets a
    different B depending on which loss it declared, and every level claim about a contrastive
    objective depends on B (NT-Xent chance is ln(2B-1)). A one-row summary would have hidden it.
    """
    sites = gen.scan()["NETT_AUX_BATCH"]
    defaults = {d for _, _, d in sites if d is not None}
    assert len(defaults) > 1, (
        "NETT_AUX_BATCH now has a single default. If that was a deliberate unification, delete "
        "this test and say so; if it is accidental, the differing defaults were load-bearing.")
    assert len(sites) > 5, f"expected many read sites, found {len(sites)}"
