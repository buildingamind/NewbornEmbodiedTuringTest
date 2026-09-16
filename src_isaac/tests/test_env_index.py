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


def test_env_flag_knobs_are_indexed():
    """⛔ `_env_flag` WAS THIS INDEX'S BLIND SPOT, and a blind spot in an ABSENCE CHECK is the
    worst kind: the staleness test above cannot see a variable the scanner's pattern never
    matches, so it passed while nine of the ten `_env_flag` knobs were missing from a document
    whose entire value is that absence means something.

    Pinned by scanning for the CALL SITES rather than by listing names, so a knob added tomorrow
    is covered without editing this test.
    """
    names = set()
    for top in gen.SCAN:
        for path in (ROOT / top).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            names |= set(re.findall(r"""_env_flag\(\s*["'](NETT_[A-Z0-9_]+)["']""",
                                    path.read_text()))
    assert len(names) >= 5, f"expected several _env_flag knobs to exist, found {sorted(names)}"
    missing = names - set(gen.scan())
    assert not missing, (
        f"{sorted(missing)} are read through _env_flag but absent from the index. The scanner's "
        "PAT must match every wrapper around os.environ, not just literal environ.get calls.")


def test_env_flag_without_an_explicit_default_is_reported_as_False_not_required():
    """⚠ `_env_flag(name)` defaults to False -- a real default. Reporting it as
    "required / no literal default" would send a reader hunting for a value they must supply."""
    src = "x = _env_flag('NETT_MADE_UP_FLAG')\ny = _env_flag('NETT_MADE_UP_TWO', True)\n"
    hits = {m.group("name3"): (m.group("default3") or "").strip() or None
            for m in gen.PAT.finditer(src) if m.group("name3")}
    assert set(hits) == {"NETT_MADE_UP_FLAG", "NETT_MADE_UP_TWO"}
    assert hits["NETT_MADE_UP_TWO"] == "True"
    assert hits["NETT_MADE_UP_FLAG"] is None      # scan() then substitutes "False"
    from_scan = gen.scan()
    assert any(d == "False" for _, _, d in from_scan["NETT_DVS_BLUR"]) or \
        any(d == "True" for _, _, d in from_scan["NETT_DVS_BLUR"])


def test_a_name_bound_to_a_constant_is_indexed():
    """⛔ `OFFSETS_ENV = "NETT_AUX_CLTT_REF_OFFSETS"` then `environ.get(self.OFFSETS_ENV, ...)`
    puts the literal nowhere near the read. Both live CLTT losses name their offsets knob this
    way, and BOTH were missing -- one of them is set by a queued wave-15 row, so the row named a
    variable the index said did not exist.

    ⚠ The stack variant is the harder half: its binding is in `cltt_ref_stack_aux.py` and the
    READ is in its parent class in another file, so a per-file resolution finds the binding, sees
    no read beside it, and drops the name silently.
    """
    found = gen.scan()
    for name in ("NETT_AUX_CLTT_REF_OFFSETS", "NETT_AUX_CLTT_STACK_OFFSETS", "NETT_REAP_TOKEN"):
        assert name in found, f"{name} is read through a constant but is absent from the index"


def test_a_helper_call_taking_a_constant_is_indexed():
    """Both indirections at once: `_env_int(ENV_VAR, DEFAULT)`. Neither the helper pattern (it
    wants a literal) nor a constant pattern restricted to `environ.get` can see it."""
    assert "NETT_KIT_THREADS" in gen.scan()


def test_a_call_split_across_lines_is_indexed():
    """⛔ THE SCANNER USED TO READ LINE BY LINE, so
    `Path(os.environ.get("NETT_CAMPAIGN_DIR",\\n    default))` matched nothing at all."""
    found = gen.scan()
    assert "NETT_CAMPAIGN_DIR" in found
    src = (ROOT / "examples" / "campaign_run.py").read_text().splitlines()
    line = found["NETT_CAMPAIGN_DIR"][0][1]
    assert "NETT_CAMPAIGN_DIR" in src[line - 1], "reported line does not contain the read"


def test_the_generator_does_not_index_its_own_patterns():
    """⛔ Whole-file scanning made this file index ITS OWN comments -- inventing a variable from a
    docstring example and filing the generator as a reader of two real knobs. An index that lists
    itself as a consumer makes the provenance column untrustworthy for every row."""
    sites = [rel for hits in gen.scan().values() for rel, _, _ in hits]
    assert not [r for r in sites if r.endswith("gen_env_index.py")]
    assert "NETT_FOO" not in gen.scan(), "a docstring example leaked into the index"


def test_reported_line_numbers_point_at_the_read():
    """⚠ Line numbers now come from a character OFFSET rather than an enumerate() counter. An
    off-by-one there is invisible in the diff and wrong in every provenance link."""
    found = gen.scan()
    checked = 0
    for name, hits in found.items():
        for rel, line, _ in hits:
            text = (ROOT / rel).read_text().splitlines()
            assert 1 <= line <= len(text), f"{name}: line {line} out of range in {rel}"
            window = "\n".join(text[max(0, line - 1):line + 3])
            # Either the literal is right there, or it is a constant/helper read -- in which case
            # the line must at least contain an environ/helper call.
            assert name in window or re.search(r"environ|getenv|_env_[a-z_]+", window), (
                f"{name}: {rel}:{line} contains neither the name nor an env read")
            checked += 1
    assert checked > 150, f"expected to check many sites, checked {checked}"
