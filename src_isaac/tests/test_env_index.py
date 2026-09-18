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
import textwrap
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
        "CALL_RE must match every wrapper around os.environ, not just literal environ.get calls.")


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


# --------------------------------------------------------------------------------------------
# ⛔ THE FOUR SHAPES BELOW ARE PINNED AS SHAPES, NOT AS KNOB NAMES. Every one of them is live on
# a wave-17 knob today, but a test that names `NETT_AUX_PATCH_BATCH` goes stale the first time
# that knob is renamed or retired, and then stops guarding the generator that broke on it. A
# synthetic corpus states the CALL-SITE SHAPE the generator must handle and outlives the names.
#
# ⚠ All four are DEFAULT-COLUMN defects, not missing-entry defects, so the absence check above
# passes throughout. A wrong default is the quieter failure: the reader believes it.


def _corpus(tmp_path, monkeypatch, files: dict[str, str]) -> dict:
    """Point the scanner at a synthetic package and return its scan()."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for rel, src in files.items():
        (pkg / rel).write_text(textwrap.dedent(src).lstrip())
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "SCAN", ("pkg",))
    return gen.scan()


def _defaults(found: dict, name: str) -> set:
    assert name in found, f"{name} never reached the index at all: {sorted(found)}"
    return {d for _, _, d in found[name] if d is not None}


def test_a_default_read_through_a_shared_base_resolves_per_CALLER(tmp_path, monkeypatch):
    """⛔ SHAPE 1: ONE READ SITE, MANY CALLERS, DIFFERENT DEFAULTS.

    A base class reads `helper(self.SOME_ENV, self.SOME_DEFAULT)` and each subclass supplies both.
    Every subclass's knob therefore resolves to the SAME read site, and printing that site's
    source text gives every one of them the identifier `self.DEFAULT_BATCH` -- so the index cannot
    answer the one question it exists for: what do I get if I do not set this? The four wave-17
    `*_BATCH` knobs sit on exactly this shape and one of them differs from the other three by 16x.
    """
    found = _corpus(tmp_path, monkeypatch, {
        "base.py": '''
            from .knobs import _env_positive_int


            class BaseTerm:
                BATCH_ENV = None
                DEFAULT_BATCH = 32

                def __init__(self):
                    self.batch = _env_positive_int(self.BATCH_ENV, self.DEFAULT_BATCH)
        ''',
        "wide.py": '''
            class WideTerm(BaseTerm):
                BATCH_ENV = "NETT_TEST_WIDE_BATCH"
                DEFAULT_BATCH = 512
        ''',
        "narrow.py": '''
            class NarrowTerm(BaseTerm):
                BATCH_ENV = "NETT_TEST_NARROW_BATCH"
                DEFAULT_BATCH = 32
        ''',
    })
    assert _defaults(found, "NETT_TEST_WIDE_BATCH") == {"512"}
    assert _defaults(found, "NETT_TEST_NARROW_BATCH") == {"32"}
    # Both still point at the real read site -- resolving the value must not lose the provenance.
    assert found["NETT_TEST_WIDE_BATCH"][0][0] == "pkg/base.py"


def test_a_default_that_is_a_function_parameter_resolves_to_its_signature(tmp_path, monkeypatch):
    """SHAPE 1b: the identifier is a PARAMETER of the enclosing function, not a class attribute.
    `decay` (NETT_AUX_EMA_DECAY) and a dozen older knobs print as bare parameter names today."""
    found = _corpus(tmp_path, monkeypatch, {
        "holder.py": '''
            class Holder:
                DECAY_ENV = "NETT_TEST_DECAY"

                def __init__(self, decay: float = 0.996):
                    self.decay = _env_unit_interval(self.DECAY_ENV, decay)
        ''',
    })
    assert _defaults(found, "NETT_TEST_DECAY") == {"0.996"}


def test_two_modules_binding_the_same_constant_name_do_not_share_provenance(tmp_path, monkeypatch):
    """⛔ SHAPE 2: CROSS-FILE CONSTANT COLLISION.

    Three wave-17 modules each bind `TEMP_ENV` in their own class. The whole-corpus resolution
    pass matches the constant NAME in every file, so each of the three knobs is filed with all
    three read sites and both defaults -- wrong provenance, and the same class of defect that
    made NETT_AUX_BATCH's real multi-default hazard indistinguishable from an artefact.

    ⚠ The cross-file pass must STAY, though: a constant bound in a subclass file and read in its
    parent (NETT_AUX_CLTT_STACK_OFFSETS) is only found that way. Own module FIRST, others only
    when the defining module reads it nowhere.
    """
    found = _corpus(tmp_path, monkeypatch, {
        "x_term.py": '''
            class XTerm:
                TEMP_ENV = "NETT_TEST_X_TEMP"

                def __init__(self):
                    self.t = _env_positive_float(self.TEMP_ENV, 0.1)
        ''',
        "y_term.py": '''
            class YTerm:
                TEMP_ENV = "NETT_TEST_Y_TEMP"

                def __init__(self):
                    self.t = _env_positive_float(self.TEMP_ENV, 0.5)
        ''',
    })
    assert _defaults(found, "NETT_TEST_X_TEMP") == {"0.1"}
    assert _defaults(found, "NETT_TEST_Y_TEMP") == {"0.5"}
    assert [rel for rel, _, _ in found["NETT_TEST_X_TEMP"]] == ["pkg/x_term.py"]
    assert [rel for rel, _, _ in found["NETT_TEST_Y_TEMP"]] == ["pkg/y_term.py"]


def test_a_constant_bound_where_it_is_not_read_still_resolves_across_files(tmp_path, monkeypatch):
    """The other half of shape 2, pinned so the fix for the collision cannot delete it."""
    found = _corpus(tmp_path, monkeypatch, {
        "parent.py": '''
            class Parent:
                OFFSETS_ENV = None
                DEFAULT_OFFSETS = "1,2"

                def __init__(self):
                    self.off = os.environ.get(self.OFFSETS_ENV, self.DEFAULT_OFFSETS)
        ''',
        "child.py": '''
            class Child(Parent):
                OFFSETS_ENV = "NETT_TEST_CHILD_OFFSETS"
                DEFAULT_OFFSETS = "2,4"
        ''',
    })
    assert [rel for rel, _, _ in found["NETT_TEST_CHILD_OFFSETS"]] == ["pkg/parent.py"]
    assert _defaults(found, "NETT_TEST_CHILD_OFFSETS") == {"'2,4'"}


def test_env_flag_through_a_constant_is_False_not_required(tmp_path, monkeypatch):
    """⛔ SHAPE 3: `_env_flag(NAME)` with no second argument defaults to False -- a real default.

    The False rule lived only in the literal branch, so the SAME call reached through a constant
    printed "(required / no literal default)" and sent the reader hunting for a value they must
    supply. A knob reported as required is a knob a launcher may refuse to omit.
    """
    found = _corpus(tmp_path, monkeypatch, {
        "flagger.py": '''
            class Flagger:
                FLAG_ENV = "NETT_TEST_FLAG"

                def __init__(self):
                    self.on = _env_flag(self.FLAG_ENV)
                    self.off = _env_flag("NETT_TEST_FLAG_LITERAL")
        ''',
    })
    assert _defaults(found, "NETT_TEST_FLAG") == {"False"}
    assert _defaults(found, "NETT_TEST_FLAG_LITERAL") == {"False"}


def test_a_closing_paren_inside_the_default_does_not_truncate_it(tmp_path, monkeypatch):
    """⛔ SHAPE 4: the default was captured with `[^)]*?`, so the FIRST `)` ended it.

    `max(1, self.n_tokens // 2)` printed as `max(1, self.n_tokens // 2` -- an expression a reader
    cannot evaluate and cannot paste, and one that looks like a scanner artefact exactly when it
    is not. Nesting must be counted, not excluded.
    """
    found = _corpus(tmp_path, monkeypatch, {
        "topg.py": '''
            class Topg:
                TOPG_ENV = "NETT_TEST_TOPG"

                def __init__(self):
                    self.g = _env_positive_int(self.TOPG_ENV, max(1, self.n_tokens // 2))
                    self.h = _env_positive_int("NETT_TEST_TOPG_LITERAL", max(1, 40 // 2))
        ''',
    })
    assert _defaults(found, "NETT_TEST_TOPG") == {"max(1, self.n_tokens // 2)"}
    assert _defaults(found, "NETT_TEST_TOPG_LITERAL") == {"max(1, 40 // 2)"}


def test_an_unresolvable_identifier_says_caller_supplied_rather_than_printing_itself(
        tmp_path, monkeypatch):
    """⚠ THE HONEST ANSWER IS AN ACCEPTABLE ANSWER; A WRONG VALUE IS NOT.

    When the default comes from something the scanner cannot see statically, the column must SAY
    so. Printing the identifier reads as a value -- `cfg.batch` in the default column looks like
    the default IS the string cfg.batch to anyone not reading the source beside it.
    """
    found = _corpus(tmp_path, monkeypatch, {
        "opaque.py": '''
            class Opaque:
                OPAQUE_ENV = "NETT_TEST_OPAQUE"

                def __init__(self, cfg):
                    self.n = _env_positive_int(self.OPAQUE_ENV, cfg.batch)
        ''',
    })
    (d,) = _defaults(found, "NETT_TEST_OPAQUE")
    assert isinstance(d, gen.CallerSupplied), f"{d!r} was printed as if it were a value"
    row = [ln for ln in gen.render(found).splitlines() if "NETT_TEST_OPAQUE" in ln][0]
    assert "caller-supplied" in row and "pkg/opaque.py:" in row


def test_no_default_anywhere_in_the_index_is_a_bare_identifier():
    """The same rule over the REAL corpus: a bare name in the default column is always either a
    resolution the generator owes the reader or a caller-supplied it must admit to."""
    ident = re.compile(r"^(?:self\.|cls\.)?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
    keywords = {"True", "False", "None"}
    bad = []
    for name, sites in gen.scan().items():
        for rel, line, d in sites:
            if d is None or d in keywords or isinstance(d, gen.CallerSupplied):
                continue
            if ident.match(d):
                bad.append(f"{name} -> {d!r} at {rel}:{line}")
    assert not bad, ("these defaults print an identifier as if it were a value:\n  "
                     + "\n  ".join(sorted(bad)))


def test_a_presence_probe_beside_a_real_read_does_not_erase_the_default(tmp_path, monkeypatch):
    """⛔ SHAPE: ONE FILE, TWO READS OF ONE KNOB -- a bare `os.environ.get(NAME)` used to REFUSE a
    knob the running configuration cannot read, and the real read that supplies the default.

    The refusal probe has no default by construction: it is asking "did a human set this?", not
    "what is the value?". Collapsing a file's reads to the FIRST one therefore made the answer
    depend on which line came first, and a probe written above the real read turned a knob with a
    documented default into `*(required / no literal default)*` -- an index row that tells a
    launcher the knob MUST be set when in fact it must not be.

    Written against the SHAPE, not the name: any file that guards a knob and then reads it.
    """
    found = _corpus(tmp_path, monkeypatch, {
        "term.py": '''
            import os
            from .knobs import _env_positive_int

            class Term:
                WIDTH_ENV = "NETT_SHAPE_WIDTH"

                def __init__(self, kind):
                    if kind != "mlp" and os.environ.get(self.WIDTH_ENV) is not None:
                        raise ValueError("that decoder does not read a width")
                    self.width = _env_positive_int(self.WIDTH_ENV, 256)
        ''',
        "knobs.py": '''
            import os

            def _env_positive_int(name, default):
                return int(os.environ.get(name, default))
        ''',
    })
    assert _defaults(found, "NETT_SHAPE_WIDTH") == {"256"}
    lines = sorted(line for _, line, _ in found["NETT_SHAPE_WIDTH"])
    assert len(lines) == 2, ("both reads belong in the provenance column -- the probe is a real "
                             f"read of the knob, it just has nothing to say about the default: "
                             f"{found['NETT_SHAPE_WIDTH']}")


def test_the_probe_ordering_does_not_decide_the_answer(tmp_path, monkeypatch):
    """The same two reads in the other order must give the same row. A rule whose answer depends
    on line order is not a rule, and this is the half of the defect that made it invisible: on
    every knob where the real read happened to come first, the index was right by luck."""
    src = '''
        import os
        from .knobs import _env_positive_int

        class Term:
            WIDTH_ENV = "NETT_SHAPE_WIDTH"

            def __init__(self, kind):
                self.width = _env_positive_int(self.WIDTH_ENV, 256)
                if kind != "mlp" and os.environ.get(self.WIDTH_ENV) is not None:
                    raise ValueError("that decoder does not read a width")
    '''
    found = _corpus(tmp_path, monkeypatch, {
        "term.py": src,
        "knobs.py": '''
            import os

            def _env_positive_int(name, default):
                return int(os.environ.get(name, default))
        ''',
    })
    assert _defaults(found, "NETT_SHAPE_WIDTH") == {"256"}
