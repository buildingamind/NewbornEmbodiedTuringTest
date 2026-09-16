#!/usr/bin/env python3
"""Generate docs/env_vars.md — every NETT_* the code reads, with where and its default.

⛔ WHY THIS EXISTS. `campaign_train.py`'s module docstring carries a "Select via env:" list of
21 variables. The code reads **144**. Nothing marks that list as partial, so it reads as an
index and is not one: **absence from it means nothing**, which is exactly the check you want an
index for. A researcher (or an agent) confirming that a knob exists finds no entry and cannot
distinguish "not documented" from "does not exist" -- and a queue row naming a variable that
does not exist launches, trains, scores, and files under the experimental label while running
the control, with no symptom anywhere.

⇒ The index is GENERATED, never hand-maintained, because a hand-maintained index drifts back
into being partial the first time someone adds a knob. `tests/test_env_index.py` regenerates it
and fails if it differs from the committed file, so the drift is caught at commit time rather
than by whoever next trusts it.

Usage:  python scripts/gen_env_index.py          # write docs/env_vars.md
        python scripts/gen_env_index.py --check  # exit 1 if the file is stale
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN = ("nett_skrl", "examples", "scripts")
OUT = ROOT / "docs" / "env_vars.md"
#: ⛔ THIS FILE MUST NOT SCAN ITSELF. Its NETT_* literals are PATTERNS and DOCSTRING EXAMPLES, not
#: reads, and whole-file scanning duly indexed its own comments -- inventing `NETT_FOO` and filing
#: the generator as a consumer of two real knobs. An index that lists itself as a reader is worse
#: than one entry wrong: it makes the provenance column untrustworthy for every row.
SELF = Path(__file__).resolve()

# `os.environ.get("X", "default")`, `os.environ["X"]`, `os.getenv("X", "default")`, and
# `_env_flag("X", default)` and any other `_env_*` helper, plus names bound to a constant.
#
# ⛔ `_env_flag` WAS THE INDEX'S OWN BLIND SPOT, AND IT IS THE WORST PLACE TO HAVE ONE. This file
# exists because "absence from the index means nothing" is the failure it was built to end -- and
# until 2026-09-16 it scanned only literal `environ.get` calls, so every knob read through the
# fleet's canonical boolean helper was ABSENT from a document that says absence is meaningful.
# Nine of the ten `_env_flag` knobs were missing; the tenth appeared only because it is also read
# through `environ.get` somewhere else, which is exactly the kind of accident that makes a gap
# look smaller than it is. ⚠ A NEW WRAPPER AROUND `os.environ` REOPENS THIS. If you add one, add
# it here in the same commit -- the staleness test cannot see a variable the pattern never matches,
# so it will pass while the index is wrong.
#
# ⛔ THAT FIX FOUND TWO MORE OF THE SAME SHAPE, BOTH LIVE:
#   * OTHER HELPERS. `_env_float("NETT_PHYSX_BASE_MIB", 512.0)` is the same move with a different
#     suffix, so the alternative below matches `_env_[a-z_]+` rather than `_env_flag` alone.
#   * MULTI-LINE CALLS. This scanned LINE BY LINE, so
#     `Path(os.environ.get("NETT_CAMPAIGN_DIR",\n        default))` matched nothing. It now scans
#     whole-file text and derives line numbers from the match offset.
# ⇒ THE LESSON, WRITTEN DOWN BECAUSE IT WILL RECUR: this index's failure mode is not "a wrong
# entry", which someone would notice -- it is a MISSING entry in a document whose contract is that
# missing means "does not exist". Every such gap is silent, and its own test passes throughout.
PAT = re.compile(
    r"""(?:environ\.get|getenv)\(\s*["'](?P<name>NETT_[A-Z0-9_]+)["']\s*(?:,\s*(?P<default>[^)]*?))?\s*\)"""
    r"""|environ\[\s*["'](?P<name2>NETT_[A-Z0-9_]+)["']\s*\]"""
    r"""|_env_[a-z_]+\(\s*["'](?P<name3>NETT_[A-Z0-9_]+)["']\s*(?:,\s*(?P<default3>[^)]*?))?\s*\)""",
    re.X,
)


#: `NAME = "NETT_FOO"` / `NAME: str = "NETT_FOO"`, at module or class level. Two live auxiliary
#: losses name their offsets knob this way and then read it as `environ.get(self.OFFSETS_ENV, ...)`,
#: which puts the literal nowhere near the read.
CONST_PAT = re.compile(
    r"""^\s*(?P<ident>[A-Z][A-Z0-9_]*)\s*(?::[^=\n]*)?=\s*["'](?P<cname>NETT_[A-Z0-9_]+)["']""",
    re.M,
)


def _line_indexer(text: str):
    """Map a character offset to a 1-based line number."""
    import bisect
    starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]
    return lambda pos: bisect.bisect_right(starts, pos)


def _norm(default: str | None) -> str | None:
    """Collapse the whitespace a multi-line call drags into the captured default."""
    if default is None:
        return None
    return " ".join(default.split()) or None


def scan() -> dict[str, list[tuple[str, int, str | None]]]:
    found: dict[str, list[tuple[str, int, str | None]]] = {}
    # ⛔ TWO PASSES, BECAUSE A CONSTANT AND ITS READ NEED NOT SHARE A FILE.
    # `cltt_ref_stack_aux.py` binds `OFFSETS_ENV = "NETT_AUX_CLTT_STACK_OFFSETS"` and the read
    # lives in its PARENT class in `cltt_ref_aux.py`. A per-file resolution finds the binding,
    # finds no read beside it, and drops the variable -- silently, which is this index's whole
    # failure mode. Collect every file's text first, then resolve bindings against all of it.
    texts: list[tuple[str, str]] = []
    for top in SCAN:
        for path in sorted((ROOT / top).rglob("*.py")):
            if "__pycache__" in path.parts or path.resolve() == SELF:
                continue
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text()
            texts.append((rel, text))
            line_of = _line_indexer(text)
            for m in PAT.finditer(text):
                name = m.group("name") or m.group("name2") or m.group("name3")
                default = (m.group("default") or m.group("default3") or "").strip() or None
                if m.group("name3") and default is None:
                    # `_env_flag(name)` with no second argument defaults to False, which is
                    # a real default and not "required" -- reporting it as required would
                    # send a reader looking for a value they must supply.
                    default = "False"
                found.setdefault(name, []).append((rel, line_of(m.start()), _norm(default)))

    # Pass 2: names bound to a constant, resolved against the whole corpus.
    for rel, text in texts:
        for cm in CONST_PAT.finditer(text):
            ident, cname = cm.group("ident"), cm.group("cname")
            # ⚠ BOTH INDIRECTIONS AT ONCE: `_env_int(ENV_VAR, DEFAULT)` is a helper call taking
            # a constant, which neither the helper pattern (it wants a literal) nor an
            # environ-only constant pattern can see. NETT_KIT_THREADS was the last one hiding
            # behind exactly this combination.
            pat = re.compile(r"(?:environ\.get|getenv|_env_[a-z_]+)\(\s*(?:self\.|cls\.)?"
                             + re.escape(ident) + r"\s*(?:,\s*(?P<d>[^)]*?))?\s*\)")
            for use_rel, use_text in texts:
                use = pat.search(use_text)
                if not use:
                    continue
                site = (use_rel, _line_indexer(use_text)(use.start()), _norm(use.group("d")))
                if site not in found.get(cname, []):
                    found.setdefault(cname, []).append(site)
    return found


def render(found) -> str:
    out = [
        "# NETT_* environment variables — GENERATED, do not edit",
        "",
        "Regenerate with `python scripts/gen_env_index.py`. `tests/test_env_index.py` fails if",
        "this file drifts from the code, so it cannot quietly go partial.",
        "",
        "⛔ **THIS IS THE AUTHORITATIVE ANSWER TO \"DOES THIS KNOB EXIST?\"** A name absent from",
        "this table is a name the code never reads. Setting it has NO EFFECT and raises nothing —",
        "the run proceeds as the control while its label says otherwise. Before putting a `NETT_*`",
        "name in a queue row, a launcher, or a message, confirm it here or with",
        "`grep -rn NETT_YOUR_NAME src_isaac/`.",
        "",
        "⚠ Defaults are the literal second argument at the read site. Where a variable is read in",
        "more than one place the defaults can differ — every site is listed rather than collapsed,",
        "because a knob with two defaults is a real hazard and a single-row summary would hide it.",
        "",
        f"{len(found)} variables.",
        "",
        "| variable | default(s) | read at |",
        "|---|---|---|",
    ]
    for name in sorted(found):
        sites = found[name]
        defaults = sorted({d for _, _, d in sites if d is not None})
        dcol = ", ".join(f"`{d}`" for d in defaults) if defaults else "*(required / no literal default)*"
        where = "<br>".join(f"`{f}:{ln}`" for f, ln, _ in sites[:6])
        if len(sites) > 6:
            where += f"<br>*(+{len(sites) - 6} more)*"
        out.append(f"| `{name}` | {dcol} | {where} |")
    out.append("")
    return "\n".join(out)


def _label(path: Path) -> str:
    """Repo-relative when it is in the repo, absolute otherwise.

    `Path.relative_to` RAISES on a path outside the root, so formatting the error message was
    itself able to crash -- and the only caller that hits it is the failure branch, i.e. the
    one path that must not have a second bug in it.
    """
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render(scan())
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            print(f"{_label(OUT)} is STALE -- run: python scripts/gen_env_index.py",
                  file=sys.stderr)
            return 1
        print(f"{_label(OUT)} is up to date")
        return 0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {_label(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
