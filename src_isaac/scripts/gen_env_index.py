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
import ast
import functools
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
#: A call to anything that reads the environment. The NAME and the DEFAULT are pulled out by
#: `_call_args` rather than by the regex, because a regex cannot count parentheses -- see below.
CALL_RE = re.compile(r"(?P<callee>environ\.get|getenv|_env_[a-z_]+)\s*\(")
SUBSCRIPT_RE = re.compile(r"""environ\[\s*["'](?P<name>NETT_[A-Z0-9_]+)["']\s*\]""")
NAME_LIT = re.compile(r"""^["'](?P<name>NETT_[A-Z0-9_]+)["']$""")
#: The first argument as a CONSTANT rather than a literal: `self.BATCH_ENV`, `cls.X`, `X`.
CONST_REF = re.compile(r"^(?:self\.|cls\.)?(?P<ident>[A-Z][A-Z0-9_]*)$")
#: A default that is nothing but a name (possibly dotted). These are the dangerous ones: printed
#: verbatim they READ AS VALUES, and the reader has no way to tell `decay` the parameter from
#: `decay` the string. Anything with an operator, a call or a literal in it is an expression and
#: is printed as written.
DOTTED = re.compile(r"^(?:self\.|cls\.)?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
KEYWORDS = {"True", "False", "None"}


class CallerSupplied(str):
    """A default the scanner could not resolve statically, carrying the expression it gave up on.

    ⚠ IT IS A `str` SUBCLASS ON PURPOSE: every consumer that sorts, sets or prints defaults keeps
    working, and `render` is the one place that has to know the difference. Saying "I could not
    resolve this, here is the read site" is an acceptable answer. Printing the identifier as
    though it were the value is not -- that is the failure this class exists to make impossible
    to reintroduce by accident.
    """


#: `NAME = "NETT_FOO"` / `NAME: str = "NETT_FOO"`, at module or class level. Two live auxiliary
#: losses name their offsets knob this way and then read it as `environ.get(self.OFFSETS_ENV, ...)`,
#: which puts the literal nowhere near the read.
CONST_PAT = re.compile(
    r"""^\s*(?P<ident>[A-Z][A-Z0-9_]*)\s*(?::[^=\n]*)?=\s*["'](?P<cname>NETT_[A-Z0-9_]+)["']""",
    re.M,
)


def _call_args(text: str, open_idx: int) -> list[str] | None:
    """Split the argument list of the call whose ``(`` sits at `open_idx`, at TOP-LEVEL commas.

    ⛔ A `)` INSIDE THE DEFAULT USED TO TRUNCATE IT. The old pattern captured the default as
    `[^)]*?`, so `_env_positive_int(self.TOPG_ENV, max(1, self.n_tokens // 2))` was indexed with
    the default `max(1, self.n_tokens // 2` -- an expression missing its closing paren, which a
    reader can neither evaluate nor paste, and which looks like a scanner artefact exactly when
    it is not. `f"{ENC}_s{SEED_OFFSET}_{datetime.now(` was the same defect on a longer default.
    Nesting has to be COUNTED; no regex can do it.

    Returns None when the call never closes (a truncated or unparseable file), so the caller
    drops the site rather than indexing a guess.
    """
    depth, quote, start, args = 0, None, open_idx + 1, []
    i = open_idx
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                args.append(text[start:i])
                return [a for a in args if a.strip()] or []
        elif ch == "," and depth == 1:
            args.append(text[start:i])
            start = i + 1
        i += 1
    return None


def _iter_calls(text: str):
    """Yield `(callee, args, offset)` for every environment read in `text`."""
    for m in CALL_RE.finditer(text):
        args = _call_args(text, m.end() - 1)
        if args:
            yield m.group("callee"), args, m.start()


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


@functools.lru_cache(maxsize=None)
def _tree(text: str):
    try:
        return ast.parse(text)
    except SyntaxError:      # a scanned file need not be importable on this interpreter
        return None


def _assigns(body) -> dict[str, str]:
    out: dict[str, str] = {}
    for st in body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            out[st.targets[0].id] = ast.unparse(st.value)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            out[st.target.id] = ast.unparse(st.value)
    return out


def _innermost(text: str, lineno: int, kinds) -> object | None:
    tree = _tree(text)
    if tree is None:
        return None
    best = None
    for node in ast.walk(tree):
        if isinstance(node, kinds) and node.lineno <= lineno <= (node.end_lineno or node.lineno):
            if best is None or node.lineno > best.lineno:
                best = node
    return best


def _class_consts(text: str, lineno: int) -> dict[str, str]:
    cls = _innermost(text, lineno, ast.ClassDef)
    return _assigns(cls.body) if cls is not None else {}


def _module_consts(text: str) -> dict[str, str]:
    tree = _tree(text)
    return _assigns(tree.body) if tree is not None else {}


def _param_default(text: str, lineno: int, ident: str) -> str | None:
    """The default of parameter `ident` in the function containing `lineno`, if it has one."""
    fn = _innermost(text, lineno, (ast.FunctionDef, ast.AsyncFunctionDef))
    if fn is None:
        return None
    a = fn.args
    positional = list(a.posonlyargs) + list(a.args)
    for i, arg in enumerate(positional):
        if arg.arg == ident:
            k = i - (len(positional) - len(a.defaults))
            return ast.unparse(a.defaults[k]) if k >= 0 else None
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        if arg.arg == ident and d is not None:
            return ast.unparse(d)
    return None


def _resolve_default(callee: str, src: str | None,
                     bind: tuple[str, int], read: tuple[str, int]) -> str | None:
    """Turn the default AS WRITTEN into a value, or into an explicit `CallerSupplied`.

    ⛔ WHY THE *CALLER'S* MODULE IS CONSULTED FIRST. `TokenWindowTerm.__init__` reads
    `_env_positive_int(self.BATCH_ENV, self.DEFAULT_BATCH)` once, and FOUR subclasses each bind
    their own `BATCH_ENV` and their own `DEFAULT_BATCH` against it. Resolving the identifier at
    the READ site gives the base class's 32 for all four and hides that one of them is 512 -- a
    16x difference in the batch every NT-Xent chance level is computed from. The binding site is
    what identifies the caller, so the class that binds the name is where its default is looked
    up; the read site is only the fallback.
    """
    if src is None:
        # `_env_flag(name)` with no second argument returns False. That is a real default, and
        # reporting it as "required" sends a reader hunting for a value they must supply. The
        # rule lives HERE rather than in one branch, because the same call reached through a
        # constant used to fall out of the literal branch and print as required.
        return "False" if callee.startswith("_env_") else None
    if src in KEYWORDS or not DOTTED.match(src):
        return src                                   # a literal, or an expression as written
    ident = src.split(".")[-1]
    head = src[:-len(ident)].rstrip(".")
    if head not in ("", "self", "cls"):
        return CallerSupplied(src)                   # `cfg.batch`: an object we cannot see
    brel, bline = bind
    rrel, rline = read
    btext, rtext = _TEXTS.get(brel, ""), _TEXTS.get(rrel, "")
    if head:
        # `self.X` / `cls.X`: an ATTRIBUTE, so the caller's class is looked at first and the
        # read site's class only as the base-class fallback.
        candidates = (_class_consts(btext, bline).get(ident),
                      _module_consts(btext).get(ident),
                      _class_consts(rtext, rline).get(ident),
                      _module_consts(rtext).get(ident))
    else:
        # ⚠ A BARE NAME FOLLOWS PYTHON SCOPING, NOT THE CLASS BODY. Inside a method, `slots`
        # is the parameter or a module global -- a class attribute of the same name is NOT in
        # scope, and consulting one would print a value the code cannot possibly read.
        candidates = (_param_default(rtext, rline, ident),
                      _module_consts(rtext).get(ident))
    for value in candidates:
        if value is not None:
            return value
    return CallerSupplied(src)


#: Every scanned file's text, so a resolution can look outside the file it is resolving in.
_TEXTS: dict[str, str] = {}


def scan() -> dict[str, list[tuple[str, int, str | None]]]:
    found: dict[str, list[tuple[str, int, str | None]]] = {}
    # ⛔ TWO PASSES, BECAUSE A CONSTANT AND ITS READ NEED NOT SHARE A FILE.
    # `cltt_ref_stack_aux.py` binds `OFFSETS_ENV = "NETT_AUX_CLTT_STACK_OFFSETS"` and the read
    # lives in its PARENT class in `cltt_ref_aux.py`. A per-file resolution finds the binding,
    # finds no read beside it, and drops the variable -- silently, which is this index's whole
    # failure mode. Collect every file's text first, then resolve bindings against all of it.
    _TEXTS.clear()
    texts: list[tuple[str, str]] = []
    const_reads: dict[str, list[tuple[str, int, str, str | None]]] = {}
    bindings: list[tuple[str, int, str, str]] = []
    for top in SCAN:
        for path in sorted((ROOT / top).rglob("*.py")):
            if "__pycache__" in path.parts or path.resolve() == SELF:
                continue
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text()
            texts.append((rel, text))
            _TEXTS[rel] = text
            line_of = _line_indexer(text)
            # ⚠ CALLS AND SUBSCRIPTS ARE MERGED BY OFFSET, NOT SCANNED IN TWO SWEEPS. Sites are
            # listed in the order they are found, so scanning all `environ.get(...)` before all
            # `environ["..."]` would report a file's sites out of line order -- a provenance
            # column that reads as if the earlier line came second.
            hits = []
            for callee, args, start in _iter_calls(text):
                default = _norm(args[1]) if len(args) > 1 else None
                first = args[0].strip()
                lit = NAME_LIT.match(first)
                if lit:
                    hits.append((start, lit.group("name"), callee, default))
                    continue
                ref = CONST_REF.match(first)
                if ref:
                    const_reads.setdefault(ref.group("ident"), []).append(
                        (rel, line_of(start), callee, default))
            hits += [(m.start(), m.group("name"), None, None)
                     for m in SUBSCRIPT_RE.finditer(text)]
            for start, name, callee, default in sorted(hits):
                line = line_of(start)
                site = (rel, line, None if callee is None else
                        _resolve_default(callee, default, (rel, line), (rel, line)))
                found.setdefault(name, []).append(site)
            for cm in CONST_PAT.finditer(text):
                bindings.append((rel, line_of(cm.start()), cm.group("ident"), cm.group("cname")))

    # Pass 2: names bound to a constant, resolved against the whole corpus.
    for brel, bline, ident, cname in bindings:
        reads = const_reads.get(ident, [])
        # ⛔ OWN MODULE FIRST. `TEMP_ENV` is bound in THREE wave-17 modules, each with its own
        # read and its own default; a whole-corpus match gave every one of those three knobs all
        # three read sites and both defaults. That is wrong provenance in the column the index
        # exists to make trustworthy -- and it is the same shape as the collision that made
        # NETT_AUX_BATCH's genuine multi-default hazard indistinguishable from an artefact.
        # The cross-file fallback stays for the subclass-binds/parent-reads case above.
        own = [r for r in reads if r[0] == brel]
        # ⛔ ONE SITE PER (FILE, DEFAULT), NOT ONE PER FILE. Collapsing to the first read in a
        # file makes whichever read comes FIRST the reported default -- and a PRESENCE PROBE
        # (`os.environ.get(NAME) is not None`, used to refuse a knob the running configuration
        # cannot read) has no default at all, so a probe written above the real read turned a
        # knob with a documented default into "*(required / no literal default)*". Measured on
        # NETT_AUX_SLOTFG_DEC_HIDDEN, whose default is 256. Keying on the default keeps the
        # probe AND the real read, and the renderer already drops the None, so the row reads
        # 256 and its provenance column names both lines -- which is what the file does.
        # ⚠ It also stops hiding the genuine two-defaults-in-one-file hazard the header warns
        # about: that case used to collapse to the first read as well.
        seen: set[tuple[str, str | None]] = set()
        for rrel, rline, callee, default in (own or reads):
            if (rrel, default) in seen:
                continue
            seen.add((rrel, default))
            site = (rrel, rline,
                    _resolve_default(callee, default, (brel, bline), (rrel, rline)))
            if site not in found.setdefault(cname, []):
                found[cname].append(site)
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
        "⚠ Defaults are the second argument at the read site, with an identifier resolved to its",
        "value — per CALLING class where a shared base reads the knob, because four `*_BATCH` knobs",
        "share one read site and do NOT share a default. A default the generator cannot resolve",
        "statically says *caller-supplied* rather than printing the identifier, which would read as",
        "a value. Where a variable is read in more than one place the defaults can differ — every",
        "site is listed rather than collapsed, because a knob with two defaults is a real hazard.",
        "",
        f"{len(found)} variables.",
        "",
        "| variable | default(s) | read at |",
        "|---|---|---|",
    ]
    for name in sorted(found):
        sites = found[name]
        defaults = sorted({d for _, _, d in sites if d is not None}, key=str)
        dcol = ", ".join(
            f"*(caller-supplied: `{d}`)*" if isinstance(d, CallerSupplied) else f"`{d}`"
            for d in defaults) or "*(required / no literal default)*"
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
