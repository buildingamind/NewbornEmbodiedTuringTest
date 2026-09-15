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

# `os.environ.get("X", "default")`, `os.environ["X"]`, `os.getenv("X", "default")`
PAT = re.compile(
    r"""(?:environ\.get|getenv)\(\s*["'](?P<name>NETT_[A-Z0-9_]+)["']\s*(?:,\s*(?P<default>[^)]*?))?\s*\)"""
    r"""|environ\[\s*["'](?P<name2>NETT_[A-Z0-9_]+)["']\s*\]""",
    re.X,
)


def scan() -> dict[str, list[tuple[str, int, str | None]]]:
    found: dict[str, list[tuple[str, int, str | None]]] = {}
    for top in SCAN:
        for path in sorted((ROOT / top).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            for i, line in enumerate(path.read_text().splitlines(), 1):
                for m in PAT.finditer(line):
                    name = m.group("name") or m.group("name2")
                    default = (m.group("default") or "").strip() or None
                    found.setdefault(name, []).append((rel, i, default))
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
