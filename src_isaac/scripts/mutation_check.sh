#!/usr/bin/env bash
# Run a suite against a deliberately broken copy of the tree, WITHOUT writing the live one.
#
# ⛔ WHY A COPY AND NOT `sed -i` + restore. Mutation checks in this campaign were done in
# place: back up the file, break it, run pytest, restore. That protects the RESULT -- and
# nothing else. `nett_skrl/runtime/task_runner.py:51` spawns mode subprocesses that
# RE-IMPORT, and the pre-push hook imports the whole tree, so for the seconds a mutant is
# live any process that starts is running deliberately broken code. On a node with live
# arms that is a self-inflicted outage, and the window is invisible afterwards: the file
# is restored, the mtime is recent, and nothing records that it was ever wrong.
#
# ⇒ A peer put it better than I did: "verify the mutant applied" protects the RESULT; it
# does nothing for the SYSTEM you mutated. The aimability has to exist BEFORE the first
# power measurement, not be added after a near miss.
#
# Usage:  scripts/mutation_check.sh <file-rel-to-src_isaac> <python-repr-old> <new> <pytest-target>...
#
# The anchor is asserted present before anything is written -- an unapplied mutant produces
# the same all-green as one the suite survived, which is the failure this whole technique
# exists to detect.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REL="$1"; OLD="$2"; NEW="$3"; shift 3
WORK="${NETT_MUTATION_DIR:-${TMPDIR:-/tmp}}/nett_mutation_$$"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK"
cp -r "$SRC" "$WORK/src_isaac"
python3 - "$WORK/src_isaac/$REL" "$OLD" "$NEW" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if old not in s:
    sys.exit(f"⛔ ANCHOR NOT FOUND in {path}. The mutant was NOT applied -- DO NOT READ "
             f"THE RESULT of any run that follows. An unapplied mutant is indistinguishable "
             f"from one the suite survived.\n  looked for: {old[:120]!r}")
open(path, "w").write(s.replace(old, new, 1))
print(f"mutant applied to the COPY at {path} (live tree untouched)")
PY
cd "$WORK/src_isaac"
PYTHONPATH=.:"${NETT_PRIVATE_SOURCE:-/home/zlaborde/code/isaac/NewbornEmbodiedTuringTest_Private/isaac_lab/source}" \
  "${NETT_PYTHON:-/home/zlaborde/code/.venv/nett-isaac/bin/python}" -m pytest "$@" -q -p no:cacheprovider || true
