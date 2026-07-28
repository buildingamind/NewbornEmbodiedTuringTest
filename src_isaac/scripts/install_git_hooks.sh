#!/usr/bin/env bash
# Point git at the version-controlled hooks in src_isaac/scripts/hooks/.
#
# Hooks live in .git/hooks, which is NOT version controlled, so a committed hook does
# nothing until git is told where to look. core.hooksPath does that in one setting and
# keeps the hooks reviewable in the repo.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
git -C "$ROOT" config core.hooksPath src_isaac/scripts/hooks
echo "core.hooksPath -> src_isaac/scripts/hooks"
echo "installed: $(ls "$ROOT/src_isaac/scripts/hooks")"
echo "bypass once with: NETT_SKIP_HOOKS=1 git push   (or git push --no-verify)"
