"""Locate the sibling repoA checkout for the cross-repo source-text tests.

Several tests here assert on the TEXT of repoA (``nett_isaac``) source files —
the in-Isaac env wiring lives in the other repo, so the only way this suite can
check that the two halves agree is to read it.

⚠ THESE TESTS VALIDATE WHATEVER CHECKOUT THIS MODULE RESOLVES, NOT NECESSARILY
THE PACKAGE UNDER TEST. Resolution is, in order:

1. ``$NETT_REPO_A`` — the repoA repository root (the directory containing
   ``isaac_lab/``).
2. the sibling default ``<workspace>/NewbornEmbodiedTuringTest_Private``.

Neither consults ``PYTHONPATH`` or the imported ``nett_isaac`` package, so if
``PYTHONPATH`` points at a *different* repoA worktree than the one resolved
here, the source-text assertions will silently describe the wrong checkout while
the behavioural tests exercise the right one. Deliberate: inferring the path
from the import would make a green run depend on which package happened to load
first, which is harder to reason about than an explicit env var. Run the suite
from the PRIMARY checkouts, or set ``NETT_REPO_A`` to match ``PYTHONPATH``.

When repoA cannot be found at all the affected tests SKIP with the attempted
path. They used to fail with "expected source file missing", which turned a
checkout-location artifact (running from a worktree) into 15 red tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: ``.../NewbornEmbodiedTuringTest/src_isaac``
SRC_ISAAC = Path(__file__).resolve().parents[1]

#: The directory holding both repositories.
#:
#: ⚠ A FIXED TWO-LEVELS-UP GUESS, AND IT IS WRONG FROM A GIT WORKTREE. Right for the
#: primary checkout (``<workspace>/NewbornEmbodiedTuringTest/src_isaac``); from a
#: worktree (``<workspace>/wt-<name>/NewbornEmbodiedTuringTest/src_isaac``) it lands on
#: the worktree parent, one level short. Prefer :func:`stimulus_library_candidates` for
#: anything that must resolve the shared, un-checked-out ``videos/`` tree; this constant
#: is kept because a skip message should still name the path this suite has always
#: documented.
WORKSPACE = SRC_ISAAC.parents[1]

#: Used when ``NETT_REPO_A`` is unset, and named for the message when nothing is found.
SIBLING_DEFAULT = WORKSPACE / "NewbornEmbodiedTuringTest_Private"

#: repoA is a SIBLING, so finding it means leaving this repository — the same boundary
#: crossing that made ``WORKSPACE`` wrong from a worktree. It survives today only by
#: luck of layout: ``wt-<name>/`` happens to hold worktrees of BOTH repos side by side,
#: so the short guess lands. Check out a worktree of repo B alone and it points at a
#: path that does not exist, and the cross-repo source-text tests all skip — silently,
#: which is how this class of bug keeps costing a debug cycle. Walk up instead.
_REPO_A_DIRNAME = "NewbornEmbodiedTuringTest_Private"

#: Env var overriding the sibling default. Point it at a repoA *repository root*.
REPO_A_ENV_VAR = "NETT_REPO_A"


def repo_a_root() -> Path:
    """Return the repoA repository root — the dir containing ``isaac_lab/``.

    Never raises: callers that must tolerate a missing repoA (e.g. building
    module-level constants in a conftest that auto-skips later) can use the path
    unconditionally, and get :data:`SIBLING_DEFAULT` to name in the message when
    nothing is on disk.

    ``$NETT_REPO_A`` still wins outright and is never probed — an explicit override
    that silently resolved somewhere else would be worse than a broken one. Below it,
    the nearest sibling that actually CONTAINS ``isaac_lab/`` wins, so a worktree at any
    depth resolves without hardcoding one. Note this does NOT infer the path from the
    imported ``nett_isaac`` package: that would make a green run depend on which package
    happened to load first, which the module docstring rejects for good reason.
    """
    override = os.environ.get(REPO_A_ENV_VAR)
    if override:
        return Path(override).expanduser()
    for parent in SRC_ISAAC.parents:
        candidate = parent / _REPO_A_DIRNAME
        if (candidate / "isaac_lab").is_dir():
            return candidate
    return SIBLING_DEFAULT


def stimulus_library_candidates() -> tuple[Path, ...]:
    """Every ``videos/binding`` at or above this checkout, nearest first.

    THE ONE PLACE THAT KNOWS HOW TO FIND THE STIMULUS LIBRARY. The library is a large,
    un-checked-out tree that lives at the WORKSPACE root, so it is shared by the primary
    checkout and every worktree alike — but :data:`WORKSPACE` cannot find it from a
    worktree (see its note). Walking up locates it from either layout without hardcoding
    a depth, and returns *candidates* rather than one path so a caller can pick the
    nearest that actually holds what it needs.

    ⚠ MEASURED 2026-08-09, both consequences of the fixed guess: the e2e media root fell
    back to repoA's two vendored fixture clips and a run reached the TEST phase before
    dying on ``O1_1Ca_1.mov`` (the clip the sheet's ``1color`` row names and repoA does
    not ship); and ``test_design.py``'s two binding-sheet tests silently SKIPPED on every
    worktree run. Same root cause, two sites — which is why this now lives here instead
    of in one caller.
    """
    seen: list[Path] = []
    for start in (SRC_ISAAC, repo_a_root()):
        for parent in [start, *start.parents]:
            candidate = parent / "videos" / "binding"
            if candidate.is_dir() and candidate not in seen:
                seen.append(candidate)
    return tuple(seen)


def stimulus_library() -> Path:
    """The nearest stimulus library, or the historical fixed guess when none is on disk.

    The fallback keeps a skip/error message naming the path this suite has always
    documented, rather than an empty string.
    """
    candidates = stimulus_library_candidates()
    return candidates[0] if candidates else WORKSPACE / "videos" / "binding"


def nett_isaac_dir() -> Path:
    """Return the ``nett_isaac`` package source directory inside repoA."""
    return repo_a_root() / "isaac_lab" / "source" / "nett_isaac"


def read_source(path: Path) -> str:
    """Read a source file for a text assertion, tolerating an absent repoA.

    A missing file when the repoA checkout *does* exist is still a hard failure —
    that means the wiring moved and the assertion is stale, which is a real
    signal. Only a wholly absent checkout is skipped, since that is a
    where-am-I-running artifact rather than a defect.
    """
    if not path.exists():
        root = repo_a_root()
        if not root.is_dir():
            pytest.skip(
                f"repoA checkout not found at {root} — set ${REPO_A_ENV_VAR} to "
                "the repository root (the directory containing isaac_lab/), or "
                "run from the primary checkouts"
            )
        raise AssertionError(f"expected source file missing: {path}")
    return path.read_text()
