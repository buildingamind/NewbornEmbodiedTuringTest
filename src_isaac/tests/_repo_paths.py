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
WORKSPACE = SRC_ISAAC.parents[1]

#: Used when ``NETT_REPO_A`` is unset.
SIBLING_DEFAULT = WORKSPACE / "NewbornEmbodiedTuringTest_Private"

#: Env var overriding the sibling default. Point it at a repoA *repository root*.
REPO_A_ENV_VAR = "NETT_REPO_A"


def repo_a_root() -> Path:
    """Return the repoA repository root — the dir containing ``isaac_lab/``.

    Never raises and never checks existence: callers that must tolerate a
    missing repoA (e.g. building module-level constants in a conftest that
    auto-skips later) can use the path unconditionally.
    """
    override = os.environ.get(REPO_A_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return SIBLING_DEFAULT


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
