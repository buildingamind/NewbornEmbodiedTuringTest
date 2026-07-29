#!/usr/bin/env python3
"""Build/verify the test prerequisites that live OUTSIDE the two repositories.

WHY THIS EXISTS. A green suite on one machine and a red suite on the next, from the same
commit, is almost never the code — it is the state a checkout does not carry. Three kinds
of state feed these tests, and only one of them is in git:

  1. THE BUILT ASSETS (chamber.usdc, chick.usdc). These ARE versioned — repo policy is
     that generated USD is source — so a checkout has them. This script only VERIFIES
     them, and deliberately does not rebuild. The reason is not that a rebuild is hard:
     it is that the chamber is a BAKED EXPERIMENTAL CONSTANT. Its emissive intensity is
     baked in, there is no per-run brightness knob, and runs trained against different
     bakes are not comparable — so re-baking it must be a deliberate act with a commit
     behind it, never a side effect of pushing. (``build_chamber.py`` is also a Kit-boot,
     two-step build: the committed asset is the builder's output PLUS
     ``add_floor_friction.py``, and the builder alone does not author ``Floor_Physics``.)
     A missing asset means a broken checkout, and the fix is named rather than guessed at.
  2. THE STIMULUS CLIPS. Experiment data, far too large to vendor in full; repoA ships
     only a small fixture pair. Cannot be built — resolved and reported.
  3. THE COMPILED STIMULI — the decoded/BC7-compressed frame cache under
     ``~/.cache/nett_frame_cache`` (``$NETT_FRAME_CACHE``). This one IS buildable, is
     outside both repos and outside the workspace, and is the only piece a fresh machine
     genuinely lacks. Building it here is not just a speed-up: the training path builds
     it lazily under a cross-process flock, so a COLD cache plus concurrent Kit boots is
     the deadlock ``prebuild_frame_cache`` was written to avoid.

Idempotent and Kit-free: a warm cache is a fast no-op, so the pre-push hook can call it
unconditionally.

    PYTHONPATH=<src_isaac>:<repoA>/isaac_lab/source python scripts/prepare_test_env.py
    ... --check     verify only; never build

Exit codes: 0 ready · 1 a prerequisite is missing and cannot be built here.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

SRC_ISAAC = Path(__file__).resolve().parents[1]

log = logging.getLogger("nett.prepare_test_env")


def _load_e2e_conftest():
    """Import ``tests/e2e/conftest.py`` as a standalone module.

    The e2e conftest already owns the sheet/media/resolution resolution, and duplicating
    any of it here would give the hook and the tests two different opinions about what
    they are running against. Loaded by path rather than by package import because
    ``tests/`` is not a package and its conftest resolves ``_repo_paths`` absolutely.
    """
    sys.path.insert(0, str(SRC_ISAAC / "tests"))
    sys.path.insert(0, str(SRC_ISAAC))
    path = SRC_ISAAC / "tests" / "e2e" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_e2e_conftest", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable in-tree
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- assets
#: repo-relative paths of the generated-but-versioned assets every run loads, and the
#: builder to name when one is absent. Verified, never rebuilt — see the module docstring.
BUILT_ASSETS: tuple[tuple[str, str], ...] = (
    ("isaac_lab/assets/chamber/chamber.usdc",
     "scripts/builds/build_chamber.py + scripts/add_floor_friction.py"),
    ("isaac_lab/assets/chick/chick.usdc", "scripts/builds/build_chick.py"),
)


def check_built_assets(repo_a: Path) -> list[str]:
    """Return one problem string per missing built asset."""
    problems = []
    for rel, builder in BUILT_ASSETS:
        if not (repo_a / rel).exists():
            problems.append(
                f"missing built asset {repo_a / rel} — restore it from git "
                f"(it is versioned), or rebuild with repoA {builder}"
            )
    return problems


# --------------------------------------------------------------------------- stimuli
def _missing_clips(sheet: Path, media_root: Path, clips: set[str]) -> list[str]:
    return sorted(c for c in clips if not (media_root / c).exists())


def prepare_stimuli(cf, check_only: bool) -> list[str]:
    """Verify the stimulus clips resolve, then build their frame cache.

    ★ THE PROBLEM/INFO SPLIT MIRRORS THE e2e TREE'S OWN SKIP GATE, and it has to: a
    prerequisite whose absence makes the tests SKIP must not block a push, or a developer
    with no stimuli at all can never push. A prerequisite whose absence makes them FAIL
    should block, and should block HERE — in a second — rather than after forty minutes of
    GPU. So:

      * sheet absent, or its whole media root absent  -> the tree skips  -> INFO;
      * media root present but INCOMPLETE             -> the tree fails  -> PROBLEM.

    That middle case is the one this file exists for. It is also the one that used to be
    survivable: while unresolvable media was a `logger.warning`, an incomplete root meant
    the tests ran against blank monitors and passed.
    """
    resolution = cf.MINIMAL_SMOKE_CFG["environment"]["input_resolution"]
    frame_format = os.environ.get("NETT_FRAME_FORMAT", "bc7").strip().lower()

    targets = [
        ("minimal", cf.DESIGN_SHEET_MINIMAL, cf.MEDIA_ROOT),
        ("full", cf.DESIGN_SHEET_FULL, cf.MEDIA_ROOT_FULL),
    ]
    problems: list[str] = []
    buildable: list[tuple[str, Path, Path, int]] = []
    for label, sheet, media_root in targets:
        if not sheet.exists():
            log.info("%s: sheet absent (%s) -- the tests using it skip", label, sheet)
            continue
        if not media_root.exists():
            log.info("%s: media root absent (%s) -- the tests using it skip",
                     label, media_root)
            continue
        clips = cf._sheet_clips(sheet)
        missing = _missing_clips(sheet, media_root, clips)
        if missing:
            problems.append(
                f"{label} sheet {sheet} names {len(missing)} clip(s) absent from "
                f"{media_root}: {', '.join(missing[:4])}"
                f"{' ...' if len(missing) > 4 else ''} -- point $NETT_MEDIA_ROOT at the "
                "stimulus library that holds them"
            )
            continue
        log.info("%s: %d clip(s) resolve under %s", label, len(clips), media_root)
        buildable.append((label, sheet, media_root, len(clips)))

    if check_only or not buildable:
        return problems
    try:
        from nett_isaac.video import prebuild_frame_cache
    except ImportError as exc:
        # nett_isaac unimportable means the e2e tree skips too (its gate imports it), so
        # this is the skip case, not a failure. Nothing to build for a tree that will not
        # run.
        log.info("nett_isaac not importable (%s) -- skipping the frame cache build", exc)
        return problems
    for label, sheet, media_root, n_clips in buildable:
        cache = prebuild_frame_cache(str(sheet), str(media_root), resolution,
                                     frame_format=frame_format)
        log.info("%s: frame cache ready (%d clips, res %d, %s) at %s",
                 label, n_clips, resolution, frame_format, cache)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify prerequisites without building anything")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

    cf = _load_e2e_conftest()
    repo_a = cf.PRIVATE_ROOT

    problems = check_built_assets(repo_a)
    problems += prepare_stimuli(cf, a.check)

    for p in problems:
        log.error("%s", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
