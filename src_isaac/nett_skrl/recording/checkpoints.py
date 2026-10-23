"""Checkpoint discovery and loading for skrl brain agents."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("nett.brain")


def pick_checkpoint(ckpt_dir: Path) -> Path | None:
    """Return the preferred skrl checkpoint in ``ckpt_dir`` for eval."""
    if not ckpt_dir.exists():
        return None

    final = ckpt_dir / "final_agent.pt"
    if final.exists():
        return final

    numbered = [
        (int(path.stem.removeprefix("agent_")), path)
        for path in ckpt_dir.glob("agent_*.pt")
        if path.stem.removeprefix("agent_").isdigit()
    ]
    if numbered:
        return max(numbered, key=lambda item: item[0])[1]

    best = ckpt_dir / "best_agent.pt"
    return best if best.exists() else None


_TEST_CHECKPOINT_NAME = re.compile(r"(final_agent|best_agent|agent_[0-9]+)\.pt")


def resolve_test_checkpoint() -> str | None:
    """``NETT_TEST_CHECKPOINT``: the checkpoint FILE NAME every brain's TEST loads, e.g. ``agent_3800.pt``.

    Opt-in (owner 2026-10-16: "add checkpoints at 1/2 and 3/4 into training ... test the checkpoint
    models"). Unset = today's choice (``pick_checkpoint``: final, else newest numbered, else best).
    TEST only -- a train resume never reads it. A bare file name from skrl's own set; anything else
    (a path, another suffix) raises, so a typo cannot fall back to the final model.
    """
    v = os.environ.get("NETT_TEST_CHECKPOINT", "").strip()
    if not v:
        return None
    if not _TEST_CHECKPOINT_NAME.fullmatch(v):
        raise ValueError(f"NETT_TEST_CHECKPOINT={v!r}: want final_agent.pt, best_agent.pt or agent_<timestep>.pt")
    return v


class MissingCheckpointError(RuntimeError):
    """TEST found a brain with no loadable checkpoint (DECISIONS §90.3)."""


def brain_checkpoint_dir(path, brain_id: int) -> Path:
    """``{path}/wandb_runs/brain_{brain_id}/checkpoints`` -- ``brain_id`` is 1-based and LOCAL."""
    return Path(path) / "wandb_runs" / f"brain_{brain_id}" / "checkpoints"


def brains_missing_final_checkpoint(path, num_brains: int, *, since: float | None = None) -> list[int]:
    """1-based brain ids with no ``final_agent.pt``, or one older than ``since`` (epoch s).

    ``final_agent.pt`` is written once, at the very end of a train subprocess
    (``RunRecorder.save_final_checkpoints``), so a fresh one is the evidence that THIS
    train subprocess finished. ``since`` excludes a file left by an earlier chunk or
    seeded by a resume.
    """
    missing = []
    for brain_id in range(1, int(num_brains) + 1):
        final = brain_checkpoint_dir(path, brain_id) / "final_agent.pt"
        if not final.exists() or (since is not None and final.stat().st_mtime < since):
            missing.append(brain_id)
    return missing


def load_latest_checkpoints(agents: list, config, *, context: str = "test") -> None:
    """Load each brain's most recent skrl checkpoint into ``agents[i]``.

    ⛔ TEST REFUSES A BRAIN WITH NOTHING TO LOAD (DECISIONS §90.3, FINDINGS §4dh.44h.143).
    This used to log "using fresh weights" at INFO and carry on, and ten arms across three
    nodes were tested -- and published -- as untrained policies, with every completion
    signal passing. A missing checkpoint or a failed load now raises
    :class:`MissingCheckpointError` after naming every affected brain. ``train`` keeps
    the fallback (a first train has nothing to resume), and so does a dry-run probe
    (it measures memory, not a policy).
    """
    refuse = context == "test" and not bool(getattr(config, "dry_run", False))
    override = resolve_test_checkpoint() if context == "test" else None
    if override:
        logger.warning("%s: NETT_TEST_CHECKPOINT=%s -- every brain loads that file, not its final model",
                       context, override)
    problems = []
    for i, agent in enumerate(agents):
        ckpt_dir = brain_checkpoint_dir(config.path, i + 1)
        if override:
            chosen = ckpt_dir / override
            if not chosen.is_file():
                # A named checkpoint that is absent is ALWAYS a refusal, dry run or not: falling back
                # to another file would test a different policy under this one's name.
                logger.error("%s: brain_%d has no %s under %s", context, i + 1, override, ckpt_dir)
                problems.append(f"brain_{i + 1}: NETT_TEST_CHECKPOINT {override} absent under {ckpt_dir}")
                continue
        else:
            chosen = pick_checkpoint(ckpt_dir)
        if chosen is None:
            if refuse:
                logger.error("%s: brain_%d has no checkpoint under %s", context, i + 1, ckpt_dir)
                problems.append(f"brain_{i + 1}: no checkpoint under {ckpt_dir}")
            else:
                logger.info("%s: no checkpoint found under %s, using fresh weights", context, ckpt_dir)
            continue
        try:
            agent.load(str(chosen))
            logger.info("%s: brain_%d loaded %s", context, i + 1, chosen.name)
        except Exception as exc:
            logger.exception("%s: failed to load %s for brain_%d", context, chosen, i + 1)
            from nett_skrl.brain.models.utils.action_std import FixedActionStdCheckpointError
            if isinstance(exc, FixedActionStdCheckpointError):
                # NETT_FIXED_ACTION_STD=1 refused this checkpoint's log_std: raise in EVERY context. A train resume
                # (CONTINUE_FROM) would otherwise carry on from fresh weights at rc 0 (DECISIONS §187 add. 9).
                raise
            if refuse:
                problems.append(f"brain_{i + 1}: load of {chosen} failed ({exc!r})")
    if problems:
        raise MissingCheckpointError(
            f"{context} refuses to run {len(problems)}/{len(agents)} brain(s) on fresh "
            f"weights: " + "; ".join(problems)
        )
