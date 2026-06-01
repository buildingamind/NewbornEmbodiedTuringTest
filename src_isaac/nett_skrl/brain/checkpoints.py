"""Checkpoint discovery and loading for skrl brain agents."""

from __future__ import annotations

import logging
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


def load_latest_checkpoints(agents: list, config, *, context: str = "test") -> None:
    """Load each brain's most recent skrl checkpoint into ``agents[i]``."""
    for i, agent in enumerate(agents):
        ckpt_dir = Path(config.path) / "wandb_runs" / f"brain_{i + 1}" / "checkpoints"
        chosen = pick_checkpoint(ckpt_dir)
        if chosen is None:
            logger.info("%s: no checkpoint found under %s, using fresh weights", context, ckpt_dir)
            continue
        try:
            agent.load(str(chosen))
            logger.info("%s: brain_%d loaded %s", context, i + 1, chosen.name)
        except Exception:
            logger.exception("%s: failed to load %s for brain_%d", context, chosen, i + 1)
