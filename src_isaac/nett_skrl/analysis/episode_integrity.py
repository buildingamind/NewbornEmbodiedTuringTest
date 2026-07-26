"""Verify a TEST run recorded exactly the episodes and steps it promised.

The eval runs a fixed TIMESTEP budget, so an under-sized budget silently truncates
the LAST episode of each env instead of failing. That is corrosive: a truncated
episode still contributes to the preference metric, and because the shortfall lands
unevenly across co-hosted brains it fabricates between-brain differences (a brain
with 6 logged rest episodes vs another's 5 scores 3/6 vs 3/5 for identical
behaviour). It is invisible in the summary numbers.

This module recomputes, from the raw per-step CSV, whether every logged episode has
exactly ``steps_per_episode`` rows, and reports the shortfall. Call it after a test
run so incompleteness is LOUD rather than silent.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("nett.analysis.episode_integrity")


@dataclass
class EpisodeIntegrity:
    """Per-run verdict. ``ok`` iff every logged episode is full length."""

    steps_per_episode: int
    total_episodes: int = 0
    complete_episodes: int = 0
    incomplete: dict[tuple[int, int], int] = field(default_factory=dict)
    #: env_id -> number of logged episodes (unequal counts skew per-brain means)
    episodes_per_env: dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.incomplete

    @property
    def missing_steps(self) -> int:
        return sum(self.steps_per_episode - n for n in self.incomplete.values())

    def summary(self) -> str:
        if self.ok:
            return (f"episode integrity OK: {self.complete_episodes} episodes, "
                    f"all exactly {self.steps_per_episode} steps")
        worst = sorted(self.incomplete.items(), key=lambda kv: kv[1])[:5]
        detail = ", ".join(f"env{e}/ep{ep}={n}" for (e, ep), n in worst)
        return (f"episode integrity FAILED: {len(self.incomplete)}/{self.total_episodes} "
                f"episodes are short of {self.steps_per_episode} steps "
                f"({self.missing_steps} rows missing). Worst: {detail}")


def check_test_log(csv_path: str | Path, steps_per_episode: int) -> EpisodeIntegrity:
    """Verify every logged TEST episode in ``csv_path`` is exactly full length."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    with Path(csv_path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("experiment.phase") != "test":
                continue
            try:
                key = (int(row["env_id"]), int(row["episode"]))
            except (KeyError, ValueError):
                continue
            counts[key] += 1

    result = EpisodeIntegrity(steps_per_episode=int(steps_per_episode))
    per_env: dict[int, int] = defaultdict(int)
    for (env_id, episode), n in counts.items():
        result.total_episodes += 1
        per_env[env_id] += 1
        if n == result.steps_per_episode:
            result.complete_episodes += 1
        else:
            result.incomplete[(env_id, episode)] = n
    result.episodes_per_env = dict(sorted(per_env.items()))
    return result


def verify_run(run_dir: str | Path, steps_per_episode: int | None = None):
    """Check every test log under ``run_dir``; log the verdict. Returns the results.

    ``steps_per_episode`` defaults to the value recorded in the run's config.yaml.
    Never raises: this is a reporting guard, not a gate on an expensive finished run.
    """
    root = Path(run_dir)
    if steps_per_episode is None:
        steps_per_episode = _steps_from_config(root)
    if not steps_per_episode:
        logger.warning("episode integrity: steps_per_episode unknown for %s; skipped", root)
        return []
    results = []
    for csv_path in sorted(root.rglob("logs/test_*.csv")):
        res = check_test_log(csv_path, steps_per_episode)
        results.append((csv_path, res))
        (logger.info if res.ok else logger.warning)("%s: %s", csv_path.name, res.summary())
        counts = set(res.episodes_per_env.values())
        if len(counts) > 1:
            logger.warning(
                "%s: UNEQUAL episodes per env %s — per-brain means are not comparable "
                "until this is even", csv_path.name, res.episodes_per_env)
    return results


def _steps_from_config(root: Path) -> int | None:
    cfg = root / "config.yaml"
    if not cfg.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(cfg.read_text()) or {}
    except Exception:  # noqa: BLE001 - reporting guard must not break analysis
        return None
    value = data.get("steps_per_episode")
    return int(value) if value else None
