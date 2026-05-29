"""Parallel environment planning helpers."""

from __future__ import annotations


def capped_num_envs(
    *,
    num_brains: int,
    preferred_envs_per_brain: int,
    max_parallel_envs: int | None,
) -> int:
    num_brains = max(1, int(num_brains))
    preferred = num_brains * max(1, int(preferred_envs_per_brain))
    if max_parallel_envs is None:
        return preferred
    cap = int(max_parallel_envs)
    if cap < num_brains:
        raise ValueError(
            f"max_parallel_envs ({cap}) must be >= num_brains ({num_brains})."
        )
    return max(num_brains, (min(preferred, cap) // num_brains) * num_brains)


def num_env_candidates(num_envs: int, num_brains: int) -> list[int]:
    num_envs = int(num_envs)
    num_brains = max(1, int(num_brains))
    return list(range(num_envs, num_brains - 1, -num_brains))
