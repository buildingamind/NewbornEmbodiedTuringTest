"""Parallel environment planning helpers.

The single source of truth for which ``num_envs`` values are legal, shared by the
TRAIN planner (``nett.NETT._estimate_envs_via_linear_model``) and the TEST selector
(``task_runner._compute_eval_num_envs``). Both must agree: a count that distorts the
render distorts it in either phase.
"""

from __future__ import annotations

import math

AUTO = "auto"


def is_square_tile_grid(num_envs: int) -> bool:
    """Does ``num_envs`` tile into a SQUARE camera grid?

    Isaac Lab tiles N cameras into ``ceil(sqrt(N)) x ceil(N / cols)``. The chick eye is
    a fisheye, and at a NON-square tile grid the lens distortion is applied over the
    full non-square canvas aspect, so the rendered eye image is distorted -- Isaac Sim
    #488. See isaac_lab/docs/known_issues.md.

    True exactly for ``N in [k^2-k+1, k^2]``: perfect squares qualify, and so does any
    count whose natural grid still comes out square (8 -> 3x3, 13 -> 4x4, 80 -> 9x9).
    52 -> 8x7 does NOT. Empty tiles are harmless (measured); only squareness matters,
    so you CANNOT pad a non-square count into a square grid -- you must pick a count
    whose natural grid is already square.
    """
    if num_envs < 1:
        return False
    cols = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / cols)
    return cols == rows


def is_valid_num_envs(num_envs: int, num_brains: int) -> bool:
    """The two hard constraints, together.

    * multiple of ``num_brains`` -- else BrainTrainer raises (each brain owns one
      contiguous env scope; see ``brain.trainer.brain_scope_sizes``);
    * square tile grid -- else the fisheye render is distorted (#488).
    """
    num_brains = max(1, int(num_brains))
    return num_envs >= 1 and num_envs % num_brains == 0 and is_square_tile_grid(num_envs)


def largest_valid_num_envs(want: int, num_brains: int) -> int | None:
    """Largest valid count at or below ``want``, or None if there is none."""
    num_brains = max(1, int(num_brains))
    for n in range(int(want), 0, -1):
        if is_valid_num_envs(n, num_brains):
            return n
    return None


def next_valid_num_envs(num_envs: int, num_brains: int) -> int:
    """Smallest valid count STRICTLY greater than ``num_envs``.

    Needed because valid counts are sparse at the bottom: from 1, doubling lands on 2,
    which tiles 2x1 and is invalid, so snapping 2 back down returns 1 again. A growth
    loop written as ``snap(2*n)`` therefore sticks at 1 forever. The next valid count
    above 1 is 3 (2x2).
    """
    num_brains = max(1, int(num_brains))
    n = int(num_envs) + 1
    while not is_valid_num_envs(n, num_brains):
        n += 1
    return n


def grow_num_envs(num_envs: int, num_brains: int) -> int:
    """The next count a doubling search should try after ``num_envs``.

    Aims for 2x, snapped DOWN to a valid count; if that would not advance (the sparse
    low end), takes the next valid count instead. Always strictly greater.
    """
    doubled = largest_valid_num_envs(2 * int(num_envs), num_brains)
    if doubled is not None and doubled > int(num_envs):
        return doubled
    return next_valid_num_envs(num_envs, num_brains)


def smallest_valid_num_envs(num_brains: int) -> int:
    """Smallest valid count. Always exists (``num_brains`` multiples grow without
    bound and every ``k^2`` is square), so this is the floor a caller must accept
    when nothing at or below its request qualifies."""
    num_brains = max(1, int(num_brains))
    n = num_brains
    while not is_square_tile_grid(n):
        n += num_brains
    return n


def snap_num_envs(want: int, num_brains: int) -> tuple[int, bool]:
    """Resolve ``want`` to the nearest valid count, preferring DOWN.

    Returns ``(num_envs, went_up)``. ``went_up=True`` means nothing at or below the
    request was legal and we had to exceed it -- the caller should warn, because the
    request is usually a VRAM ceiling and exceeding it risks OOM. We do it anyway:
    the alternative is a distorted render or a crash in BrainTrainer, and silently
    corrupt data is worse than a loud OOM.
    """
    best = largest_valid_num_envs(want, num_brains)
    if best is not None:
        return best, False
    return smallest_valid_num_envs(num_brains), True


def select_test_num_envs(
    want: int, num_brains: int, total_test_episodes: int
) -> tuple[int, bool]:
    """Resolve the TEST-phase num_envs from a ceiling. Returns ``(num_envs, went_up)``.

    Pure, so the orchestrator can record the value it will get without building a Task,
    and the worker can compute the same answer independently. Both MUST agree -- the
    recorded value is the reproducibility contract.

    Divisibility of the episode total is NOT required (overflow episodes run unlogged;
    see ``LogChannel.new_episode(logged=False)``), but a divisor is preferred WITHIN
    the chosen tile-grid band ``[k^2-k+1, k^2]``: every count in that band tiles k x k,
    so it costs nothing to avoid the overflow entirely when we can.
    """
    num_brains = max(1, int(num_brains))
    best, went_up = snap_num_envs(want, num_brains)
    if went_up:
        return best, True
    k = math.ceil(math.sqrt(best))
    band_lo = k * k - k + 1
    clean = next(
        (
            n
            for n in range(best, band_lo - 1, -1)
            if is_valid_num_envs(n, num_brains) and total_test_episodes % n == 0
        ),
        None,
    )
    return (clean or best), False


def capped_num_envs(
    *,
    num_brains: int,
    preferred_envs_per_brain: int,
    max_parallel_envs: int | str | None,
) -> int:
    """Total env rows the recipe asks for, clamped to an explicit integer cap.

    ``max_parallel_envs="auto"`` is NOT resolved here -- it means "let measured VRAM
    decide", which needs a dry run, so the caller (NETT.single_run) handles it and
    passes None through to get the recipe's preferred count as the probe baseline.
    """
    num_brains = max(1, int(num_brains))
    preferred = num_brains * max(1, int(preferred_envs_per_brain))
    if max_parallel_envs is None or max_parallel_envs == AUTO:
        return preferred
    cap = int(max_parallel_envs)
    if cap < num_brains:
        raise ValueError(
            f"max_parallel_envs ({cap}) must be >= num_brains ({num_brains})."
        )
    return max(num_brains, (min(preferred, cap) // num_brains) * num_brains)


def num_env_candidates(num_envs: int, num_brains: int) -> list[int]:
    """Descending VALID counts at or below ``num_envs``, for the fallback dry-run scan.

    Only valid counts are offered: the scan stops at the first candidate whose dry run
    succeeds, so an invalid one here would be silently ADOPTED -- fitting in VRAM says
    nothing about whether the render is distorted. E.g. ``(9, 3)`` yields ``[9, 3]``,
    not ``[9, 6, 3]``: 6 tiles 3x2 and would distort the fisheye.
    """
    num_envs = int(num_envs)
    num_brains = max(1, int(num_brains))
    return [
        n
        for n in range(num_envs, num_brains - 1, -num_brains)
        if is_valid_num_envs(n, num_brains)
    ]
