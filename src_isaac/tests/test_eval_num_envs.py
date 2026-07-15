"""Test-phase num_envs selection.

The load-bearing property is the SQUARE TILE GRID: Isaac Lab tiles N cameras into
ceil(sqrt(N)) x ceil(N/cols), and at a non-square grid the fisheye eye render is
distorted (Isaac Sim #488). A bad pick does not crash -- it silently corrupts every
test frame -- so it is pinned here.
"""

from __future__ import annotations

import types

import pytest

from nett_skrl.runtime.task_runner import _compute_eval_num_envs, _is_square_tile_grid


def _task(total_test_rows: int, episodes_test: int, num_brains: int = 1,
          max_parallel_envs: int | None = 16, condition: str = "Object1"):
    logger = types.SimpleNamespace(warning=lambda *a, **k: None)
    config = types.SimpleNamespace(
        num_brains=num_brains,
        episodes={"test": episodes_test},
        condition=condition,
        max_parallel_envs=max_parallel_envs,
        logger=logger,
    )
    env = types.SimpleNamespace(iterations_per_test_episode={condition: total_test_rows})
    return types.SimpleNamespace(config=config, agent=types.SimpleNamespace(env=env))


# --- the grid predicate ---------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 4, 7, 8, 9, 13, 16, 25, 80])
def test_square_grids(n):
    """Perfect squares, plus the ones whose ceil-grid still lands square:
    3->2x2, 7->3x3, 8->3x3, 13->4x4, 80->9x9.

    NOTE known_issues.md lists 3 among the values to avoid, but by the mechanism it
    documents (grid = ceil(sqrt(N)) x ceil(N/cols)) 3 tiles 2x2 -- square -- for the
    same reason 8 tiles 3x3 and is called out there as empirically fine. The grid
    shape is the thing that matters; treat that list as informal."""
    assert _is_square_tile_grid(n)


@pytest.mark.parametrize("n", [2, 5, 6, 12, 20, 26, 40, 52, 104])
def test_non_square_grids(n):
    """52 -> 8x7 is the one that bit us: NETT_TEST_ENVS=64 used to resolve to it."""
    assert not _is_square_tile_grid(n)


# --- selection ------------------------------------------------------------


def test_default_cap_is_max_parallel_envs():
    # 52 test rows x 20 episodes = 1040; 16 divides it and is 4x4.
    assert _compute_eval_num_envs(_task(52, 20, max_parallel_envs=16)) == 16


def test_forced_value_never_returns_a_non_square_grid(monkeypatch):
    """REGRESSION: NETT_TEST_ENVS=64 used to return 52 (8x7) -> distorted fisheye.
    It must now fall back to the largest VALID count instead."""
    monkeypatch.setenv("NETT_TEST_ENVS", "64")
    n = _compute_eval_num_envs(_task(52, 20))
    assert n == 16, f"expected the largest square-grid divisor <=64, got {n}"
    assert _is_square_tile_grid(n)


def test_forced_value_unlocks_the_wide_grid(monkeypatch):
    """80 divides 1040 AND is 9x9 -> 13 batches instead of 65."""
    monkeypatch.setenv("NETT_TEST_ENVS", "80")
    assert _compute_eval_num_envs(_task(52, 20)) == 80


def test_forced_value_is_a_ceiling_not_a_target(monkeypatch):
    """Never exceed the request -- it is the caller's VRAM ceiling."""
    monkeypatch.setenv("NETT_TEST_ENVS", "100")
    n = _compute_eval_num_envs(_task(52, 20))
    assert n == 80 and n <= 100


def test_pick_always_divides_total_and_respects_num_brains(monkeypatch):
    monkeypatch.setenv("NETT_TEST_ENVS", "64")
    n = _compute_eval_num_envs(_task(52, 20, num_brains=4))
    total = 52 * 20
    assert total % n == 0 and n % 4 == 0 and _is_square_tile_grid(n)


def test_falls_back_to_one_rather_than_render_distorted():
    """2 test episodes: 2 tiles 2x1 (non-square), so the only valid pick is 1 (1x1).
    Slow, but undistorted -- never trade the measurement for throughput."""
    assert _compute_eval_num_envs(_task(1, 2, max_parallel_envs=16)) == 1


def test_never_leaves_empty_episode_slots():
    """Whatever we pick must divide the total, or NETTEnv gets empty slots."""
    for rows, eps in ((52, 20), (13, 4), (5, 5), (3, 9)):
        n = _compute_eval_num_envs(_task(rows, eps))
        assert (rows * eps) % n == 0
