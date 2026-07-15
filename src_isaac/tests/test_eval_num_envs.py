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
    64 itself is 8x8 (square), so it is now returned as-is; divisibility is no
    longer required because overflow episodes are discarded."""
    monkeypatch.setenv("NETT_TEST_ENVS", "64")
    n = _compute_eval_num_envs(_task(52, 20))
    assert n == 64, f"expected the largest square-grid count <=64, got {n}"
    assert _is_square_tile_grid(n)


def test_forced_value_unlocks_the_wide_grid(monkeypatch):
    """80 divides 1040 AND is 9x9 -> 13 batches instead of 65."""
    monkeypatch.setenv("NETT_TEST_ENVS", "80")
    assert _compute_eval_num_envs(_task(52, 20)) == 80


def test_forced_value_is_a_ceiling_not_a_target(monkeypatch):
    """Never exceed the request -- it is the caller's VRAM ceiling. 100 is itself
    10x10 (square), so it is taken; 100 does not divide 1040 and that is fine now
    (the 60 overflow episodes are discarded)."""
    monkeypatch.setenv("NETT_TEST_ENVS", "100")
    n = _compute_eval_num_envs(_task(52, 20))
    assert n == 100 and n <= 100


def test_pick_respects_num_brains_and_square_grid(monkeypatch):
    """The two HARD constraints survive the divisibility relaxation. (Dividing the
    total is now only a preference -- see the tile-band test.)"""
    monkeypatch.setenv("NETT_TEST_ENVS", "64")
    n = _compute_eval_num_envs(_task(52, 20, num_brains=4))
    assert n % 4 == 0 and _is_square_tile_grid(n)


def test_falls_back_to_one_rather_than_render_distorted():
    """2 test episodes: 2 tiles 2x1 (non-square), so the only valid pick is 1 (1x1).
    Slow, but undistorted -- never trade the measurement for throughput."""
    assert _compute_eval_num_envs(_task(1, 2, max_parallel_envs=16)) == 1


def test_overflow_is_bounded_by_one_batch():
    """A non-divisor pick leaves at most one partial final batch -- never a whole
    wasted batch, and never fewer episodes than the experiment needs."""
    import math

    for rows, eps in ((52, 20), (13, 4), (5, 5), (3, 9)):
        total = rows * eps
        n = _compute_eval_num_envs(_task(rows, eps))
        overflow = math.ceil(total / n) * n - total
        assert 0 <= overflow < n


# --- divisibility relaxation + overflow discard ---------------------------
# num_envs no longer has to divide the episode total: the eval runs
# ceil(total/num_envs)*num_envs episodes and the surplus (index >= total) is
# muted (nett_env_cfg.test_total_episodes -> LogChannel.new_episode(logged=False)),
# so every design row still contributes exactly episodes_test episodes.


def _simulate_row_counts(n_rows, episodes_test, num_envs, discard=True):
    """Mirror of the runtime schedule: env i runs strided episode indices
    i, i+N, i+2N, ...; conditions.current() maps index -> index % n_rows."""
    import math
    from collections import Counter

    total = n_rows * episodes_test
    per_env = math.ceil(total / num_envs)
    counts = Counter()
    for env in range(num_envs):
        for k in range(per_env):
            ep = env + k * num_envs
            if discard and ep >= total:
                continue
            counts[ep % n_rows] += 1
    return counts


@pytest.mark.parametrize("num_envs", [16, 64, 80, 100, 196])
def test_every_condition_runs_exactly_episodes_test_times(num_envs):
    """THE invariant: no skips, no over-sampling, at ANY num_envs."""
    counts = _simulate_row_counts(52, 20, num_envs, discard=True)
    assert len(counts) == 52, f"only {len(counts)}/52 design rows covered"
    assert set(counts.values()) == {20}, f"unbalanced: {sorted(set(counts.values()))}"


@pytest.mark.parametrize("num_envs", [64, 100, 196])
def test_without_the_discard_non_divisors_are_imbalanced(num_envs):
    """Why the discard exists: the raw strided schedule over-samples the first
    rows whenever num_envs does not divide the total."""
    counts = _simulate_row_counts(52, 20, num_envs, discard=False)
    assert set(counts.values()) != {20}


def test_non_divisor_is_now_allowed_and_maximizes_parallelism(monkeypatch):
    """64 (8x8) does not divide 1040, but is square and now permitted:
    65 sequential batches -> 17."""
    monkeypatch.setenv("NETT_TEST_ENVS", "64")
    n = _compute_eval_num_envs(_task(52, 20))
    assert n == 64 and _is_square_tile_grid(n)


def test_prefers_a_clean_divisor_within_the_same_tile_band(monkeypatch):
    """80 and 81 both tile 9x9 (identical parallelism/VRAM), but 80 divides 1040 --
    take it and avoid overflow for free."""
    monkeypatch.setenv("NETT_TEST_ENVS", "81")
    assert _compute_eval_num_envs(_task(52, 20)) == 80


def test_still_never_returns_a_non_square_grid(monkeypatch):
    """Relaxing divisibility must NOT relax the square-grid (#488) constraint."""
    for want in ("52", "72", "104"):
        monkeypatch.setenv("NETT_TEST_ENVS", want)
        assert _is_square_tile_grid(_compute_eval_num_envs(_task(52, 20)))
