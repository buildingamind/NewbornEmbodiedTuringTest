"""Guard: a truncated TEST episode must be detected, not silently averaged in."""
from __future__ import annotations

import csv

from nett_skrl.analysis.episode_integrity import check_test_log

_HEADER = ["env_id", "episode", "step", "agent.x", "experiment.phase"]


def _write(path, episodes):
    """episodes: {(env, ep): n_steps}"""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_HEADER)
        for (env, ep), n in episodes.items():
            for s in range(n):
                w.writerow([env, ep, s, 0.0, "test"])


def test_all_complete_is_ok(tmp_path):
    p = tmp_path / "test_x.csv"
    _write(p, {(0, 0): 50, (0, 1): 50, (1, 0): 50})
    res = check_test_log(p, 50)
    assert res.ok
    assert res.total_episodes == 3 and res.complete_episodes == 3
    assert res.missing_steps == 0
    assert "OK" in res.summary()


def test_truncated_episode_is_flagged(tmp_path):
    p = tmp_path / "test_x.csv"
    _write(p, {(0, 0): 50, (0, 1): 47, (1, 0): 49})
    res = check_test_log(p, 50)
    assert not res.ok
    assert res.incomplete == {(0, 1): 47, (1, 0): 49}
    assert res.missing_steps == 3 + 1
    assert "FAILED" in res.summary()


def test_unequal_episodes_per_env_is_visible(tmp_path):
    # The real defect that biased per-brain means: one env logs an extra episode.
    p = tmp_path / "test_x.csv"
    _write(p, {(0, 0): 50, (0, 1): 50, (0, 2): 50, (1, 0): 50, (1, 1): 50})
    res = check_test_log(p, 50)
    assert res.ok, "all episodes are full length"
    assert res.episodes_per_env == {0: 3, 1: 2}, "unequal counts must be reported"


def test_non_test_rows_are_ignored(tmp_path):
    p = tmp_path / "test_x.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_HEADER)
        for s in range(10):
            w.writerow([0, 0, s, 0.0, "train"])
        for s in range(50):
            w.writerow([0, 1, s, 0.0, "test"])
    res = check_test_log(p, 50)
    assert res.ok and res.total_episodes == 1
