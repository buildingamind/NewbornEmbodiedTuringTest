"""Test-preference aggregation is per BRAIN (model), not per env.

A brain is one model instance; several envs may belong to it, and how many is a
THROUGHPUT knob (_compute_eval_num_envs). So the reported statistic must not move
when that knob moves. It used to: rows were bucketed by env_id, so n_brains came
out as num_envs and correct_pct was a mean-of-ratios over an arbitrary partition.
"""

from __future__ import annotations

import csv

from nett_skrl.analysis.api import _test_preference_rows
from nett_skrl.brain.trainer import brain_id_per_env, brain_scope_sizes

HALF_X = 21.0
HEADER = ["env_id", "episode", "step", "agent.x", "agent.y", "agent.angle",
          "head.flexion", "head.lateral", "left.monitor", "right.monitor",
          "correct.monitor", "experiment.phase", "imprint.cond", "test.cond",
          "brain_id"]  # brain_id LAST, matching nett_isaac.log_channel._HEADER


def _write(tmp_path, rows, name="test_Object1_0.csv"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    with p.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)
    return p


def _row(brain, env, ep, step, x, correct="left"):
    return [env, ep, step, x, 0.0, 0.0, 0.0, 0.0,
            "a.mp4", "b.mp4", correct, "test", "Object1", "rest", brain]


# --- ownership rule -------------------------------------------------------


def test_scopes_are_contiguous_and_cover_every_env():
    assert brain_scope_sizes(16, 4) == [4, 4, 4, 4]
    assert brain_id_per_env(16, 4) == [0]*4 + [1]*4 + [2]*4 + [3]*4


def test_brains_never_share_an_env():
    m = brain_id_per_env(80, 4)
    assert len(m) == 80 and len(set(m)) == 4
    for b in range(4):                      # each brain's envs are one contiguous block
        idx = [i for i, x in enumerate(m) if x == b]
        assert idx == list(range(min(idx), max(idx) + 1))


# --- the invariant that matters -------------------------------------------


def test_statistic_is_invariant_to_env_count(tmp_path):
    """ONE brain, same 8 episodes, spread over 2 envs vs 8 envs -> same correct_pct.

    x=-20 is the correct (left) outer third; x=+20 is the wrong side.
    """
    def rows_for(n_envs):
        out = []
        for ep in range(8):
            env = ep % n_envs
            x = -20.0 if ep < 6 else 20.0      # 6 of 8 episodes on the correct side
            out.append(_row(0, env, ep, 0, x))
        return out
    a = _test_preference_rows(_write(tmp_path / "a", rows_for(2)), "Object1", HALF_X)
    b = _test_preference_rows(_write(tmp_path / "b", rows_for(8)), "Object1", HALF_X)
    assert len(a) == 1 and len(b) == 1, "one brain -> exactly one row, whatever the env count"
    assert a[0][4] == b[0][4] == 0.75, (a, b)


def test_one_row_per_brain_not_per_env(tmp_path):
    """4 brains x 4 envs -> 4 rows (was: 16, mislabelled n_brains=16)."""
    rows = []
    for env in range(16):
        rows.append(_row(brain_id_per_env(16, 4)[env], env, env, 0, -20.0))
    out = _test_preference_rows(_write(tmp_path, rows), "Object1", HALF_X)
    assert len(out) == 4
    assert sorted(r[2] for r in out) == ["0", "1", "2", "3"]  # csv -> str


def test_brains_are_not_pooled_together(tmp_path):
    """Brain 0 always correct, brain 1 always wrong -> 1.0 and 0.0, never averaged."""
    rows = [_row(0, 0, 0, 0, -20.0), _row(0, 1, 1, 0, -20.0),
            _row(1, 2, 2, 0, 20.0),  _row(1, 3, 3, 0, 20.0)]
    out = {r[2]: r[4] for r in _test_preference_rows(_write(tmp_path, rows), "Object1", HALF_X)}
    assert out == {"0": 1.0, "1": 0.0}  # csv -> str


def test_pooled_ratio_not_mean_of_ratios(tmp_path):
    """Brain 0: env A 3/3 correct, env B 1/3 correct -> pooled 4/6 = 0.667,
    NOT (1.0 + 0.333)/2 = 0.667... use an asymmetric split so they differ."""
    rows = [_row(0, 0, e, 0, -20.0) for e in range(3)]          # env 0: 3/3
    rows += [_row(0, 1, 3, 0, -20.0)] + [_row(0, 1, e, 0, 20.0) for e in (4, 5, 6)]  # env 1: 1/4
    out = _test_preference_rows(_write(tmp_path, rows), "Object1", HALF_X)
    assert len(out) == 1
    assert abs(out[0][4] - 4/7) < 1e-9, f"expected pooled 4/7, got {out[0][4]}"


def test_falls_back_to_env_id_for_legacy_csvs(tmp_path):
    """Pre-brain_id CSVs still parse (single-brain runs: env_id == the only brain)."""
    p = tmp_path / "legacy.csv"
    hdr = [h for h in HEADER if h != "brain_id"]
    with p.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr)
        w.writerow(_row(0, 0, 0, 0, -20.0)[:-1])
    out = _test_preference_rows(p, "Object1", HALF_X)
    assert len(out) == 1 and out[0][4] == 1.0
