"""The cross-engine estimator must stay the one Unity actually used.

WHY THESE TESTS. ``nett_skrl.analysis.unity_parity`` exists to put an Isaac number and a
published Unity number in the same column. Everything it is for rests on the claim that it
computes Unity's statistic -- so the strongest test available is to recompute Unity's own
published table from Unity's own raw logs and demand agreement. That is what
``validate_against_unity`` does, and it is exercised here.

The load-bearing detail is the NESTING (episode -> agent -> across agents). Flat pooling over
steps agrees with Unity to 0.0036 and the nesting to 0.0011 -- close enough that a silent
regression to flat pooling would look plausible while shifting conditions by the size of the
effects being argued about. ``test_nesting_is_not_flat_pooling`` pins it with synthetic data
where the two differ by construction, so it fails whether or not Unity's bundle is present.

The Unity bundle lives outside both repos, so bundle-dependent tests SKIP when it is absent
-- the same contract as the repoA source-text tests in ``_repo_paths``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from nett_skrl.analysis import unity_parity as up

_BUNDLE = up.DEFAULT_UNITY_BUNDLE
requires_bundle = pytest.mark.skipif(
    not (_BUNDLE / "analysis/_0000_CNNTests/cnn_light/stats_across_all_agents.csv").exists(),
    reason=f"Unity jan22 bundle not present at {_BUNDLE}")

_HALF_X = up.DEFAULT_CHAMBER_HALF_X          # 33.15 -> outer third at |x| > 11.05
_OUT = _HALF_X / 3.0 + 1.0                   # comfortably inside an outer third
_MID = 0.0                                   # middle third: excluded entirely


def _row(x: float, correct: str, episode: str, cond: str = "1shape", env: str = "0"):
    return {"agent.x": f"{x}", "correct.monitor": correct, "episode": episode,
            "env_id": env, "test.cond": cond, "experiment.phase": "test", "brain_id": "0"}


def _score(rows, **kw):
    return up.agent_scores(rows, x_field="agent.x",
                           episode_fields=("env_id", "episode"), **kw)


# --------------------------------------------------------------- the nesting
def test_nesting_is_not_flat_pooling():
    """Episodes are weighted equally, NOT by their step count.

    Episode 0: 1 step, correct.   Episode 1: 3 steps, all wrong.
    nested  -> mean(1.0, 0.0)              = 0.50
    flat    -> 1 correct of 4 steps        = 0.25
    A regression to flat pooling turns this 0.50 into 0.25.
    """
    rows = [_row(_OUT, "right", "0")]
    rows += [_row(_OUT, "left", "1") for _ in range(3)]
    assert _score(rows)["1shape"] == pytest.approx(0.50)


def test_parallel_envs_are_separate_episodes():
    """On the Isaac side an episode is (env_id, episode), not episode alone.

    Envs run concurrently and share episode numbers, so keying on ``episode`` alone merges
    one episode per env into a single fragment and silently reweights the average.
    """
    rows = [_row(_OUT, "right", "0", env="0")]                       # env 0: 1/1 correct
    rows += [_row(_OUT, "left", "0", env="1") for _ in range(3)]     # env 1: 0/3 correct
    assert _score(rows)["1shape"] == pytest.approx(0.50)
    # Keyed on episode alone these four rows would be ONE episode -> 0.25.
    merged = up.agent_scores(rows, x_field="agent.x", episode_fields=("episode",))
    assert merged["1shape"] == pytest.approx(0.25)


def test_middle_third_is_excluded_from_both_numerator_and_denominator():
    """A middle-third step must not count as incorrect -- it must not count at all."""
    rows = [_row(_OUT, "right", "0")] + [_row(_MID, "left", "0") for _ in range(9)]
    assert _score(rows)["1shape"] == pytest.approx(1.0)


def test_condition_with_no_outer_third_steps_is_absent_not_zero():
    """An agent that never left the middle has NO score, which is not the same as 0.0."""
    assert "1shape" not in _score([_row(_MID, "left", "0")])


def test_immobile_agents_are_counted_not_silently_dropped(tmp_path):
    """n=3 out of a 4-agent arm must leave a visible trace.

    Real case (2026-07-29, arm A offset 1): a policy with a normal training reward reversed
    into the back wall and sat there for all 520k test steps. It scores nothing, so the arm
    silently reported n=3 -- indistinguishable in the table from a 3-agent arm.
    """
    movers = [_write_run(tmp_path, f"m{i}", [_row(_OUT, "right", "0")]) for i in range(3)]
    stuck = _write_run(tmp_path, "stuck", [_row(_MID, "right", "0")])
    recs = up.score_isaac_runs([*movers, stuck])
    assert recs[0]["n"] == 3, "only the movers are scored"
    assert recs[0]["n_immobile"] == 1, "the immobile agent must be counted"
    assert "EXCLUDED as immobile" in up.format_table("t", recs)


def test_all_agents_immobile_is_not_reported_as_a_null_result(tmp_path):
    stuck = [_write_run(tmp_path, f"s{i}", [_row(_MID, "right", "0")]) for i in range(3)]
    recs = up.score_isaac_runs(stuck)
    assert recs == []
    assert "NO AGENTS SCORED" in up.format_table("t", recs)


def test_torn_row_is_dropped_rather_than_scored():
    rows = [_row(_OUT, "right", "0"), {**_row(_OUT, "left", "0"), "agent.x": ""}]
    assert _score(rows)["1shape"] == pytest.approx(1.0)


def test_geometry_comes_from_the_shared_definition():
    """The outer-third rule must be the one the rest of the analysis uses, not a copy.

    Pins the import seam: if ``unity_parity`` grows its own ``half_x / 3``, changing the
    shared definition would stop moving this estimator and the two engines drift apart.
    """
    src = (up.__file__ and open(up.__file__).read()) or ""
    assert "in_correct_chamber_third" in src, "must reuse api.in_correct_chamber_third"
    assert "/ 3" not in src.replace("half_x / 3", ""), "no second copy of the third rule"


def test_chamber_half_x_is_threaded_through():
    """A wider chamber moves the threshold, so a step can leave the outer third."""
    rows = [_row(12.0, "right", "0")]                 # outer third at half_x=33.15
    assert _score(rows)["1shape"] == pytest.approx(1.0)
    assert "1shape" not in _score(rows, chamber_half_x=100.0)   # threshold 33.3 > 12.0


# --------------------------------------------------------------- pooling separate runs
def _write_run(tmp_path, name: str, rows: Sequence[Mapping[str, str]]) -> Path:
    path = tmp_path / f"{name}.csv"
    fields = ["env_id", "episode", "agent.x", "correct.monitor", "experiment.phase",
              "test.cond", "brain_id"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    return path


def test_pooling_keeps_separate_runs_as_separate_agents(tmp_path):
    """A one-brain-per-GPU arm has n agents, not 1 -- ``brain_id`` is file-local.

    Every run in such a fan-out writes ``brain_id=0``; the offset is in the directory name.
    Grouping the concatenated rows by ``brain_id`` would merge four agents into one, driving
    sd to nan/0 and |t| up -- a significant-looking result from a clerical merge. Two runs
    scoring 1.0 and 0.0 must give mean 0.5 with n=2, not one agent at 0.5.
    """
    a = _write_run(tmp_path, "run_a", [_row(_OUT, "right", "0")])
    b = _write_run(tmp_path, "run_b", [_row(_OUT, "left", "0")])
    pooled = {r["condition"]: r for r in up.score_isaac_runs([a, b])}["1shape"]
    assert pooled["n"] == 2, "each run must contribute its own agent"
    assert float(pooled["mean"]) == pytest.approx(0.5)
    assert float(pooled["sd"]) == pytest.approx(math.sqrt(0.5), abs=1e-6)


def test_pooling_one_run_matches_scoring_it_directly(tmp_path):
    rows = [_row(_OUT, "right", "0"), _row(_OUT, "left", "1")]
    path = _write_run(tmp_path, "run", rows)
    assert up.score_isaac_runs([path]) == up.score_isaac(path)


def test_multi_brain_file_still_yields_one_agent_per_brain(tmp_path):
    """The 7-brains-in-one-process layout must keep working through the same seam."""
    rows = [{**_row(_OUT, "right", "0"), "brain_id": "0"},
            {**_row(_OUT, "left", "0"), "brain_id": "1"}]
    path = _write_run(tmp_path, "multi", rows)
    assert len(up.isaac_agents(path)) == 2


# --------------------------------------------------------------- the side-lock screen
def test_a_fully_side_locked_agent_scores_exactly_chance():
    """The artifact the |sp| column exists to expose.

    An agent parked on the right wall scores 1.0 whenever the target is right and 0.0 when it
    is left, so on a counterbalanced design its mean is EXACTLY 0.5 -- identical to
    indifference. ``|sp| = 1.0`` is the only thing that separates them, which is why the two
    are computed in one pass and printed together.
    """
    rows = [_row(_OUT, "right", "0"), _row(_OUT, "left", "1")]
    stats = up.agent_stats(rows, x_field="agent.x", episode_fields=("env_id", "episode"))
    assert stats["1shape"]["score"] == pytest.approx(0.5)
    assert stats["1shape"]["side_preference"] == pytest.approx(1.0)


def test_side_preference_is_aggregated_absolute_so_opposite_locks_do_not_cancel(tmp_path):
    """Two agents locked on OPPOSITE walls are both locked, not jointly unbiased."""
    right = _write_run(tmp_path, "r", [_row(_OUT, "right", "0"), _row(_OUT, "left", "1")])
    left = _write_run(tmp_path, "l", [_row(-_OUT, "right", "0"), _row(-_OUT, "left", "1")])
    rec = {r["condition"]: r for r in up.score_isaac_runs([right, left])}["1shape"]
    assert rec["sp_abs_mean"] == pytest.approx(1.0), "signed preferences must not cancel"
    assert rec["sp_abs_max"] == pytest.approx(1.0)


def test_the_table_always_carries_the_side_preference_and_flags_a_lock(tmp_path):
    """"No mean without its |sp|" has to be enforced by the printer, not by discipline."""
    runs = [_write_run(tmp_path, f"a{i}", [_row(_OUT, "right", "0"), _row(_OUT, "left", "1")])
            for i in range(3)]
    table = up.format_table("t", up.score_isaac_runs(runs))
    assert "|sp|avg" in table and "|sp|max" in table
    assert "SIDE-LOCK" in table, "a locked arm must be called out, not left to be spotted"


# --------------------------------------------------- Unity's aggregated per-episode results
_RESULT_FIELDS = ["Episode", "left.monitor", "right.monitor", "correct.monitor",
                  "experiment.phase", "imprint.cond", "test.cond",
                  "left_steps", "right_steps", "middle_steps", "filename", "agent"]


def _write_unity_results(tmp_path, rows: Sequence[Mapping[str, object]]):
    """``rows``: dicts of episode, correct, left, right (+ optional cond/imprint/agent)."""
    path = tmp_path / "test_results.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_RESULT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({"Episode": r["episode"], "left.monitor": "L.webm",
                        "right.monitor": "R.webm", "correct.monitor": r["correct"],
                        "experiment.phase": r.get("phase", "test"),
                        "imprint.cond": r.get("imprint", "Object1"),
                        "test.cond": r.get("cond", "1shape"),
                        "left_steps": r["left"], "right_steps": r["right"],
                        "middle_steps": r.get("middle", 0),
                        "filename": "x.csv", "agent": r.get("agent", "1")})
    return path


def test_unity_results_reader_uses_the_same_nesting(tmp_path):
    """Episodes are weighted equally here too -- the counts are pre-reduced, the nesting is not.

    Episode 0: 1 correct step. Episode 1: 3 incorrect steps.
    nested -> mean(1.0, 0.0) = 0.50; flat over steps -> 1/4 = 0.25.
    """
    path = _write_unity_results(tmp_path, [
        {"episode": 0, "correct": "right", "left": 0, "right": 1},
        {"episode": 1, "correct": "right", "left": 3, "right": 0}])
    stats = up.unity_results_agents(path)
    assert len(stats) == 1
    assert stats[0]["1shape"]["score"] == pytest.approx(0.50)
    assert stats[0]["1shape"]["side_preference"] == pytest.approx((1 - 3) / 4)


def test_unity_results_agent_is_imprint_condition_and_index(tmp_path):
    """One file holds every imprint condition; a brain per condition is its own subject.

    Keying on ``agent`` alone merges them and halves n -- in
    ``archive/Compendium1.2.15`` that is 10 agents reported as 5, averaged across imprinting.
    """
    path = _write_unity_results(tmp_path, [
        {"episode": 0, "correct": "right", "left": 0, "right": 1, "imprint": "Object1"},
        {"episode": 0, "correct": "left", "left": 0, "right": 1, "imprint": "Object2"}])
    assert len(up.unity_results_agents(path)) == 2
    rec = {r["condition"]: r for r in up.score_unity_results(path)}["1shape"]
    assert rec["n"] == 2 and float(rec["mean"]) == pytest.approx(0.5)


def test_unity_results_episode_with_no_outer_third_steps_is_skipped(tmp_path):
    """An empty denominator is undefined; scoring it 0.5 would invent an episode."""
    path = _write_unity_results(tmp_path, [
        {"episode": 0, "correct": "right", "left": 0, "right": 1},
        {"episode": 1, "correct": "right", "left": 0, "right": 0, "middle": 500}])
    assert up.unity_results_agents(path)[0]["1shape"]["score"] == pytest.approx(1.0)


def test_unity_results_reader_ignores_non_test_rows(tmp_path):
    path = _write_unity_results(tmp_path, [
        {"episode": 0, "correct": "right", "left": 0, "right": 1},
        {"episode": 1, "correct": "left", "left": 9, "right": 0, "phase": "train"}])
    assert up.unity_results_agents(path)[0]["1shape"]["score"] == pytest.approx(1.0)


_ARCHIVE = Path.home() / "code" / "analysis" / "archive" / "Compendium1.2.15"


@pytest.mark.skipif(not (_ARCHIVE / "0" / "test_results.csv").exists(),
                    reason=f"Unity-era archive not present at {_ARCHIVE}")
@pytest.mark.parametrize("variant,binding,sp_avg,n", [
    ("0", 0.516, 0.103, 10),      # policy head None -> a LINEAR head
    ("1", 0.724, 0.070, 10),      # [64, 64]
    ("2", 0.674, 0.042, 9),       # [64, 64], ent_coef 0
    ("3", 0.723, 0.050, 10),      # [64, 64, 64], ent_coef 0
    ("4", 0.599, 0.080, 10),      # [64, 64] + binocular
])
def test_reader_reproduces_the_archive_policy_head_table(variant, binding, sp_avg, n):
    """The substantive archive finding, pinned to the module rather than to a scratch script.

    ``archive/Compendium1.2.15`` is an internal control on the policy head: one script, one
    Unity executable, one seed set, ``POLICY_LAYERS`` the only substantive variable. The
    linear head sits at chance on binding (0.516) and two hidden layers lift it to ~0.72 with
    low side bias -- replicated in ``Compendium1.2.15T``. That result is the reason the Isaac
    campaign moved off the linear head, so the numbers it rests on are asserted here.
    """
    rec = {r["condition"]: r for r in
           up.score_unity_results(_ARCHIVE / variant / "test_results.csv")}["binding"]
    assert float(rec["mean"]) == pytest.approx(binding, abs=0.001)
    assert float(rec["sp_abs_mean"]) == pytest.approx(sp_avg, abs=0.001)
    assert int(rec["n"]) == n


# --------------------------------------------------------------- statistics
def test_t_test_matches_a_known_value():
    t, p = up.t_test_1samp([0.6, 0.7, 0.8], mu=0.5)
    assert t == pytest.approx(3.4641, abs=1e-3)       # (0.7-0.5)/(0.1/sqrt(3))
    assert p == pytest.approx(0.0742, abs=1e-3)       # two-sided, df=2


@pytest.mark.parametrize("vals", [[0.5], [], [0.6, 0.6, 0.6]])
def test_t_test_is_undefined_rather_than_infinite(vals):
    """n<2 or zero variance must report nan, not a spurious significant result."""
    t, p = up.t_test_1samp(vals)
    assert math.isnan(t) and math.isnan(p)


# --------------------------------------------------------------- against Unity itself
@requires_bundle
def test_estimator_reproduces_unitys_published_table():
    """The whole justification for the module, asserted rather than asserted-in-prose."""
    worst = up.validate_against_unity()
    assert worst["mean"] < 0.002, f"estimator disagrees with Unity's `avgs`: {worst}"
    assert worst["tval"] < 1.0, f"t statistic disagrees with Unity's: {worst}"
    assert worst["pval"] < 0.01, f"p value disagrees with Unity's: {worst}"


@requires_bundle
def test_unity_shape_conditions_are_at_chance():
    """The substantive cross-engine finding: Unity's OWN CNN is shape-blind.

    Isaac's shape results were read as a port defect until this was checked. If a future
    change to the estimator makes Unity's shape conditions significant, the comparison that
    conclusion rests on has broken -- so it is pinned here.
    """
    table = {r["condition"]: r for r in up.score_unity()}
    for cond in ("1shape", "2shape", "binding"):
        assert float(table[cond]["pval"]) > 0.05, f"Unity {cond} should be at chance"
        assert float(table[cond]["mean"]) == pytest.approx(0.5, abs=0.02)
    for cond in ("1color", "2color"):
        assert float(table[cond]["pval"]) < 0.05, f"Unity {cond} should be significant"


@requires_bundle
def test_unity_reader_drops_the_interleaved_trace_lines():
    """Unity's logger writes bare 'OnActionReceived' lines between data rows."""
    path = next((_BUNDLE / up._UNITY_RESULTS).glob("brain_*/logs/test_*.csv"))
    rows = up.read_unity_log(path)
    assert rows, "no rows parsed"
    assert all("agent.x" in r for r in rows), "header keys must be stripped of padding"
    assert all(r["agent.x"] not in ("", "OnActionReceived") for r in rows)


@requires_bundle
def test_verdict_is_robust_to_the_outer_third_threshold():
    """Documented insensitivity: the threshold is not quietly tuning the result."""
    base = {r["condition"]: float(r["mean"]) for r in up.score_unity()}
    for half_x in (30.15, 36.0):
        alt = {r["condition"]: float(r["mean"]) for r in up.score_unity(chamber_half_x=half_x)}
        for cond, value in base.items():
            assert abs(alt[cond] - value) < 0.005, f"{cond} moved at half_x={half_x}"


@requires_bundle
def test_cli_refuses_when_the_bundle_is_missing(tmp_path, capsys):
    """Without Unity there is no comparison -- it must not print a table anyway."""
    assert up.main(["--unity-bundle", str(tmp_path / "nope")]) == 2
    assert "not comparable" in capsys.readouterr().out
