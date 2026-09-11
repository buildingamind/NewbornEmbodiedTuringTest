"""The capture-mode imprinting readout, and the joins it must not get wrong.

⛔ MEASURED, NOT INVENTED. The fixtures below use the real schema of
`fork-1/logs/test_fork-1_0.csv` from 3DCNN_parsing_fork-1_off0_0831_160636: 15 named
columns, `(env_id, episode, step)` unique across 560,000 rows, and a `Rest` condition
that shows `2A_00.mov` opposite `White.mov` on every one of its 21,000 steps. A fixture
written from the reader's assumptions can only confirm them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_harness",
                                               REPO / "examples" / "replay_harness.py")
rh = importlib.util.module_from_spec(_spec)
sys.modules["replay_harness"] = rh
_spec.loader.exec_module(rh)

COLUMNS = ("env_id,episode,step,agent.x,agent.y,agent.angle,head.flexion,head.lateral,"
           "left.monitor,right.monitor,correct.monitor,experiment.phase,imprint.cond,"
           "test.cond,brain_id")


def row(env, ep, step, x, cond, left, right, correct):
    return (f"{env},{ep},{step},{x},0.0,0.0,0.0,0.0,{left},{right},{correct},"
            f"test,fork-1,{cond},0")


def write_csv(tmp_path, rows, name="test_fork-1_0.csv"):
    p = tmp_path / name
    p.write_text(COLUMNS + "\n" + "\n".join(rows) + "\n")
    return p


def rest_rows(env, ep, n, x):
    """Rest: the imprinted object on the LEFT, blank on the right."""
    return [row(env, ep, s, x, "Rest", "2A_00.mov", "White.mov", "left") for s in range(n)]


def blob_for(keys):
    return {"keys": np.array(keys, dtype=np.int32)}


# --- the exposure set ------------------------------------------------------------------

def test_memory_is_built_from_rest_frames_viewing_the_object_not_the_blank(tmp_path):
    """Rest shows the object on ONE side. Frames from the blank side are not exposure."""
    rows = (rest_rows(0, 0, 4, -20.0)      # left side, sees 2A_00 -> exposure
            + rest_rows(0, 1, 3, +20.0))   # right side, sees White -> NOT exposure
    csv = write_csv(tmp_path, rows)
    keys = [(0, 0, s) for s in range(4)] + [(0, 1, s) for s in range(3)]
    mem, eps, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert len(mem) == 4, "only the frames facing the imprinted object are exposure"
    assert rep["rest_blank_side"] == 3
    assert mem == [0, 1, 2, 3]


def test_an_empty_exposure_set_refuses_rather_than_scoring(tmp_path):
    """⛔ No memory means NO SCORE. Returning 0.5 would look like chance performance."""
    csv = write_csv(tmp_path, rest_rows(0, 0, 5, +20.0))   # all on the blank side
    keys = [(0, 0, s) for s in range(5)]
    with pytest.raises(SystemExit, match="EMPTY"):
        rh.build_capture_pairs(blob_for(keys), csv, verbose=False)


def test_middle_of_chamber_frames_are_evidence_about_neither_monitor(tmp_path):
    """At 300 degrees both screens are in view from the middle; those frames are not a
    view of either, and counting them as exposure would blend the novel object in."""
    csv = write_csv(tmp_path, rest_rows(0, 0, 3, -20.0) + rest_rows(0, 1, 3, -2.0))
    keys = [(0, 0, s) for s in range(3)] + [(0, 1, s) for s in range(3)]
    mem, _, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert len(mem) == 3
    assert rep["rest_frames"] == 6, "the middle frames were seen, just not used"


# --- the scored episodes ---------------------------------------------------------------

def scorable_episode(env, ep, cond="Novel Familiar", n=4, correct="right"):
    """An episode the agent crossed: n frames each side.

    ⚠ `correct` matters: a corpus whose every episode shares one correct side cannot
    distinguish a side-locked agent from a discriminating one, and build_capture_pairs
    refuses it. Real captures from this driver are 100% target-left, which is why that
    refusal exists -- so fixtures must be balanced deliberately, not by luck.
    """
    left, right = ("1A_00.mov", "2B_00.mov") if correct == "right" else ("2B_00.mov", "1A_00.mov")
    return ([row(env, ep, s, -20.0, cond, left, right, correct) for s in range(n)]
            + [row(env, ep, n + s, +20.0, cond, left, right, correct) for s in range(n)])


def mirrored(env, ep, cond="Novel Familiar", n=4):
    """The opposite-side partner, so a fixture corpus has a non-constant answer key."""
    return scorable_episode(env, ep, cond=cond, n=n, correct="left")


def test_a_parked_episode_is_dropped_not_scored_as_a_coin_flip(tmp_path):
    """⛔ THE CENTRAL EXCLUSION. The agent is parked ~80% of steps; an episode it never
    left has frames of one monitor and none of the other. There is no contest to run."""
    rows = (rest_rows(9, 0, 3, -20.0) + [
        row(0, 1, s, -20.0, "Novel Familiar", "1A_00.mov", "2B_00.mov", "right")
        for s in range(10)]
        # a balanced partner, so the corpus is refused for PARKING and not for a
        # constant answer key -- otherwise this test would pass for the wrong reason
        + mirrored(3, 0))
    csv = write_csv(tmp_path, rows)
    # env 0's only episode is global id 1, so the capture names it as local index 0.
    keys = ([(9, 0, s) for s in range(3)] + [(0, 0, s) for s in range(10)]
            + [(3, 0, s) for s in range(8)])
    _, eps, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert [e[0] for e in eps] == ["Novel Familiar"], "only the crossing episode scores"
    assert rep["episodes_total"] == 2 and rep["episodes_scorable"] == 1
    assert rep["dropped_one_sided"] == {"Novel Familiar": 1}, "the parked one is dropped"


def test_the_drop_is_reported_per_condition(tmp_path):
    """The exclusion is not uniform across conditions, so one total would hide it."""
    rows = rest_rows(9, 0, 3, -20.0) + scorable_episode(0, 0) + mirrored(2, 0) + [
        row(1, 0, s, +20.0, "Both Familiar", "1A_00.mov", "2A_00.mov", "right")
        for s in range(6)]
    csv = write_csv(tmp_path, rows)
    keys = ([(9, 0, s) for s in range(3)] + [(0, 0, s) for s in range(8)]
            + [(2, 0, s) for s in range(8)] + [(1, 0, s) for s in range(6)])
    _, eps, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    # Two crossing episodes score (the pair that balances the answer key); the parked
    # Both Familiar one is dropped, and the drop is reported UNDER ITS OWN CONDITION.
    assert rep["episodes_scorable"] == 2
    assert rep["dropped_one_sided"] == {"Both Familiar": 1}


def test_rest_is_never_scored_as_a_test_episode(tmp_path):
    """Rest is the exposure set. Scoring it would fit and test on the same frames."""
    csv = write_csv(tmp_path, rest_rows(0, 0, 4, -20.0) + scorable_episode(1, 0)
                    + mirrored(2, 0))
    keys = ([(0, 0, s) for s in range(4)] + [(1, 0, s) for s in range(8)]
            + [(2, 0, s) for s in range(8)])
    _, eps, _ = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert [c for c, _, _ in eps] == ["Novel Familiar", "Novel Familiar"]


def test_episodes_are_keyed_on_env_AND_episode_not_episode_alone(tmp_path):
    """⛔ `episode` restarts per env. Keying on it alone merges 112 envs' episode 3 into
    one, so frames from different trials -- different monitors -- pool into one contest."""
    csv = write_csv(tmp_path, rest_rows(9, 0, 3, -20.0)
                    + scorable_episode(0, 3) + mirrored(1, 3))
    # Both envs' single episode is global id 3, so each is local index 0.
    keys = ([(9, 0, s) for s in range(3)] + [(0, 0, s) for s in range(8)]
            + [(1, 0, s) for s in range(8)])
    _, eps, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert rep["episodes_total"] == 2, "same episode id, different env, different trial"
    assert len(eps) == 2


# --- the join itself -------------------------------------------------------------------

def test_a_capture_that_matches_nothing_refuses(tmp_path):
    """A capture joined to the wrong run's log yields an empty set, and an empty set
    would otherwise flow through as 'no episodes to score' -- exit 0, nothing measured.

    Caught by the same RATE check that catches the 1.25% collision case: 0% is simply
    its limit, and both have the identical fix (point at the capture's own log).
    """
    csv = write_csv(tmp_path, rest_rows(0, 0, 3, -20.0))
    with pytest.raises(SystemExit, match="ONLY 0 of 1"):
        rh.build_capture_pairs(blob_for([(77, 77, 77)]), csv, verbose=False)


def test_partial_join_is_counted_not_silently_dropped(tmp_path):
    csv = write_csv(tmp_path, rest_rows(0, 0, 3, -20.0) + scorable_episode(1, 0)
                    + mirrored(2, 0))
    keys = ([(0, 0, s) for s in range(3)] + [(1, 0, s) for s in range(8)]
            + [(2, 0, s) for s in range(8)] + [(5, 5, 5)])
    _, _, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert rep["unjoined"] == 1 and rep["joined"] == 19


def test_a_summary_csv_is_refused_by_name_not_by_position(tmp_path):
    """⛔ analysis/test/test_preferences.csv has no agent.x. An awk survey in this
    campaign read it positionally and reported a silent 1.0000."""
    p = tmp_path / "test_preferences.csv"
    p.write_text("imprint.cond,test.cond,percent.correct\nfork-1,Novel Familiar,0.42\n")
    with pytest.raises(SystemExit, match="missing"):
        rh.load_test_labels(p)


def test_duplicate_join_keys_refuse_rather_than_keeping_the_last(tmp_path):
    csv = write_csv(tmp_path, rest_rows(0, 0, 3, -20.0) + rest_rows(0, 0, 3, +20.0))
    with pytest.raises(SystemExit, match="duplicate"):
        rh.load_test_labels(csv)


def test_default_csv_refuses_the_summary_path(tmp_path):
    (tmp_path / "fork-1" / "logs").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no test_"):
        rh.default_test_csv(str(tmp_path), "fork-1")


def test_default_csv_finds_the_per_step_log(tmp_path):
    logs = tmp_path / "fork-1" / "logs"
    logs.mkdir(parents=True)
    (logs / "test_fork-1_0.csv").write_text(COLUMNS + "\n")
    assert rh.default_test_csv(str(tmp_path), "fork-1") == logs / "test_fork-1_0.csv"


# --- the rule itself -------------------------------------------------------------------

class ToyEncoder:
    """Returns the frame's own mean as its feature, so cosine is exactly controllable."""

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def _prepare_image(self, x):
        return x

    def encode_prepared(self, x):
        import torch
        return torch.as_tensor(x, dtype=torch.float32).reshape(len(x), -1)


def test_the_rule_picks_the_side_closer_to_the_exposure_memory():
    """The whole readout in one assertion: memory is 'ones', the right side is 'ones',
    the left is orthogonal, correct is 'right' -> the rule must score 1.0."""
    obs = np.zeros((6, 4), dtype=np.float32)
    obs[0:2] = [1, 1, 0, 0]        # exposure
    obs[2:4] = [0, 0, 1, 1]        # left  -- orthogonal to memory
    obs[4:6] = [1, 1, 0, 0]        # right -- identical to memory
    episodes = [("Novel Familiar", {"left": [2, 3], "right": [4, 5]}, "right")]
    out = rh.capture_readout(ToyEncoder(), obs, [0, 1], episodes)
    assert out == {"Novel Familiar": (1.0, 1)}


def test_the_rule_is_wrong_when_the_novel_side_is_the_closer_one():
    """The same machinery must be able to score ZERO, or it is not measuring anything."""
    obs = np.zeros((6, 4), dtype=np.float32)
    obs[0:2] = [1, 1, 0, 0]
    obs[2:4] = [1, 1, 0, 0]        # left matches memory
    obs[4:6] = [0, 0, 1, 1]
    episodes = [("Novel Familiar", {"left": [2, 3], "right": [4, 5]}, "right")]
    assert rh.capture_readout(ToyEncoder(), obs, [0, 1], episodes) == {
        "Novel Familiar": (0.0, 1)}


def test_the_episode_count_travels_with_the_accuracy():
    """⛔ An accuracy without its n is the defect this campaign spent a day on."""
    obs = np.zeros((10, 4), dtype=np.float32)
    obs[0:2] = [1, 1, 0, 0]
    for s in (2, 4, 6, 8):
        obs[s:s + 2] = [1, 1, 0, 0] if s % 4 == 0 else [0, 0, 1, 1]
    episodes = [("Novel Familiar", {"left": [2, 3], "right": [4, 5]}, "right"),
                ("Novel Familiar", {"left": [6, 7], "right": [8, 9]}, "right")]
    (_, n), = rh.capture_readout(ToyEncoder(), obs, [0, 1], episodes).values()
    assert n == 2


# --- the join RATE, and the collision that a zero-check cannot see -----------------------

def test_a_tiny_join_built_from_key_collisions_is_refused(tmp_path):
    """⛔ THE ONE THAT NEARLY SHIPPED. Joined against the SOURCE run's log instead of the
    capture's own, a real 5,120-frame capture matched 64 frames -- 1.25%, every one of
    them labelled Rest, all of them key COLLISIONS with a different execution. A
    zero-check passes that, and those 56 frames become the imprinting memory.
    """
    csv = write_csv(tmp_path, rest_rows(0, 0, 4, -20.0))
    keys = [(0, 0, s) for s in range(4)] + [(0, 900 + i, 0) for i in range(96)]
    with pytest.raises(SystemExit, match="4.00%|ONLY 4 of 100"):
        rh.build_capture_pairs(blob_for(keys), csv, verbose=False)


def test_a_complete_join_is_accepted(tmp_path):
    csv = write_csv(tmp_path, rest_rows(0, 0, 4, -20.0) + scorable_episode(1, 0)
                    + mirrored(2, 0))
    keys = ([(0, 0, s) for s in range(4)] + [(1, 0, s) for s in range(8)]
            + [(2, 0, s) for s in range(8)])
    mem, eps, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert rep["joined"] == rep["captured"] == 20 and len(eps) == 2


def test_the_capture_s_own_log_is_preferred_over_the_source_run_s(tmp_path):
    """A capture is a REPLAY with its own env/episode numbering; the source run's log
    describes a different execution and is the wrong file even though it exists."""
    src = tmp_path / "source_run"
    (src / "fork-1" / "logs").mkdir(parents=True)
    (src / "fork-1" / "logs" / "test_fork-1_0.csv").write_text(COLUMNS + "\n")
    cap_dir = tmp_path / "capture_out"
    own = cap_dir / "source_run" / "fork-1" / "logs"
    own.mkdir(parents=True)
    (own / "test_fork-1_0.csv").write_text(COLUMNS + "\n")
    npz = cap_dir / "obs_source_run_fork-1.npz"
    got = rh.default_test_csv(str(src), "fork-1", capture=npz)
    assert got == own / "test_fork-1_0.csv", "must prefer the capture's own tree"


def test_falling_back_to_the_source_run_says_so(tmp_path, capsys):
    """The fallback is a near-certain wrong join, so it must never be silent."""
    src = tmp_path / "source_run"
    (src / "fork-1" / "logs").mkdir(parents=True)
    (src / "fork-1" / "logs" / "test_fork-1_0.csv").write_text(COLUMNS + "\n")
    cap_dir = tmp_path / "capture_out"
    cap_dir.mkdir()
    rh.default_test_csv(str(src), "fork-1", capture=cap_dir / "obs_x.npz")
    assert "near-zero join" in capsys.readouterr().out


# --- the local/global episode translation -----------------------------------------------

def test_a_per_env_episode_counter_is_translated_to_the_log_s_global_ids(tmp_path):
    """⛔ THE DEFECT THAT MADE THE DOCUMENTED JOIN A 1.25% JOIN. capture_observations
    numbers episodes per env from 0 (`FrameAlignment.on_done`); the log numbers them
    globally, so env 0's episodes are 0, 112, 224 ... Only local 0 coincides, and every
    later episode silently fails to match.
    """
    rows = []
    for local, gep in enumerate((0, 112, 224)):        # env 0's episodes, strided
        rows += [row(0, gep, s, -20.0, "Rest", "2A_00.mov", "White.mov", "left")
                 for s in range(3)]
    csv = write_csv(tmp_path, rows)
    assert rh.episode_index_map(csv) == {(0, 0): 0, (0, 1): 112, (0, 2): 224}
    # A capture naming local episodes 0,1,2 must reach all three, not just the first.
    keys = [(0, local, s) for local in range(3) for s in range(3)]
    mem, _, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert rep["joined"] == 9, "all three episodes must join, not just local 0"
    assert len(mem) == 9


def test_the_untranslated_join_would_have_been_one_episode(tmp_path):
    """The counterfactual, pinned: keyed on the raw local index against global ids, only
    episode 0 matches -- which is what a 5,120-frame capture matching 64 frames was."""
    rows = []
    for gep in (0, 112, 224):
        rows += [row(0, gep, s, -20.0, "Rest", "2A_00.mov", "White.mov", "left")
                 for s in range(3)]
    csv = write_csv(tmp_path, rows)
    raw = rh.load_test_labels(csv)
    hits = sum(1 for local in range(3) for s in range(3) if (0, local, s) in raw)
    assert hits == 3, "untranslated, only local==global==0 matches: 1 episode of 3"


# ---------------------------------------------------------------------------
# Minimum detectable shift. Added because on 2026-09-11 I published "no imprinting
# signal on the one cell where a signal would have meant something" from a cell whose
# MDE was 0.256 -- it would have returned the same answer if imprinting were strong.
# ---------------------------------------------------------------------------
from replay_harness import minimum_detectable_shift  # noqa: E402


def test_the_cell_that_produced_the_false_negative():
    """⛔ THE EXACT CELL AND THE EXACT CLAIM. n=30, trained 0.4333, chance 0.500."""
    mde = minimum_detectable_shift(30)
    assert round(mde, 3) == 0.256
    assert abs(0.4333 - 0.5) < mde, "the observed shift is INSIDE the noise floor"
    # It separates chance only from these extremes:
    assert round(0.5 + mde, 3) == 0.756 and round(0.5 - mde, 3) == 0.244


def test_the_brain_level_mde_exceeds_the_parameter_range():
    """⛔⛔ THE POINT THAT KILLS 'more episodes will fix it'. Episodes shrink the
    within-brain term, which is not binding. At the corpus's ~2 effective policies the
    detectable shift is wider than [0, 1] -- nothing is detectable at ANY effect size,
    and it takes roughly 20 brains to bring it inside a usable range."""
    assert minimum_detectable_shift(2) > 0.5     # 0.991
    assert minimum_detectable_shift(4) > 0.5     # 0.700
    assert minimum_detectable_shift(7) > 0.5     # 0.529 -- seven brains is STILL not enough
    assert minimum_detectable_shift(20) < 0.35   # 0.313


def test_more_units_is_monotonically_better_and_zero_is_infinite():
    prev = float("inf")
    for n in (1, 2, 5, 10, 50, 200):
        m = minimum_detectable_shift(n)
        assert m < prev
        prev = m
    assert minimum_detectable_shift(0) == float("inf"), "an empty cell detects nothing"
