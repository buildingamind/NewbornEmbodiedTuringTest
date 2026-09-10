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

def scorable_episode(env, ep, cond="Novel Familiar", n=4):
    """An episode the agent crossed: n frames each side."""
    return ([row(env, ep, s, -20.0, cond, "1A_00.mov", "2B_00.mov", "right")
             for s in range(n)]
            + [row(env, ep, n + s, +20.0, cond, "1A_00.mov", "2B_00.mov", "right")
               for s in range(n)])


def test_a_parked_episode_is_dropped_not_scored_as_a_coin_flip(tmp_path):
    """⛔ THE CENTRAL EXCLUSION. The agent is parked ~80% of steps; an episode it never
    left has frames of one monitor and none of the other. There is no contest to run."""
    rows = rest_rows(9, 0, 3, -20.0) + [
        row(0, 1, s, -20.0, "Novel Familiar", "1A_00.mov", "2B_00.mov", "right")
        for s in range(10)]
    csv = write_csv(tmp_path, rows)
    keys = [(9, 0, s) for s in range(3)] + [(0, 1, s) for s in range(10)]
    _, eps, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert eps == [], "a one-sided episode cannot be scored"
    assert rep["episodes_total"] == 1 and rep["episodes_scorable"] == 0
    assert rep["dropped_one_sided"] == {"Novel Familiar": 1}


def test_the_drop_is_reported_per_condition(tmp_path):
    """The exclusion is not uniform across conditions, so one total would hide it."""
    rows = rest_rows(9, 0, 3, -20.0) + scorable_episode(0, 1) + [
        row(1, 2, s, +20.0, "Both Familiar", "1A_00.mov", "2A_00.mov", "right")
        for s in range(6)]
    csv = write_csv(tmp_path, rows)
    keys = ([(9, 0, s) for s in range(3)] + [(0, 1, s) for s in range(8)]
            + [(1, 2, s) for s in range(6)])
    _, eps, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert rep["episodes_scorable"] == 1
    assert rep["dropped_one_sided"] == {"Both Familiar": 1}


def test_rest_is_never_scored_as_a_test_episode(tmp_path):
    """Rest is the exposure set. Scoring it would fit and test on the same frames."""
    csv = write_csv(tmp_path, rest_rows(0, 0, 4, -20.0) + scorable_episode(1, 1))
    keys = [(0, 0, s) for s in range(4)] + [(1, 1, s) for s in range(8)]
    _, eps, _ = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert [c for c, _, _ in eps] == ["Novel Familiar"]


def test_episodes_are_keyed_on_env_AND_episode_not_episode_alone(tmp_path):
    """⛔ `episode` restarts per env. Keying on it alone merges 112 envs' episode 3 into
    one, so frames from different trials -- different monitors -- pool into one contest."""
    csv = write_csv(tmp_path, rest_rows(9, 0, 3, -20.0)
                    + scorable_episode(0, 3) + scorable_episode(1, 3))
    keys = ([(9, 0, s) for s in range(3)] + [(0, 3, s) for s in range(8)]
            + [(1, 3, s) for s in range(8)])
    _, eps, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert rep["episodes_total"] == 2, "same episode id, different env, different trial"
    assert len(eps) == 2


# --- the join itself -------------------------------------------------------------------

def test_a_capture_that_matches_nothing_refuses(tmp_path):
    """A capture joined to the wrong run's log yields an empty set, and an empty set
    would otherwise flow through as 'no episodes to score' -- exit 0, nothing measured."""
    csv = write_csv(tmp_path, rest_rows(0, 0, 3, -20.0))
    with pytest.raises(SystemExit, match="NOT ONE"):
        rh.build_capture_pairs(blob_for([(77, 77, 77)]), csv, verbose=False)


def test_partial_join_is_counted_not_silently_dropped(tmp_path):
    csv = write_csv(tmp_path, rest_rows(0, 0, 3, -20.0) + scorable_episode(1, 1))
    keys = [(0, 0, s) for s in range(3)] + [(1, 1, s) for s in range(8)] + [(5, 5, 5)]
    _, _, rep = rh.build_capture_pairs(blob_for(keys), csv, verbose=False)
    assert rep["unjoined"] == 1 and rep["joined"] == 11


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
