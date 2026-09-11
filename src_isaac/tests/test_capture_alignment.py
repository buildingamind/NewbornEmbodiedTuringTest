"""Frame-keeping and action alignment for examples/capture_observations.py.

⛔ WHY THIS FILE EXISTS. `capture_observations.py` has never completed a
full-schedule run (its own module docstring says so), and the two things it now
has to get right fail SILENTLY:

  * a strided capture (`--window 1`) contains no two adjacent frames, so an
    action-conditioned objective trained on it has no (obs_t, a_t, obs_t+1)
    triple anywhere -- the file looks fine and the loss has nothing to learn from;
  * attaching an action to the frame it PRODUCED rather than the frame it ACTED
    ON shifts every action by one step, does not crash, and trains a predictor on
    the action that followed its target.

Neither shows up as an error. Both show up as a plausible, wrong model. The
bookkeeping is therefore extracted into `FrameAlignment`, which is pure integer
logic and testable here without booting Kit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "examples"))

from capture_observations import FrameAlignment


def _run(num_envs, every, window, n_steps, done_at=()):
    """Drive the alignment the way Capture does and return what it recorded."""
    a = FrameAlignment(num_envs, every, window)
    kept, acted = [], []
    a.on_reset()
    for i in range(n_steps):
        wanted = a.wanted()
        a.begin_record()
        for e in wanted:
            key = a.key(e)
            kept.append(key)
            a.mark_kept(e, key)
        a.advance()
        if i in done_at:
            for e in range(num_envs):
                a.on_done(e)
            continue
        acted.extend(k for _e, k in a.action_targets())
    return kept, acted


# --- frame keeping ---------------------------------------------------------

def test_window_one_reproduces_the_original_strided_capture():
    kept, _ = _run(1, every=20, window=1, n_steps=61)
    assert [s for _e, _ep, s in kept] == [0, 20, 40, 60]


def test_window_keeps_consecutive_frames_at_each_sampling_point():
    kept, _ = _run(1, every=20, window=3, n_steps=45)
    assert [s for _e, _ep, s in kept] == [0, 1, 2, 20, 21, 22, 40, 41, 42]


def test_window_equal_to_every_is_a_contiguous_capture():
    kept, _ = _run(1, every=4, window=4, n_steps=9)
    assert [s for _e, _ep, s in kept] == list(range(9))


def test_window_larger_than_every_is_refused():
    with pytest.raises(ValueError, match="overlap"):
        FrameAlignment(1, every=4, window=5)


def test_every_env_is_tracked_independently():
    kept, _ = _run(3, every=5, window=2, n_steps=6)
    for env_id in range(3):
        assert [s for e, _ep, s in kept if e == env_id] == [0, 1, 5]


# --- the property the file is FOR ------------------------------------------

def test_window_one_yields_no_usable_transition_pair():
    """The bug that motivated --window: strided frames are never adjacent."""
    kept, acted = _run(1, every=20, window=1, n_steps=61)
    present = set(kept)
    pairs = [k for k in acted if (k[0], k[1], k[2] + 1) in present]
    assert pairs == [], "a strided capture must not appear to contain transitions"


def test_window_two_yields_usable_transition_pairs():
    kept, acted = _run(1, every=20, window=2, n_steps=42)  # 41 must be REACHED for the 40->41 pair
    present = set(kept)
    pairs = [k for k in acted if (k[0], k[1], k[2] + 1) in present]
    assert [s for _e, _ep, s in pairs] == [0, 20, 40]


# --- action alignment ------------------------------------------------------

def test_action_attaches_to_the_frame_it_acts_on_not_the_one_it_produces():
    _kept, acted = _run(1, every=1, window=1, n_steps=4)
    # Frames 0..3 are all kept. The action passed at the i-th step acts on frame i.
    assert [s for _e, _ep, s in acted] == [0, 1, 2, 3]


def test_no_action_is_recorded_for_a_discarded_frame():
    _kept, acted = _run(1, every=10, window=1, n_steps=25)
    # Only frames 0, 10, 20 exist, so only those may carry an action.
    assert [s for _e, _ep, s in acted] == [0, 10, 20]


def test_a_transition_never_spans_a_reset():
    """The first action of episode n+1 must not attach to the last frame of n."""
    _kept, acted = _run(1, every=1, window=1, n_steps=6, done_at=(2,))
    # step 2 ends the episode; nothing may be attached across that boundary.
    assert (0, 0, 2) not in acted
    episodes = {ep for _e, ep, _s in acted}
    assert episodes == {0, 1}


def test_episode_counter_advances_and_steps_restart():
    kept, _ = _run(1, every=1, window=1, n_steps=5, done_at=(1,))
    assert kept == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 2)]


def test_action_targets_empty_before_any_frame_is_kept():
    a = FrameAlignment(2, every=10, window=1)
    a.on_reset()
    assert a.action_targets() == []


# --- completeness: the file must say whether it is a prefix -----------------
#
# ⛔ A SHORT CAPTURE IS WELL-FORMED, HAS A NON-ZERO PAIR COUNT, AND IS UNUSABLE.
# The test schedule is ORDERED -- conditions grouped, target-left design rows
# first -- so a capture that stops early covers some conditions and not others.
# Measured 2026-08-11: `--episodes 2` ran 104 of 1040 episodes and produced 104
# target-left episodes and 0 target-right. Nothing in the saved file said so.
#
# ⚠ A crash or kill is NOT this failure mode: the npz is written only after
# brain.test() returns, so an interrupted capture leaves no file at all. The
# cases that DO produce a misleading file are an --episodes that does not match
# the source run, a schedule that ends early, and --max-frames.

def _is_prefix(requested, seen, source, truncated_by_cap=False):
    """Mirror of the driver's completeness rule (capture_observations.main)."""
    return bool(
        truncated_by_cap
        or (source is not None and requested != source)
        or seen < requested
    )


def test_matching_episode_count_is_not_a_prefix():
    assert _is_prefix(requested=20, seen=20, source=20) is False


def test_short_request_against_the_source_run_is_a_prefix():
    """The 2026-08-11 measured failure: --episodes 2 against a 1040-episode run."""
    assert _is_prefix(requested=2, seen=2, source=20) is True


def test_schedule_ending_early_is_a_prefix_even_when_the_request_matched():
    assert _is_prefix(requested=20, seen=13, source=20) is True


def test_max_frames_truncation_is_a_prefix():
    assert _is_prefix(requested=20, seen=20, source=20, truncated_by_cap=True) is True


def test_unknown_source_count_still_catches_an_early_finish():
    """A run config without a test count must not silently pass a short capture."""
    assert _is_prefix(requested=20, seen=5, source=None) is True
    assert _is_prefix(requested=20, seen=20, source=None) is False


# ---------------------------------------------------------------------------
# num_envs resolution. The capture of 2026-09-11 recorded env 0 and nothing else
# because `getattr(env, "num_envs", 1)` missed on a gymnasium>=1.0 wrapper and the
# DEFAULT won. These tests exist because that failure is invisible in the output
# file: a 1-env capture and a 112-env capture that lost 111 envs are the same bytes.
# ---------------------------------------------------------------------------
import gymnasium as gymn  # noqa: E402
import numpy as np  # noqa: E402

from capture_observations import resolve_num_envs  # noqa: E402


class _Base(gymn.Env):
    observation_space = gymn.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)
    action_space = gymn.spaces.Discrete(2)

    def __init__(self, num_envs=None):
        if num_envs is not None:
            self.num_envs = num_envs


def test_gymnasium_wrapper_really_does_drop_attribute_forwarding():
    """⛔ THE PREMISE OF THE BUG, ASSERTED RATHER THAN ASSUMED. If a future gymnasium
    restores `__getattr__`, the old expression starts working and this test tells the
    reader why the resolver looks over-built."""
    assert "__getattr__" not in vars(gymn.Wrapper)
    wrapped = gymn.Wrapper(_Base(num_envs=112))
    assert getattr(wrapped, "num_envs", 1) == 1, "the silent default, reproduced"
    assert resolve_num_envs(wrapped) == 112, "the resolver sees through the wrapper"


def test_declared_count_is_used_and_cross_checked():
    assert resolve_num_envs(gymn.Wrapper(_Base(num_envs=112)), 112) == 112
    # Declared alone, on an env that cannot be asked, is accepted.
    assert resolve_num_envs(gymn.Wrapper(_Base()), 112) == 112


def test_a_disagreement_is_a_refusal_not_a_preference():
    with pytest.raises(SystemExit, match="disagreement"):
        resolve_num_envs(gymn.Wrapper(_Base(num_envs=112)), 1)
    with pytest.raises(SystemExit, match="disagreement"):
        resolve_num_envs(gymn.Wrapper(_Base(num_envs=1)), 112)


def test_no_source_at_all_refuses_rather_than_defaulting_to_one():
    """⛔ THE DEFAULT IS THE DEFECT. `1` is a legitimate value, so a default of 1 makes
    'nobody knows' indistinguishable from 'one env' -- and the capture cannot tell."""
    with pytest.raises(SystemExit, match="Refusing to default to 1"):
        resolve_num_envs(gymn.Wrapper(_Base()), None)


def test_single_env_is_still_reachable_when_it_is_the_truth():
    assert resolve_num_envs(gymn.Wrapper(_Base(num_envs=1))) == 1
    assert resolve_num_envs(gymn.Wrapper(_Base(num_envs=1)), 1) == 1


# ---------------------------------------------------------------------------
# The two axes a capture can be truncated on. `is_prefix` has only ever watched the
# EPISODE axis; the ROW axis has no witness in the npz at all, which is how a capture
# spent 40 minutes of a card producing 4.2 GB of one pose and one side on 2026-09-11.
# ---------------------------------------------------------------------------
import csv as _csv  # noqa: E402
import shutil  # noqa: E402

from capture_observations import (  # noqa: E402
    _design_coverage, expected_episodes_per_env,
)


def test_the_episode_axis_conversion_is_per_env_not_per_row():
    """⛔ THE FALSE ALARM, PINNED. 56 rows x 20 per row / 112 envs = 10 per env. The old
    comparison read the 20 as a per-env count and called a complete capture a prefix."""
    assert expected_episodes_per_env(56, 20, 112) == 10
    assert expected_episodes_per_env(56, 20, 56) == 20, "one env per row -> they coincide"
    assert expected_episodes_per_env(56, 40, 112) == 20


def test_the_conversion_refuses_degenerate_inputs_rather_than_dividing():
    for args in ((0, 20, 112), (56, 0, 112), (56, 20, 0), (-1, 20, 112)):
        assert expected_episodes_per_env(*args) == 0, f"{args} must not produce a threshold"


def _write_log(path, rows):
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["env_id", "episode", "step", "left.monitor", "right.monitor",
                    "correct.monitor", "test.cond"])
        for i, (cond, left, right, correct) in enumerate(rows):
            w.writerow([0, i, 0, left, right, correct, cond])


def test_design_coverage_separates_a_covered_sheet_from_a_truncated_one(tmp_path):
    """⛔ BOTH CASES ARE REAL FILES FROM 2026-09-11, reduced. The grouped run wrote
    560,000 log rows every one of which said correct.monitor=left."""
    full = tmp_path / "full"; full.mkdir()
    _write_log(full / "test_fork-1_0.csv", [
        ("Both Familiar", f"2A_{p}.mov", f"1A_{p}.mov", side)
        for p in ("00", "30", "60") for side in ("left", "right")])
    assert _design_coverage(full) == (6, {"left", "right"})

    trunc = tmp_path / "trunc"; trunc.mkdir()
    _write_log(trunc / "test_fork-1_0.csv",
               [("Both Familiar", "2A_00.mov", "1A_00.mov", "left")] * 40)
    rows, sides = _design_coverage(trunc)
    assert (rows, sides) == (1, {"left"}), "one pose, one side -- the failure, detected"


def test_absence_is_reported_as_not_measured_not_as_zero_coverage(tmp_path):
    """⚠ A caller must be able to tell 'no log found' from 'the log shows nothing', or an
    unmeasured capture reads as a catastrophically truncated one."""
    empty = tmp_path / "empty"; empty.mkdir()
    assert _design_coverage(empty) == (0, set())


def test_a_log_with_the_wrong_schema_is_refused_rather_than_read_positionally(tmp_path):
    """⛔ Read by NAME or not at all: a positional read of a second schema is how an
    earlier analysis in this campaign got every implied n wrong with nothing looking wrong."""
    odd = tmp_path / "odd"; odd.mkdir()
    with open(odd / "test_x_0.csv", "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["env_id", "episode", "step", "monitor_a", "monitor_b"])
        w.writerow([0, 0, 0, "2A_00.mov", "1A_00.mov"])
    assert _design_coverage(odd) == (0, set())


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "design_coverage"


@pytest.mark.parametrize("name,rows,sides", [
    ("grouped_8of56.csv", 8, {"left"}),
    ("strided_56of56.csv", 56, {"left", "right"}),
])
def test_design_coverage_on_the_real_logs_of_both_2026_09_11_captures(
        tmp_path, name, rows, sides):
    """⭐ REAL BYTES, BOTH OUTCOMES. A regression test holding only the healthy log is a
    test of the case that cannot fail, so the failing capture's own log is committed
    beside it: same driver, same source run, same day, one env var apart.

    ⚠ WHAT THE REDUCTION PRESERVES AND WHAT IT DOES NOT. Each fixture is the first row of
    each distinct (cond, left, right, correct) tuple from a 560,001-row log -- so it
    preserves the statistic under test exactly, and preserves NOTHING about visit counts,
    episode lengths or step ordering. Do not compute anything else from these files. The
    full-size originals are under ~/nett/scratch/chicken/capture/ (the failing one renamed
    UNUSABLE_8of56rows_..., with a README saying what it reproduces).
    """
    d = tmp_path / "logs"; d.mkdir()
    shutil.copy(FIXTURES / name, d / "test_fork-1_0.csv")
    assert _design_coverage(d) == (rows, sides)
