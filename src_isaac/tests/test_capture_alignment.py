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
