"""`current_frame_stack` under a non-RGB per-frame channel count.

⚠ The point of the change under test is that a body wrapper can make one frame something
other than 3 channels (`dvs_polarity` emits an ON/OFF PAIR). The load-bearing property is
therefore NOT "the new knob works" -- it is "nothing moved for the arms that never set it",
pinned below against a verbatim copy of the pre-change implementation.
"""

from __future__ import annotations

import pytest
import torch

from nett_skrl.brain.aux.cltt_views import (
    DEFAULT_CHANNELS_PER_FRAME,
    current_frame_stack,
    resolve_channels_per_frame,
)


def _legacy(prepared: torch.Tensor) -> torch.Tensor:
    """The implementation as it stood before per-frame channels became a parameter."""
    channels = prepared.shape[1]
    if channels < 3 or channels % 3:
        raise ValueError(f"CLTT requires T-major RGB channels, got {channels}")
    return prepared[:, -3:].repeat(1, channels // 3, *([1] * (prepared.ndim - 2)))


def _ramp(c, b=2, h=4, w=5):
    """Channel-distinguishable input: every channel is a different constant."""
    return torch.arange(c, dtype=torch.float32).view(1, c, 1, 1).expand(b, c, h, w).contiguous()


# ---------------------------------------------------------------- no regression

@pytest.mark.parametrize("c", [3, 6, 9, 12])
def test_rgb_default_matches_the_pre_change_implementation(c):
    x = _ramp(c)
    assert torch.equal(current_frame_stack(x), _legacy(x))


@pytest.mark.parametrize("c", [1, 2, 4, 5, 7, 8])
def test_non_multiples_of_three_still_raise_by_default(c):
    x = _ramp(c)
    with pytest.raises(ValueError):
        _legacy(x)
    with pytest.raises(ValueError, match="CHANNELS_PER_FRAME"):
        current_frame_stack(x)


def test_default_is_three():
    assert DEFAULT_CHANNELS_PER_FRAME == 3
    assert resolve_channels_per_frame() == 3


# ---------------------------------------------------------------- the two-channel path

def test_four_channels_as_two_frames_of_two():
    x = _ramp(4)                                   # channels 0,1 | 2,3  (oldest | current)
    out = current_frame_stack(x, channels_per_frame=2)
    assert out.shape == x.shape
    # ⛔ The CURRENT frame -- the LAST cpf channels -- repeated. Taking the first would make
    # the view the OLDEST frame and silently shift every offset by one stack depth.
    assert out[:, :, 0, 0].tolist() == [[2.0, 3.0, 2.0, 3.0]] * x.shape[0]


def test_two_channels_unstacked_is_the_identity():
    x = _ramp(2)
    assert torch.equal(current_frame_stack(x, channels_per_frame=2), x)


def test_six_channels_of_two_repeats_three_times():
    out = current_frame_stack(_ramp(6), channels_per_frame=2)
    assert out[:, :, 0, 0][0].tolist() == [4.0, 5.0, 4.0, 5.0, 4.0, 5.0]


def test_odd_channel_count_under_cpf_two_raises():
    with pytest.raises(ValueError, match="got 5"):
        current_frame_stack(_ramp(5), channels_per_frame=2)


def test_gradients_flow_through_the_view():
    x = _ramp(4).requires_grad_(True)
    current_frame_stack(x, channels_per_frame=2).sum().backward()
    # Only the CURRENT frame is selected, so the oldest frame must receive NO gradient.
    assert x.grad[:, :2].abs().sum() == 0
    assert x.grad[:, 2:].abs().sum() > 0


# ---------------------------------------------------------------- the knob

def test_env_var_is_read(monkeypatch):
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "2")
    assert resolve_channels_per_frame() == 2
    assert current_frame_stack(_ramp(4)).shape == (2, 4, 4, 5)


def test_explicit_argument_beats_the_env_var(monkeypatch):
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "2")
    assert resolve_channels_per_frame(3) == 3


def test_empty_env_var_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "")
    assert resolve_channels_per_frame() == 3


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_non_positive_knob_is_rejected(monkeypatch, raw):
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", raw)
    with pytest.raises(ValueError, match=">= 1"):
        resolve_channels_per_frame()


def test_non_integer_knob_is_rejected(monkeypatch):
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "two")
    with pytest.raises(ValueError, match="must be an integer"):
        resolve_channels_per_frame()


def test_the_aux_reads_stack_depth_through_the_same_knob(monkeypatch):
    # ⛔ `cltt_ref_aux` derived num_frames as `shape[1] // 3` and REPORTS it in its one-time
    # log line as the arm's stack depth. Under cpf=2 that line would have published 4//3 = 1.
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "2")
    from nett_skrl.brain.aux import cltt_ref_aux
    assert 4 // cltt_ref_aux.resolve_channels_per_frame() == 2
