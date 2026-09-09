"""Offline replay harness: framestacking, the readout rule, and its guards.

⛔ WHY THE GUARDS MATTER MORE THAN THE MATH HERE. This harness is designed to be
CHEAP, which means it will be run often and its numbers quoted casually. Three
ways it can produce a confident wrong answer, all silent:

  * building a DEFAULT-configured encoder because the registry key was misread --
    caught in development: `spec.get("encoder_kwargs")` returned {} and built a
    509,312-parameter ViViT where the fleet's is 694,016. It did not raise;
  * reporting the trained readout WITHOUT the untrained baseline, which credits
    the objective for whatever raw pixels already gave away. Measured on the
    fixture: an untrained encoder scores 0.59 on one background and 0.20 on
    another, neither near chance;
  * training on a capture that contains no transitions, or a biased prefix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "examples"))

import replay_harness as rh


# --- framestacking ---------------------------------------------------------

def test_framestack_depth_one_is_identity():
    f = np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)
    assert np.array_equal(rh.framestack(f, 1), f)


def test_framestack_concatenates_consecutive_frames_on_the_channel_axis():
    f = np.zeros((5, 2, 2, 3), dtype=np.uint8)
    for i in range(5):
        f[i] = i
    out = rh.framestack(f, 2)
    assert out.shape == (4, 2, 2, 6)
    # row i must hold frame i in the first 3 channels and frame i+1 in the last 3
    for i in range(4):
        assert (out[i, ..., :3] == i).all()
        assert (out[i, ..., 3:] == i + 1).all()


def test_framestack_shortens_the_stream_by_depth_minus_one():
    f = np.zeros((10, 1, 1, 3), dtype=np.uint8)
    assert len(rh.framestack(f, 3)) == 8


# --- the readout rule ------------------------------------------------------

class _ToyEncoder:
    """Encodes a frame to its mean per channel. Enough to exercise the rule."""

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def _prepare_image(self, x):
        import torch
        return torch.as_tensor(x).float()

    def encode_prepared(self, x):
        return x.reshape(x.shape[0], -1)


def _frames(value, n=4, c=3):
    return np.full((n, 2, 2, c), value, dtype=np.float32)


def test_readout_picks_the_member_closer_to_the_training_memory():
    """The imprinting rule: whichever pair member looks more like what was reared."""
    enc = _ToyEncoder()
    train = _frames(1.0)
    # target identical to training, distractor orthogonal in sign
    pairs = [(_frames(1.0), -_frames(1.0), "target is familiar")]
    out = rh.familiarity_readout(enc, train, pairs)
    assert out["target is familiar"] == pytest.approx(1.0)


def test_readout_is_reversed_when_the_distractor_is_the_familiar_one():
    enc = _ToyEncoder()
    train = _frames(1.0)
    pairs = [(-_frames(1.0), _frames(1.0), "distractor is familiar")]
    out = rh.familiarity_readout(enc, train, pairs)
    assert out["distractor is familiar"] == pytest.approx(0.0)


def test_readout_uses_no_test_labels_and_restores_training_mode():
    """A supervised probe would answer an easier question than the chicks are asked."""
    enc = _ToyEncoder()
    enc.train()
    rh.familiarity_readout(enc, _frames(1.0), [(_frames(1.0), _frames(0.5), "x")])
    assert enc.training is True, "readout must leave the encoder as it found it"


def test_readout_compares_equal_numbers_of_frames_from_each_member():
    """Clips differ in length; an unequal contest would weight one member more."""
    enc = _ToyEncoder()
    out = rh.familiarity_readout(
        enc, _frames(1.0), [(_frames(1.0, n=9), -_frames(1.0, n=3), "ragged")])
    assert out["ragged"] == pytest.approx(1.0)


# --- registry guard --------------------------------------------------------

def test_build_encoder_refuses_a_spec_without_cfg(monkeypatch):
    """Reading the wrong key silently builds a different model. It must raise."""
    import campaign_train
    monkeypatch.setitem(campaign_train.MODELS, "_toy", {"encoder": "compact_vivit"})
    with pytest.raises(SystemExit, match="no 'cfg'"):
        rh.build_encoder("_toy", (80, 128, 6), seed=0)


def test_build_encoder_applies_the_registry_cfg():
    """ViViT at the live eye is 694,016 parameters; a default build was 509,312."""
    enc = rh.build_encoder("ViViT", (80, 128, 6), seed=0)
    assert sum(p.numel() for p in enc.parameters()) == 694_016
