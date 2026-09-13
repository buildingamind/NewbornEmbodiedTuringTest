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


def test_capture_stream_temporal_neighbours_and_ragged_tail(monkeypatch, tmp_path):
    """Exercise main's actual capture plumbing, including its memory metadata."""
    import torch

    keys = np.array([(env, 0, step) for step in range(5) for env in (7, 19)] + [(7, 0, 5)])
    obs = np.arange(len(keys), dtype=np.float32).reshape(-1, 1, 1, 1)
    actions = np.arange(len(keys) * 2).reshape(-1, 2)
    capture = tmp_path / "capture.npz"
    np.savez(capture, obs=obs, keys=keys, actions=actions,
             n_transition_pairs=8, run_dir="unused", condition="unused")
    monkeypatch.setattr(sys, "argv", ["replay", "--capture", str(capture),
                                     "--test-csv", str(tmp_path / "test.csv"), "--seeds", "1"])
    monkeypatch.setattr(rh, "build_capture_pairs", lambda *a: ([], [], {}))
    monkeypatch.setattr(rh, "capture_readout", lambda *a: {})
    monkeypatch.setattr(rh, "build_encoder", lambda *a: torch.nn.Linear(1, 1))
    calls = []

    def train(encoder, aux_kind, stream, stream_actions, *args, **kwargs):
        row_ids = stream[..., 0, 0, 0].numpy().astype(int)
        layout = keys[row_ids]
        assert np.all(layout[1:, :, 0] == layout[:-1, :, 0]), "temporal neighbours switch env"
        assert np.all(np.diff(layout[..., 2], axis=0) == 1), "temporal step is not +1"
        assert tuple(stream.shape[:2]) == (5, 2)
        np.testing.assert_array_equal(stream_actions, actions[row_ids])
        np.testing.assert_array_equal(kwargs["keys"], layout)
        calls.append(True)
        return None, [1.0]

    monkeypatch.setattr(rh, "train_offline", train)
    rh.main()
    assert calls == [True]


def test_capture_layout_sorts_each_environment_and_preserves_gaps(capsys):
    import torch
    from nett_skrl.brain.aux.cltt_ref_aux import episode_window_starts

    keys = np.array([(8, 2, 0), (3, 0, 3), (8, 1, 1), (3, 0, 0),
                     (8, 1, 0), (3, 0, 1), (8, 2, 1)])
    obs = np.arange(len(keys)).reshape(-1, 1, 1, 1)
    stream, actions, layout = rh.reshape_capture_stream(obs, keys)
    assert stream.shape[:2] == (3, 2) and actions is None
    assert "dropped 1 incomplete tail rows" in capsys.readouterr().out
    np.testing.assert_array_equal(layout[:, 0], [(3, 0, 0), (3, 0, 1), (3, 0, 3)])
    np.testing.assert_array_equal(layout[:, 1], [(8, 1, 0), (8, 1, 1), (8, 2, 0)])
    valid = episode_window_starts(rh.ReplayMemory(stream, keys=layout), (1,))
    assert torch.equal(valid, torch.tensor([[True, True], [False, False]]))


@pytest.mark.parametrize("kind", ["vicreg", "cltt", "simclr", "gwm", "eoo", "motok"])
def test_offline_batches_train_corpus_and_register_only_head(monkeypatch, kind):
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.delenv("NETT_AUX_BATCH", raising=False)
        encoder = rh.build_encoder('compact_cnn:{"features_dim":16}', (6, 16, 16), 4)
        generator = torch.Generator().manual_seed(21)
        obs = torch.randint(0, 256, (8, 2, 6, 16, 16), dtype=torch.uint8, generator=generator)
        if kind == "motok":
            # MoTok is a screening candidate, deliberately outside AUX_LOSSES.
            from nett_skrl.brain.aux.motok_aux import MoTokAuxLoss
            monkeypatch.setattr(rh, "resolve_aux", lambda name: (MoTokAuxLoss, "screening"))
        factory, _ = rh.resolve_aux(kind)
        cls = type(factory(encoder))
        compute = cls.compute
        batches = []

        def record(self, enc, batch):
            assert batch.shape == (4, 6, 16, 16)
            assert len(torch.unique(batch.flatten(1), dim=0)) == 4
            batches.append(batch.clone())
            return compute(self, enc, batch)

        monkeypatch.setattr(cls, "compute", record)
        adam = torch.optim.Adam
        registered = []
        initial_params = []

        def optimizer(params, **kwargs):
            registered.extend(params)
            initial_params.extend(p.detach().clone() for p in params)
            assert len(params) == len({id(p) for p in params})
            return adam(params, **kwargs)

        monkeypatch.setattr(torch.optim, "Adam", optimizer)
        before = [p.detach().clone() for p in encoder.parameters()]
        aux, losses = rh.train_offline(encoder, kind, obs, None, 2, 3e-4, 4, batch=4)
        assert set(map(id, registered)) == set(map(id, encoder.parameters())) | set(map(id, aux.head.parameters()))
        assert all(np.isfinite(loss) and loss > 0 for loss in losses)
        assert any(not torch.equal(a, b) for a, b in zip(initial_params, registered))
        if kind in ("vicreg", "cltt", "simclr"):
            assert any(not torch.equal(a, b) for a, b in zip(before, encoder.parameters()))
        assert not torch.equal(batches[0], batches[1])
    finally:
        torch.set_num_threads(threads)
