"""CPU tests for the parameter-free expert optical flow and the dorsal factory.

Two defects motivate this file and neither could be caught by a shape or parameter check:

  * **A SIGN INVERSION.** A true +4 px shift was recovered as -4. Magnitudes and spatial
    structure are identical either way, so every summary statistic the wrapper logs --
    flow_absmax, the spatial-std ratio, the reconstruction loss -- reads exactly the same
    with the direction of motion reversed. Only a KNOWN translation separates them.
  * **A HALF-RIGHT DROP-IN.** ``ExpertBlockFlow.forward`` was aliased to ``forward_single``.
    ``Small3DCNNDorsal`` deliberately has two contracts: ``forward_single`` returns one
    tensor (GWM / gwm_seg) and ``forward`` returns ``(F_fwd, F_bwd)`` (EoO). The alias made
    it a drop-in for one caller and a silent mis-shape for the other.
"""
from __future__ import annotations

import torch

from nett_skrl.brain.aux.dual_stream import Small3DCNNDorsal, make_dorsal
from nett_skrl.brain.aux.expert_flow import ExpertBlockFlow


def _scene(h=48, w=64, seed=0):
    """A textured frame. Block matching needs texture; a flat field has no correspondence."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g)


def _interior(t, margin=16):
    """Score away from the border, where torch.roll's wrapped band is not a translation."""
    return t[..., margin:-margin, margin:-margin]


def test_a_known_translation_is_recovered_with_the_RIGHT_SIGN():
    """⛔ THE REGRESSION TEST FOR THE SIGN BUG. Direction, not just magnitude."""
    flow = ExpertBlockFlow(radius=12, stride=1, patch=8)
    frame = _scene()
    for dx, dy in [(4, 0), (8, 0), (-6, 0), (0, 4), (0, -4), (0, 0)]:
        moved = torch.roll(frame, shifts=(dy, dx), dims=(2, 3))
        out = flow.forward_single(frame, moved)
        got_dx = float(_interior(out[:, 0]).median())
        got_dy = float(_interior(out[:, 1]).median())
        assert abs(got_dx - dx) <= 1.0, f"dx {dx}: recovered {got_dx}"
        assert abs(got_dy - dy) <= 1.0, f"dy {dy}: recovered {got_dy}"


def test_forward_matches_the_learned_dorsals_two_tuple_contract():
    """EoO unpacks ``forward`` into (F_fwd, F_bwd); GWM calls ``forward_single``."""
    frame_t, frame_t1 = _scene(seed=1), _scene(seed=2)

    learned = Small3DCNNDorsal()
    ref_fwd, ref_bwd = learned(frame_t, frame_t1)
    ref_single = learned.forward_single(frame_t, frame_t1)

    expert = ExpertBlockFlow()
    out = expert(frame_t, frame_t1)
    assert isinstance(out, tuple) and len(out) == 2, (
        "forward must return (F_fwd, F_bwd) -- the alias to forward_single returned a "
        "single tensor, which EoO would have unpacked along its BATCH axis")
    fwd, bwd = out
    assert fwd.shape == ref_fwd.shape and bwd.shape == ref_bwd.shape
    assert expert.forward_single(frame_t, frame_t1).shape == ref_single.shape


def test_the_backward_flow_is_a_second_search_not_a_negation():
    """Occlusion makes the two directions genuinely differ -- which is what EoO's
    consistency term reads. A backward flow computed as -forward would make that term
    identically zero and the check vacuous."""
    expert = ExpertBlockFlow(radius=8, stride=1, patch=8)
    frame_t = _scene(seed=3)
    frame_t1 = torch.roll(frame_t, shifts=(0, 5), dims=(2, 3))
    frame_t1[..., 10:20, 10:20] = 0.0          # an occluder present only at t+1
    fwd, bwd = expert(frame_t, frame_t1)
    assert not torch.allclose(bwd, -fwd, atol=1e-3)


def test_it_has_no_parameters_at_all():
    """The whole compliance argument: an ALGORITHM, not frozen pretrained weights."""
    assert list(ExpertBlockFlow().parameters()) == []


def test_the_factory_swaps_only_when_asked(monkeypatch):
    monkeypatch.delenv("NETT_EXPERT_FLOW", raising=False)
    assert isinstance(make_dorsal(), Small3DCNNDorsal)
    for spelling in ("1", "true", "YES", "on"):
        monkeypatch.setenv("NETT_EXPERT_FLOW", spelling)
        assert isinstance(make_dorsal(), ExpertBlockFlow), spelling
    monkeypatch.setenv("NETT_EXPERT_FLOW", "0")
    assert isinstance(make_dorsal(), Small3DCNNDorsal)
