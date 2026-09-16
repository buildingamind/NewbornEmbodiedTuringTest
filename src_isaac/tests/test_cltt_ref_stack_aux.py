"""The MOTION half of the 3DCNN CLTT pair: whole-stack views, and the guard they need back.

⭐ Every assertion here is about the ONE thing that differs from ``cltt_ref``. The shared machinery
(projector geometry, temperature, sampler, reset-awareness, NT-Xent) is already covered by
test_cltt_ref_aux.py and is NOT re-asserted -- a second copy of those would drift.
"""

import gymnasium as gym
import pytest
import torch

from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss
from nett_skrl.brain.aux.cltt_ref_stack_aux import CLTTReferenceStackAuxLoss
from nett_skrl.brain.encoders.compact_3dcnn import Compact3DCNN

from test_cltt_ref_aux import FakeMemory

EXPECT = 10


def _enc(num_frames=2, height=16, width=16):
    space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3 * num_frames), dtype="uint8")
    return Compact3DCNN(space, features_dim=32, num_frames=num_frames, conv_dim=8)


def _frames(t_max=24, num_envs=2, num_frames=2, height=16, width=16):
    torch.manual_seed(0)
    return torch.rand(t_max, num_envs, height * width * 3 * num_frames) * 255.0


def _loss(monkeypatch=None, offsets=None):
    if monkeypatch is not None and offsets is not None:
        monkeypatch.setenv("NETT_AUX_CLTT_STACK_OFFSETS", offsets)
    enc = _enc()
    aux = AUX_LOSSES["cltt_ref_stack"](enc)
    aux.attach_memory(FakeMemory(_frames()))
    return enc, aux


def test_it_is_registered_and_is_a_cltt_ref_subclass():
    enc = _enc()
    aux = AUX_LOSSES["cltt_ref_stack"](enc)
    assert isinstance(aux, CLTTReferenceStackAuxLoss)
    assert isinstance(aux, CLTTReferenceAuxLoss), "must inherit the shared machinery, not copy it"


def test_the_incumbent_is_untouched():
    """⛔ THE WHOLE POINT OF SUBCLASSING. Already-scored ViT-CLTT-Ref and SimCLR-CLTT-Ref arms keep
    their definition; if this ever fails, a published arm's meaning has changed retroactively."""
    assert CLTTReferenceAuxLoss.OFFSETS_ENV == "NETT_AUX_CLTT_REF_OFFSETS"
    assert CLTTReferenceAuxLoss.DEFAULT_OFFSETS == "1,2"
    assert CLTTReferenceStackAuxLoss.OFFSETS_ENV == "NETT_AUX_CLTT_STACK_OFFSETS"
    assert CLTTReferenceStackAuxLoss.DEFAULT_OFFSETS == "2,4"


def test_views_are_WHOLE_STACKS_and_carry_motion():
    """The single behavioural difference, asserted on the tensors rather than on the docstring."""
    enc, aux = _loss()
    aux.num_frames = 2
    prepared = torch.arange(2 * 6 * 4 * 4, dtype=torch.float32).reshape(2, 6, 4, 4)
    out = aux._make_views([prepared], enc)
    assert torch.equal(out[0], prepared), "whole-stack views must pass through unmodified"
    B, CT, H, W = out[0].shape
    slots = out[0].view(B, CT // 3, 3, H, W)
    assert not torch.equal(slots[:, 0], slots[:, 1]), (
        "the T slots must DIFFER -- that difference is the motion the Conv3d reads, and it is "
        "exactly what cltt_ref's repeated still frame destroys")


def test_the_static_adapter_is_what_the_incumbent_still_does():
    """Direction-symmetric: assert the OTHER branch too, or this suite only proves one side."""
    enc = _enc()
    ref = AUX_LOSSES["cltt_ref"](enc)
    prepared = torch.arange(2 * 6 * 4 * 4, dtype=torch.float32).reshape(2, 6, 4, 4)
    out = ref._make_views([prepared], enc)
    B, CT, H, W = out[0].shape
    slots = out[0].view(B, CT // 3, 3, H, W)
    assert torch.equal(slots[:, 0], slots[:, 1]), "cltt_ref must still be static"


def test_default_offsets_are_multiples_of_the_discovered_depth():
    enc, aux = _loss()
    aux.compute(enc, torch.empty(0))
    assert aux.num_frames == 2
    assert all(k % aux.num_frames == 0 for k in aux.offsets), aux.offsets


def test_it_computes_a_finite_loss_with_backbone_gradient():
    enc, aux = _loss()
    value = aux.compute(enc, torch.empty(0))
    assert torch.is_tensor(value) and torch.isfinite(value), value
    assert value.requires_grad


def test_a_shared_frame_offset_is_REFUSED(monkeypatch):
    """⛔ offset 1 at T=2 makes the two views share f(t). The guard must fire on the REALISED
    depth, which is why it lives after discovery rather than in __init__."""
    enc, aux = _loss(monkeypatch, "1")
    with pytest.raises(ValueError, match="share a literally identical frame"):
        aux.compute(enc, torch.empty(0))


def test_the_refusal_names_a_workable_offset(monkeypatch):
    """A refusal that does not say what to do instead gets worked around, not fixed."""
    enc, aux = _loss(monkeypatch, "3")
    with pytest.raises(ValueError) as exc:
        aux.compute(enc, torch.empty(0))
    assert "2,4" in str(exc.value), str(exc.value)


def test_the_registry_entry_is_one_factor_from_plain_3dcnn():
    """⛔ THE ARM'S WHOLE VALUE IS THAT `aux` IS THE ONLY DIFFERENCE. If cfg ever drifts, the
    contrast silently becomes 'adds the aux AND something else' and no result would reveal it."""
    from examples.campaign_train import MODELS
    stack, ref, base = MODELS["3DCNN-CLTT-Stack"], MODELS["3DCNN-CLTT-Ref"], MODELS["3DCNN"]
    assert stack["aux"] == "cltt_ref_stack" and ref["aux"] == "cltt_ref"
    assert base.get("aux") is None
    for field in ("encoder", "cfg", "framestack"):
        assert stack[field] == base[field] == ref[field], field
    assert stack["aux_weight"] == ref["aux_weight"] == 1.0


def test_case_count():
    """⛔ A suite that does not count itself reports green over cases it no longer runs."""
    import test_cltt_ref_stack_aux as m
    ran = len([n for n in dir(m) if n.startswith("test_")])
    assert ran == EXPECT, (
        f"RAN {ran} CASES, EXPECTED {EXPECT} -- a case was lost, or added without updating "
        f"EXPECT. Write {ran}.")
