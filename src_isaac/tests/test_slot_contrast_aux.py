"""CPU tests for CNN-SlotContrast (`nett_skrl.brain.aux.slot_contrast_aux`).

PROMOTED out of `examples/candidates/` on 2026-09-15 and registered as `slot_contrast`, so
these tests now guard a LAUNCHABLE objective rather than a screening candidate.
Two failure modes dominate and both are silent:

  * reading a POOLED global vector instead of the pre-pool feature map -- slot attention
    still runs, still trains, still reports a loss, for a method that no longer has spatial
    positions to compete over;
  * the degenerate optimum the reference is protected from by a frozen pretrained DINOv2 and
    we are not -- constant slots drive the contrastive term to ~0 with nothing segmented,
    and the loss falls the whole way.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]

from nett_skrl.brain.aux import slot_contrast_aux as slotc  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402


def _encoder(channels=3, h=80, w=128):
    space = gym.spaces.Box(low=0, high=255, shape=(h, w, channels), dtype="uint8")
    return encoder_mapping["nature_cnn"](space)


class _FakeMemory:
    """A rollout buffer the episode-contiguity helpers can actually read.

    ⛔ `terminated`/`truncated` are NOT decoration. Since promotion the sampler refuses to
    draw a window that crosses an episode reset, and it discovers resets from these two
    tensors (or from replay `keys`). A double that omitted them would make every test here
    exercise the raise-path instead of the loss -- passing, while measuring nothing.
    Pass `done_at` to place a boundary and check the sampler actually honours it.
    """

    def __init__(self, obs, done_at=None):
        t, envs = obs.shape[0], obs.shape[1]
        terminated = torch.zeros(t, envs, 1, dtype=torch.bool)
        if done_at is not None:
            terminated[done_at, :, 0] = True
        self.tensors = {"observations": obs,
                        "terminated": terminated,
                        "truncated": torch.zeros(t, envs, 1, dtype=torch.bool)}
        self.memory_size = obs.shape[0]
        self.memory_index = obs.shape[0]
        self.filled = True


def _obs(n=6, envs=2, h=80, w=128, c=3):
    return torch.randint(0, 255, (n, envs, h, w, c), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# 1. The feature map is read before the pool -- or not at all
# ---------------------------------------------------------------------------


def test_features_keep_their_geometry():
    enc = _encoder()
    feats = slotc.spatial_features(enc, torch.zeros(2, 3, 80, 128))
    assert feats.dim() == 4, "a pooled/flattened tensor has no positions to attend over"
    assert feats.shape[0] == 2 and feats.shape[2] > 1 and feats.shape[3] > 1


def test_an_encoder_without_a_conv_trunk_is_refused_not_substituted():
    """Falling back to encoder(obs) would produce slots over a single global vector."""
    class NoTrunk(nn.Module):
        observation_space = gym.spaces.Box(low=0, high=255, shape=(80, 128, 3), dtype="uint8")

        def forward(self, x):
            return torch.zeros(x.shape[0], 128)

    with pytest.raises(TypeError, match="different method"):
        slotc.spatial_features(NoTrunk(), torch.zeros(1, 3, 80, 128))


def test_a_trunk_with_no_pool_boundary_is_refused():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())

    with pytest.raises(TypeError, match="no pool/flatten boundary"):
        slotc.spatial_features(Odd(), torch.zeros(1, 3, 16, 16))


def test_the_cut_is_located_by_type_not_by_index():
    """An index would survive a reorder of the trunk and read the wrong tensor silently."""
    class Reordered(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),   # one extra stage
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            )

    feats = slotc.spatial_features(Reordered(), torch.zeros(1, 3, 80, 128))
    assert feats.shape[1] == 16 and feats.shape[2:] == (20, 32)


# ---------------------------------------------------------------------------
# 2. What is and is not in the parameter set
# ---------------------------------------------------------------------------


def test_the_encoder_is_not_in_the_aux_parameter_set():
    """The harness optimises list(encoder.parameters()) + list(aux.parameters()).

    An encoder held as a plain attribute of an nn.Module is REGISTERED, which would hand
    Adam the same tensors twice. The first version of the candidate did exactly that and
    reported its own size as 803,297 instead of 137,025.
    """
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    enc_ids = {id(p) for p in enc.parameters()}
    assert not (enc_ids & {id(p) for p in aux.parameters()})


def test_the_ema_target_is_not_in_the_aux_parameter_set():
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    aux._update_target(enc)                       # materialises the copy
    assert aux._target, "target should exist after the first update"
    tgt_ids = {id(p) for p in aux._target[0].parameters()}
    assert not (tgt_ids & {id(p) for p in aux.parameters()})


def test_slot_count_does_not_change_the_parameter_count():
    """Slot attention shares weights across slots, so K is free to pick on the scene."""
    enc = _encoder()
    sizes = {k: sum(p.numel() for p in slotc.SlotContrastAuxLoss(enc, slots=k).parameters())
             for k in (4, 6, 8)}
    assert len(set(sizes.values())) == 1, sizes


# ---------------------------------------------------------------------------
# 3. The EMA target is a target, not a free variable
# ---------------------------------------------------------------------------


def test_the_target_receives_no_gradient():
    """The whole port turns on this. A target the loss can move gives the trivial optimum
    gwm_aux.py records: it trains, converges, and reports a LOWER loss than the correct
    version."""
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))
    aux.compute(enc, obs[0]).backward()
    assert aux._target
    assert all(p.grad is None and not p.requires_grad for p in aux._target[0].parameters())


def test_the_target_tracks_the_trunk_but_lags_it():
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc, ema=0.5)
    aux._update_target(enc)
    before = next(aux._target[0].parameters()).clone()
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(1.0)
    aux._update_target(enc)
    after = next(aux._target[0].parameters())
    live = next(enc.parameters())
    assert not torch.allclose(after, before), "the target must move toward the trunk"
    assert not torch.allclose(after, live), "and must not arrive in one step"


def test_the_no_detach_ablation_is_reachable_by_env_knob(monkeypatch):
    """The pre-registered ablation must not require editing the file that gets shipped."""
    monkeypatch.setenv("NETT_SLOTC_NO_DETACH", "1")
    assert slotc.SlotContrastAuxLoss(_encoder()).no_detach is True
    monkeypatch.setenv("NETT_SLOTC_NO_DETACH", "0")
    assert slotc.SlotContrastAuxLoss(_encoder()).no_detach is False


def test_every_boolean_knob_accepts_the_fleet_spellings(monkeypatch):
    for spelling in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("NETT_SLOTC_DIAG", spelling)
        assert slotc.SlotContrastAuxLoss(_encoder()).diag is True, spelling
    for spelling in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("NETT_SLOTC_DIAG", spelling)
        assert slotc.SlotContrastAuxLoss(_encoder()).diag is False, spelling


# ---------------------------------------------------------------------------
# 4. The degeneracy diagnostic -- the check the reference does not need
# ---------------------------------------------------------------------------


def test_collapsed_slots_show_near_zero_mask_variance(monkeypatch):
    """If slots ignore the image, their masks stop depending on the input -- while the
    contrastive term goes to ~0 and the loss reads as progress."""
    monkeypatch.setenv("NETT_SLOTC_DIAG", "1")
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))

    # Real inputs: masks differ across the batch.
    aux.compute(enc, obs[0])
    live = aux.last_mask_variance
    assert live > 0.0

    # Collapse it by hand: zero the key/value projections so attention logits are
    # input-independent, which is the failure the diagnostic exists to see.
    with torch.no_grad():
        aux.attn.to_k.weight.zero_()
        aux.attn.to_v.weight.zero_()
    aux.compute(enc, obs[0])
    assert aux.last_mask_variance < live / 100, (aux.last_mask_variance, live)


def test_the_diagnostic_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("NETT_SLOTC_DIAG", raising=False)
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))
    aux.compute(enc, obs[0])
    assert aux.last_mask_variance == -1.0, "sentinel, not a measured zero"


# ---------------------------------------------------------------------------
# 5. It trains, and it says so when it cannot
# ---------------------------------------------------------------------------


def test_the_loss_reaches_the_trunk():
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))
    aux.compute(enc, obs[0]).backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "no gradient reached the encoder"


def test_no_memory_raises_rather_than_scoring_zero():
    """⛔ THE CANDIDATE RETURNED 0.0 HERE AND THAT WAS THE DEFECT.

    A zero enters the caller's running mean as a real value, so "there was no usable
    (t, t+1) window" and "the objective scored 0" become the same series -- and a loss
    sitting at zero reads as converged, not as never-ran. Promotion changed it to a raise.
    """
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    with pytest.raises(RuntimeError, match="attach_memory"):
        aux.compute(enc, _obs()[0])
    assert aux.last_terms is None

    aux.attach_memory(_FakeMemory(_obs()))
    aux.compute(enc, _obs()[0])
    assert aux.last_terms is not None and len(aux.last_terms) == 3


def test_the_temporal_pair_never_crosses_an_episode_reset():
    """The pair must come from ONE episode. A pair spanning a reset shows a teleport.

    With a boundary after step 2 of a 6-step buffer, every returned (t, t+1) pair must sit
    wholly inside [0,2] or wholly inside [3,5]. The candidate's bare `randint` could return
    (2,3) -- the last frame of one episode against the first of the next -- and nothing
    downstream could tell, because a teleport is a perfectly well-formed image pair.
    """
    enc = _encoder()
    obs = _obs(n=6)
    aux = slotc.SlotContrastAuxLoss(enc)
    aux.attach_memory(_FakeMemory(obs, done_at=2))
    mem = aux._memory
    for _ in range(64):
        a, b = aux._temporal_pair(enc, obs[0])
        # Recover the slab by identity against the buffer rather than trusting an index.
        starts = [t for t in range(obs.shape[0] - len(a) + 1)
                  for e in range(obs.shape[1])
                  if torch.equal(obs[t:t + len(a), e], a)]
        assert starts, "returned slab is not a contiguous slice of the buffer"
        for t0 in starts:
            assert not (t0 <= 2 < t0 + len(a)), (
                f"slab [{t0}, {t0 + len(a)}) straddles the reset after step 2")


def test_it_declares_that_it_draws_its_own_windows():
    assert slotc.SlotContrastAuxLoss.needs_memory is True


def test_the_temporal_pair_is_moved_to_the_encoders_device():
    """⛔ campaign_train forces NETT_MEMORY_DEVICE=cpu for EVERY framestacked arm, and this
    loss is only ever declared on framestacked arms. So the buffer slice is on the CPU while
    the encoder may be on the GPU. The candidate returned the slice untouched and would have
    raised "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be
    the same" on the first aux call of every real arm. Device-agnostic form of the check: the
    pair must come back on the encoder's device whatever device the buffer is on.
    """
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    aux.attach_memory(_FakeMemory(_obs()))
    want = next(enc.parameters()).device
    a, b = aux._temporal_pair(enc, _obs()[0])
    assert a.device == want and b.device == want, (
        f"pair came back on {a.device}/{b.device}, encoder is on {want}")
