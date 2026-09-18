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

import random
import statistics
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


def test_slot_count_changes_the_parameter_count_by_exactly_the_per_slot_init():
    """⛔ INVERTED 2026-09-18, AND DELIBERATELY NOT DELETED. This asserted the OPPOSITE --
    "K does not change the parameter count" -- which was true only because `mu`/`log_sigma`
    were `(1, 1, slot_dim)`: ONE distribution broadcast to every slot. That is the defect that
    made identical slots an absorbing state (softmax over slots + identical slots => uniform
    attention => identical update), and it is now fixed to `(1, K, slot_dim)`.

    ⭐ THE INVERTED TEST IS STRICTLY STRONGER THAN THE ONE IT REPLACES. The old form passed for
    ANY K-independent parameterisation, including the broken one; this pins the slope to the
    exact per-slot cost, so a future revert to a shared init FAILS here rather than silently
    restoring the collapse. A test that merely allowed the new count would not have done that.
    """
    enc = _encoder()
    sizes = {k: sum(p.numel() for p in slotc.SlotContrastAuxLoss(enc, slots=k).parameters())
             for k in (4, 6, 8)}
    slot_dim = slotc.SlotContrastAuxLoss(enc, slots=4).slot_dim
    per_slot = 2 * slot_dim                       # mu and log_sigma, one row each per slot
    assert sizes[6] - sizes[4] == 2 * per_slot, sizes
    assert sizes[8] - sizes[6] == 2 * per_slot, sizes
    assert len(set(sizes.values())) == 3, sizes


def test_slots_do_not_collapse_when_the_init_noise_vanishes():
    """⛔⛔ THE REGRESSION TEST FOR THE ABSORBING STATE, and the reason the fix exists.

    `attn = logits.softmax(dim=1)` is a softmax OVER SLOTS, so identical slots give every
    position a uniform 1/K attention and an identical GRU update: once the slots coincide they
    can never separate. With a SHARED `mu`, the only thing that ever distinguished slot k from
    slot j was the random draw -- so the partition decays to nothing as `log_sigma` trains down.

    Measured on chicken with the shared init, across-slot variance of the attention map:
        log_sigma  0.0 -> 5.02e-03   -6.0 -> 4.32e-06   -12.0 -> 2.68e-11   (i.e. one slot)
    and with the per-slot init it PLATEAUS at ~4.7e-04 instead of decaying.

    ⚠ The assertion is about the LIMIT, not about a magnitude: driving `log_sigma` to -12 makes
    the init noise negligible, so anything left is carried by the learned per-slot means. A
    shared init scores ~0 here by construction, which is what makes this discriminating.
    """
    import torch
    torch.manual_seed(1)
    sa = slotc.SlotAttention(64, 64, 4)
    with torch.no_grad():
        sa.log_sigma.fill_(-12.0)                 # init noise ~ 0: only learned means remain
    torch.manual_seed(0)
    with torch.no_grad():
        _, attn = sa(torch.randn(8, 40, 64))
    across_slot = float(attn.var(dim=1).mean())
    assert across_slot > 1e-5, (
        f"slots collapsed to one: across-slot attention variance {across_slot:.3e}. "
        "This is what a shared (1, 1, slot_dim) init produces (~1e-11).")


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
        aux.head.attn.to_k.weight.zero_()
        aux.head.attn.to_v.weight.zero_()
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


# -- the `head` interface ppo_aux reaches for ------------------------------------------
#
# ⛔ THESE GUARD A BUILD DEFECT THAT COST ROW 14 OF THE 14-ARM WAVE. `slot_contrast` was
# promoted into AUX_LOSSES without a `head`, and ppo_aux reaches for `aux.head` at four
# sites -- checkpoint_modules (:235), the optimiser param group (:252) and both
# grad-norm-clip chains (:422, :425). Only the first is guarded by aux kind, so the arm
# died with AttributeError ~60s in, before a training step. The tests below fix the
# instance AND sweep the class.


def _ppo_aux_head_sites(aux, enc):
    """Exactly what ppo_aux does with `aux.head`, in the order it does it."""
    import itertools
    return {
        ":235 checkpoint_modules": lambda: aux.head,
        ":252 optimizer group": lambda: list(aux.head.parameters()),
        ":422 grad-clip shared": lambda: list(itertools.chain(enc.parameters(), aux.head.parameters())),
        ":425 grad-clip split": lambda: list(itertools.chain(enc.parameters(), aux.head.parameters())),
    }


def test_every_ppo_aux_head_site_resolves():
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    for name, site in _ppo_aux_head_sites(aux, enc).items():
        site()  # AttributeError here is the build defect, at the site that raises it


def test_the_optimiser_group_covers_the_whole_trainable_set():
    """⛔ THE CONTROL THAT FAILS FOR `self.head = self.attn`.

    That one-liner satisfies all four sites and builds, trains and files results with
    78,081 decoder + 4,608 positional params receiving no optimiser group and no grad
    clip -- an arm that is a partial ablation and reports as the method. Construction
    succeeding is NOT the property under test; coverage of the trainable set is.
    """
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    opt = torch.optim.Adam(list(enc.parameters()), lr=1e-4)
    opt.add_param_group({"params": list(aux.head.parameters())})   # ppo_aux.py:252, verbatim

    optimised = {id(p) for g in opt.param_groups for p in g["params"]}
    assert {id(p) for p in aux.parameters()} <= optimised, "a trainable tensor is in no param group"
    for name, mod in (("decoder", aux.head.decoder), ("attn", aux.head.attn)):
        missing = [n for n, p in mod.named_parameters() if id(p) not in optimised]
        assert not missing, f"{name}: {missing} never reaches the optimiser"
    assert id(aux.head.pos) in optimised, "the positional grid never reaches the optimiser"


def test_the_head_carries_the_encoder_neither_directly_nor_through_the_ema_target():
    """The harness optimises list(encoder.parameters()) + list(aux.parameters()); a head
    that reached either would hand Adam the same tensors twice."""
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    head_ids = {id(p) for p in aux.head.parameters()}
    assert not (head_ids & {id(p) for p in enc.parameters()})
    aux._update_target(enc)
    assert aux._target, "control is vacuous unless the target actually exists"
    assert not (head_ids & {id(p) for p in aux._target[0].parameters()})


def test_the_head_holds_every_trainable_tensor_the_module_has():
    """`aux.head.parameters()` and `aux.parameters()` must be the SAME set: ppo_aux
    optimises the former, the promotion note counts the latter."""
    aux = slotc.SlotContrastAuxLoss(_encoder())
    assert {id(p) for p in aux.head.parameters()} == {id(p) for p in aux.parameters()}
    # ⛔ 137_025 until 2026-09-18. `mu`/`log_sigma` went from (1, 1, slot_dim) to (1, K, slot_dim),
    # which adds 2 * (K - 1) * slot_dim = 2 * 5 * 64 = 640 at the default K=6. The literal is
    # UPDATED rather than loosened: this number is what the promotion note counts, so a test that
    # stopped pinning it exactly would stop catching the thing it exists to catch.
    assert sum(p.numel() for p in aux.head.parameters()) == 137_665


def test_the_whole_trainable_set_lands_on_the_encoders_device():
    """⛔ THE DEFECT CPU TESTS CANNOT SEE. `pos` was built by nn.Parameter(torch.zeros(...))
    with no `.to(device)`, and nothing calls `aux.to(device)` -- `_build_slot_contrast`
    returns the module as-is. On a GPU `_decode` added a cuda `slots` to a cpu `pos`.
    Asserted against the encoder's device so it is a real check on either host.
    """
    enc = _encoder()
    if torch.cuda.is_available():
        enc = enc.cuda()
    aux = slotc.SlotContrastAuxLoss(enc)
    want = next(enc.parameters()).device
    off = [n for n, p in aux.head.named_parameters() if p.device != want]
    assert not off, f"{off} are not on {want}"


def test_every_registered_aux_kind_exposes_a_head():
    """⛔ FIX THE INSTANCE, SWEEP THE CLASS. ppo_aux:252 and :422/:425 are NOT guarded by
    kind, so any registered loss without a `head` dies exactly the way slot_contrast did.

    Built, not grepped: a `self.head = ...` line in the source proves nothing about the
    object the builder returns (slot_contrast had three such lines and no head). The
    A sweep that silently covers 7 of 11 kinds reports as a class check and is an
    instance check, so each kind gets the host shape it actually declares:

      * eoo / gwm read the framestack and refuse a 3-channel space;
      * eoo_dual / gwm_dual demand a >=6-channel SPACE but probe the host with a
        3-channel tensor (dual_stream.py:219) -- the host consumes single frames
        spatially while the stack declares two. That is production's shape, not this
        test's invention: I74/I75 built and trained on insect, so that probe resolved
        against the real encoder.
    """
    from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
    import gymnasium as gym

    def _framestacked_host():
        """Convs sized for one RGB frame; observation_space declaring two."""
        enc = _encoder(channels=3)
        enc.observation_space = gym.spaces.Box(low=0, high=255, shape=(80, 128, 6), dtype="uint8")
        return enc

    missing, unbuilt = [], []
    def _vit_host():
        """A TOKEN host. The wave-17 kinds (cltt_ref + a token term) REFUSE an encoder with no
        token path by design -- `token_features.spatial_tokens` raises rather than reshaping a
        pooled vector -- so a CNN-only sweep would report them as unbuildable and stop checking
        them. Their real rows all run `compact_vit`."""
        space = gym.spaces.Box(low=0, high=255, shape=(80, 128, 6), dtype="uint8")
        return encoder_mapping["compact_vit"](
            space, features_dim=32, patch_size=16, embed_dim=32, depth=1, num_heads=2)

    for kind, build in sorted(AUX_LOSSES.items()):
        aux = None
        for host in (lambda: _encoder(channels=3), lambda: _encoder(channels=6),
                     _framestacked_host, _vit_host):
            try:
                aux = build(host())
                break
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
        if aux is None:
            unbuilt.append(f"{kind} ({last})")
        elif not hasattr(aux, "head"):
            missing.append(kind)

    assert not missing, f"registered aux kinds with no .head: {missing}"
    assert not unbuilt, (
        "the sweep could not construct these, so it did NOT check them: " + "; ".join(unbuilt)
    )
    assert len(AUX_LOSSES) >= 11, f"registry shrank to {len(AUX_LOSSES)}; is this sweep still real?"


def test_the_bare_mask_variance_cannot_GRADE_input_dependence_at_init():
    """⛔ THE CONFOUND THE NULL EXISTS FOR, AND IT IS NARROWER THAN "the measure is broken".

    `test_collapsed_slots_show_near_zero_mask_variance` above is still true: zeroing to_k/to_v
    makes every logit 0, attention uniform and identical for every batch element, so the bare
    variance collapses. The bare measure DOES detect the total-collapse ENDPOINT.

    What it cannot do is GRADE input dependence, because `SlotAttention` draws
    `mu + log_sigma.exp() * randn(b, K, D)` -- a random slot init PER BATCH ELEMENT -- so
    across-batch variance has two sources:

        input dependence (the signal)  and  slot-init noise (which SHRINKS as log_sigma trains)

    The decisive input is a batch of IDENTICAL images: zero input-dependent variance by
    construction, so whatever the bare statistic reports there is pure noise. At fresh init it
    reports the SAME ORDER as on distinct images -- i.e. no discriminating power at all.

    ⇒ That is why a FALLING mask_variance early in training cannot be read as degeneracy: it is
    also exactly what the noise term shrinking looks like. The insect seat measured that fall in
    7/7 brains over 4 updates and correctly declined to call it degeneracy; this test records the
    instrument-level reason, which is a defect in this diagnostic rather than a limit on their
    reading.
    """
    import torch
    from nett_skrl.brain.aux.slot_contrast_aux import SlotAttention

    torch.manual_seed(0)
    B, K, D, N = 32, 4, 64, 72
    sa = SlotAttention(in_dim=D, slot_dim=D, slots=K).eval()

    torch.manual_seed(1)
    distinct = torch.randn(B, N, D)
    identical = distinct[:1].expand(B, N, D).contiguous()

    def bare(x):
        with torch.no_grad():
            return float(sa(x)[1].var(dim=0).mean())

    v_distinct, v_identical = bare(distinct), bare(identical)

    # Identical inputs have no input-dependent variance, yet the bare statistic reports a value
    # of the same order -- so the bare number is not a measure of input dependence at init.
    assert v_identical > 0.5 * v_distinct, (
        f"identical-input variance {v_identical:.3e} vs distinct {v_distinct:.3e}: the bare "
        f"measure now DOES separate them at init, so this test's premise has changed -- re-derive "
        f"whether mask_variance_null is still needed before deleting it.")


def test_the_null_is_emitted_and_reaches_the_reader(monkeypatch):
    """A diagnostic that never reaches `last_scalars` is not a diagnostic (ppo_aux.py:442)."""
    monkeypatch.setenv("NETT_SLOTC_DIAG", "1")
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))
    aux.compute(enc, obs[0])
    for key in ("mask_variance", "mask_variance_null", "mask_variance_excess"):
        assert key in aux.last_scalars, f"{key} never reaches last_scalars"
    assert aux.last_scalars["mask_variance_null"] >= 0.0, "null kept its negative sentinel"
    assert aux.last_scalars["mask_variance_excess"] == pytest.approx(
        aux.last_scalars["mask_variance"] - aux.last_scalars["mask_variance_null"])


# -- loss_ss discriminability ----------------------------------------------------------
#
# ⛔ THE DISCHARGE CONDITION FOR ROW 14'S BLOCKER, STATED BY THE RESEARCHER IN ADVANCE:
# building what the blocker names discharges it IFF the instrument SEPARATES the states it
# must separate -- red on the constant-slot collapse, and distinguishable from the
# no-correspondence case. "The null is a null" is necessary and NOT sufficient.


def _slots(b, k, d, kind, seed=0):
    """Three constructed regimes, as unit-normalised slot sets (s1, s2)."""
    g = torch.Generator().manual_seed(seed)
    if kind == "healthy":
        # input-dependent, and slot identity is stable across t
        s1 = torch.randn(b, k, d, generator=g)
        s2 = s1 + 0.01 * torch.randn(b, k, d, generator=g)
    elif kind == "degenerate":
        # constant per-slot vectors: slot k is the SAME vector for every batch element, at
        # both t and t+1. Nothing is segmented; the temporal pairing is nonetheless perfect.
        proto = torch.randn(1, k, d, generator=g)
        s1 = proto.expand(b, k, d).contiguous()
        s2 = proto.expand(b, k, d).contiguous()
    elif kind == "nocorr":
        # independent draws: no temporal correspondence at all (this is ALSO the init state)
        s1 = torch.randn(b, k, d, generator=g)
        s2 = torch.randn(b, k, d, generator=g)
    else:
        raise ValueError(kind)
    n = torch.nn.functional.normalize
    return n(s1, p=2.0, dim=-1).reshape(-1, d), n(s2, p=2.0, dim=-1).reshape(-1, d)


@pytest.mark.parametrize("kind", ["healthy", "degenerate", "nocorr"])
def test_the_diagnostic_separates_the_three_regimes(kind):
    """⛔ THE TEST THAT DISCHARGES THE BLOCKER, OR DOES NOT."""
    b, k, d = 32, 4, 16
    s1, s2 = _slots(b, k, d, kind)
    r = slotc.slot_contrast_diagnostics(s1, s2, 0.1, k)
    chance = r["ss_chance"]
    assert r["ss_batch"] == b
    if kind == "healthy":
        assert r["ss_pos_acc"] > 0.9, r
        assert r["ss_pos_acc_x_batch"] > 0.5 * b, "healthy must sit far above the 1/B collapse"
    elif kind == "degenerate":
        # ⭐ pos_acc ~= 1/B, and CRUCIALLY it is ABOVE both nulls -- which is why cltt_ref's
        # "pos_acc <= shuffled_acc" rule cannot be carried across.
        assert abs(r["ss_pos_acc"] - 1.0 / b) < 0.5 / b, r
        assert r["ss_pos_acc_x_batch"] == pytest.approx(1.0, abs=0.5), r
        assert r["ss_pos_acc"] > r["ss_shuffled_acc_within"], (
            "the collapse SATISFIES cltt_ref's rule; that is the whole reason for this test")
    else:
        assert r["ss_pos_acc"] < 8 * chance, r
        assert r["ss_pos_acc_x_batch"] < 0.5 * b


def test_the_collapse_is_EXACT_where_no_correspondence_only_brushes_the_value():
    """⛔ THE REAL SEPARATOR IS EXACTNESS, NOT AN ORDERING. Both the constant-slot collapse and
    the no-correspondence state give a LOW pos_acc, and they call for opposite actions -- one is
    a dead arm, the other is the correct STARTING state. What tells them apart is that the
    collapse is PINNED at exactly 1.0 / exactly 0.0 by construction (B columns tie, argmax takes
    the first), while no-correspondence is a noisy quantity that merely passes through.
    """
    b, k, d = 32, 4, 16
    for seed in range(40):
        deg = slotc.slot_contrast_diagnostics(*_slots(b, k, d, "degenerate", seed), 0.1, k)
        assert abs(deg["ss_pos_acc_x_batch"] - 1.0) < 1e-6, (seed, deg)
        assert abs(deg["ss_input_dependence"]) < 1e-6, (seed, deg)
    exact = sum(
        1 for seed in range(40)
        if abs(slotc.slot_contrast_diagnostics(
            *_slots(b, k, d, "nocorr", seed), 0.1, k)["ss_pos_acc_x_batch"] - 1.0) < 1e-6)
    assert exact < 8, f"no-correspondence hit exactly 1.0 in {exact}/40 -- exactness is not rare"


def test_a_SINGLE_read_cannot_tell_the_collapse_from_no_correspondence():
    """⛔⛔⛔ THIS PINS A REFUTATION OF MY OWN CLAIM, SO IT CANNOT COME BACK.

    I asserted, from ONE seed, that `ss_input_dependence` and `across == pos_acc` separate the
    constant-slot collapse from no-correspondence. A 200-seed sweep by the researcher seat found
    the first holds 59/200 -- WORSE THAN A COIN FLIP IN THE DIRECTION I CLAIMED -- and the second
    132/200. Degenerate sits at exactly 0 while no-correspondence is noise centred slightly BELOW
    zero (mean -0.0011), so the collapse is ABOVE nocorr more often than below: the opposite
    ordering. My single seed happened to draw nocorr at +0.0078.

    ⇒ The lesson is not "that claim was wrong", it is that ONE DRAW OF A NOISY QUANTITY READ AS A
    STRUCTURAL CONSTANT looks exactly like a discriminator when it lands where your story wants
    it. This test fails if anyone re-derives the ordering claim from a small sample.
    """
    b, k, d = 32, 4, 16
    n = 60
    holds = sum(
        1 for seed in range(n)
        if slotc.slot_contrast_diagnostics(*_slots(b, k, d, "degenerate", seed), 0.1, k
                                           )["ss_input_dependence"]
        < slotc.slot_contrast_diagnostics(*_slots(b, k, d, "nocorr", seed), 0.1, k
                                          )["ss_input_dependence"])
    assert holds < 0.9 * n, (
        f"input_dependence ordering held {holds}/{n}; it is NOT a single-read discriminator "
        "against no-correspondence, and treating it as one is the error this test exists for")


def test_the_two_nulls_are_not_one_null():
    """⛔ A SINGLE SHIFT-BY-ONE NULL AVERAGES TWO DIFFERENT QUESTIONS. At K=4 it lands on
    another slot of the SAME image 3 times in 4 and crosses to a different image once, so it
    estimates neither. They are emitted separately and must be able to disagree."""
    b, k, d = 16, 4, 16
    g = torch.Generator().manual_seed(3)
    # scenes differ; slots WITHIN a scene are identical -> across is easy, within is impossible
    scene = torch.randn(b, 1, d, generator=g).expand(b, k, d).contiguous()
    n = torch.nn.functional.normalize
    s = n(scene, p=2.0, dim=-1).reshape(-1, d)
    r = slotc.slot_contrast_diagnostics(s, s, 0.1, k)
    assert r["ss_shuffled_acc_within"] > r["ss_shuffled_acc_across"], (
        "slots identical within a scene must show up as a WITHIN-null hit, not an across one")


def test_a_null_on_a_degenerate_axis_is_a_sentinel_not_a_number():
    """K=1 makes the within-null the positive itself; B=1 does the same for across. Returning
    a NUMBER there would read as perfect agreement between a statistic and its own null."""
    d = 8
    one_slot = slotc.slot_contrast_diagnostics(torch.randn(6, d), torch.randn(6, d), 0.1, 1)
    assert one_slot["ss_shuffled_acc_within"] == -1.0, one_slot
    one_img = slotc.slot_contrast_diagnostics(torch.randn(4, d), torch.randn(4, d), 0.1, 4)
    assert one_img["ss_shuffled_acc_across"] == -1.0, one_img


def test_every_ss_field_reaches_last_scalars_with_the_diag_off(monkeypatch):
    """⛔ A DIAGNOSTIC THAT NEVER REACHES ppo_aux.py:442 IS NOT A DIAGNOSTIC -- the defect this
    module already had once. With the diag OFF the fields must still be PRESENT and carry the
    negative sentinel, so "off" and "on and degenerate" cannot both present as absent."""
    monkeypatch.delenv("NETT_SLOTC_DIAG", raising=False)
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))
    aux.compute(enc, torch.empty(0))
    for f in ("ss_pos_acc", "ss_shuffled_acc_within", "ss_shuffled_acc_across",
              "ss_chance", "ss_batch", "ss_pos_acc_x_batch", "ss_pos_sim", "ss_neg_sim"):
        assert f in aux.last_scalars, f"{f} never reaches a reader"
        assert aux.last_scalars[f] == -1.0, f"{f} should be the sentinel with the diag off"


def test_the_diag_on_path_writes_real_ss_values(monkeypatch):
    monkeypatch.setenv("NETT_SLOTC_DIAG", "1")
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))
    aux.compute(enc, torch.empty(0))
    assert aux.last_scalars["ss_batch"] > 0
    assert 0.0 <= aux.last_scalars["ss_pos_acc"] <= 1.0
    assert aux.last_scalars["ss_chance"] > 0.0


# ---------------------------------------------------------------------------
# Row 14's REGISTERED HEALTHY FLOOR.
#
# The floor is calibrated against the two nulls, never against a constructed
# "healthy" fixture -- a fixture built as s2 = s1 + noise has correspondence by
# fiat and can only ever confirm the threshold fitted to it. Both nulls here are
# derivable without running an arm, which is what makes the floor registrable in
# advance. These tests pin the derivation so the registered numbers cannot drift
# away from the code that produces them.
# ---------------------------------------------------------------------------

_FLOOR_DEP = 0.05  # median ss_input_dependence, final 20 logged updates
_FLOOR_XB = 1.0  # median ss_pos_acc_x_batch must be STRICTLY above this


def _chance_draw(batch: int, slots: int, seed: int, dim: int = 64) -> dict:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(batch * slots, dim, generator=gen)
    b = torch.randn(batch * slots, dim, generator=gen)
    return slotc.slot_contrast_diagnostics(a, b, 0.1, slots)


@pytest.mark.parametrize("batch,slots", [(32, 4), (32, 3), (16, 4)])
def test_x_batch_is_hits_over_slots_so_the_chance_null_is_closed_form(batch, slots):
    """x_batch = hits/K exactly, which is what makes the null analytic.

    Without this identity the null is a simulation and its tail is only ever as
    good as the seed count. With it, P(x_batch >= t) is a Binomial survival
    function -- independent of B, and in `hits` units independent of K as well.
    """
    for seed in range(40):
        x = _chance_draw(batch, slots, seed)["ss_pos_acc_x_batch"]
        hits = round(x * slots)
        assert abs(x - hits / slots) < 1e-6, f"x_batch={x} is not a multiple of 1/{slots}"


def test_the_collapse_fails_the_floor_deterministically_not_probabilistically():
    """The degenerate solution must fail the floor at EVERY seed, not merely usually.

    This is the defect the first floor had: `median hits >= 3` has chance-null
    p < 5e-6 and the constant-slot collapse attains hits = K = 4, so it PASSED.
    A floor must clear the score a non-policy attains, not the score chance attains.

    Reuses `_slots(..., "degenerate")` rather than building a second constant-slot
    fixture. The first draft of this test did build its own, collapsing all B*K
    vectors onto ONE -- which also destroys the K slot prototypes and scores at
    chance (x_batch = 1/K), not at the registered 1.0. Two constructions of "the
    collapse" would have drifted apart silently.
    """
    for seed in range(50):
        d = slotc.slot_contrast_diagnostics(*_slots(32, 4, 64, "degenerate", seed=seed), 0.1, 4)
        assert d["ss_input_dependence"] == pytest.approx(0.0, abs=1e-9)
        assert d["ss_pos_acc_x_batch"] == pytest.approx(1.0, abs=1e-9)
        # ... and therefore fails both conjuncts, with no margin to argue about.
        assert not d["ss_input_dependence"] >= _FLOOR_DEP
        assert not d["ss_pos_acc_x_batch"] > _FLOOR_XB


def test_the_floor_catches_a_total_collapse_that_the_kill_rule_does_not():
    """A degeneracy the KILL RULE misses, and the floor does not.

    If the K slot prototypes collapse onto each other too, every row of `sim` is
    identical, the argmax ties resolve to column 0, and the read is x_batch = 1/K
    -- indistinguishable from chance. The kill rule requires x_batch == 1.0 and so
    NEVER FIRES on this state. The floor fails it on both conjuncts.

    Recorded because it bounds what the kill rule is: a detector for ONE degenerate
    manifold, not for degeneracy.
    """
    dim = 64
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        one = torch.nn.functional.normalize(torch.randn(1, dim, generator=gen), p=2.0, dim=-1)
        flat = one.expand(32 * 4, dim).contiguous()
        d = slotc.slot_contrast_diagnostics(flat, flat.clone(), 0.1, 4)
        assert d["ss_pos_acc_x_batch"] == pytest.approx(0.25, abs=1e-9)  # == 1/K, i.e. chance
        assert not abs(d["ss_pos_acc_x_batch"] - 1.0) < 1e-6, "kill rule would fire; it must not"
        assert not (d["ss_input_dependence"] >= _FLOOR_DEP and d["ss_pos_acc_x_batch"] > _FLOOR_XB)


@pytest.mark.parametrize("batch,slots", [(32, 4), (32, 3), (16, 4), (48, 4)])
def test_chance_clears_neither_conjunct_of_the_block_median_floor(batch, slots):
    """No-correspondence is the CORRECT state at init, so it must fail the floor.

    ⚠ The floor is a MEDIAN over the final 20 logged updates, and this test must
    match that unit. Asserting it per-update fails honestly -- single reads at
    B=16 reach input_dep = 0.0625, above the 0.05 threshold -- because a single
    update is MORE variable than the median, not less. Testing the per-update
    value would have condemned a floor that is sound at the unit it is read at.
    """
    deps, xbs = [], []
    for seed in range(400):
        d = _chance_draw(batch, slots, seed)
        deps.append(d["ss_input_dependence"])
        xbs.append(d["ss_pos_acc_x_batch"])
    rng = random.Random(7)
    for _ in range(300):
        idx = [rng.randrange(len(deps)) for _ in range(20)]
        med_dep = statistics.median(deps[i] for i in idx)
        med_xb = statistics.median(xbs[i] for i in idx)
        assert med_dep < _FLOOR_DEP
        assert not (med_dep >= _FLOOR_DEP and med_xb > _FLOOR_XB)


def test_input_dependence_alone_is_a_weak_conjunct_and_the_kill_rule_needs_both():
    """Pins WHY the kill rule is a conjunction.

    `input_dep == 0` alone fires on roughly a third of no-correspondence draws.
    If anyone ever simplifies the kill rule to that half, this fails.
    """
    fires = sum(abs(_chance_draw(32, 4, s)["ss_input_dependence"]) < 1e-6 for s in range(400))
    assert 0.15 < fires / 400 < 0.50, f"input_dep==0 fired {fires}/400; the rule's shape assumed ~0.31"


# ---------------------------------------------------------------------------
# The t+1 pass must START FROM the t slots (FINDINGS §4bv, §4bv.1)
# ---------------------------------------------------------------------------
# ⛔ THE DEFECT THIS PINS, MEASURED ON A SHIPPED ARM. `compute()` called `_slots_for` twice
# with no `slots_init`, so t+1 drew a FRESH random init while the loss target is `torch.eye`
# ("slot k at t matches slot k at t+1"). `mu`/`log_sigma` are (1, 1, slot_dim) -- ONE
# distribution shared by every slot -- so slot index carries no identity and the diagonal was
# arbitrary. The gradient was NOT inert: it shaped the encoder with noise. I77 ran this way.
# The reference builds the correspondence by RECURRENCE (init drawn once per sequence, carried
# forward), not by a predictor.


def _record_inits(aux, monkeypatch):
    """Record the `slots_init` every attention pass receives, in call order."""
    seen, real = [], aux.head.attn.forward

    def spy(tokens, slots_init=None, return_init=False):
        seen.append(slots_init)
        return real(tokens, slots_init, return_init=return_init)

    monkeypatch.setattr(aux.head.attn, "forward", spy)
    return seen


def test_the_t_plus_1_pass_starts_from_the_t_slots(monkeypatch):
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))
    seen = _record_inits(aux, monkeypatch)
    aux.compute(enc, obs[0])

    assert len(seen) == 2, f"expected one pass per frame, got {len(seen)}"
    assert seen[0] is None, "the t pass draws the init; it must not inherit one"
    assert seen[1] is not None, (
        "the t+1 pass drew a FRESH init: the eye() target then names a correspondence "
        "nothing created, and the gradient still reaches the encoder")


def test_the_t_plus_1_init_is_the_t_slots_themselves_not_a_copy(monkeypatch):
    # ⛔ Identity, not allclose: a detached or cloned init would silently cut the recurrence's
    # gradient, which the reference does not do. `is` cannot pass by coincidence.
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))
    captured = {}
    real = aux.head.attn.forward
    seen = []

    def spy(tokens, slots_init=None, return_init=False):
        out = real(tokens, slots_init, return_init=return_init)
        seen.append(slots_init)
        captured.setdefault("first_slots", out[0] if isinstance(out, tuple) else out)
        return out

    monkeypatch.setattr(aux.head.attn, "forward", spy)
    aux.compute(enc, obs[0])
    assert seen[1] is captured["first_slots"], (
        "the t+1 init must BE the t slots tensor, so the gradient flows along the recurrence")


def test_the_t_plus_1_init_carries_gradient(monkeypatch):
    enc = _encoder()
    aux = slotc.SlotContrastAuxLoss(enc)
    obs = _obs()
    aux.attach_memory(_FakeMemory(obs))
    seen = _record_inits(aux, monkeypatch)
    aux.compute(enc, obs[0])
    assert seen[1].requires_grad, "a detached init makes the recurrence a stop-gradient"
