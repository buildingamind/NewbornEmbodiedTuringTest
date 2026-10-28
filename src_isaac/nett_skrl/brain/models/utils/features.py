"""Feature extraction helpers for NETT skrl models."""

from __future__ import annotations

import contextlib
import os

import torch


def _decouple_encoder() -> bool:
    """True when the RL objective must NOT shape the encoder.

    ⭐ THE HYPOTHESIS THIS EXISTS FOR (owner, 2026-09-15): representations are
    formed by the AUXILIARY objective and the reward signal is only needed for
    the actor/critic. Today both gradients reach the same shared encoder, so an
    aux loss teaching object structure competes with an RL loss teaching
    whatever predicts reward -- and in this environment that is the BACKGROUND,
    which occupies 93-96% of the stimulus frame against an object at ~4%.

    ⇒ With this set, the encoder is shaped by the aux loss ALONE and the heads
    learn to read it. With no aux loss configured it degenerates to a frozen
    randomly-initialised encoder, which is a useful floor in its own right: it
    says how much RL-learning the encoder buys over a random conv feature bank.
    """
    return os.environ.get("NETT_DECOUPLE_ENCODER", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


_CACHE_ATTR = "_nett_shared_feature_cache"

#: ⭐ NETT_DECOUPLE_TRUNK (owner, 2026-10-28, BASELINE detached SC257 / Mask75): detach the RL path at
#: the AUX'S READ POINT instead of at the encoder output. SC257 and the masked-reconstruction aux read
#: the backbone BEFORE its readout (NatureCNN.linear; CompactViT's CLS -> head), so with
#: NETT_DECOUPLE_ENCODER=1 nothing would ever train that readout. With this knob the RL forward runs
#: the trunk under no_grad and the readout with grad: the aux alone shapes the trunk, PPO trains the
#: readout + the policy / value heads. The aux path is untouched (it calls the encoder directly).
#: Needs an encoder with `forward_trunk_detached` (NatureCNN, CompactViT); refused with
#: NETT_DECOUPLE_ENCODER, which already detaches the whole encoder.
DECOUPLE_TRUNK_ENV = "NETT_DECOUPLE_TRUNK"


def _decouple_trunk() -> bool:
    on = os.environ.get(DECOUPLE_TRUNK_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    if on and _decouple_encoder():
        raise ValueError("NETT_DECOUPLE_TRUNK=1 and NETT_DECOUPLE_ENCODER=1 are contradictory; set one")
    return on


def _trunk_detached_forward(enc, x):
    fn = getattr(enc, "forward_trunk_detached", None)
    if fn is None:
        raise TypeError(f"NETT_DECOUPLE_TRUNK=1 needs an encoder with forward_trunk_detached; "
                        f"{type(enc).__name__} has none")
    return fn(x)

#: ⭐ FIX 2b OF DECISIONS §111 (owner, 2026-10-14, "Proceed with 1 + 2 + 2b"). Set on an encoder by
#: AuxLossPPO ONLY when its aux never reads the RL forward's graph (`independent_of_ppo_graph`,
#: the CLTT-Ref family). With NETT_DECOUPLE_ENCODER=1 the RL path's encoder output is detached
#: below, so building its graph bought nothing -- yet the cache kept that graph alive until the
#: optimizer step. Marked + decoupled => the encoder pass runs under no_grad: same values, same
#: (zero) RL gradient into the encoder, no graph. "0" in NETT_RL_ENCODER_NO_GRAD restores the
#: graph-building pass (a control).
RL_NO_GRAD_ATTR = "_nett_rl_forward_no_grad"
RL_NO_GRAD_ENV = "NETT_RL_ENCODER_NO_GRAD"


def _rl_forward_no_grad(enc) -> bool:
    if getattr(enc, RL_NO_GRAD_ATTR, False) is not True or not _decouple_encoder():
        return False
    raw = os.environ.get(RL_NO_GRAD_ENV, "").strip().lower()
    if raw in ("", "1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{RL_NO_GRAD_ENV}={raw!r} is not a boolean; refusing to guess.")


# ⛔⛔⛔ THE CACHE IS OFF UNTIL AN AGENT OPTS IN, AND THAT IS NOT CAUTION -- IT IS A CORRECTNESS
# REQUIREMENT THE EXISTING SUITE CAUGHT. Caching makes the actor and critic share ONE autograd
# graph. That is safe only where every consumer of the shared tensor is backwarded TOGETHER (or
# with retain_graph). skrl's SAC / DDPG / TD3 call `.backward()` SEPARATELY on the critic loss and
# then the policy loss, so the second traverses a graph the first already freed:
#     RuntimeError: Trying to backward through the graph a second time
# ⇒ The maths is identical under caching; the GRAPH LIFETIME is not, and no amount of
# "the gradients are the same" reasoning reaches that. tests/test_brain.py's SAC and DDPG cases
# went red on the first version of this change, which is exactly what they are for.
# ⇒ `shared_feature_cache()` is entered by ppo_aux alone, which owns a single fused backward
# (or a split one with retain_graph=True) AND the clear at each optimizer step.
_CACHE_ACTIVE = False


@contextlib.contextmanager
def shared_feature_cache(*modules):
    """Enable encoder-output reuse for the enclosed block, and ALWAYS release it on exit.

    A context manager rather than a flag, so an exception mid-update cannot leave the cache set
    with a live graph in it -- the failure mode would be a memory leak that no test detects.
    """
    global _CACHE_ACTIVE
    prev = _CACHE_ACTIVE
    _CACHE_ACTIVE = True
    try:
        yield
    finally:
        _CACHE_ACTIVE = prev
        clear_feature_cache(*modules)


def _cache_enabled() -> bool:
    """True only inside `shared_feature_cache()` and only when the kill-switch is unset.

    ⛔ NETT_DISABLE_FEATURE_CACHE IS A CONTROL, NOT A CONVENIENCE. "Caching is gradient-identical"
    is worth nothing unless the other branch can still be RUN, so an A/B is measurable rather than
    argued. tests/test_feature_cache.py exercises BOTH branches.
    """
    if not _CACHE_ACTIVE:
        return False
    return os.environ.get("NETT_DISABLE_FEATURE_CACHE", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }


def clear_feature_cache(*modules) -> None:
    """Drop any cached encoder output. MUST be called once per optimizer step.

    ⛔ NOT OPTIONAL, AND ITS ABSENCE COSTS MEMORY RATHER THAN CORRECTNESS -- which is the
    dangerous direction, because the run still produces right answers while holding a whole
    encoder graph past the step that needed it. The cached tensor carries its autograd graph;
    left in place it pins ~15 GiB at a 500-sample minibatch, i.e. it would make the very problem
    this cache exists to solve WORSE while every test still passed.
    """
    for m in modules:
        if m is not None and getattr(m, _CACHE_ATTR, None) is not None:
            setattr(m, _CACHE_ATTR, None)
        # NETT_MIRROR_EQUIV's mirrored-pass entry (never set when the flag is off).
        if m is not None and getattr(m, "_nett_shared_feature_cache_mirror", None) is not None:
            setattr(m, "_nett_shared_feature_cache_mirror", None)


def features_forward(model, inputs):
    """Run NETT feature extractor + MLP trunk for skrl model inputs.

    ⭐ WHY THE ENCODER OUTPUT IS CACHED (2026-09-17). With `shared_encoder: True` the actor and
    the critic hold the SAME encoder instance (models/builder.py) but each calls this function
    independently, so one PPO minibatch ran the encoder TWICE on identical observations and kept
    BOTH autograd graphs alive until the single backward. Measured on UnityViT at the campaign
    eye: 37.2 MiB of retained activations PER SAMPLE, so a 500-sample minibatch held ~14.8 GiB
    per pass -- the second pass is what put a ONE-BRAIN, 16-env arm over a 24 GiB card.

    ⇒ Caching is GRADIENT-IDENTICAL, not an approximation. With one shared `feats` node autograd
    SUMS the incoming gradients from the policy and value paths before traversing the encoder
    once, and d(L_pi + L_V)/d(theta_enc) is that same sum either way. It is what
    `builder.py` already says it is imitating (SB3 `share_features_extractor=True`).

    ⚠ The TRUNK is per-head and is NOT cached -- only `model.encoder`'s output is. Caching the
    trunk output would feed the critic the actor's value basis, which is a different model.

    ⚠ It also removes a latent defect: for an encoder with BatchNorm (`simclr_cltt` alone on this
    fleet) the two passes applied TWO momentum updates to the running statistics from ONE batch,
    an effective momentum of 0.19 where the config asks 0.1. Outputs and gradients were identical;
    only the eval-time buffers differed, and measured the effect is 0.0003-0.012 activation-sd
    units -- real, and immaterial. See FINDINGS 4bt/4bt.1.
    """
    x = inputs.get("observations")
    if x is None:
        x = inputs.get("states")
    if x is None:
        raise KeyError("skrl model inputs must include 'observations' or 'states'.")
    target_device = next(model.encoder.parameters()).device
    if x.device != target_device:
        x = x.to(target_device, non_blocking=True)
    # Do NOT normalize here: every encoder's forward() calls _prepare_image()
    # which owns the HWC->CHW layout + single /255. Normalizing here too
    # double-applied the non-idempotent /255 (pixels reached the CNN at ~1/255
    # magnitude, crushing color). Removing it requires the value-head gain fix
    # (value_critic.py, gain=1.0) so the correctly-scaled features don't
    # destabilize the critic.
    # ⛔ KEY ON TENSOR IDENTITY (`is`), NEVER ON VALUE. The actor is called with
    # `{**inputs, "taken_actions": ...}` -- a NEW dict holding the SAME tensor object -- so
    # identity is exactly the right test and costs nothing. An equality or hash key would be a
    # full tensor read per call, and a shape/device key would COLLIDE across different
    # minibatches with the same shape, silently feeding one minibatch's features to the next.
    enc = model.encoder
    _hit = getattr(enc, _CACHE_ATTR, None) if _cache_enabled() else None
    if _hit is not None and _hit[0] is x:
        feats = _hit[1]
    else:
        if _decouple_trunk():
            feats = _trunk_detached_forward(enc, x)
        elif _rl_forward_no_grad(enc):
            with torch.no_grad():
                feats = enc(x)
        else:
            feats = enc(x)
        if _cache_enabled():
            setattr(enc, _CACHE_ATTR, (x, feats))
    if _decouple_encoder():
        # ⛔ DETACH HERE AND NOWHERE ELSE. This is the single chokepoint the RL
        # path takes into the encoder; every auxiliary loss calls model.encoder
        # DIRECTLY, so detaching here removes the reward gradient and leaves the
        # auxiliary gradient untouched. Detaching inside the encoder, or freezing
        # requires_grad, would cut both and silently turn every aux arm into a
        # frozen-encoder arm.
        # ⛔ `.detach()` returns a NEW tensor and leaves the cached one attached, which is
        # required: the auxiliary losses call `model.encoder` directly and must still reach a
        # graph. Assigning the detached tensor back into the cache would silently convert every
        # aux arm into a frozen-encoder arm -- the exact failure the note above forbids.
        feats = feats.detach()
    return model.trunk(feats)


# ── MIRROR-EQUIVARIANT POLICY (NETT_MIRROR_EQUIV; presenter team, 2026-10-22). ─────────────────────
# Failing parsing seeds fail through a constant LEFT/RIGHT side bias of the deterministic policy.
# With NETT_MIRROR_EQUIV=1 the Gaussian MEAN and the VALUE are made exactly mirror-equivariant /
# invariant by symmetrising over the left-right image flip F (the width axis) and the action
# mirror M (negate turn = actions[..., 0]; move = actions[..., 1] is mirror-invariant):
#     mu_sym(o) = 0.5 * (mu(o) + M mu(F o)),     V_sym(o) = 0.5 * (V(o) + V(F o))
# Exact for ANY network because F and M are involutions; nothing is asked of the encoder.
# log_std is state-independent and untouched. Unset / "0" = the original code path, statement for
# statement (features_forward above is not modified). No parameters are added: checkpoints load strict.
MIRROR_ENV = "NETT_MIRROR_EQUIV"
#: Opt-in activation recompute (torch.utils.checkpoint, non-reentrant) for the encoder passes of the
#: mirrored forward. Gradient-identical; trades ~1/3 more encoder compute in the PPO update for
#: storing only the uint8 inputs instead of the conv activations (the mirror doubles those).
MIRROR_CKPT_ENV = "NETT_MIRROR_CKPT"
_MIRROR_CACHE_ATTR = "_nett_shared_feature_cache_mirror"
#: Every encoder attribute that can hold a cached (x, feats) pair with a live graph. A deepcopy of an encoder
#: (EMA / slot targets) must neutralise ALL of them: under NETT_MIRROR_EQUIV the mirrored pass's slot holds a
#: non-leaf tensor too, and Tensor.__deepcopy__ refuses one (workspace DECISIONS §187).
FEATURE_CACHE_ATTRS = (_CACHE_ATTR, _MIRROR_CACHE_ATTR)


def _strict_flag(name: str, raw: str) -> bool:
    raw = raw.strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{name}={raw!r} is not a boolean; refusing to guess.")


# ⚠ Each knob is read with a VISIBLE `os.environ.get(CONST, ...)` at its call site, not inside the helper:
# scripts/gen_env_index.py resolves `NAME = "NETT_..."` constants only where they are passed to the read
# itself, and a read hidden behind a parameter left both knobs out of docs/env_vars.md (DECISIONS §187).
def mirror_equiv_enabled() -> bool:
    """True when NETT_MIRROR_EQUIV is set on. Read per call, like NETT_DECOUPLE_ENCODER."""
    return _strict_flag(MIRROR_ENV, os.environ.get(MIRROR_ENV, ""))


def mirror_ckpt_enabled() -> bool:
    return _strict_flag(MIRROR_CKPT_ENV, os.environ.get(MIRROR_CKPT_ENV, ""))


def mirror_obs(x: torch.Tensor, observation_space) -> torch.Tensor:
    """Left-right mirror of a batch of image observations (flip of the WIDTH axis).

    Flat (B, prod(shape)) input is viewed as (B, *observation_space.shape) -- the layout
    prepare_image_tensor uses -- and returned 4-D in that layout (the encoders accept 4-D).
    Both CHW (W last) and HWC (W second to last) are handled; the layout is read with
    body.observation.image_layout, the rule every encoder uses.
    """
    from ....body.observation import image_layout

    if x.ndim == 2:
        shape = tuple(int(s) for s in getattr(observation_space, "shape", ()))
        if len(shape) != 3:
            raise ValueError(f"{MIRROR_ENV}: flat observations need a 3-D image space, got {shape!r}")
        n = shape[0] * shape[1] * shape[2]
        if x.shape[1] != n:
            raise ValueError(f"{MIRROR_ENV}: flat observation width {x.shape[1]} != prod{shape} = {n}")
        img = x.view(x.shape[0], *shape)
    elif x.ndim == 4:
        img = x
        shape = tuple(int(s) for s in x.shape[1:])
    else:
        raise ValueError(f"{MIRROR_ENV}: expected a (B, D) or (B, C, H, W)/(B, H, W, C) batch, got {tuple(x.shape)}")
    axis = -1 if image_layout(shape) == "chw" else -2
    return img.flip(axis)


def _encode_one(enc, x):
    if _decouple_trunk():
        if mirror_ckpt_enabled() and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(_trunk_detached_forward, enc, x, use_reentrant=False)
        return _trunk_detached_forward(enc, x)
    if _rl_forward_no_grad(enc):
        with torch.no_grad():
            return enc(x)
    if mirror_ckpt_enabled() and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(enc, x, use_reentrant=False)
    return enc(x)


def features_forward_mirrored(model, inputs):
    """``features_forward`` for o AND its mirror F o: returns (trunk(enc(o)), trunk(enc(F o))).

    Two SEQUENTIAL same-shape encoder calls, not one 2B batch: same kernels per sample in both
    passes (so the symmetrisation is bit-exact where the backend is deterministic) and only B
    float inputs alive at a time. Cache / no-grad / decouple handling as in features_forward.
    """
    x = inputs.get("observations")
    if x is None:
        x = inputs.get("states")
    if x is None:
        raise KeyError("skrl model inputs must include 'observations' or 'states'.")
    target_device = next(model.encoder.parameters()).device
    if x.device != target_device:
        x = x.to(target_device, non_blocking=True)
    enc = model.encoder
    space = getattr(enc, "observation_space", None) or model.observation_space
    _hit = getattr(enc, _CACHE_ATTR, None) if _cache_enabled() else None
    _hitm = getattr(enc, _MIRROR_CACHE_ATTR, None) if _cache_enabled() else None
    if _hit is not None and _hit[0] is x and _hitm is not None and _hitm[0] is x:
        feats, feats_m = _hit[1], _hitm[1]
    else:
        feats = _encode_one(enc, x)
        feats_m = _encode_one(enc, mirror_obs(x, space))
        if _cache_enabled():
            setattr(enc, _CACHE_ATTR, (x, feats))
            setattr(enc, _MIRROR_CACHE_ATTR, (x, feats_m))
    if _decouple_encoder():
        feats = feats.detach()
        feats_m = feats_m.detach()
    return model.trunk(feats), model.trunk(feats_m)


def mirror_symmetrize_mean(mean: torch.Tensor, mean_m: torch.Tensor) -> torch.Tensor:
    """0.5 * (mu(o) + M mu(F o)) with M = diag(-1, +1): turn (index 0) antisymmetric, move symmetric.

    Written as differences/sums (not a sign multiply) so the turn component is exactly
    antisymmetric in IEEE arithmetic: fl(a - b) == -fl(b - a), and is exactly 0 when a == b.
    """
    if mean.shape[-1] != 2:
        raise ValueError(f"{MIRROR_ENV}: the mirror M is defined for the 2-D [turn, move] action only; "
                         f"got {mean.shape[-1]} action dims (an enabled neck DOF needs its own sign).")
    turn = 0.5 * (mean[..., :1] - mean_m[..., :1])
    move = 0.5 * (mean[..., 1:] + mean_m[..., 1:])
    return torch.cat([turn, move], dim=-1)
