"""Feature extraction helpers for NETT skrl models."""

from __future__ import annotations

import contextlib
import os


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
