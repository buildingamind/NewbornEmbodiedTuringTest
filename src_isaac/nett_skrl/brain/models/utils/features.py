"""Feature extraction helpers for NETT skrl models."""

from __future__ import annotations

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


def features_forward(model, inputs):
    """Run NETT feature extractor + MLP trunk for skrl model inputs."""
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
    feats = model.encoder(x)
    if _decouple_encoder():
        # ⛔ DETACH HERE AND NOWHERE ELSE. This is the single chokepoint the RL
        # path takes into the encoder; every auxiliary loss calls model.encoder
        # DIRECTLY, so detaching here removes the reward gradient and leaves the
        # auxiliary gradient untouched. Detaching inside the encoder, or freezing
        # requires_grad, would cut both and silently turn every aux arm into a
        # frozen-encoder arm.
        feats = feats.detach()
    return model.trunk(feats)
