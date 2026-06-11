"""Feature extraction helpers for NETT skrl models."""

from __future__ import annotations


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
    return model.trunk(model.encoder(x))
