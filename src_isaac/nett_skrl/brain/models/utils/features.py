"""Feature extraction helpers for NETT skrl models."""

from __future__ import annotations

from ....observation import prepare_image_tensor


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
    x = prepare_image_tensor(x, model.observation_space, device=target_device)
    return model.trunk(model.encoder(x))
