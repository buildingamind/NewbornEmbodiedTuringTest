"""NETT-owned PPO behaviour that must NOT live as an in-place edit of skrl.

Previously the time-limit-bootstrap device fix was a hand-edit of
``site-packages/skrl/agents/torch/ppo/ppo.py`` ("# NETT patch:"). That edit is
invisible to source control, so a fresh ``pip install skrl==2.1.0`` silently
dropped it and the opt-in bootstrap path then crashed cuda-vs-cpu. Expressing it
as a mixin here makes it reproducible and version-checkable.
"""
from __future__ import annotations

import functools
from contextlib import contextmanager

import torch


@contextmanager
def strict_determinism():
    """Run the PPO learning path under STRICT deterministic algorithms.

    The ambient policy (set in runtime.task.set_seeds) is
    ``use_deterministic_algorithms(True, warn_only=True)`` — warn-not-raise — so
    Isaac's RTX render/camera CUDA kernels, which have no deterministic
    implementation, don't kill a run. But warn_only also SILENTLY tolerates any
    nondeterministic op inside the gradient step (the AdaptiveAvgPool2d backward
    that was replaced by DeterministicAvgPool2d was one such op). Wrapping the
    PPO update in strict mode makes any *future* nondeterministic op in the
    learning path fail loud instead of silently corrupting reproducibility, then
    restores the render-tolerant policy on exit.
    """
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(True, warn_only=True)


def strict_update(update_fn):
    """Decorator: run a PPO ``update`` under :func:`strict_determinism`."""

    @functools.wraps(update_fn)
    def wrapper(self, *args, **kwargs):
        with strict_determinism():
            return update_fn(self, *args, **kwargs)

    return wrapper


class NETTBootstrapMixin:
    """Device-safe time-limit (truncation) bootstrapping.

    skrl's ``PPO.record_transition`` computes ``discount * V(next) * truncated``
    and adds it into ``rewards``. ``V(next)`` is on the agent's compute device
    (cuda) while ``rewards`` / ``truncated`` may be on cpu (HybridDeviceMemory
    stores image observations on cpu), which raises a device-mismatch. Aligning
    the reward tensors to the agent device BEFORE delegating to skrl removes the
    need for any library edit.

    Only active when ``cfg.time_limit_bootstrap`` is True. On the default config
    (``time_limit_bootstrap=False``) this is an exact pass-through — zero
    behaviour change — so it does not affect standard runs.
    """

    def record_transition(self, *, rewards, truncated, **kwargs):
        if getattr(self, "training", False) and getattr(self.cfg, "time_limit_bootstrap", False):
            dev = self.device
            if rewards.device != dev:
                rewards = rewards.to(dev)
            if truncated.device != dev:
                truncated = truncated.to(dev)
        return super().record_transition(rewards=rewards, truncated=truncated, **kwargs)
