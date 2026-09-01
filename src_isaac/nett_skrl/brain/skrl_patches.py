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
    restores the AMBIENT policy on exit.

    Exit restores whatever was in force on entry rather than hardcoding
    warn_only=True: under the NETT_STRICT_DETERMINISM diagnostic the ambient
    policy is itself strict, and forcing warn_only back would silently disarm the
    diagnostic after the first PPO update -- i.e. exactly the window where the
    run-to-run divergence accumulates.
    """
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(True, warn_only=was_warn_only)


@contextmanager
def relaxed_determinism():
    """Allow nondeterministic CUDA kernels for the duration of this block.

    ⛔ WHY THIS EXISTS. ``strict_determinism`` (above) wraps the whole PPO update in
    ``warn_only=False``, so ANY op in the learning path without a deterministic kernel
    RAISES. The motion auxiliary losses (EoO, GWM) warp frames with ``F.grid_sample``, and
    ``grid_sampler_2d_backward_cuda`` has no deterministic implementation -- so every
    EoO/GWM arm dies at its FIRST optimizer step:

        RuntimeError: grid_sampler_2d_backward_cuda does not have a deterministic
        implementation, but you set 'torch.use_deterministic_algorithms(True)'

    That is the guard working as designed, not a misconfiguration -- ``compact_vit.py:235``
    records the identical death for a spatial-pooled ViT. But it makes six of the eight
    priority arms unrunnable, so the OWNER RULED (2026-08-28) to exempt the auxiliary
    backward ONLY, keeping PPO-proper strict.

    ⚠ THE EXEMPTION IS A SCOPE, NOT A SWITCH. It restores the AMBIENT policy (warn-not-raise)
    rather than disabling determinism, and it restores whatever was in force on entry --
    so under the NETT_STRICT_DETERMINISM diagnostic this narrows to a no-op rather than
    silently disarming the diagnostic, which is the same trap ``strict_determinism``'s own
    docstring warns about.

    ⚠ WHAT THE RESULTING RUNS MAY AND MAY NOT CLAIM: bitwise run-to-run reproducibility no
    longer holds for the auxiliary gradient path of any arm that uses a nondeterministic
    kernel there. PPO-proper's backward is untouched and still strict. Any reproducibility
    statement about an EoO/GWM arm must carry that limit.
    """
    # ⛔ SAVE AND RESTORE BOTH FLAGS, not just warn_only. `strict_determinism` above restores
    # with the first argument hardcoded True, so a caller whose ambient state had determinism
    # DISABLED exits the context with it ENABLED -- a leak that is benign in production (task.py
    # always enables it) but poisons any standalone probe, and makes the NEXT case in a probe
    # loop fail with the PREVIOUS case's state. Reported by seat:lion, who lost a gwm result to
    # exactly that leak and read it as a gwm defect. Mirroring the shape would inherit the bug.
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


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
