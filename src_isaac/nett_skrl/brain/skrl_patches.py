"""NETT-owned PPO behaviour that must NOT live as an in-place edit of skrl.

Previously the time-limit-bootstrap device fix was a hand-edit of
``site-packages/skrl/agents/torch/ppo/ppo.py`` ("# NETT patch:"). That edit is
invisible to source control, so a fresh ``pip install skrl==2.1.0`` silently
dropped it and the opt-in bootstrap path then crashed cuda-vs-cpu. Expressing it
as a mixin here makes it reproducible and version-checkable.
"""
from __future__ import annotations

import functools
import logging
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


def unique_parameters(parameters):
    """Yield each distinct parameter tensor once, in first-seen order.

    ``itertools.chain(policy.parameters(), value.parameters())`` repeats every
    tensor the two models SHARE. ``nn.Module.parameters()`` de-duplicates within
    one module, never across two, so a shared encoder appears twice.
    """
    seen, out = set(), []
    for p in parameters:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


@contextmanager
def deduplicated_grad_clip():
    """Make ``nn.utils.clip_grad_norm_`` ignore repeated parameter objects.

    ⛔ WHY A PATCH AND NOT AN EDIT. The offending call is INSIDE upstream skrl
    (``skrl/agents/torch/ppo/ppo.py``, the ``self.policy is not self.value``
    branch of ``_update``), so there is no override seam narrower than copying
    the whole method -- which would pin us to one skrl version. Patching the
    function skrl looks up at CALL time is version-independent: skrl does
    ``from torch import nn`` and then ``nn.utils.clip_grad_norm_(...)``, which
    resolves the attribute on each call.

    ⚠ This is NOT an in-place edit of site-packages. The previous time-limit
    bootstrap fix was, and a reinstall silently dropped it -- see this module's
    docstring. The patch lives here, is restored on exit, and is covered by tests.
    """
    original = torch.nn.utils.clip_grad_norm_

    @functools.wraps(original)
    def clip(parameters, *args, **kwargs):
        if isinstance(parameters, torch.Tensor):
            return original(parameters, *args, **kwargs)
        return original(unique_parameters(parameters), *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = clip
    try:
        yield
    finally:
        torch.nn.utils.clip_grad_norm_ = original


def deduped_clip_update(update_fn):
    """Decorator: run a PPO ``update`` under :func:`deduplicated_grad_clip`."""

    @functools.wraps(update_fn)
    def wrapper(self, *args, **kwargs):
        with deduplicated_grad_clip():
            return update_fn(self, *args, **kwargs)

    return wrapper


class NETTSharedEncoderMixin:
    """Remove the duplicated shared encoder from skrl's optimizer.

    ⛔ THE DEFECT, MEASURED 2026-09-14 ON THE REAL AGENT (PPO, shared_encoder=True,
    features_dim=64): skrl builds the optimizer from
    ``itertools.chain(self.policy.parameters(), self.value.parameters())``. With
    ``cfg.shared_encoder=True`` the encoder is the SAME object in both models, so the
    optimizer holds 29 parameters of which only 21 are distinct -- the 8 encoder
    tensors appear twice. torch warns at construction ("optimizer contains a parameter
    group with duplicate parameters") and that warning was never acted on.

    ⚠ A DUPLICATED PARAMETER IS STEPPED TWICE PER ``optimizer.step()``. Measured:
    after 3 updates a duplicated parameter has moved EXACTLY 2x as far as a deduped
    one, and its Adam step counter reads 6 rather than 3. So the shared encoder ran at
    twice the heads' effective learning rate, on a double-advanced bias-correction
    schedule.

    ⚠ THIS IS A SECOND DEFECT, DISTINCT FROM THE GRAD-CLIP DOUBLE COUNT, and the
    standing "Adam is scale-invariant to a constant rescaling" argument does NOT cover
    it: two sequential Adam updates with the momentum state advanced twice is not a
    rescaling of one gradient.

    The de-duplication MUTATES ``param_groups`` in place rather than rebuilding the
    optimizer, so the learning-rate scheduler and ``checkpoint_modules["optimizer"]``
    keep pointing at the same object.

    ⚠ RESUMING A PRE-FIX CHECKPOINT WILL RAISE, deliberately. Its optimizer state
    carries the pre-fix parameter count, and torch refuses a group-size mismatch. That
    is the correct behaviour -- silently accepting it would resume a run under
    different optimisation than it started with -- but it means a pre-fix run cannot be
    continued by post-fix code, only re-run or left as it is.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nett_deduped_parameters = self._dedupe_optimizer_parameters()

    def _dedupe_optimizer_parameters(self) -> int:
        # Duck-typed, not isinstance: skrl agents are constructed with stand-in
        # optimizers in several tests, and a mixin that crashes on a test double would
        # make the double the thing under test.
        groups = getattr(getattr(self, "optimizer", None), "param_groups", None)
        if not groups:
            return 0
        seen, removed = set(), 0
        for group in groups:
            kept = []
            for p in group["params"]:
                if id(p) in seen:
                    removed += 1
                    continue
                seen.add(id(p))
                kept.append(p)
            group["params"] = kept
        if removed:
            logging.getLogger("nett.brain.skrl_patches").info(
                "shared encoder: dropped %d duplicate parameter tensor(s) from the "
                "optimizer; they would each have been stepped twice per update", removed,
            )
        return removed


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
