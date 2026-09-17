"""ONE stop-gradient EMA copy of the trunk, shared by every term of a composite row.

Every wave-17 token objective needs a target computed from an encoder the loss also trains, and
the wave-17 plan's collapse note is explicit: without a stop-grad EMA target the optimum is a
constant z. `slot_contrast_aux.py` established the form (deepcopy, `requires_grad_(False)`,
`tp = ema·tp + (1−ema)·sp`, buffers copied, held in a list so it is not registered) and this is
that form, lifted to ONE holder above the losses.

⛔ WHY IT IS NOT PER-LOSS. `SlotContrastAuxLoss._update_target` builds a deepcopy per aux
INSTANCE. A composition row (cltt_ref + slots + ego) would then hold two or three copies of the
trunk -- 800K parameters each, plus their activations during the teacher forward -- and, worse,
the terms would be trained against DIFFERENT targets that drift apart, so "the same teacher
tokens" in two diagnostics would not be the same tokens. The composite (`with_cltt_ref.py`)
owns one holder, steps it ONCE per compute, and hands it to each term.

⛔ THE COPY IS MADE AT CONSTRUCTION, NOT LAZILY AT THE FIRST UPDATE, AND THAT IS A CORRECTNESS
REQUIREMENT ON A GPU ARM. Inside `AuxLossPPO.update` the actor/critic have already run under
`shared_feature_cache`, so the encoder carries `_nett_shared_feature_cache = (x, feats)` where
`feats` is a NON-LEAF tensor holding the minibatch graph. `copy.deepcopy(module)` copies
`__dict__`, and `Tensor.__deepcopy__` raises `RuntimeError: Only Tensors created explicitly by
the user ... support the deepcopy protocol` for a non-leaf tensor. A lazy copy therefore dies at
update 1 with the cache on -- and never on CPU, where the cache path is inactive in tests.
Constructing here (AuxLossPPO builds the aux before any update) avoids it, and the cache slot is
additionally neutralised around the copy so a late construction cannot resurrect it.

⛔ OBJECTIVE CHANGE 2026-09-17 (owner, workspace DECISIONS): `cltt_ref` now excludes each anchor's OWN FRAME from its negatives, so every cltt_ref arm trained before this commit ran a different objective and is NOT comparable to one trained after it.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ..models.utils.features import _CACHE_ATTR
from .knobs import _env_unit_interval


class EMATeacher:
    """A frozen EMA copy of the trunk. NOT an nn.Module, and it holds its copy in a list.

    Both facts keep the copy out of every parameter set that matters: assigning an nn.Module to
    an attribute of an nn.Module registers it, which would put 800K frozen parameters into
    `aux.parameters()` -- and `head` is what AuxLossPPO hands the optimizer, so the failure would
    be a second copy of the trunk in the optimizer rather than a crash.
    """

    DECAY_ENV = "NETT_AUX_EMA_DECAY"

    def __init__(self, encoder: nn.Module, decay: float = 0.996) -> None:
        self.decay = _env_unit_interval(self.DECAY_ENV, decay)
        self.updates = 0
        prev = getattr(encoder, _CACHE_ATTR, None)
        if prev is not None:
            setattr(encoder, _CACHE_ATTR, None)
        try:
            target = copy.deepcopy(encoder).eval()
        finally:
            if prev is not None:
                setattr(encoder, _CACHE_ATTR, prev)
        for p in target.parameters():
            p.requires_grad_(False)
        setattr(target, _CACHE_ATTR, None)
        self._target: list[nn.Module] = [target]

    @property
    def module(self) -> nn.Module:
        return self._target[0]

    @torch.no_grad()
    def step(self, encoder: nn.Module) -> None:
        """EMA the target toward the live trunk. Called ONCE per composite compute.

        ⚠ Once per COMPUTE, not once per term: two terms stepping the same holder would apply
        the decay twice per minibatch, i.e. an effective decay of 0.996² = 0.992, and nothing
        downstream would say so. The composite owns the call.
        """
        target = self._target[0]
        for tp, sp in zip(target.parameters(), encoder.parameters()):
            tp.mul_(self.decay).add_(sp.detach(), alpha=1.0 - self.decay)
        for tb, sb in zip(target.buffers(), encoder.buffers()):
            tb.copy_(sb)
        self.updates += 1

    @torch.no_grad()
    def tokens(self, prepared: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Teacher tokens (B, N, D) and the grid, from an already-prepared image. Never grad."""
        from .token_features import spatial_tokens
        tokens, grid = spatial_tokens(self._target[0], prepared)
        return tokens.detach(), grid
