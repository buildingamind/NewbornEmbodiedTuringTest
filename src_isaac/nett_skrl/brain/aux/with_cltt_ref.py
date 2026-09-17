"""Compose a NEW auxiliary term with the incumbent `cltt_ref`, so a row differs from its control
by exactly ONE term.

Wave 17's rows 02-04 (notes/researcher/wave17-object-centric.md) are "ViT-CLTT-Ref + X". The
control row runs `CLTTReferenceAuxLoss` alone; each candidate runs this module, whose loss is

    L_aux = w_ref * L_cltt_ref + L_term          (w_ref = NETT_AUX_CLTT_REF_WEIGHT, default 1.0)

and AuxLossPPO multiplies the sum by aux_weight as usual. At the default the cltt_ref addend is
the control's objective unchanged, built by the same class from the same knobs, so the contrast
is the added term. ⚠ Moving w_ref away from 1.0 breaks that: the row then differs from the
control in TWO places, and must be read as such.

THE AuxLossPPO CONTRACT THIS SATISFIES (ppo_aux.py, read before changing anything here)
- ``head``: the ONLY parameters AuxLossPPO registers with the optimizer (`add_param_group`) and
  adds to the grad-norm clip. ``head`` is therefore a ModuleList of BOTH heads -- a term whose
  parameters live anywhere else is silently NEVER UPDATED, while its loss still falls through
  the encoder. ⛔ Two corollaries, both checked at construction:
    * no head parameter may be an ENCODER parameter: the policy group already owns those.
      torch's `add_param_group` would raise "some parameters appear in more than one parameter
      group" -- but only when a real optimizer exists, deep inside agent construction, and
      without naming the term; this refuses at the term's own construction, with the reason;
    * the two heads must not share a parameter.
  A stop-gradient EMA target (the collapse guard P2/P3 need) must NOT be in ``head``: keep it
  as a buffer or a module outside ``head`` with requires_grad False, updated by the term.
- ``needs_memory`` / ``attach_memory``: forwarded to cltt_ref always, and to the term when the
  term declares ``needs_memory``.
- ``last_scalars``: merged, every key prefixed (``cltt_ref_*``, ``<term>_*``) plus the two
  unweighted addends (``cltt_ref_loss``, ``<term>_loss``) and the weight. "Differs from the
  control by one term" is only auditable if the cltt_ref addend's value is logged beside the
  control's. Prefixes use ``_``, not ``/``: ppo_aux emits ``Loss / Aux {kind} {key}``, and a
  slash would nest a new tensorboard hierarchy.
- ``last_window_turn``: forwarded from the term if it publishes one, so
  ``ppo_aux.track_transit_mask`` sees a transit-weighted term exactly as it sees vicreg_tt.

ORDER. cltt_ref is computed FIRST, so it consumes the global RNG stream from the same state it
would in the control at the start of each minibatch's aux call; the term draws after it.

⛔ OBJECTIVE CHANGE 2026-09-17 (owner, workspace DECISIONS): `cltt_ref` now excludes each anchor's OWN FRAME from its negatives, so every cltt_ref arm trained before this commit ran a different objective and is NOT comparable to one trained after it.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from .cltt_ref_aux import CLTTReferenceAuxLoss
from .knobs import _env_positive_float

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class WithCLTTRef(nn.Module):
    """``w_ref * cltt_ref + term``, exposing the single ``head`` AuxLossPPO optimizes."""

    needs_memory = True
    CLTT_REF_WEIGHT_ENV = "NETT_AUX_CLTT_REF_WEIGHT"

    def __init__(self, encoder: nn.Module, term: nn.Module, name: str) -> None:
        super().__init__()
        if not _NAME.match(name or "") or name.startswith("cltt_ref"):
            raise ValueError(
                f"WithCLTTRef: term name {name!r} must be lowercase snake_case and must not "
                f"start with 'cltt_ref' (its scalars would collide with the reference term's)."
            )
        for attr in ("head", "compute"):
            if not hasattr(term, attr):
                raise TypeError(f"WithCLTTRef: term {type(term).__name__} has no `{attr}`.")
        if not isinstance(term.head, nn.Module):
            raise TypeError(
                f"WithCLTTRef: {type(term).__name__}.head must be an nn.Module holding every "
                f"trainable parameter of the term; got {type(term.head).__name__}."
            )
        self.name = name
        self.cltt_ref = CLTTReferenceAuxLoss(encoder)
        self.term = term
        self.cltt_ref_weight = _env_positive_float(self.CLTT_REF_WEIGHT_ENV, 1.0)

        enc_ids = {id(p) for p in encoder.parameters()}
        ref_ids = {id(p) for p in self.cltt_ref.head.parameters()}
        term_params = list(term.head.parameters())
        if not term_params:
            raise ValueError(
                f"WithCLTTRef: {type(term).__name__}.head has no parameters. A term with nothing "
                f"to train adds only encoder pressure; if that is intended, say so with a "
                f"distinct class rather than an empty head."
            )
        if any(id(p) in enc_ids for p in term_params) or ref_ids & enc_ids:
            raise ValueError(
                "WithCLTTRef: a head contains ENCODER parameters. The policy optimizer group "
                "already owns them; registering them again via `head` steps them twice."
            )
        if any(id(p) in ref_ids for p in term_params):
            raise ValueError("WithCLTTRef: the term's head shares parameters with cltt_ref's head.")
        # ⛔ THE ONE HEAD. See the module docstring: anything not reachable from here is never
        # optimized.
        self.head = nn.ModuleList([self.cltt_ref.head, term.head])

        self._memory = None
        self.last_scalars: dict = {}
        # ⛔ ONE EMA TEACHER, BUILT HERE AND NOW. Every wave-17 token target is computed from an
        # encoder the loss also trains, so it needs a stop-grad EMA copy or its optimum is a
        # constant z. One holder above the terms, because a per-term deepcopy would (a) hold two
        # or three copies of the trunk on a composition row and (b) give the terms targets that
        # DRIFT APART, so "the same teacher tokens" in two diagnostics would not be.
        # ⛔ AT CONSTRUCTION, NOT AT THE FIRST COMPUTE: inside AuxLossPPO.update the encoder
        # carries the shared feature cache, whose tensor is non-leaf, and deepcopy raises on it.
        # See ema_teacher.py.
        self._teacher = None
        if getattr(term, "needs_teacher", False):
            from .ema_teacher import EMATeacher
            self._teacher = EMATeacher(encoder)
            term.attach_teacher(self._teacher)

    def attach_memory(self, memory) -> None:
        self._memory = memory
        self.cltt_ref.attach_memory(memory)
        if getattr(self.term, "needs_memory", False):
            self.term.attach_memory(memory)

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        ref = self.cltt_ref.compute(encoder, observations)
        # ⚠ ONE EMA STEP PER COMPUTE, not one per term: two terms stepping it would apply the
        # decay twice per minibatch (0.996^2 = 0.992) with nothing downstream saying so.
        if self._teacher is not None:
            self._teacher.step(encoder)
        term = self.term.compute(encoder, observations)
        scalars = {
            "cltt_ref_loss": float(ref.detach()),
            f"{self.name}_loss": float(term.detach()),
            "cltt_ref_weight": float(self.cltt_ref_weight),
            # ⚠ Emitted ALWAYS, sentinel when there is no teacher: "this row has no EMA target"
            # and "the EMA never stepped" are different facts, and a missing series reads as
            # neither. A frozen teacher (updates not rising) is a dead target.
            "ema_updates": float(self._teacher.updates) if self._teacher is not None else -9.0,
        }
        for prefix, source in (("cltt_ref", self.cltt_ref), (self.name, self.term)):
            for key, value in (getattr(source, "last_scalars", None) or {}).items():
                scalars[f"{prefix}_{key}"] = float(value)
        self.last_scalars = scalars
        turn = getattr(self.term, "last_window_turn", None)
        if turn is not None:
            self.last_window_turn = float(turn)
        return self.cltt_ref_weight * ref + term
