"""`cltt_ref+embdiv_unif`: the attached CLTT-Ref term PLUS the EMBDIV uniformity term (BL2, owner 2026-11-09).

The BL2 "<trunk>+CLTT-Ref+UNIF" labels (examples/campaign_train.py) carry BOTH objectives on one shared encoder:

```
L_aux = L_cltt_ref + lambda * L_unif          lambda = NETT_AUX_EMBDIV_WEIGHT (finite float > 0, default 0.01)
total = PPO + aux_weight * L_aux              aux_weight = the label's 1.0, REQUIRED to be exactly 1.0
```

so the run's loss is PPO + 1.0 * L_cltt_ref + lambda * L_unif: CLTT keeps the weight of every CLTTA arm and the
uniformity term enters at lambda, as on a "+UNIF" arm whose NETT_AUX_WEIGHT_OVERRIDE is lambda.

THE TWO ADDENDS ARE THE EXISTING TERMS, UNCHANGED.
* ``self.cltt`` is the object ``AUX_LOSSES["cltt_ref"]`` builds for this encoder (CLTTReferenceAuxLoss, every
  NETT_AUX_CLTT_* / NETT_AUX_BATCH / NETT_CLTT_REF_DEDUP knob read by it exactly as on a CLTTA arm).
* ``self.unif`` is ``EmbDivTerm(encoder, "embdiv_unif")`` (embdiv_aux.py; NETT_AUX_EMBDIV_T / _BATCH /
  _DIAG_EVERY / _FORM read by it exactly as on a "+UNIF" arm; it admits CompactViT, NatureCNN, Compact3DCNN and
  CompactViViT only, the four BL2 trunks).

WEIGHTS. AuxLossPPO multiplies compute() by its ``_aux_weight``. That weight is CLTT's, so it must be exactly 1.0:
``check_agent`` (called by AuxLossPPO.__init__ right after the build) REFUSES any other value -- an
NETT_AUX_WEIGHT_OVERRIDE would silently rescale CLTT, and campaign_train also refuses it on these labels. lambda
is this module's own knob, NETT_AUX_EMBDIV_WEIGHT; its default (0.01, the "+UNIF" labels' aux_weight) exists so
the registry sweep (tests/test_slot_contrast_aux.py) can build the key without env. campaign_train refuses
NETT_AUX_EMBDIV_WEIGHT on every other label (a plain "+UNIF" arm's lambda stays NETT_AUX_WEIGHT_OVERRIDE).

ORDER / RNG. Every minibatch computes CLTT FIRST, then the uniformity term: CLTT draws its episode window from the
global torch RNG exactly as on a CLTTA arm, and the uniformity term's randperm subsample draws after it.

GRAPH INDEPENDENCE (``independent_of_ppo_graph = True``; DECISIONS §111 fix 2 / 2b). CLTT encodes frames it draws
from the attached memory and ignores ``observations``. The uniformity term reads ``observations`` -- the PPO
minibatch's raw memory tensor, which carries no graph -- prepares it under no_grad and re-encodes it through
``encoder.encode_prepared`` (encoders/base.py), i.e. the encoder's own ``forward``: never ``features_forward``, so
it neither reads nor fills the shared feature cache and ignores RL_NO_GRAD_ATTR. Its graph and the PPO graph
therefore share only parameter LEAVES, so AuxLossPPO may run this aux AFTER the PPO backward with no retain_graph,
and under NETT_DECOUPLE_ENCODER=1 run the RL encoder pass under no_grad: the uniformity gradient still reaches
the encoder through its own forward. ⚠ MEMORY: the aux phase holds CLTT's graph (B + max(offsets) frames) AND
the uniformity term's (NETT_AUX_EMBDIV_BATCH frames, 64) until the one aux backward, so the update's peak is
max(PPO, CLTT + unif) where a CLTTA arm's is max(PPO, CLTT).

THE AuxLossPPO CONTRACT (ppo_aux.py):
* ``head`` IS CLTT's head (the same module): the param group AuxLossPPO adds to the optimizer, the grad clip and
  any checkpoint see exactly a CLTTA arm's head, in the same order (the uniformity term has no parameters).
* ``needs_memory`` / ``attach_memory`` -> CLTT (the uniformity term needs none).
* ``last_scalars``: CLTT's under its own names (as a CLTTA arm's tfevents) plus ``cltt_ref_loss``, ``unif_loss``
  and ``unif_B`` (both terms UNWEIGHTED) and ``unif_weight`` (lambda); a collision refuses.
* ``after_update`` -> the uniformity term's, which writes ``embdiv_diag.csv`` exactly as on a "+UNIF" arm (same
  columns, form "uniformity", one row per diag update), with ``aux_weight`` = lambda and ``loss_weighted`` =
  lambda x loss_raw (EmbDivTerm.diag_weight).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .embdiv_aux import EmbDivTerm
from .knobs import _env_positive_float

#: aux key -> the label family "<trunk>+CLTT-Ref+UNIF".
KEY = "cltt_ref+embdiv_unif"
#: lambda, the uniformity term's weight inside this aux (module docstring WEIGHTS).
UNIF_WEIGHT_ENV = "NETT_AUX_EMBDIV_WEIGHT"
#: The scalars this module adds beside CLTT's own.
OWN_SCALARS = ("cltt_ref_loss", "unif_loss", "unif_B", "unif_weight")


class CLTTRefUnifAux(nn.Module):
    """``L_cltt_ref + lambda * L_unif`` on one encoder (module docstring)."""

    needs_memory = True
    #: See the module docstring GRAPH INDEPENDENCE: neither term reads the PPO forward's graph.
    independent_of_ppo_graph = True

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        from .ppo_aux import AUX_LOSSES      # lazily: ppo_aux imports this module lazily from its builder
        self.cltt = AUX_LOSSES["cltt_ref"](encoder)        # FIRST: its head init draws from the RNG, as on CLTTA
        self.unif = EmbDivTerm(encoder, "embdiv_unif")
        if self.unif.form != "uniformity" or list(self.unif.parameters()):
            raise ValueError("cltt_ref+embdiv_unif: the embdiv term must be the parameter-free uniformity form.")
        self.unif_weight = _env_positive_float(UNIF_WEIGHT_ENV, 0.01)
        #: embdiv_diag.csv's aux_weight column (and loss_weighted) report lambda, not AuxLossPPO's 1.0.
        self.unif.diag_weight = self.unif_weight
        #: ⛔ THE ONE HEAD AuxLossPPO OPTIMIZES: CLTT's, the same module (module docstring).
        self.head = self.cltt.head
        self.last_scalars: dict = {}

    def check_agent(self, agent) -> None:
        """AuxLossPPO.__init__ hook: the agent's aux weight is CLTT's and must be exactly 1.0."""
        weight = float(getattr(agent, "_aux_weight", float("nan")))
        if weight != 1.0:
            raise ValueError(
                f"{KEY}: the aux weight multiplies L_cltt_ref + lambda * L_unif, so it is CLTT's weight and must be "
                f"exactly 1.0 (a CLTTA arm's); got {weight!r}. Set the uniformity weight with {UNIF_WEIGHT_ENV}, "
                f"never with NETT_AUX_WEIGHT / NETT_AUX_WEIGHT_OVERRIDE.")

    def attach_memory(self, memory) -> None:
        self.cltt.attach_memory(memory)

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        ref = self.cltt.compute(encoder, observations)      # FIRST (ORDER / RNG)
        unif = self.unif.compute(encoder, observations)
        scalars = dict(self.cltt.last_scalars)
        clash = sorted(set(scalars) & set(OWN_SCALARS))
        if clash:
            raise ValueError(f"{KEY}: cltt_ref publishes {clash}, which this aux's own scalars would overwrite.")
        scalars.update({"cltt_ref_loss": float(ref.detach()), "unif_loss": float(self.unif.last_scalars["loss"]),
                        "unif_B": float(self.unif.last_scalars["B"]), "unif_weight": float(self.unif_weight)})
        self.last_scalars = scalars
        return ref + self.unif_weight * unif

    def after_update(self, agent, *, timestep: int, timesteps: int, ppo: dict | None = None) -> dict | None:
        """The uniformity term's per-update collapse diagnostics (embdiv_diag.csv), at aux_weight = lambda."""
        return self.unif.after_update(agent, timestep=timestep, timesteps=timesteps, ppo=ppo)


def build(encoder: nn.Module) -> CLTTRefUnifAux:
    """The AUX_LOSSES builder for ``cltt_ref+embdiv_unif``."""
    return CLTTRefUnifAux(encoder)
