"""NETT_ENC_LR_SCALE: the encoder and the aux head step at ``lr x scale``, the policy/value heads at ``lr``.

LRP2 (workspace FINDINGS add. 327-329, notes/researcher2/lrp2): the add. 327 "2x LR" doubled the encoder
step and the heads' step together. This knob separates them.

* UNSET (the default) changes nothing: ``split_encoder_group`` is never called.
* SET: the encoder's parameters (the unique set over ``policy.encoder`` and ``value.encoder``) leave the
  PPO param group(s) IN PLACE (xsp2_aux.exclude_from_optimizer, so ``checkpoint_modules["optimizer"]``
  keeps the same object). They join the aux head's group, which is set to ``base x scale``. The heads keep
  ``base``. Under NETT_SHARED_ENCODER_DUP=1 each encoder tensor keeps both of its slots (stepped twice per update,
  as the DUP recipe always was) in the new group. NETT_DIAG_LR_ANNEAL=linear anneals each group from its own base (ppo_metrics._lr_anneal_base).
  One banner line states both lrs.

Refused, with the reason, rather than trained without the split:
* a non-positive, non-finite or unparsable value;
* an agent that is not AuxLossPPO, or an AuxLossPPO with no active aux term (agent_factory / AuxLossPPO);
* a term that owns its optimizer (xsp2 / slot_recon / stp): those already set their own LR scale;
* a skrl learning_rate_scheduler (KLAdaptiveLR writes ONE lr into every group);
* an encoder with no trainable parameter in the optimizer (frozen: the scale would act on nothing);
* a resumed optimizer (Adam state present at the first update): the saved groups carry the parent's lrs.
"""
from __future__ import annotations

import math
import os

ENC_LR_SCALE_ENV = "NETT_ENC_LR_SCALE"
ENC_LR_KEY = "Learning / Encoder learning rate"


def enc_lr_scale() -> float | None:
    """The NETT_ENC_LR_SCALE value; None when unset or empty."""
    raw = os.environ.get(ENC_LR_SCALE_ENV, "").strip()
    if not raw:
        return None
    try:
        scale = float(raw)
    except ValueError:
        raise ValueError(f"{ENC_LR_SCALE_ENV}={raw!r}: expected a positive number (unset = one lr for all)") from None
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{ENC_LR_SCALE_ENV}={raw!r}: expected a positive finite number")
    return scale


def encoder_parameters(agent) -> list:
    """Unique parameters of ``policy.encoder`` and ``value.encoder``, in first-seen order."""
    seen, out = set(), []
    for role in ("policy", "value"):
        enc = getattr(getattr(agent, role, None), "encoder", None)
        if enc is None:
            continue
        for p in enc.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def split_encoder_group(optimizer, enc_params, head_params, scale: float) -> dict:
    """Move ``enc_params`` (in place) out of the PPO group(s) into the aux head's group, and set that group to base x scale.

    The head's group is the LAST group and holds exactly ``head_params``: AuxLossPPO calls this right after adding it.
    Returns {"base", "encoder_lr", "encoder_tensors", "head_tensors", "frozen_tensors"} for the banner and tests.
    """
    from .xsp2_aux import exclude_from_optimizer

    groups = optimizer.param_groups
    head = [id(p) for p in head_params]
    if len(groups) < 2 or [id(p) for p in groups[-1]["params"]] != head:
        raise ValueError(f"{ENC_LR_SCALE_ENV}: the last param group is not exactly the aux head's")
    lrs = {float(g["lr"]) for g in groups}
    if len(lrs) != 1:
        raise ValueError(f"{ENC_LR_SCALE_ENV}: the optimizer already holds groups at different lrs {sorted(lrs)}")
    base = lrs.pop()
    # Each encoder tensor moves with its MULTIPLICITY: under NETT_SHARED_ENCODER_DUP=1 (the BL2 recipe) skrl's chain holds
    # the shared encoder twice and Adam steps it twice per update; it stays twice in the new group, so DUP keeps its
    # meaning and only the lr changes. Deduped (the default) it is once.
    count = {}
    for g in groups[:-1]:
        for p in g["params"]:
            count[id(p)] = count.get(id(p), 0) + 1
    enc = [p for p in enc_params if id(p) in count]
    if not any(p.requires_grad for p in enc):
        raise ValueError(f"{ENC_LR_SCALE_ENV}: the encoder has no trainable parameter in the optimizer "
                         "(frozen encoder?); the scale would act on nothing")
    moved = [p for p in enc for _ in range(count[id(p)])]
    removed = exclude_from_optimizer(optimizer, enc)
    if removed != len(moved):
        raise ValueError(f"{ENC_LR_SCALE_ENV}: removed {removed} encoder tensors, expected {len(moved)}")
    if any(not g["params"] for g in groups[:-1]):
        raise ValueError(f"{ENC_LR_SCALE_ENV}: a param group holds only encoder parameters; no head would keep the base lr")
    groups[-1]["params"].extend(moved)
    groups[-1]["lr"] = base * scale
    return {"base": base, "encoder_lr": base * scale, "encoder_tensors": len(enc), "head_tensors": len(head),
            "frozen_tensors": sum(not p.requires_grad for p in enc), "encoder_slots": len(moved)}


def refuse_resumed(optimizer) -> None:
    """At the first update: a scaled optimizer that already holds Adam state was loaded from a checkpoint."""
    state = getattr(optimizer, "state", None) or {}
    if any(state.get(p) for g in optimizer.param_groups for p in g["params"]):
        raise ValueError(f"{ENC_LR_SCALE_ENV} on a resumed optimizer: the loaded param groups carry the parent's "
                         "lrs, so the split cannot be stated; run it fresh")
