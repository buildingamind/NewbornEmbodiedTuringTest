"""fp16 loss scaling for the encoder-level autocast knobs (NETT_AMP, NETT_MATCHED_AMP).

The fleet's mixed precision is ENCODER-LEVEL: an encoder's forward runs under CUDA autocast and returns
fp32 features (NatureCNN and CompactViT read NETT_AMP, MatchedEncoder reads NETT_MATCHED_AMP), so the
policy/value heads, the distribution and GAE stay fp32. bf16 has fp32's exponent range and needs no loss
scaling. fp16 does not: small gradients underflow to zero in the fp16 backward. This module gives the
fp16 encoders the standard remedy -- a torch GradScaler on the optimizer that steps their graphs.

⭐ NO PARALLEL UPDATE LOOP. skrl's PPO already owns ``self.scaler`` (built DISABLED, since
``cfg.mixed_precision`` is never set here) and both update paths step through it:
skrl's own ``_update`` (MetricsPPO) and AuxLossPPO.update (PPO and aux losses alike) run
``scaler.scale(loss).backward() -> scaler.unscale_(optimizer) -> clip_grad_norm_ -> scaler.step ->
scaler.update``. A disabled scaler is a pass-through; this module only swaps in an ENABLED one, so
unscale-before-clip and skip-on-inf/nan come from that existing code. ``cfg.mixed_precision`` is left
False on purpose: it would also autocast the heads and the value/GAE path, which no bf16 row does.

⛔ REFUSALS (a precision that is asked for and cannot be honoured is never ignored):
  * a knob set to fp16 that no encoder of the agent reads (e.g. NETT_AMP=fp16 on a 3DCNN, or
    NETT_MATCHED_AMP=fp16 on a CompactViT);
  * an fp16 encoder under an agent class whose update does not step through ``self.scaler``;
  * an fp16 encoder with an aux term that steps its OWN optimizer (``owns_optimizer``, e.g. xsp2):
    that optimizer would step unscaled fp16 graphs.

Default (both knobs unset), "off" and "bf16" leave the agent exactly as before: the disabled skrl scaler
and no extra checkpoint key. Under fp16 the scaler's state is checkpointed as ``grad_scaler`` beside
the optimizer, so a resume restores the loss scale.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger("nett")

#: The checkpoint_modules key of the enabled scaler (absent when fp16 is off).
SCALER_CHECKPOINT_KEY = "grad_scaler"


def fp16_requested_knobs() -> list[str]:
    """The AMP knobs whose value spells fp16, each read exactly as its encoders read it (NETT_AMP by
    NatureCNN's parser; NETT_MATCHED_AMP stripped + lowercased, as matched_amp_dtype does, but without its
    typo refusal, so an unused NETT_MATCHED_AMP on a non-matched row stays as inert as it always was)."""
    from .encoders.nature_cnn import FP16_SPELLINGS, _amp_dtype
    knobs = []
    if _amp_dtype() is torch.float16:
        knobs.append("NETT_AMP")
    if os.environ.get("NETT_MATCHED_AMP", "bf16").strip().lower() in FP16_SPELLINGS:
        knobs.append("NETT_MATCHED_AMP")
    return knobs


def amp_encoders(agent) -> list[nn.Module]:
    """Every distinct module in the agent's models whose class declares an AMP_KNOB."""
    seen: dict[int, nn.Module] = {}
    for model in getattr(agent, "models", {}).values():
        if isinstance(model, nn.Module):
            for m in model.modules():
                if getattr(type(m), "AMP_KNOB", None) and id(m) not in seen:
                    seen[id(m)] = m
    return list(seen.values())


def configure_fp16_grad_scaler(agent) -> bool:
    """Enable loss scaling on ``agent`` when one of its encoders autocasts to float16; refuse what fp16
    cannot cover (module docstring). Returns True when the scaler was enabled. Idempotent."""
    requested = fp16_requested_knobs()
    encoders = amp_encoders(agent)
    names = sorted({type(e).__name__ for e in encoders}) or ["<no encoder with a precision knob>"]
    for knob in requested:
        if not any(type(e).AMP_KNOB == knob for e in encoders):
            raise ValueError(
                f"{knob}={os.environ.get(knob)!r} asks for fp16, but this agent's encoder ({', '.join(names)}) "
                f"does not read {knob}, so it would run at another precision while the row records fp16. "
                f"fp16 paths: NETT_AMP -> NatureCNN, CompactViT; NETT_MATCHED_AMP -> MatchedEncoder.")
    fp16 = [e for e in encoders if e.amp_dtype() is torch.float16]
    if not fp16:
        return False

    from .aux.ppo_aux import AuxLossPPO
    from .ppo_metrics import MetricsPPO
    if not isinstance(agent, (MetricsPPO, AuxLossPPO)):
        raise ValueError(
            f"fp16 encoder ({', '.join(sorted({type(e).__name__ for e in fp16}))}) under "
            f"{type(agent).__name__}: only MetricsPPO and AuxLossPPO step through the GradScaler. "
            f"Unset the fp16 knob ({', '.join(requested)}).")
    aux = getattr(agent, "_aux", None)
    if aux is not None and getattr(aux, "owns_optimizer", False):
        raise ValueError(
            f"fp16 encoder with aux {getattr(agent, '_aux_kind', type(aux).__name__)!r}, which steps its own "
            f"optimizer outside the GradScaler; refusing to step unscaled fp16 gradients.")

    scaler = getattr(agent, "scaler", None)
    if not (isinstance(scaler, torch.amp.GradScaler) and scaler.is_enabled()):
        scaler = torch.amp.GradScaler(device=torch.device(agent.device).type, enabled=True)
        agent.scaler = scaler
    agent.checkpoint_modules[SCALER_CHECKPOINT_KEY] = scaler
    logger.info("fp16 autocast (%s) on %s: GradScaler ENABLED (init scale %.0f), checkpointed as %r.",
                ", ".join(requested), ", ".join(sorted({type(e).__name__ for e in fp16})),
                scaler.get_scale(), SCALER_CHECKPOINT_KEY)
    return True
