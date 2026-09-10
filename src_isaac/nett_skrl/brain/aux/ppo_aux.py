"""PPO subclass that adds an optional SimCLR auxiliary loss to the update step.

The auxiliary loss shapes the SHARED visual encoder backbone with gradient
pressure that is INDEPENDENT of the (weak) RL reward, fixing transformer
representation collapse (see ``brain/aux/simclr_aux.py``).

It is added inside skrl PPO's existing minibatch optimization step (so it shares
the same optimizer and AMP scaler). The projection head's parameters are added
to the optimizer as a new param group, so they are optimized alongside the
policy/value networks.

When the aux loss is disabled (the default), this subclass is not used at all
and behavior is byte-for-byte identical to stock skrl PPO.
"""

from __future__ import annotations

import os

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl import config, logger

from ..skrl_patches import NETTBootstrapMixin, relaxed_determinism, strict_update

from .simclr_aux import SimCLRAuxLoss


def _build_simclr(encoder):
    return SimCLRAuxLoss(encoder)


def _build_vicreg(encoder):
    from .vicreg_aux import VICRegAuxLoss
    return VICRegAuxLoss(encoder)


def _build_cltt(encoder):
    """CLTT: temporal-contrastive NT-Xent. The positive pair is the framestack's own
    (first, last) frame, so this REQUIRES framestack=True and refuses without it."""
    from .cltt_aux import CLTTAuxLoss
    return CLTTAuxLoss(encoder)


def _build_cltt_ref(encoder):
    from .cltt_ref_aux import CLTTReferenceAuxLoss
    return CLTTReferenceAuxLoss(encoder)


def _build_vicreg_tt(encoder):
    from .vicreg_tt_aux import VICRegTemporalAuxLoss
    return VICRegTemporalAuxLoss(encoder)


def _build_eoo(encoder):
    from .eoo_aux import EoOAuxLoss
    return EoOAuxLoss(encoder)


def _build_gwm(encoder):
    from .gwm_aux import GWMAuxLoss
    return GWMAuxLoss(encoder)


#: The ONE place a loss becomes reachable. `NETT_AUX_LOSS` is matched against these
#: keys; anything else raises in ``AuxLossPPO.__init__`` rather than disabling itself.
#: Builders import lazily so an unrelated import error in one loss cannot take down
#: every arm that uses a different one.
AUX_LOSSES = {
    "simclr": _build_simclr,
    # ⛔ 'cltt' was declared by two MODELS entries since 2026-08-28 and was NEVER here,
    # so SimCLR-CLTT and ViT-CLTT raised at construction and could not be launched at
    # all. The raise was correct -- it is why neither ever trained as silent vanilla
    # PPO -- but the registry entry was simply missing. Added 2026-09-02.
    "cltt": _build_cltt,
    "cltt_ref": _build_cltt_ref,
    "vicreg": _build_vicreg,
    "vicreg_tt": _build_vicreg_tt,
    "eoo": _build_eoo,
    "gwm": _build_gwm,
}


def track_transit_mask(agent, last: float | None, cumulative: float, seen: int) -> None:
    """Publish the transit-mask sampler's STATE, not only its successes.

    The mask has four outcomes and three of them sample uniformly. If only the
    working one wrote a scalar, an engaged mask and a mask that silently fell
    back to uniform would both reach tensorboard as an absent series -- and an
    absent series reads as benign. So the state is always emitted:

        0 engaged   1 mask off   2 no actions tensor   3 no commanded rotation
        4 never ran

    ``Loss / Aux transit |turn|`` carries the magnitude, averaged over the
    minibatches that actually engaged (never over a state code, which would
    name nothing), and is absent exactly when the mask never engaged.
    """
    if last is None:
        return
    agent.track_data("Loss / Aux transit state", 0.0 if last >= 0.0 else -last)
    if seen:
        agent.track_data("Loss / Aux transit |turn|", cumulative / seen)


class AuxLossPPO(NETTBootstrapMixin, PPO):
    """skrl PPO with an optional SimCLR auxiliary loss on the shared encoder.

    Extra config (read from the PPO_CFG-style cfg object or set post-construction):
      - ``aux_loss``: "none" | "simclr"
      - ``aux_weight``: float multiplier for the aux loss term.

    These are injected by ``agent_factory.build_agents`` from the
    ``NETT_AUX_LOSS`` / ``NETT_AUX_WEIGHT`` environment variables.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Allow passing aux_loss / aux_weight via kwargs without polluting PPO.
        self._aux_kind = str(kwargs.pop("aux_loss", "none") or "none")
        self._aux_weight = float(kwargs.pop("aux_weight", 0.0) or 0.0)
        super().__init__(*args, **kwargs)

        self._aux = None
        if self._aux_kind == "none":
            return

        # ⛔★★★★★ THIS SELECTION USED TO BE `if kind in ("simclr", "vicreg")` WITH NO
        # ELSE AND NO LOG LINE. Any other kind -- "cltt" among them -- fell through
        # silently: `self._aux` stayed None, `update()` added 0.0, and the arm trained
        # as VANILLA PPO for its whole run while every log line said aux=<kind>. That is
        # a SILENT NO-OP AT THE SELECTION POINT, and it is the fifth distinct path by
        # which this campaign has disabled an objective without saying so.
        #
        # An unknown kind now RAISES. A misspelled or unregistered loss must cost a
        # startup error, never a GPU-night that scores like a scientific negative.
        # To add a loss: one entry in AUX_LOSSES. There is no second place to edit.
        if self._aux_kind not in AUX_LOSSES:
            raise ValueError(
                f"AuxLossPPO: unknown aux loss {self._aux_kind!r}. "
                f"Registered: {sorted(AUX_LOSSES)}. Set NETT_AUX_LOSS to one of these, "
                f"or to 'none'. Refusing to train with the objective silently absent."
            )

        # ⚠ A DECLARED OBJECTIVE WITH ZERO WEIGHT IS THE SAME NO-OP WEARING A LABEL.
        # `aux_loss * 0.0` contributes nothing to the backward pass, so the run is
        # indistinguishable from plain PPO except in the log. Refuse it. An ablation
        # that genuinely wants weight 0 must say so out loud via NETT_AUX_ALLOW_ZERO=1.
        if self._aux_weight <= 0.0:
            if os.environ.get("NETT_AUX_ALLOW_ZERO", "").strip() not in ("1", "true", "yes"):
                raise ValueError(
                    f"AuxLossPPO: aux loss {self._aux_kind!r} is declared but "
                    f"aux_weight={self._aux_weight}. A zero-weight objective contributes "
                    f"NOTHING to the backward pass -- this run would be plain PPO with a "
                    f"misleading label. Set NETT_AUX_WEIGHT>0, or NETT_AUX_ALLOW_ZERO=1 "
                    f"if a zero-weight ablation is genuinely intended."
                )
            logger.warning(
                "AuxLossPPO: %s DECLARED WITH WEIGHT 0 and NETT_AUX_ALLOW_ZERO set. "
                "This run is plain PPO; the aux objective contributes nothing.",
                self._aux_kind,
            )
            return

        # ⚠ A ONE-SAMPLE AUX BATCH IS A SILENT ZERO-GRADIENT PATH FOR THIS WHOLE
        # FAMILY (raised by seat:insect from the MoTok port, where the VQ update is
        # degenerate below batch 2). It is the same defect that made CLTT identically
        # 0.0 for the entire campaign: at B=1 a contrastive softmax has one logit,
        # -log(1) = 0; a batch variance is 0 or NaN; a per-slot mean equals its own
        # sample. The objective still evaluates and still returns a number. Refuse it
        # once, here, rather than in each loss.
        aux_batch = int(os.environ.get("NETT_AUX_BATCH", "0") or 0)
        if aux_batch == 1:
            raise ValueError(
                "AuxLossPPO: NETT_AUX_BATCH=1. Every aux objective in this package is "
                "degenerate at batch 1 -- contrastive terms reduce to -log(1)=0, batch "
                "variances to 0 or NaN, per-slot means to their own sample. The loss "
                "would return a confident number and teach nothing. Use >=2."
            )

        encoder = getattr(self.policy, "encoder", None)
        if encoder is None:
            # ⚠ RAISE, do not warn-and-continue. The old code returned here, which
            # produced the same silent vanilla-PPO run the registry above exists to stop.
            raise ValueError(
                f"AuxLossPPO: aux loss {self._aux_kind!r} requested but the policy has "
                f"no .encoder to shape. Refusing to train with the objective absent."
            )
        self._aux = AUX_LOSSES[self._aux_kind](encoder)
        if getattr(self._aux, "needs_memory", False):
            if self.memory is None:
                raise ValueError(
                    f"AuxLossPPO: aux loss {self._aux_kind!r} requires memory "
                    "because it draws its own temporal windows."
                )
            self._aux.attach_memory(self.memory)
        # Register the projection/expander head's params with the optimizer so
        # they are trained alongside the policy/value networks.
        if self.optimizer is not None:
            self.optimizer.add_param_group(
                {"params": list(self._aux.head.parameters())}
            )
        self._aux_encoder = encoder
        logger.info(
            "AuxLossPPO: %s aux loss ENABLED (weight=%s).", self._aux_kind, self._aux_weight
        )

    # The body below mirrors stock skrl PPO.update; the only change is the added
    # aux-loss term folded into the backward() of the existing optimizer step.
    @strict_update
    def update(self, *, timestep: int, timesteps: int) -> None:
        if self._aux is None:
            # Disabled path: identical to stock PPO (zero overhead, no risk).
            return super().update(timestep=timestep, timesteps=timesteps)

        from skrl.agents.torch.ppo.ppo import compute_gae

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_aux_loss = 0.0
        # Per-term accumulator for aux losses that expose a decomposition via
        # `last_terms`. Additive and optional: an aux without it logs nothing extra.
        cumulative_aux_terms = [0.0, 0.0, 0.0]
        aux_terms_seen = 0
        # ⛔ GENERIC SCALAR CHANNEL, ADDED BECAUSE ITS ABSENCE MADE A DIAGNOSTIC INERT.
        # `last_terms` above takes exactly three values and names them invariance /
        # variance / covariance -- VICReg's decomposition and nothing else. cltt_ref's
        # NT-Xent diagnostic therefore had nowhere to publish: it computed pos_acc,
        # chance, pos_sim, neg_sim and the batch size into `self.last_diag`, and NOTHING
        # IN THIS FILE EVER READ IT. The instrument ran, cost its compute, and reached no
        # log, no tfevents and no reader -- a detector wired to no actuator. Any aux may
        # now expose `last_scalars: dict[str, float]` and have it averaged and tracked.
        cumulative_aux_scalars: dict = {}
        aux_scalars_seen = 0
        cumulative_inv_temporal = 0.0
        cumulative_inv_control = 0.0
        inv_pairs_seen = 0
        # Transit-mask telemetry. `window_turn_last` keeps the LAST value rather than
        # only an average because the sentinels are states, not magnitudes, and a mean
        # over a state code names nothing.
        cumulative_window_turn = 0.0
        window_turns_seen = 0
        window_turn_last: float | None = None

        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in self.memory.sample(
                names=self._tensors_names, batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": self._state_preprocessor(sampled_states, train=not epoch),
                    }

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                        break

                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _ = self.value.act(inputs, role="value")
                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # ---- auxiliary SimCLR contrastive loss (shapes encoder) ----
                    aux_loss = self._aux_weight * self._aux.compute(
                        self._aux_encoder, sampled_observations
                    )

                self.optimizer.zero_grad()
                # ⛔ SPLIT BACKWARD (owner ruling, 2026-08-28). This was ONE fused
                # `.backward()` over all four losses, run entirely inside
                # `strict_determinism()` (skrl_patches.strict_update wraps `update`). The
                # motion aux losses warp with F.grid_sample, and
                # `grid_sampler_2d_backward_cuda` has no deterministic kernel, so every
                # EoO/GWM arm raised at its FIRST optimizer step and six of the eight
                # priority arms could not run at all.
                #
                # Splitting is exact, not an approximation: gradients ACCUMULATE into
                # `.grad`, and grad(a + b) == grad(a) + grad(b), so two backwards over the
                # addends deposit the same gradients as one backward over the sum. The AMP
                # scaler applies the same scale factor to both, so the scaled gradients sum
                # identically too. `retain_graph=True` is required on the first because the
                # encoder trunk is shared with the aux branch.
                #
                # What this buys: PPO-proper's backward stays STRICT -- a future
                # nondeterministic op in the policy/value path still fails loud, which is
                # the whole point of the guard. Only the auxiliary term's backward runs
                # under the ambient warn-not-raise policy.
                #
                # NETT_AUX_STRICT=1 restores the old fused, fully-strict backward, so the
                # pre-ruling behaviour remains reachable as a control.
                _ppo_loss = policy_loss + entropy_loss + value_loss
                if (
                    torch.is_tensor(aux_loss)
                    and aux_loss.requires_grad
                    and os.environ.get("NETT_AUX_STRICT", "").strip() not in ("1", "true", "yes")
                ):
                    self.scaler.scale(_ppo_loss).backward(retain_graph=True)
                    with relaxed_determinism():
                        self.scaler.scale(aux_loss).backward()
                else:
                    self.scaler.scale(_ppo_loss + aux_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        params = itertools.chain(self.policy.parameters(), self._aux.head.parameters())
                    else:
                        params = itertools.chain(
                            self.policy.parameters(), self.value.parameters(), self._aux.head.parameters()
                        )
                    nn.utils.clip_grad_norm_(params, self.cfg.grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                cumulative_aux_loss += float(aux_loss.detach())
                _scalars = getattr(self._aux, "last_scalars", None)
                if isinstance(_scalars, dict) and _scalars:
                    for _k, _v in _scalars.items():
                        cumulative_aux_scalars[_k] = (
                            cumulative_aux_scalars.get(_k, 0.0) + float(_v))
                    aux_scalars_seen += 1
                _terms = getattr(self._aux, "last_terms", None)
                if _terms is not None and len(_terms) == 3:
                    for _i, _t in enumerate(_terms):
                        cumulative_aux_terms[_i] += float(_t)
                    aux_terms_seen += 1
                _it = getattr(self._aux, "last_inv_temporal", None)
                _ic = getattr(self._aux, "last_inv_control", None)
                if _it is not None and _ic is not None:
                    cumulative_inv_temporal += float(_it)
                    cumulative_inv_control += float(_ic)
                    inv_pairs_seen += 1
                _wt = getattr(self._aux, "last_window_turn", None)
                if _wt is not None:
                    window_turn_last = float(_wt)
                    if window_turn_last >= 0.0:
                        cumulative_window_turn += window_turn_last
                        window_turns_seen += 1
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        n = self.cfg.learning_epochs * self.cfg.mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / n)
        self.track_data("Loss / Value loss", cumulative_value_loss / n)
        # ⛔ THE KIND IS INTERPOLATED, NOT HARDCODED. This label was `Aux (SimCLR)` for every aux,
        #    so 217 of 217 aux-carrying tfevents across the fleet read `SimCLR` regardless of what
        #    actually ran (eoo 0 · gwm 0 · cltt 0 · vicreg 0, over 910 tfevents / 63 roots).
        #    ⚠ It does NOT retroactively fix those files -- a tfevents records the label the code
        #    held when it was written -- and `aux_effective.py` stays the authoritative path, since
        #    it reads the trainer's own `AuxLossPPO: <kind> aux loss ENABLED` line, not tfevents.
        #    ⚠ THE TAG IS LOWERCASE, and deliberately not prettified. `_aux_kind` is an AUX_LOSSES
        #    key -- `simclr`, `cltt`, `vicreg`, `eoo`, `gwm` -- so a genuine SimCLR arm now writes
        #    `Loss / Aux (simclr) loss`, NOT the old `(SimCLR)`. A display mapping here would put a
        #    second hardcoded label in the file this change exists to de-hardcode. Consumers must
        #    match case-insensitively: a full-string cased key goes QUIET on a post-boundary arm
        #    rather than wrong, so a mixed corpus reads as "no such arms exist" (lion registered
        #    this before the change landed, against `tools/tfevents_auxkey.py`).
        self.track_data(f"Loss / Aux ({self._aux_kind}) loss", cumulative_aux_loss / n)
        # UNWEIGHTED decomposition, when the aux exposes one. The invariance term is
        # the only one a temporal objective's view construction can move: variance and
        # covariance are computed per view and are indifferent to which frames the pair
        # came from. If invariance is a rounding error beside the other two, a null
        # result is about the COEFFICIENTS, not about the pairing -- so this is logged
        # rather than argued about afterwards.
        if aux_scalars_seen:
            # ⚠ Averaged over minibatches and emitted EVERY update, not once at startup.
            # A quantity that varies across updates and is logged once is a number about
            # update 1 wearing the name of a constant: cltt_ref's batch B is
            # min(max_samples, t_max - max(offsets)), and t_max is the memory FILL INDEX
            # until the buffer fills -- so B grows early in training. A protocol that
            # reads chance = 2*ln(2B-1) once and compares later updates against it is
            # comparing to a moving line. Measured 2026-09-10: assumed B=96, realised
            # B≈45 at update 1.
            for _k, _v in sorted(cumulative_aux_scalars.items()):
                self.track_data(f"Loss / Aux {self._aux_kind} {_k}", _v / aux_scalars_seen)
        if aux_terms_seen:
            for _name, _val in zip(
                ("invariance", "variance", "covariance"), cumulative_aux_terms
            ):
                self.track_data(f"Loss / Aux term ({_name})", _val / aux_terms_seen)
        # GATE A: the same invariance term against a second augmentation of the ANCHOR
        # (the incumbent's construction) on the same batch. Their ratio isolates what
        # the TIME OFFSET contributes, separately from what the coefficient weights it by.
        if inv_pairs_seen:
            _t = cumulative_inv_temporal / inv_pairs_seen
            _c = cumulative_inv_control / inv_pairs_seen
            self.track_data("Loss / Aux inv (temporal)", _t)
            self.track_data("Loss / Aux inv (control)", _c)
            if _c > 0:
                self.track_data("Loss / Aux inv ratio", _t / _c)
        track_transit_mask(self, window_turn_last, cumulative_window_turn, window_turns_seen)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / n)
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        # SB3/Unity-parity health metrics (KL divergence, clip fraction,
        # explained variance, entropy) — same as the stock MetricsPPO path.
        from ..ppo_metrics import track_ppo_health_metrics
        track_ppo_health_metrics(self)
