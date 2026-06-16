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

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl import config, logger

from .simclr_aux import SimCLRAuxLoss


class AuxLossPPO(PPO):
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
        if self._aux_kind in ("simclr", "vicreg") and self._aux_weight > 0.0:
            encoder = getattr(self.policy, "encoder", None)
            if encoder is None:
                logger.warning("AuxLossPPO: policy has no .encoder; aux loss disabled.")
                return
            if self._aux_kind == "vicreg":
                from .vicreg_aux import VICRegAuxLoss
                self._aux = VICRegAuxLoss(encoder)
            else:
                self._aux = SimCLRAuxLoss(encoder)
            # Register the projection/expander head's params with the optimizer so
            # they are trained alongside the policy/value networks.
            if self.optimizer is not None:
                self.optimizer.add_param_group(
                    {"params": list(self._aux.head.parameters())}
                )
            self._aux_encoder = encoder
            logger.info(
                f"AuxLossPPO: {self._aux_kind} aux loss ENABLED (weight={self._aux_weight})."
            )

    # The body below mirrors stock skrl PPO.update; the only change is the added
    # aux-loss term folded into the backward() of the existing optimizer step.
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
                self.scaler.scale(policy_loss + entropy_loss + value_loss + aux_loss).backward()

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
        self.track_data("Loss / Aux (SimCLR) loss", cumulative_aux_loss / n)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / n)
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
