"""PPO health metrics — parity with the original SB3/Unity tensorboard.

skrl's stock PPO logs only ``Loss / Policy loss``, ``Loss / Value loss``,
``Policy / Standard deviation`` (+ entropy/LR when configured). The original
NETT (stable-baselines3) tensorboard additionally logged the standard PPO
diagnostics ``approx_kl``, ``clip_fraction``, ``explained_variance``,
``entropy_loss``, ``learning_rate`` and ``clip_range``.

``track_ppo_health_metrics`` recovers those for the skrl stack and logs them
under the same ``Loss / …`` and ``Policy / …`` groups skrl already uses, so
they appear alongside the existing curves in wandb/tensorboard. It runs once
per update, after the optimisation epochs, and is strictly best-effort: any
failure is swallowed so logging can never perturb training.

``explained_variance`` is read straight from the stored rollout tensors (no
extra forward pass). ``KL divergence``/``clip fraction``/``entropy`` need the
post-update policy evaluated on the rollout, so they are recomputed in a single
pass that is chunked into the same ``mini_batches`` as the update — bounding the
memory footprint to what training already used (important at high resolution).
"""
from __future__ import annotations

import os

import torch
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo import ppo as _skrl_ppo_mod

# ---------------------------------------------------------------------------
# Time-limit (partial-episode) bootstrapping — env-gated by NETT_DIAG_PEB.
#
# skrl's stock compute_gae offers only:
#   time_limit_bootstrap=False -> NOT isolated: advantage propagates ("leaks")
#       across truncation boundaries, and the truncation step bootstraps with
#       values[i+1] (= V(reset state), since Isaac auto-resets).
#   time_limit_bootstrap=True  -> isolated, but NO bootstrap: value target at a
#       truncation = reward only (the "0-value bug" — treats a time-limit as a
#       true terminal).
# Neither is correct for fixed-length (time-limit) episodes. The correct handling
# (Pardo et al. 2018) is: ISOLATE (no cross-episode leak) AND bootstrap the cut-off
# future with a real value estimate. Two variants:
#   NETT_DIAG_PEB=B : isolate + bootstrap with V(next/reset) = values[i+1].
#                     return_T = r_T + gamma*V(reset).  ("default minus the leak")
#   NETT_DIAG_PEB=A : isolate + bootstrap with the truncation step's OWN value
#                     V(states_T) = values[T] as a STATE-SPECIFIC proxy for
#                     V(true final state) (one action away; no extra camera render).
#                     return_T = r_T + gamma*V(states_T).
# Default (unset) -> STOCK skrl compute_gae (byte-for-byte). Option B was the
#   default 2026-07-01..07-05 on a theoretical argument + a weak EV trend, but a
#   post-reward-fix re-ablation (4 seeds) found B does NOT beat stock skrl on
#   explained variance (B EV 0.607 vs stock 0.652, p=0.47) — the original "B best"
#   came from the reward-bug regime and did not reproduce. So the default reverted
#   to stock. B/A remain OPT-IN (NETT_DIAG_PEB=B/A) for the theoretical-correctness
#   case; B = isolate + bootstrap V(reset); A = V(states_T) proxy (its faithful
#   form needs the true final-observation render + VRAM).
# ---------------------------------------------------------------------------
_orig_compute_gae = _skrl_ppo_mod.compute_gae


def _peb_compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
    time_limit_bootstrap: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Diagnostic one-shot capture (env-gated, no-op by default): dump the raw
    # GAE inputs for one rollout so the buffer-layout invariance test can re-lay
    # out the identical transitions offline. Harmless when NETT_DIAG_DUMP unset.
    _dump = os.environ.get("NETT_DIAG_DUMP")
    if _dump and not os.path.exists(_dump):
        torch.save({
            "rewards": rewards.detach().cpu(), "terminated": terminated.detach().cpu(),
            "truncated": truncated.detach().cpu(), "values": values.detach().cpu(),
            "last_values": last_values.detach().cpu(),
            "discount_factor": discount_factor, "lambda_coefficient": lambda_coefficient,
            "time_limit_bootstrap": time_limit_bootstrap,
        }, _dump)
    mode = os.environ.get("NETT_DIAG_PEB", "off")  # default = stock skrl (see below)
    if mode not in ("B", "A"):
        return _orig_compute_gae(
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            values=values,
            last_values=last_values,
            discount_factor=discount_factor,
            lambda_coefficient=lambda_coefficient,
            time_limit_bootstrap=time_limit_bootstrap,
        )
    memory_size = rewards.shape[0]
    trunc = truncated.to(rewards.dtype)
    if mode == "B":
        # V(next/reset) per step: values[i+1] for i<T-1, else last_values
        boot = torch.cat([values[1:], last_values.unsqueeze(0)], dim=0)
    else:  # mode == "A": V(states_i) — the agent's own value at the truncation step
        boot = values
    rewards = rewards + discount_factor * boot * trunc
    not_done = (terminated | truncated).logical_not()  # isolate: reset GAE at every truncation
    advantage = 0
    advantages = torch.zeros_like(rewards)
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


# Install the wrapper. It is a no-op when NETT_DIAG_PEB is unset/"off" (delegates to
# the original), so default behaviour is unchanged. Stock PPO calls the module-global
# compute_gae; ppo_aux does a call-time `from ...ppo import compute_gae` — both resolve
# this patched attribute.
_skrl_ppo_mod.compute_gae = _peb_compute_gae


def track_ppo_health_metrics(agent: PPO) -> None:
    """Log SB3-parity PPO diagnostics on ``agent`` (best-effort, never raises)."""
    cfg = agent.cfg

    # explained variance: 1 - Var(returns - values) / Var(returns), from the
    # rollout tensors skrl already stored (values are the pre-update estimates).
    try:
        values = agent.memory.get_tensor_by_name("values").flatten()
        returns = agent.memory.get_tensor_by_name("returns").flatten()
        var_returns = returns.var(unbiased=False)
        if var_returns > 0:
            ev = 1.0 - (returns - values).var(unbiased=False) / var_returns
            agent.track_data("Loss / Explained variance", float(ev))
    except Exception:  # pragma: no cover - diagnostics must not break training
        pass

    # clip range: a constant, always available (no forward pass).
    try:
        agent.track_data("Policy / Clip range", float(cfg.ratio_clip))
    except Exception:  # pragma: no cover
        pass

    # KL divergence, clip fraction, entropy: one chunked recompute of the
    # post-update policy on the rollout (mirrors the update's memory footprint).
    # Each quantity is tracked independently so that a model that does not
    # implement (e.g.) entropy still yields KL and clip fraction.
    try:
        ratio_clip = float(cfg.ratio_clip)
        names = ["observations", "states", "actions", "log_prob"]
        kl_sum = clip_sum = ent_sum = 0.0
        batches = ent_batches = 0
        device = agent.device
        with torch.no_grad():
            for obs, states, actions, old_log_prob in agent.memory.sample_all(
                names=names, mini_batches=cfg.mini_batches
            ):
                # the rollout buffer may live in CPU RAM (hybrid memory); move the
                # minibatch onto the policy's device before the forward pass.
                # ``states`` is None when the env has no separate state space.
                obs = obs.to(device)
                actions = actions.to(device)
                old_log_prob = old_log_prob.to(device)
                states = states.to(device) if states is not None else states
                inputs = {
                    "observations": agent._observation_preprocessor(obs),
                    "states": agent._state_preprocessor(states),
                    "taken_actions": actions,
                }
                _, outputs = agent.policy.act(inputs, role="policy")
                log_ratio = outputs["log_prob"] - old_log_prob
                ratio = log_ratio.exp()
                # skrl's approximate KL: E[(r - 1) - log r]  (>= 0, low-variance)
                kl_sum += float(((ratio - 1.0) - log_ratio).mean())
                # SB3's clip fraction: fraction of samples outside the clip band
                clip_sum += float(((ratio - 1.0).abs() > ratio_clip).float().mean())
                batches += 1
                try:
                    ent_sum += float(agent.policy.distribution(role="policy").entropy().mean())
                    ent_batches += 1
                except Exception:  # pragma: no cover - model may not expose entropy
                    pass
        if batches:
            agent.track_data("Loss / KL divergence", kl_sum / batches)
            agent.track_data("Loss / Clip fraction", clip_sum / batches)
        if ent_batches:
            agent.track_data("Policy / Entropy", ent_sum / ent_batches)
    except Exception:  # pragma: no cover - diagnostics must not break training
        import logging
        logging.getLogger("nett").debug("PPO health metrics (KL/clip/entropy) skipped", exc_info=True)

    # learning rate (always, matching SB3 which logged it every update).
    try:
        scheduler = getattr(agent, "scheduler", None)
        lr = scheduler.get_last_lr()[0] if scheduler else agent.optimizer.param_groups[0]["lr"]
        agent.track_data("Learning / Learning rate", float(lr))
    except Exception:  # pragma: no cover
        pass


class MetricsPPO(PPO):
    """skrl PPO that also logs the SB3/Unity-parity health metrics.

    Behaviourally identical to ``skrl.agents.torch.ppo.PPO`` — it only adds
    tensorboard/wandb scalars after each update.
    """

    def update(self, *, timestep: int, timesteps: int) -> None:
        # Optional entropy-coefficient annealing (env-gated; default = no change).
        # Constant high entropy inflates the policy std without bound on long runs
        # (measured: sigma 1.1->7.4 over 2M steps -> random policy). Annealing from
        # a high start (escape the side-lock local optimum early) down to the
        # configured final value (let sigma settle so the policy commits) fixes that.
        import os as _os
        _es = _os.environ.get("NETT_DIAG_ENT_START")
        if _es is not None:
            if not hasattr(self, "_ent_end"):
                self._ent_end = float(self.cfg.entropy_loss_scale)  # configured final
                self._ent_start = float(_es)
            frac = min(1.0, max(0.0, timestep / max(1, timesteps)))
            self.cfg.entropy_loss_scale = self._ent_start + (self._ent_end - self._ent_start) * frac
        super().update(timestep=timestep, timesteps=timesteps)
        track_ppo_health_metrics(self)
