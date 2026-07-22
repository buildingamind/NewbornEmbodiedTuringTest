"""skrl agent construction for NETT brains."""

from __future__ import annotations

from typing import Any

import os
import torch
import torch.nn as nn
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

from .hybrid_memory import HybridDeviceMemory, Uint8StatesMemory, resolve_memory_device

from .aux import AuxLossPPO
from .experiment import apply_experiment_cfg
from .models import build_models_for_algorithm
from .registry import algorithm_spec
from ..recording.wandb import attach_wandb_init_hook


def _aux_loss_settings() -> tuple[str, float]:
    """Read the optional auxiliary-loss config from the environment.

    NETT_AUX_LOSS   = "none" (default) | "simclr"
    NETT_AUX_WEIGHT = float (default 0.0)

    Returns ("none", 0.0) when disabled so the default RL path is unchanged.
    """
    kind = os.environ.get("NETT_AUX_LOSS", "none").strip().lower()
    try:
        weight = float(os.environ.get("NETT_AUX_WEIGHT", "0.0"))
    except ValueError:
        weight = 0.0
    if kind == "none" or weight <= 0.0:
        return "none", 0.0
    return kind, weight


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def encoder_cfg_for(brain) -> dict[str, Any]:
    kwargs = brain.encoder_cfg.as_kwargs()
    kwargs.pop("trainable", None)
    return kwargs


def default_algorithm_cfg(brain):
    """Return a fresh skrl algorithm config from the algorithm registry."""
    return algorithm_spec(brain.algorithm).cfg_cls()


def memory_size_for(brain) -> int:
    """Use rollout-sized memory for on-policy skrl agents."""
    return brain.algorithm_cfg.agent_memory_size()


def build_agents(brain, env, device: torch.device, *, config=None) -> list:
    """Build one skrl agent per brain, each owning a contiguous env scope."""
    obs_space, act_space = env.observation_space, env.action_space
    encoder_kwargs = encoder_cfg_for(brain)
    spec = algorithm_spec(brain.algorithm)
    num_agents = int(
        getattr(config, "num_brains", None)
        or getattr(brain, "num_brains", None)
        or env.num_envs
    )
    if num_agents < 1:
        raise ValueError(f"num_brains must be >= 1; got {num_agents}.")
    if env.num_envs % num_agents != 0:
        raise ValueError(
            f"env.num_envs ({env.num_envs}) must be divisible by num_brains ({num_agents})."
        )
    scope = env.num_envs // num_agents
    # Scale rollouts and memory_size so updates fire on `rollouts` transitions
    # regardless of num_envs. Without this, skrl's _rollout counter increments
    # by 1 per env.step() call, so the update fires after rollouts × scope
    # transitions instead of rollouts transitions.
    base_rollouts = memory_size_for(brain)
    scaled_rollouts = max(1, base_rollouts // scope)
    # P5 (determinism): capture the base seed set at task start (set_seeds ->
    # torch.manual_seed). Model weight init below is re-seeded per brain from this,
    # so it is reproducible regardless of how much global torch RNG the Isaac env
    # build happened to consume between set_seeds and here.
    base_seed = torch.initial_seed()
    # GLOBAL brain id = brain_id_offset + local id. The task seed is now condition-only
    # (runtime/task.py), so keying weight init by the global id makes single-brain
    # offset-b initialise identically to brain b of a multi-brain run.
    brain_offset = int(getattr(config, "brain_id_offset", 0) or 0)
    agents = []
    for brain_id in range(num_agents):
        cfg = default_algorithm_cfg(brain)
        brain.algorithm_cfg.apply_to(cfg, spec)
        if scope > 1 and hasattr(cfg, "rollouts"):
            cfg.rollouts = scaled_rollouts

        # Value-target standardization is REQUIRED for learning here: turning it
        # OFF (to match SB3, which doesn't standardize values) empirically broke
        # learning — reward went flat and policy std rose instead of converging.
        # The RunningStandardScaler keeps the critic/advantages well-scaled given
        # the correctly-normalized [0,1] image input. Keep it ON.
        if hasattr(cfg, "value_preprocessor"):
            cfg.value_preprocessor = RunningStandardScaler
            cfg.value_preprocessor_kwargs = {"size": 1, "device": device}

        if config is not None:
            apply_experiment_cfg(
                cfg,
                brain=brain,
                config=config,
                brain_id=brain_id + 1,
            )

        # Rollout buffer placement: FOLLOWS THE COMPUTE DEVICE (changed 2026-07-21).
        # See resolve_memory_device() for the rule and its rationale.
        mem_device = resolve_memory_device(device)
        on_compute_device = torch.device(mem_device) == torch.device(device)
        if not on_compute_device:
            # Storage off the compute device (CPU buffer for high-res runs):
            # use the hybrid memory so PPO's GAE/minibatch math stays on-device.
            memory = HybridDeviceMemory(
                memory_size=scaled_rollouts,
                num_envs=scope,
                device=mem_device,
                compute_device=device,
            )
        else:
            # On-GPU buffer, uint8 image states BY DEFAULT (changed 2026-07-21).
            # 1/4 the VRAM, numerically transparent (see Uint8StatesMemory), and it
            # is what makes the on-GPU default fit: float32 would need 22.7 GB of a
            # 23 GB card at 2 jobs/GPU, uint8 needs 13.9. Set NETT_UINT8_BUFFER=0 to
            # store float32 — only viable at 1 job/GPU or low rollout counts.
            mem_cls = (
                RandomMemory
                if os.environ.get("NETT_UINT8_BUFFER", "1") == "0"
                else Uint8StatesMemory
            )
            memory = mem_cls(
                memory_size=scaled_rollouts,
                num_envs=scope,
                device=mem_device,
            )
        # Deterministic per-brain weight-init seed keyed by the GLOBAL brain id
        # (brain_offset + local), so it is decorrelated across brains AND identical for
        # the same global brain in either topology (single-brain vs multi-brain).
        global_brain_id = brain_offset + brain_id
        torch.manual_seed(base_seed + global_brain_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + global_brain_id)
        models = build_models_for_algorithm(
            spec,
            encoder_cls=brain.encoder,
            encoder_kwargs=encoder_kwargs,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            cfg=brain.model_cfg,
        )
        if not brain.encoder_cfg.trainable:
            for model in models.values():
                if hasattr(model, "encoder"):
                    freeze(model.encoder)

        # Optional auxiliary self-supervised loss (opt-in via env vars). Only
        # wired for PPO; when disabled, the stock algorithm class is used and
        # behavior is byte-for-byte identical to before.
        aux_kind, aux_weight = _aux_loss_settings()
        agent_cls = brain.algorithm
        agent_kwargs = {}
        if aux_kind != "none":
            from skrl.agents.torch.ppo import PPO as _SkrlPPO
            if agent_cls is _SkrlPPO or issubclass(agent_cls, _SkrlPPO):
                agent_cls = AuxLossPPO
                agent_kwargs = {"aux_loss": aux_kind, "aux_weight": aux_weight}
            else:
                import logging
                logging.getLogger("nett").warning(
                    "NETT_AUX_LOSS set but algorithm is not PPO; aux loss ignored."
                )

        # Stock PPO -> MetricsPPO so the SB3/Unity-parity health metrics
        # (KL divergence, clip fraction, explained variance, entropy, LR) are
        # logged. Exact-class check leaves AuxLossPPO and non-PPO algorithms
        # untouched; MetricsPPO is behaviourally identical to PPO.
        from skrl.agents.torch.ppo import PPO as _SkrlPPO
        if agent_cls is _SkrlPPO:
            from .ppo_metrics import MetricsPPO
            agent_cls = MetricsPPO

        agent = agent_cls(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            **agent_kwargs,
        )
        agent._nett_brain_id = brain_id + 1   # 1-based; used for unified-wandb namespacing
        if config is not None:
            attach_wandb_init_hook(agent)
        agents.append(agent)
    return agents
