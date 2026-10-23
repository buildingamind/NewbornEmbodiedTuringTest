"""Model factory helpers for NETT skrl agents."""

from __future__ import annotations

import json
import os

import torch.nn as nn

from .deterministic_actor import DeterministicActor
from .gaussian_actor import GaussianActor
from .model_cfg import ModelCfg
from .multivariate_gaussian_actor import MultivariateGaussianActor
from .q_critic import QCritic
from .value_critic import ValueCritic
from .utils.features import MIRROR_ENV, mirror_ckpt_enabled, mirror_equiv_enabled
from .utils.init import encoder_owns_init, init_encoder


def build_models_for_algorithm(
    spec,
    *,
    encoder_cls,
    encoder_kwargs,
    observation_space,
    action_space,
    device,
    cfg: ModelCfg,
) -> dict[str, nn.Module]:
    """Build the skrl model dictionary required by an ``AlgorithmSpec``.

    When ``cfg.shared_encoder`` is True, a single encoder instance is created
    and its reference is shared between the actor and all critic models — the
    same behaviour as SB3's ``share_features_extractor=True``. The shared
    encoder and its trunk are each initialised once; each model head gets its
    own independent MLP trunk so the critic can still learn a different value
    basis from the actor.
    """
    # Build one shared encoder if requested; otherwise each model builds its own.
    shared_enc = None
    if cfg.shared_encoder:
        shared_enc = encoder_cls(observation_space, **encoder_kwargs)
        shared_enc.to(device)
        if cfg.orthogonal_init and not encoder_owns_init(shared_enc):
            init_encoder(shared_enc, cfg.hidden_gain)

    model_kwargs = {
        "encoder_cls": encoder_cls,
        "encoder_kwargs": encoder_kwargs,
        "observation_space": observation_space,
        "action_space": action_space,
        "device": device,
        "cfg": cfg,
        "shared_encoder": shared_enc,
    }

    def actor():
        if spec.actor_type == "gaussian":
            # NETT DEFAULTS to the correlated multivariate actor (ModelCfg). Explicit
            # "gaussian" -- and legacy explicit None -- select the independent
            # per-component actor; None is deliberately still accepted so old configs
            # and checkpoints keep resolving to the diagonal head they were built with.
            if cfg.actor_distribution == "multivariate_gaussian":
                cls = MultivariateGaussianActor
            elif cfg.actor_distribution in (None, "gaussian"):
                cls = GaussianActor
            else:
                raise ValueError(
                    f"Unknown actor_distribution {cfg.actor_distribution!r}; "
                    "expected None, 'gaussian', or 'multivariate_gaussian'."
                )
        else:
            cls = DeterministicActor
        return cls(**model_kwargs)

    def critic():
        cls = ValueCritic if spec.critic_type == "value" else QCritic
        return cls(**model_kwargs)

    models: dict[str, nn.Module] = {}
    for key in spec.model_keys:
        models[key] = actor() if "policy" in key else critic()
    if mirror_equiv_enabled():
        _check_mirror_models(models)
    _announce_nas_encoder(shared_enc if shared_enc is not None else next(
        (m.encoder for m in models.values() if hasattr(m, "encoder")), None))
    return models


def _check_mirror_models(models) -> None:
    """NETT_MIRROR_EQUIV is implemented in GaussianActor (diagonal: M maps it to itself) and ValueCritic
    only. A correlated actor (tril_offdiag would need its sign flipped under M), a deterministic actor
    or a Q critic would run UNSYMMETRISED under a mirror label -- refuse instead."""
    bad = {k: type(m).__name__ for k, m in models.items() if type(m) not in (GaussianActor, ValueCritic)}
    if bad:
        raise ValueError(f"{MIRROR_ENV}=1 supports GaussianActor + ValueCritic only; got {bad}")
    for k, m in models.items():
        if isinstance(m, GaussianActor) and m.num_actions != 2:
            raise ValueError(f"{MIRROR_ENV}=1 needs the 2-D [turn, move] action; {k} has {m.num_actions}")
    # stdout, not a logger (the spawn child's logger is not bridged): the run log's proof the flag reached the models.
    print(f"[NETT mirror] {MIRROR_ENV}=1: policy mean / value symmetrised over the width flip, "
          f"turn (action 0) negated, models={sorted(models)} ckpt={int(mirror_ckpt_enabled())}", flush=True)


def _announce_nas_encoder(encoder) -> None:
    """NAS labels (examples/campaign_train.py NAS_LABELS): one line per encoder build, naming the cfg
    and its EXACT parameter count. Inert unless NETT_NAS_CFG is set, which campaign_train refuses on
    every registered label.

    stdout, not a logger: the spawn child's logger is not bridged, and a NAS label files every search
    point under one model key, so this line is the record of which encoder the arm actually built.
    """
    raw = os.environ.get("NETT_NAS_CFG")
    if raw is None or encoder is None:
        return
    label = os.environ.get("NETT_MODEL", "")
    cfg = json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))
    print(f"[NETT NAS] label={label} cfg={cfg} "
          f"encoder_params={sum(p.numel() for p in encoder.parameters())}", flush=True)
