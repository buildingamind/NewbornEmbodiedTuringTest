"""Shared encoder and MLP trunk wiring for NETT skrl models."""

from __future__ import annotations

from ..model_cfg import ModelCfg
from .feature_attn import feature_attn_trunk
from .init import orthogonal_init
from .mlp import mlp_trunk


class FeatureBackbone:
    def _build_backbone(
        self,
        encoder_cls,
        encoder_kwargs,
        observation_space,
        cfg: ModelCfg,
        shared_encoder=None,
    ) -> int:
        # observation_space is already CHW — ChannelsFirst wrapper guarantees this.
        if shared_encoder is not None:
            # Reuse a pre-built encoder (shared weights with another model).
            # Skip encoder init — it was already initialised when the shared
            # instance was created.
            self.encoder = shared_encoder
        else:
            self.encoder = encoder_cls(observation_space, **encoder_kwargs)
            if cfg.orthogonal_init:
                self.encoder.apply(lambda module: orthogonal_init(module, cfg.hidden_gain))
        if cfg.feature_attn is not None:
            # Feature-token attention head (feature_attn.py). It replaces the MLP trunk, so a
            # non-empty hidden_sizes alongside it is ambiguous and refuses. It initialises
            # itself; the orthogonal pass is NOT applied (see feature_attn.py, INIT).
            if list(cfg.hidden_sizes):
                raise ValueError(
                    f"feature_attn replaces the MLP trunk; hidden_sizes must be [] with it, got {list(cfg.hidden_sizes)}")
            self.trunk, last = feature_attn_trunk(int(self.encoder.features_dim), dict(cfg.feature_attn))
            # stdout, not a logger: the spawn child's logger is not bridged, and this line is
            # how a close-time check proves the head reached the arm (one per model).
            print(f"[NETT head] feature_attn tokens={self.trunk.num_tokens} group={self.trunk.group} "
                  f"dim={self.trunk.dim} blocks={len(self.trunk.blocks)} "
                  f"params={sum(p.numel() for p in self.trunk.parameters())}", flush=True)
            return last
        self.trunk, last = mlp_trunk(
            int(self.encoder.features_dim),
            list(cfg.hidden_sizes),
            cfg.activation,
        )
        if cfg.orthogonal_init:
            self.trunk.apply(lambda module: orthogonal_init(module, cfg.hidden_gain))
        return last
