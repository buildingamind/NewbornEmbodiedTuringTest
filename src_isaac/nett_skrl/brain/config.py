"""Brain-side config objects with explicit defaults and passthrough kwargs."""

from __future__ import annotations

from typing import Any


def _split_kwargs(data: dict[str, Any], known: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    values = {name: data.pop(name) for name in tuple(data) if name in known}
    extras = dict(data.pop("extra", {}) or {})
    extras.update(data)
    return values, extras


class FlexibleCfg:
    """Base for configs that accept known fields plus arbitrary passthrough kwargs."""

    known_fields: set[str] = set()

    def __init__(self, **kwargs) -> None:
        values, extras = _split_kwargs(dict(kwargs), self.known_fields)
        for name, value in values.items():
            setattr(self, name, value)
        self.extra = extras

    @classmethod
    def from_value(cls, value=None):
        if isinstance(value, cls):
            return value
        if isinstance(value, FlexibleCfg):
            return cls(**value.as_dict())
        return cls(**dict(value or {}))

    def known_items(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.known_fields}

    def as_dict(self, *, include_none: bool = False) -> dict[str, Any]:
        data = {
            name: value
            for name, value in self.known_items().items()
            if include_none or value is not None
        }
        data.update(self.extra)
        return data


class EncoderCfg(FlexibleCfg):
    known_fields = {"features_dim", "trainable"}

    def __init__(
        self,
        features_dim: int = 512,
        trainable: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(features_dim=int(features_dim), trainable=bool(trainable), **kwargs)

    def as_kwargs(self) -> dict[str, Any]:
        return self.as_dict()


class AlgorithmCfg(FlexibleCfg):
    known_fields = {"learning_rate"}

    def __init__(
        self,
        learning_rate=1e-5,
        **kwargs,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate if callable(learning_rate) else float(learning_rate),
            **kwargs,
        )

    def apply_to(self, cfg, spec) -> None:
        if self.learning_rate is not None and hasattr(cfg, "learning_rate"):
            cfg.learning_rate = self.learning_rate
        for name, value in self.extra.items():
            setattr(cfg, name, value)

    def agent_memory_size(self) -> int:
        raise NotImplementedError

    def dry_run_timesteps(self) -> int:
        return self.agent_memory_size()

    def envs_per_brain_for(self, steps_per_episode: int) -> int:
        return 1


class OnPolicyAlgorithmCfg(AlgorithmCfg):
    known_fields = AlgorithmCfg.known_fields | {
        "rollouts",
        "mini_batches",
        "value_loss_scale",
        "grad_norm_clip",
    }

    def __init__(
        self,
        learning_rate=1e-5,
        rollouts: int = 8000,
        mini_batches: int = 16,
        value_loss_scale: float | None = 0.25,
        grad_norm_clip: float | None = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            rollouts=int(rollouts),
            mini_batches=int(mini_batches),
            value_loss_scale=float(value_loss_scale) if value_loss_scale is not None else None,
            grad_norm_clip=float(grad_norm_clip) if grad_norm_clip is not None else None,
            **kwargs,
        )

    def apply_to(self, cfg, spec) -> None:
        for name in ("rollouts", "mini_batches"):
            if hasattr(cfg, name):
                setattr(cfg, name, getattr(self, name))
        if spec.cls.__name__ == "PPO":
            for name, value in self._ppo_defaults().items():
                if name not in {"rollouts", "mini_batches"} and hasattr(cfg, name):
                    setattr(cfg, name, value)
        super().apply_to(cfg, spec)

    def _ppo_defaults(self) -> dict[str, Any]:
        values = {}
        if self.value_loss_scale is not None:
            values["value_loss_scale"] = self.value_loss_scale
        if self.grad_norm_clip is not None:
            values["grad_norm_clip"] = self.grad_norm_clip
        values["rollouts"] = self.rollouts
        values["mini_batches"] = self.mini_batches
        return values

    def rollout_batch_size(self) -> int:
        return max(1, self.rollouts // max(1, self.mini_batches))

    def agent_memory_size(self) -> int:
        return int(self.rollouts)

    def envs_per_brain_for(self, steps_per_episode: int) -> int:
        return max(1, self.rollout_batch_size() // max(1, steps_per_episode))


class OffPolicyAlgorithmCfg(AlgorithmCfg):
    known_fields = AlgorithmCfg.known_fields | {
        "batch_size",
        "memory_size",
        "gradient_steps",
        "learning_starts",
    }

    def __init__(
        self,
        learning_rate=1e-5,
        batch_size: int = 64,
        memory_size: int = 8000,
        gradient_steps: int | None = None,
        learning_starts: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            memory_size=memory_size,
            gradient_steps=int(gradient_steps) if gradient_steps is not None else None,
            learning_starts=int(learning_starts) if learning_starts is not None else None,
            **kwargs,
        )

    def apply_to(self, cfg, spec) -> None:
        for name in ("batch_size", "gradient_steps", "learning_starts"):
            value = getattr(self, name)
            if value is not None and hasattr(cfg, name):
                setattr(cfg, name, value)
        super().apply_to(cfg, spec)

    def agent_memory_size(self) -> int:
        return int(self.memory_size)

    def dry_run_timesteps(self) -> int:
        return max(int(self.batch_size), int(self.memory_size))

    def envs_per_brain_for(self, steps_per_episode: int) -> int:
        return max(1, int(self.batch_size) // max(1, steps_per_episode))


def algorithm_cfg_from(value, spec) -> AlgorithmCfg:
    cfg_cls = (
        OffPolicyAlgorithmCfg
        if spec.family == "continuous_off_policy"
        else OnPolicyAlgorithmCfg
    )
    return cfg_cls.from_value(value)


class RewardCfg(FlexibleCfg):
    known_fields = {"beta", "kappa", "gamma", "weight", "trainable"}

    def __init__(
        self,
        beta: float = 0.2,
        kappa: float = 0.0,
        gamma: float = 0.99,
        weight: float = 1.0,
        trainable: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            beta=float(beta),
            kappa=float(kappa),
            gamma=float(gamma),
            weight=float(weight),
            trainable=bool(trainable),
            **kwargs,
        )

    def as_kwargs(self) -> dict[str, Any]:
        data = self.as_dict()
        data.pop("weight", None)
        data.pop("trainable", None)
        return data
