"""Wrapper registry — string names to ``gym.Wrapper`` classes.

Mirrors the brain registry contract (``brain/registry.py``): names are
resolved lazily so importing :mod:`nett_skrl` doesn't require OpenCV /
torchvision unless a wrapper is actually used.
"""

from __future__ import annotations

import gymnasium as gym

_WRAPPER_SPECS: dict[str, tuple[str, str]] = {
    "dvs": ("nett_skrl.wrappers.dvs", "DVS"),
    "retina": ("nett_skrl.wrappers.retina", "Retina"),
    "video": ("nett_skrl.wrappers.video", "Video"),
}

wrapper_list: list[str] = list(_WRAPPER_SPECS)


def _load_wrapper(name: str) -> type[gym.Wrapper]:
    import importlib

    try:
        module_name, class_name = _WRAPPER_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"wrapper should be one of {sorted(_WRAPPER_SPECS)}; got {name!r}.") from exc
    cls = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(cls, gym.Wrapper):
        raise TypeError(f"wrapper {name!r} did not resolve to a gym.Wrapper subclass.")
    return cls


def _validate_wrapper(value: str | type[gym.Wrapper]) -> type[gym.Wrapper]:
    if isinstance(value, str):
        return _load_wrapper(value)
    if isinstance(value, type) and issubclass(value, gym.Wrapper):
        return value
    raise TypeError(f"wrapper should be one of {sorted(_WRAPPER_SPECS)} or a gym.Wrapper subclass.")


def validate_wrappers(
    wrappers: list[str | type[gym.Wrapper]] | None,
) -> list[type[gym.Wrapper]]:
    """Resolve a list of wrapper specs to wrapper classes."""
    if not wrappers:
        return []
    return [_validate_wrapper(w) for w in wrappers]
