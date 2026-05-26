"""Observation wrappers + the wrapper-name registry.

``Body`` is gone; wrappers attach to the env directly inside
``isaac_mode_runner._run_single_mode``. Import wrapper classes lazily by name
through :mod:`nett_skrl.wrappers.registry` (``validate_wrappers``) or by
PascalCase via ``from nett_skrl.wrappers import DVS, Retina, Video``.
"""

from __future__ import annotations

from .registry import _WRAPPER_SPECS, validate_wrappers, wrapper_list  # noqa: F401


def __getattr__(name: str):
    import importlib

    for spec_name, (module_name, class_name) in _WRAPPER_SPECS.items():
        if name == class_name or name == spec_name:
            return getattr(importlib.import_module(module_name), class_name)
    raise AttributeError(name)


__all__ = ["validate_wrappers", "wrapper_list"] + sorted(
    {class_name for _, class_name in _WRAPPER_SPECS.values()}
)
