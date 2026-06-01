"""Observation wrappers used by the NETT-skrl body."""

from __future__ import annotations

from .channels_first import ChannelsFirst
from .registry import _WRAPPER_SPECS, validate_wrappers, wrapper_list


def __getattr__(name: str):
    import importlib

    for spec_name, (module_name, class_name) in _WRAPPER_SPECS.items():
        if name == class_name or name == spec_name:
            return getattr(importlib.import_module(module_name), class_name)
    raise AttributeError(name)


__all__ = ["ChannelsFirst", "validate_wrappers", "wrapper_list"] + sorted(
    {class_name for _, class_name in _WRAPPER_SPECS.values()}
)
