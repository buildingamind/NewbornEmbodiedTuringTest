"""Body module for NETT-skrl."""

from .body import Body
from .wrappers import validate_wrappers, wrapper_list


def __getattr__(name: str):
    if name in {"DVS", "Retina", "Video"}:
        from . import wrappers

        return getattr(wrappers, name)
    raise AttributeError(name)

__all__ = [
    "Body",
    "DVS",
    "Retina",
    "Video",
    "validate_wrappers",
    "wrapper_list",
]
