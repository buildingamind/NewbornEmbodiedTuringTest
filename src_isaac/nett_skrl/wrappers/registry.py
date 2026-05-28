"""Compatibility registry for body wrappers.

Prefer :mod:`nett_skrl.body.wrappers.registry` for new code.
"""

from nett_skrl.body.wrappers.registry import (
    _WRAPPER_SPECS,
    validate_wrappers,
    wrapper_list,
)

__all__ = ["_WRAPPER_SPECS", "validate_wrappers", "wrapper_list"]
