"""Fallback classes for legacy rewards not implemented in the Isaac backend."""

from __future__ import annotations


class UnsupportedIntrinsicReward:
    """Explicit shim for legacy intrinsic rewards not implemented natively yet."""

    reward_name = "legacy intrinsic reward"

    def __init__(self, *args, **kwargs):
        raise ImportError(
            f"{self.reward_name} is not bundled with nett_skrl. Register a "
            "skrl-compatible implementation with register_reward(name, cls)."
        )

    @classmethod
    def named(cls, name: str):
        return type(name, (cls,), {"reward_name": name})
