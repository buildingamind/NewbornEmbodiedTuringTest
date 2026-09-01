"""Wrapper registry for the NETT-skrl body."""

from __future__ import annotations

import gymnasium as gym

_WRAPPER_SPECS: dict[str, tuple[str, str]] = {
    "dvs": ("nett_skrl.body.wrappers.dvs", "DVS"),
    "framestack": ("nett_skrl.body.wrappers.framestack", "FrameStack"),
    # ⛔ THE MoTok ARM IS NOT AN (encoder, aux) ARM AND CANNOT BE EXPRESSED AS ONE.
    # In Unity (trainParsing.py:338, seg_wrappers.py:213) MoTok holds its OWN model
    # and OWN AdamW, and its mask MULTIPLIES THE OBSERVATION -- it hands the policy a
    # segmented image rather than shaping the policy's representation. Behind the aux
    # interface a faithful port would train a parallel network and the policy encoder
    # would receive NOTHING, while the run logged aux=motok. This entry is the seat
    # that lets the faithful arm exist at all.
    # ⚠ ORDER MATTERS: body.py:53 applies wrappers in list order, innermost first, so
    # this must precede "framestack" -- MoTok is SINGLE-FRAME (get_masks ignores
    # frame_next), so it masks one frame and framestack then stacks masked frames.
    "motok_seg": ("nett_skrl.body.wrappers.motok_seg", "MoTokSeg"),
    "retina": ("nett_skrl.body.wrappers.retina", "Retina"),
    "video": ("nett_skrl.body.wrappers.video", "Video"),
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
    """Resolve body wrapper specs to wrapper classes."""
    if not wrappers:
        return []
    return [_validate_wrapper(w) for w in wrappers]
