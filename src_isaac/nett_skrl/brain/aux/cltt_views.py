"""Single-frame CLTT views adapted to the host encoder's T-major input."""

import os

import torch

#: Channels in ONE frame. 3 for RGB, which every arm before the DVS work used.
DEFAULT_CHANNELS_PER_FRAME = 3


def resolve_channels_per_frame(value: int | None = None) -> int:
    """Explicit argument wins; otherwise ``NETT_AUX_CLTT_CHANNELS_PER_FRAME``; otherwise 3.

    ⚠ This exists because a body wrapper can change how many channels one frame carries --
    ``dvs_polarity`` emits an (ON, OFF) PAIR, so a 2-frame stack is 4 channels, not 6 -- while
    the CLTT views code assumed RGB. Left at 3 the assumption does not mis-slice quietly; it
    raises, because 4 % 3 != 0. The knob is how an arm states the truth instead.

    ⛔ RESIDUAL HAZARD, UNGUARDED AND DELIBERATELY SO. Divisibility does not pin this value:
    cpf=2 against a 6-channel RGB stack divides evenly and would slice two of three COLOUR
    channels and call them a frame -- wrong, and silent. The obvious cross-check, "does
    channels//cpf equal the body's stack depth", was written and then REMOVED: the body's
    depth is only recoverable from ``NETT_FRAMESTACK_N`` when nothing passed ``n_stack``
    explicitly, so the check fired on correct 3-frame arms (tests/test_cltt_ref_aux.py has
    three). A guard that refuses valid input is worse than none. What bounds this instead is
    that the knob is set per-arm in the same ``env:`` block that selects the wrapper, and a
    MISSING setting fails loudly (4 % 3) rather than quietly.
    """
    if value is not None:
        return int(value)
    raw = os.environ.get("NETT_AUX_CLTT_CHANNELS_PER_FRAME")
    if raw is None or raw == "":
        return DEFAULT_CHANNELS_PER_FRAME
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"NETT_AUX_CLTT_CHANNELS_PER_FRAME must be an integer; got {raw!r}."
        ) from exc
    if n < 1:
        raise ValueError(f"NETT_AUX_CLTT_CHANNELS_PER_FRAME must be >= 1; got {n}.")
    return n


def current_frame_stack(prepared: torch.Tensor, channels_per_frame: int | None = None) -> torch.Tensor:
    """Repeat the current frame across the encoder's required temporal slots.

    Preparation orders channels [oldest frame, ..., current frame]. Selecting only
    the current frame keeps t and t+1 distinct even when their original stacks
    overlap. Repetition preserves the supplied encoder's input geometry; it is
    a static view, so a motion encoder sees no within-view motion.
    """
    cpf = resolve_channels_per_frame(channels_per_frame)
    channels = prepared.shape[1]
    if channels < cpf or channels % cpf:
        raise ValueError(
            f"CLTT requires T-major frames of {cpf} channels, got {channels}. "
            "Set NETT_AUX_CLTT_CHANNELS_PER_FRAME to this arm's per-frame channel count "
            "(3 for RGB, 2 for dvs_polarity)."
        )
    depth = channels // cpf
    return prepared[:, -cpf:].repeat(1, depth, *([1] * (prepared.ndim - 2)))
