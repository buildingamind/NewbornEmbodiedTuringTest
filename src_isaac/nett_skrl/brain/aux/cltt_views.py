"""Single-frame CLTT views adapted to the host encoder's T-major RGB input."""

import torch


def current_frame_stack(prepared: torch.Tensor) -> torch.Tensor:
    """Repeat the current RGB frame across the encoder's required temporal slots.

    Preparation orders channels [oldest RGB, ..., current RGB]. Selecting only
    the current frame keeps t and t+1 distinct even when their original stacks
    overlap. Repetition preserves the supplied encoder's input geometry; it is
    a static view, so a motion encoder sees no within-view motion.
    """
    channels = prepared.shape[1]
    if channels < 3 or channels % 3:
        raise ValueError(f"CLTT requires T-major RGB channels, got {channels}")
    return prepared[:, -3:].repeat(1, channels // 3, *([1] * (prepared.ndim - 2)))
