"""Consistency guard between the framestack wrapper's depth and an encoder's ``num_frames``.

⛔⛔⛔ WHY THIS EXISTS. The framestack depth (``FrameStack.n_stack``) and the temporal
encoders' ``num_frames`` are set in two different places -- a body wrapper and an encoder
cfg -- and NOTHING checked that they agreed. While both were hardwired to 2 they could not
disagree. ``NETT_FRAMESTACK_N`` (added 2026-09-02) makes disagreement reachable.

A mismatch is SILENT AND CATASTROPHIC. ``compact_3dcnn`` computes
``base_channels = total_channels // num_frames`` and then reshapes with
``x.view(B, num_frames, base_channels, H, W)``. Give it 12 channels (4 RGB frames) while it
believes ``num_frames=2`` and it derives ``base_channels=6``, reshapes to (B,2,6,H,W), and
feeds the Conv3d a volume in which "colour" is half of one frame and half of another. The
PARAMETER COUNT IS UNCHANGED, so no capacity check, no shape error, and no log line can
see it -- the run simply learns from scrambled input and scores like a scientific negative.

That is not hypothetical: the same class of defect (a C-major read of a T-major tensor)
already shipped in ``compact_3dcnn.forward`` and handed two of three colour channels no
temporal signal at all. A startup error is cheap; a scrambled GPU-night is not.
"""

from __future__ import annotations


def validate_framestack_depth(total_channels: int, num_frames: int, encoder_name: str) -> int:
    """Return ``int(num_frames)``, or raise if it cannot describe ``total_channels``.

    The observation arriving at a temporal encoder is ``base_channels * num_frames``
    channels. ``base_channels`` must divide evenly and must be a plausible per-frame image
    depth (1 = grayscale, 3 = RGB, 4 = RGBA). Anything else means the wrapper and the
    encoder disagree about how many frames are stacked.
    """
    n = int(num_frames)
    if n < 1:
        raise ValueError(f"{encoder_name}: num_frames must be >= 1; got {num_frames!r}.")
    if total_channels % n:
        raise ValueError(
            f"{encoder_name}: observation has {total_channels} channels, which is not "
            f"divisible by num_frames={n}. The framestack depth and this encoder's "
            f"num_frames disagree. Set NETT_FRAMESTACK_N and the encoder cfg's "
            f"'num_frames' from the same value."
        )
    base = total_channels // n
    if base not in (1, 3, 4):
        raise ValueError(
            f"{encoder_name}: observation has {total_channels} channels and num_frames={n}, "
            f"implying {base} channels per frame -- not a plausible image depth (1/3/4). "
            f"The framestack depth and this encoder's num_frames disagree; a reshape on "
            f"these values would SCRAMBLE TIME INTO COLOUR without changing the parameter "
            f"count or raising. Refusing to build. Likely fix: num_frames="
            f"{total_channels // 3} to match NETT_FRAMESTACK_N, or framestack=False."
        )
    return n
