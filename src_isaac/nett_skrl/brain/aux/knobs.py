"""STRICT environment knobs for the wave-17 auxiliary losses.

⛔ WHY NOT THE EXISTING `_env_flag` COPIES. Every copy in this package (nett.py, simclr_aux,
cltt_ref_aux, slot_contrast_aux, dvs_polarity) returns False for ANY unrecognised spelling, so
`NETT_X=ture`, `NETT_X=enabled` or `NETT_X=2` runs the arm with the component OFF while the
launch config says it is on -- the silent no-op this fleet has paid for repeatedly. Those copies
are left exactly as they are (already-scored arms keep their definition); new knobs use these,
which accept the same ON spellings, an explicit OFF set, and RAISE on everything else.

⚠ THE LEADING UNDERSCORE IS LOAD-BEARING, not style. `scripts/gen_env_index.py` finds knobs read
through a helper by matching `_env_[a-z_]+(...)`; a helper named `env_positive_float` is
INVISIBLE to it, and this index's contract is that absence means the knob does not exist. Named
this way, `docs/env_vars.md` lists every knob read through these.
"""

from __future__ import annotations

import math
import os

_ON = frozenset({"1", "true", "yes", "on"})
_OFF = frozenset({"", "0", "false", "no", "off"})


#: A statistic that could not be computed. ⛔ NOT 0.0 and NOT silence: a diagnostic that
#: reports a plausible zero where it failed to measure is indistinguishable from one that
#: measured and found nothing, and those are opposite facts. Defined HERE, in the leaf both
#: `token_term` and `cltt_ref_aux` already import, because a second copy is a second convention
#: the day one of them is edited.
NOT_MEASURED = -9.0


def _env_flag_strict(name: str, default: bool = False) -> bool:
    """Boolean knob: unset -> ``default``; on/off spellings as `_env_flag`; anything else raises."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean. Use one of {sorted(_ON)} to enable or "
        f"{sorted(v for v in _OFF if v)} (or empty) to disable. Refusing to guess, because a "
        f"misspelled ON that reads as OFF trains the arm without its component."
    )


def _env_choice(name: str, default: str, options: tuple[str, ...]) -> str:
    """One of a fixed set of spellings; unset -> ``default``; anything else raises.

    ⛔ THE NEAREST-MATCH TEMPTATION IS THE WHOLE REASON THIS RAISES. A knob that selects an
    ARCHITECTURE has no safe fallback: quietly resolving "conv_sbd" or "pixel" to the default
    trains a different model than the config names, and the arm is then filed under the wrong
    cell of a screen whose entire value is that its cells differ by one factor.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if value in options:
        return value
    raise ValueError(
        f"{name}={raw!r} is not one of {list(options)}. Refusing to guess: this knob selects "
        f"which model is trained, and a misspelling that resolved to the default would put the "
        f"arm in the wrong cell of the comparison it was launched for."
    )


def _env_positive_int(name: str, default: int) -> int:
    """Integer knob > 0; unset -> ``default``; anything else raises."""
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(
            f"{name}={raw!r} must be an integer > 0. A zero or negative batch, offset or token "
            f"count would make the objective degenerate while still returning a number."
        )
    return value


def _env_nonneg_float(name: str, default: float) -> float:
    """Finite float >= 0; unset -> ``default``; anything else raises.

    For knobs whose zero is MEANINGFUL (an identity bias of 0 is a real configuration), unlike
    a loss weight, where zero silently deletes the term.
    """
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw.strip())
    except ValueError:
        value = float("nan")
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name}={raw!r} must be a finite number >= 0.")
    return value


def _env_unit_interval(name: str, default: float) -> float:
    """Float in [0, 1]; unset -> ``default``; anything else raises.

    For EMA decays and mixture fractions, where a value outside [0, 1] is not a strong setting
    but a different (and usually divergent) objective.
    """
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw.strip())
    except ValueError:
        value = float("nan")
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name}={raw!r} must be a finite number in [0, 1].")
    return value


def _env_positive_float(name: str, default: float) -> float:
    """Finite, strictly positive float knob; unset -> ``default``; anything else raises.

    ⚠ Zero is refused on purpose: a zero weight on a declared term deletes it from the backward
    while every log line still names it (the same trap AuxLossPPO refuses for aux_weight=0).
    """
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw.strip())
    except ValueError:
        value = float("nan")
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(
            f"{name}={raw!r} must be a finite number > 0. A zero, negative or non-numeric "
            f"weight would silently remove or invert a declared loss term."
        )
    return value
