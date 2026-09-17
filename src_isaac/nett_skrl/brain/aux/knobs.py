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
