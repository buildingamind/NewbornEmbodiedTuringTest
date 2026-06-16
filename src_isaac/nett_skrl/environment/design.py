"""Design-sheet helpers (Isaac Lab variant).

Replaces the Unity YAML probe with direct CSV inspection. Returns the same
``dict[condition_name, num_test_episodes]`` shape the legacy callers expect.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def get_experiment_design(design_sheet: str | Path) -> dict[str, int]:
    """Read a NETT design CSV and return {imprint_condition: num_test_episodes}.

    Counts rows whose Phase column starts with ``"test"`` (case-insensitive)
    per imprint condition; this matches the Unity ``--imprint-condition``
    semantics where train rows imprint and test rows evaluate.
    """
    path = Path(design_sheet)
    if not path.exists():
        raise FileNotFoundError(f"Design sheet not found: {path}")
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # binding sheets use "ImprintCondition"; parsing/viewinvariance sheets
            # use the spaced "Imprinting Condition" header — accept either.
            imprint = (row.get("ImprintCondition") or row.get("Imprinting Condition") or "").strip()
            phase = (row.get("Phase") or "").strip().lower()
            if not imprint:
                continue
            seen.add(imprint)
            if phase.startswith("test"):
                counts[imprint] += 1
    # Every imprint condition that appears in the sheet gets an entry, even
    # if zero test rows (legacy parity: caller decides what to do with 0).
    return {c: counts.get(c, 0) for c in seen}


def validate_conditions(
    valid: dict[str, int], requested: list[str] | None
) -> list[str]:
    """Resolve user-requested conditions against the design-sheet's set."""
    if requested is None:
        return list(valid.keys())
    invalid = set(requested) - set(valid.keys())
    if invalid:
        raise ValueError(
            f"Conditions not in design sheet: {sorted(invalid)}. "
            f"Available: {sorted(valid.keys())}"
        )
    return requested
