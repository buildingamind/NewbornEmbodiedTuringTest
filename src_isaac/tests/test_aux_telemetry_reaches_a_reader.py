"""Every value an aux loss publishes must have a READER. Structural, not behavioural.

⛔ WHY THIS FILE EXISTS. `nt_xent_diagnostics` computed pos_acc, chance, pos_sim, neg_sim
and the batch size into `self.last_diag` on every call. `grep -rn "last_diag"` over the
whole package returned exactly one line -- the assignment. `ppo_aux` reads `last_terms`,
which takes exactly three values and names them invariance/variance/covariance (VICReg's
decomposition), so a dict had no channel it could ever reach. The diagnostic ran, cost its
forward passes, and produced no log line, no tfevents scalar, and no reader.

⛔ AND ITS 34 TESTS ALL PASSED, because they tested the FUNCTION and never the PATH. A
unit test that calls a diagnostic directly proves that it computes. It cannot notice that
nothing reads the attribute it writes. The producer looks correct from its own side, the
consumer looks complete from its own side, and neither says the other is missing.

⇒ This is the cheapest check that would have caught it, and it catches the whole class
rather than the instance: an aux may publish whatever it likes, but if no line anywhere
in the package READS the name, the value does not exist.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

AUX_DIR = Path(__file__).resolve().parents[1] / "nett_skrl" / "brain" / "aux"
PKG = Path(__file__).resolve().parents[1] / "nett_skrl"

#: `self.last_x = ...` or `self.last_x: T = ...` -- the publication sites.
ASSIGN = re.compile(r"^\s*self\.(last_\w+)\s*[:=]")


def published_names() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for path in sorted(AUX_DIR.glob("*.py")):
        for line in path.read_text().splitlines():
            m = ASSIGN.match(line)
            if m:
                out.setdefault(m.group(1), [])
                if path not in out[m.group(1)]:
                    out[m.group(1)].append(path)
    return out


def read_sites(name: str) -> list[str]:
    """Every mention of `name` in the package that is NOT an assignment to it."""
    hits = []
    for path in sorted(PKG.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if name not in line:
                continue
            m = ASSIGN.match(line)
            if m and m.group(1) == name:
                continue          # a write, not a read
            hits.append(f"{path.name}:{i}")
    return hits


def test_there_is_something_to_check():
    """⛔ An empty scan passes every assertion below it. Establish the n first."""
    names = published_names()
    assert len(names) >= 3, f"expected several last_* channels, found {sorted(names)}"


@pytest.mark.parametrize("name", sorted(published_names()))
def test_every_published_value_has_a_reader(name):
    readers = read_sites(name)
    assert readers, (
        f"`self.{name}` is assigned in "
        f"{[p.name for p in published_names()[name]]} and READ NOWHERE in nett_skrl.\n"
        f"It costs whatever it costs to compute and reaches no log, no tfevents and no "
        f"reader. Either wire it to a consumer (ppo_aux reads `last_scalars` for "
        f"arbitrary dicts) or delete it. This is exactly how `last_diag` shipped inert.")


def test_the_generic_scalar_channel_is_the_one_ppo_aux_consumes():
    """`last_terms` takes three values with fixed names; anything else needs
    `last_scalars`. If that consumer is ever removed, every dict publisher goes silent
    again, and the failure is invisible from the publisher's side."""
    ppo = (AUX_DIR / "ppo_aux.py").read_text()
    assert 'getattr(self._aux, "last_scalars", None)' in ppo, \
        "the generic dict channel's CONSUMER is gone; publishers will emit into nothing"
    assert "track_data" in ppo.split('getattr(self._aux, "last_scalars", None)')[1], \
        "last_scalars is read but never tracked -- read without publish is still inert"
