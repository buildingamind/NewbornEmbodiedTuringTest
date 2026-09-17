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


# =============================================================================================
# ⛔ A MEAN OVER A MIXTURE OF MEASUREMENTS AND SENTINELS IS NEITHER OF THEM. Measured on the
# apparatus: fg_objectness_corr published as -5.545, outside the range a correlation can take.

from nett_skrl.brain.aux.knobs import NOT_MEASURED  # noqa: E402
from nett_skrl.brain.aux.ppo_aux import (  # noqa: E402
    accumulate_aux_scalar,
    aggregate_aux_scalars,
)


def _run(samples: dict, seen: int) -> dict:
    sums, fired = {}, {}
    for _ in range(seen):
        pass
    for key, values in samples.items():
        assert len(values) == seen
    for i in range(seen):
        for key, values in samples.items():
            accumulate_aux_scalar(sums, fired, key, values[i])
    return dict(aggregate_aux_scalars(sums, fired, seen))


def test_a_scalar_that_fires_on_a_subset_gets_its_conditional_mean_and_its_fire_rate():
    """The measured instance, reconstructed: 61 of 160 minibatches measured r = +0.06 and the
    other 99 emitted the sentinel. The old aggregation published
    0.38125*0.06 + 0.61875*(-9) = -5.545 -- the sentinel FRACTION wearing a correlation's name."""
    seen = 160
    values = [0.06] * 61 + [NOT_MEASURED] * 99
    old_behaviour = sum(values) / seen
    assert old_behaviour == pytest.approx(-5.545, abs=1e-3)      # the published defect
    out = _run({"fg_objectness_corr": values}, seen)
    assert out["fg_objectness_corr"] == pytest.approx(0.06)
    assert out["fg_objectness_corr fire_rate"] == pytest.approx(61 / 160)
    assert out["sentinel_keys"] == 1.0
    assert -1.0 <= out["fg_objectness_corr"] <= 1.0


def test_a_scalar_that_never_fires_does_not_acquire_a_number():
    out = _run({"shift_rho": [NOT_MEASURED] * 8}, 8)
    assert out["shift_rho"] == NOT_MEASURED
    assert out["shift_rho fire_rate"] == 0.0
    assert out["sentinel_keys"] == 1.0


def test_a_scalar_that_always_fires_is_bitwise_what_the_old_aggregation_published():
    """⚠ THE REGRESSION GUARD. Most scalars are unconditional, and their published series must
    not move by a bit -- otherwise this fix becomes a discontinuity in every arm's history."""
    seen = 160
    values = [0.1 + i * 1e-3 for i in range(seen)]
    old_sum = 0.0
    for v in values:
        old_sum += v
    out = _run({"pos_acc": values}, seen)
    assert out["pos_acc"] == old_sum / seen              # same sum, same divisor, same order
    assert "pos_acc fire_rate" not in out                # not emitted when it is 1
    assert out["sentinel_keys"] == 0.0


def test_the_summary_count_is_what_makes_an_absent_fire_rate_readable():
    """`fire_rate` is emitted only when it is not 1, so something must distinguish "fired
    everywhere" from "not emitted". sentinel_keys is that something, and it goes out always."""
    seen = 4
    out = _run({"always": [1.0] * 4,
                "sometimes": [1.0, NOT_MEASURED, 3.0, NOT_MEASURED],
                "never": [NOT_MEASURED] * 4}, seen)
    assert out["always"] == 1.0 and "always fire_rate" not in out
    assert out["sometimes"] == 2.0 and out["sometimes fire_rate"] == 0.5
    assert out["never"] == NOT_MEASURED and out["never fire_rate"] == 0.0
    assert out["sentinel_keys"] == 2.0


def test_the_window_turn_mask_states_do_not_enter_the_average():
    """⛔ MASK_OFF is -1.0 and a correlation of exactly -1.0 is attainable, so the aggregator
    cannot be taught to treat -1.0 as a sentinel without silently dropping real measurements
    from some other key. The conversion happens at the one place that knows what -1.0 means."""
    from nett_skrl.brain.aux.action_windows import MASK_NO_MOTION, MASK_OFF
    from nett_skrl.brain.aux.token_term import window_turn_scalar

    assert window_turn_scalar(MASK_OFF) == NOT_MEASURED
    assert window_turn_scalar(MASK_NO_MOTION) == NOT_MEASURED
    assert window_turn_scalar(0.0) == 0.0
    assert window_turn_scalar(0.31) == pytest.approx(0.31)
    out = _run({"window_turn": [window_turn_scalar(v) for v in (0.31, MASK_OFF, 0.29, MASK_OFF)]}, 4)
    assert out["window_turn"] == pytest.approx(0.30)
    assert out["window_turn fire_rate"] == 0.5
