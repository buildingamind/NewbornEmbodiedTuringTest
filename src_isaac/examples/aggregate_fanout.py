"""Aggregate N single-brain replication runs into a mean-per-condition table
and compare to the Unity Object1 baseline + ideal ChickData targets.

Reads each run's analysis/test/test_preferences.csv (cols: imprint,
test_condition, brain_id, n_steps, correct_pct, then the side-lock
diagnostics side_preference / pct_target_left / pct_target_right / verdict)
and averages correct_pct per test condition across all seeds (each run = 1
independent brain).

NOTE: correct_pct alone cannot tell a side-locked brain from a wandering one --
both average to ~0.5. For binding-style comparisons read ``verdict`` (or
``learn_fraction`` in the run's summary.json) instead of this table.

Usage:
  python examples/aggregate_fanout.py --output ~/nett_replicate_out \
      --encoder nature_cnn --tag wave1 --seeds 8 --seed-start 0
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# Unity Object1 baseline (typical, 4-brain mean) + ideal ChickData binding.csv.
UNITY = {
    "rest": 0.90, "1color": 0.674, "2color": 0.805, "1shape": 0.509,
    "2shape": 0.501, "1shape&color": 0.660, "2shape&color": 0.807, "binding": 0.622,
}
CHICK_IDEAL = {
    "1color": 0.658, "1shape": 0.611, "1shape&color": 0.710, "2color": 0.784,
    "2shape": 0.703, "2shape&color": 0.807, "binding": 0.673,
}
ORDER = ["rest", "1color", "2color", "1shape", "2shape", "1shape&color",
         "2shape&color", "binding"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="~/nett_replicate_out")
    p.add_argument("--encoder", default="nature_cnn")
    p.add_argument("--tag", required=True)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=0)
    return p.parse_args()


def load_run(run_dir: Path) -> dict[str, float]:
    """Return {test_condition: correct_pct} for one run (mean over its brains)."""
    pref = run_dir / "analysis" / "test" / "test_preferences.csv"
    if not pref.exists():
        # fall back to any test_preferences.csv under the run dir
        found = list(run_dir.glob("**/test_preferences.csv"))
        if not found:
            return {}
        pref = found[0]
    by_cond: dict[str, list[float]] = {}
    with open(pref) as fh:
        for row in csv.DictReader(fh):
            tc = row["test_condition"]
            v = float(row["correct_pct"])
            if v > 1.5:            # stored as percent -> normalize to fraction
                v /= 100.0
            by_cond.setdefault(tc, []).append(v)
    return {tc: sum(vs) / len(vs) for tc, vs in by_cond.items()}


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    a = parse_args()
    output = Path(a.output).expanduser()
    seeds = range(a.seed_start, a.seed_start + a.seeds)
    per_seed = {}
    for s in seeds:
        run_dir = output / f"{a.encoder}_{a.tag}_s{s}"
        vals = load_run(run_dir)
        if vals:
            per_seed[s] = vals
        else:
            print(f"  [warn] no test_preferences for seed {s} ({run_dir.name})")

    print(f"\n=== {a.encoder} / tag={a.tag} : {len(per_seed)}/{a.seeds} brains with results ===")
    print(f"{'condition':<14}{'mean':>8}{'std':>8}{'n':>4}   {'Unity':>7}  {'ideal':>7}  verdict")
    agg = {}
    for tc in ORDER:
        xs = [per_seed[s][tc] for s in per_seed if tc in per_seed[s]]
        m, sd = mean_std(xs)
        agg[tc] = (m, sd, len(xs))
        u = UNITY.get(tc)
        ci = CHICK_IDEAL.get(tc, float("nan"))
        verdict = ""
        if tc == "rest":
            verdict = "LEARNED" if m >= 0.85 else ("WEAK" if m >= 0.7 else "NOT-LEARNED")
        elif u is not None and not math.isnan(m):
            verdict = "ok" if m >= u - 0.05 else ("low" if m < u - 0.1 else "near")
        us = f"{u:.3f}" if u is not None else "   -  "
        cis = f"{ci:.3f}" if not math.isnan(ci) else "   -  "
        print(f"{tc:<14}{m:>8.3f}{sd:>8.3f}{len(xs):>4}   {us:>7}  {cis:>7}  {verdict}")
    # per-seed rest for collapse visibility
    print("\nper-seed rest (collapse check):")
    for s in sorted(per_seed):
        r = per_seed[s].get("rest", float("nan"))
        print(f"  seed {s}: rest={r:.3f}  {'COLLAPSE' if r < 0.7 else ''}")
    return agg


if __name__ == "__main__":
    main()
