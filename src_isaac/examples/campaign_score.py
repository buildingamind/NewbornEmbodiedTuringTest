"""Score the campaign with the SAME metric Unity used, from the Isaac raw test logs.

Reproduces the authoritative Unity metric (the analysis repo's
compute_results.py::compute_success_rates) so Isaac and Unity compare like-for-like:

  UNITY FINAL-POSITION metric: for each test condition, group test-phase rows by
  (env_id, episode), take the LAST step of the episode, count success if
  sign(agent.x) matches correct.monitor (left -> x<0, right -> x>0). Score =
  successes / episodes. Chance = 0.5. The imprinting-preference headline is the
  "rest"/"Rest" condition (imprinted object vs blank White monitor).

Group dirs are campaign_train.py outputs named ``<exp>_<slug(model)>`` (e.g.
``binding_CNN``); each holds the brain-offset run dirs ``*_off*``. Per group we
aggregate mean/std/n_brains per test condition, plus per-run train wall time from
campaign_timing.json. The encoder (for the Unity baseline lookup) and the episode
length (for the completed-episode filter) are read from each run's config.yaml, so
this is correct at any steps_per_episode (campaign default 256, not orch's 500).

Usage:
  python examples/campaign_score.py [--root ~/nett_campaign] [--json out.json]
  python examples/campaign_score.py --validate <one_run_dir>   # sanity on a single run
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

# Unity baselines (final-position metric), reconstructed by the Unity analyst from the
# archived Unity results (final-position "CompendiumGoodCheck5"). rest = imprinting.
# Keyed by (encoder, experiment); models without a published baseline print "-".
UNITY = {
    ("nature_cnn", "binding"): {"rest": (0.939, 0.085, 8), "binding": (0.495, 0.050, 8)},
    ("compact_vit", "binding"): {"rest": (0.999, 0.005, 18), "binding": (0.617, 0.14, 18)},
    ("nature_cnn", "parsing"): {"Rest": (0.998, 0.008, 30), "Imprinted Object Familiar": (0.766, 0.136, 30),
                                 "Novel Familiar": (0.273, 0.139, 30)},
    ("compact_vit", "parsing"): {"Rest": (1.0, 0.0, 5), "Imprinted Object Familiar": (0.938, 0.079, 5),
                                  "Novel Familiar": (0.058, 0.074, 5)},
}

EXPERIMENTS = ("binding", "parsing", "viewinvariance")


def unity_scores(test_csv: str, final_step: int, completed_only: bool = True) -> dict[str, tuple[float, int]]:
    """Unity compute_success_rates on one raw test log, per test.cond.

    Returns {cond: (score, n_episodes)}. With completed_only (DEFAULT), scores ONLY
    episodes whose last step reached ``final_step`` (= steps_per_episode - 1) — required
    when a test was STOPPED mid-run, since an unfinished episode's last-written row is a
    mid-episode position, not the agent's final imprinting choice.
    """
    by_cond: dict[str, list] = defaultdict(list)
    with open(test_csv, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("experiment.phase") or "").strip() != "test":
                continue
            by_cond[(r.get("test.cond") or "").strip()].append(r)
    out: dict[str, tuple[float, int]] = {}
    for cond, crows in by_cond.items():
        eps: dict[tuple, list] = defaultdict(list)
        for r in crows:
            eps[(r["env_id"], r["episode"])].append(r)
        succ = []
        for ep_rows in eps.values():
            last = max(ep_rows, key=lambda r: int(r["step"]))
            if completed_only and int(last["step"]) != final_step:
                continue  # episode was cut off by the stop — not a valid final position
            try:
                ax = float(last["agent.x"])
            except (ValueError, KeyError):
                continue
            cm = (last.get("correct.monitor") or "").strip()
            if cm == "left":
                succ.append(1 if ax < 0 else 0)
            elif cm == "right":
                succ.append(1 if ax > 0 else 0)
        if succ:
            out[cond] = (sum(succ) / len(succ), len(succ))
    return out


def _run_dirs(group_dir: str) -> list[str]:
    return sorted(d for d in glob.glob(f"{group_dir}/*_off*") if os.path.isdir(d))


def _run_config(run_dir: str) -> dict:
    p = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(p):
        try:
            return yaml.safe_load(Path(p).read_text()) or {}
        except Exception:
            return {}
    return {}


def group_meta(group_dir: str) -> tuple[str | None, str | None, int]:
    """Return (encoder, experiment, final_step) for a group, read from the first run's
    config.yaml. experiment falls back to the leading token of the group dir name."""
    encoder, steps = None, 256
    for rd in _run_dirs(group_dir):
        cfg = _run_config(rd)
        if cfg:
            encoder = (cfg.get("brain", {}) or {}).get("encoder", encoder)
            steps = int(cfg.get("steps_per_episode", steps) or steps)
            break
    name = os.path.basename(group_dir.rstrip("/"))
    exp = next((e for e in EXPERIMENTS if name.startswith(f"{e}_")), None)
    return encoder, exp, max(1, steps) - 1


def score_group(group_dir: str, final_step: int) -> dict:
    """Aggregate the brain-offset runs of one group: per-brain per-condition
    final-position score over that brain's COMPLETED test episodes, then mean/std
    across brains, plus total episodes per cond and per-run train wall time."""
    per_cond: dict[str, list[float]] = defaultdict(list)   # brain-mean scores
    per_cond_eps: dict[str, int] = defaultdict(int)        # total completed episodes
    wall_train: list[float] = []
    n_runs = 0
    for rd in _run_dirs(group_dir):
        test_csvs = glob.glob(f"{rd}/*/logs/test_*.csv")
        if not test_csvs:
            continue
        n_runs += 1
        merged: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for tc in test_csvs:
            for cond, (val, n) in unity_scores(tc, final_step).items():
                merged[cond].append((val, n))
        for cond, vn in merged.items():
            tot_n = sum(n for _, n in vn)
            if tot_n:
                per_cond[cond].append(sum(v * n for v, n in vn) / tot_n)
                per_cond_eps[cond] += tot_n
        # train wall time (campaign_timing.json; orch_timing.json for legacy runs)
        for fname in ("campaign_timing.json", "orch_timing.json"):
            src = os.path.join(rd, fname)
            if os.path.isfile(src):
                try:
                    wall_train.append(json.load(open(src)).get("train_secs", 0.0))
                except Exception:
                    pass
                break
    agg = {cond: {"mean": round(st.mean(v), 4),
                  # ⚠ None, NOT 0.0, at a single brain. A spread of "0.0000" printed
                  # beside a mean reads as PERFECT CONSISTENCY when it means WE HAVE
                  # ONE SAMPLE -- and this campaign is about to run many arms at low n,
                  # where that is the common case rather than the edge case. n_brains
                  # is right there in the row, but a number that can be quoted out of
                  # its row will be. Same family as torch/numpy's opposite variance
                  # defaults: the degenerate case must not return the answer that
                  # looks best.
                  "std": round(st.pstdev(v), 4) if len(v) > 1 else None,
                  "n_brains": len(v), "n_episodes": per_cond_eps[cond]}
           for cond, v in sorted(per_cond.items())}
    return {"n_runs": n_runs, "unity_metric": agg,
            "train_wall_secs": {"per_run": [round(x, 1) for x in wall_train],
                                "mean": round(st.mean(wall_train), 1) if wall_train else None,
                                "max": round(max(wall_train), 1) if wall_train else None}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/nett_campaign"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--validate", default=None, help="score a single run dir's test CSVs and exit")
    a = ap.parse_args()

    if a.validate:
        _, _, final_step = group_meta(str(Path(a.validate).parent))
        for tc in glob.glob(f"{a.validate}/*/logs/test_*.csv") or glob.glob(f"{a.validate}/logs/test_*.csv"):
            print(tc)
            for cond, (v, n) in unity_scores(tc, final_step).items():
                print(f"  {cond:32} {v:.3f}  (n={n} completed eps)")
        return 0

    report = {}
    for group_dir in sorted(glob.glob(f"{a.root}/*")):
        name = os.path.basename(group_dir)
        if name.startswith("_") or not os.path.isdir(group_dir):
            continue
        encoder, exp, final_step = group_meta(group_dir)
        if exp is None:
            continue
        g = score_group(group_dir, final_step)
        if g["n_runs"] == 0:
            continue
        g["encoder"], g["experiment"] = encoder, exp
        report[name] = g

    # Print comparison table.
    print(f"\n{'group':30} {'cond':26} {'Isaac(final-pos)':>18} {'eps':>6} {'Unity':>14}")
    print("-" * 104)
    for name, g in sorted(report.items()):
        ub = UNITY.get((g["encoder"], g["experiment"]), {})
        for cond, cd in g["unity_metric"].items():
            um = ub.get(cond)
            us = f"{um[0]:.3f}±{um[1]:.3f}" if um else "-"
            flag = ""
            if um and cond.lower() == "rest":
                flag = "  <== IMPRINTING" + ("  >=Unity" if cd["mean"] >= um[0] - 0.05 else "  below")
            # std is None at a single brain (see score_group) -- print it as "n/a"
            # rather than 0.000, and NEVER let the format crash: a scoreboard that
            # raises on its own degenerate row is a scoreboard nobody sees.
            sd = f"{cd['std']:.3f}" if cd["std"] is not None else "  n/a"
            print(f"{name:30} {cond:26} {cd['mean']:.3f}±{sd} ({cd['n_brains']:d}b) "
                  f"{cd['n_episodes']:6d}   {us:>14}{flag}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
