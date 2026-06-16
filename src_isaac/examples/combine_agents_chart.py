"""Build the combined ALL-AGENTS test-preference bar chart across N separate
single-brain runs — the equivalent of the original Unity code's
``all_imprinting_conds_test.png`` (one bar per test condition, one dot per
agent, chick reference band overlaid).

Because each seed is its own single-brain run dir, we merge them: each seed's
test CSV has its ``env_id`` column collapsed to the seed index (so the seed
counts as ONE agent = ONE dot, pooling its parallel test envs/episodes), then
all seeds are dropped into one ``<imprint>/logs/`` dir and the stock Isaac
``test_viz`` is run on it — reusing the exact, baseline-matching plotting
(colors, error bars, 50% chance line, chick band).

Usage:
  python examples/combine_agents_chart.py --output ~/nett_replicate_out \
      --encoder nature_cnn --tag clean --seeds 8 --imprint Object1 --chick binding
Produces:  <output>/_combined_<encoder>_<tag>/analysis_test/test_preference_<imprint>.png
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from nett_skrl.analysis.api import test_viz


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="~/nett_replicate_out")
    p.add_argument("--encoder", default="nature_cnn")
    p.add_argument("--tag", default="clean")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--imprint", default="Object1")
    p.add_argument("--chick", default="binding", help="ChickData/<name>.csv overlay")
    p.add_argument("--run-pattern", default="{encoder}_{tag}_s{seed}",
                   help="run dir name template")
    return p.parse_args()


def collapse_env_id(src_csvs: list[Path], dst: Path, agent_id: int) -> int:
    """Concatenate a seed's test CSV(s) into dst, rewriting env_id -> agent_id.

    Returns number of data rows written. The collapse makes the whole seed
    aggregate into a single (env_id, test_cond) bucket downstream = one dot.
    """
    n = 0
    header = None
    with open(dst, "w", newline="") as out_fh:
        writer = None
        for src in src_csvs:
            with open(src) as in_fh:
                reader = csv.DictReader(in_fh)
                if writer is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(out_fh, fieldnames=header)
                    writer.writeheader()
                for row in reader:
                    row["env_id"] = str(agent_id)
                    writer.writerow(row)
                    n += 1
    return n


def main():
    a = parse_args()
    output = Path(a.output).expanduser()
    merged_root = output / f"_combined_{a.encoder}_{a.tag}"
    logs = merged_root / a.imprint / "logs"
    if merged_root.exists():
        shutil.rmtree(merged_root)
    logs.mkdir(parents=True)

    seeds = range(a.seed_start, a.seed_start + a.seeds)
    used = 0
    for s in seeds:
        run = output / a.run_pattern.format(encoder=a.encoder, tag=a.tag, seed=s)
        src_logs = run / a.imprint / "logs"
        src_csvs = sorted(src_logs.glob(f"test_{a.imprint}_*.csv"))
        if not src_csvs:
            print(f"  [warn] no test CSV for seed {s} ({run.name}) — skipping")
            continue
        dst = logs / f"test_{a.imprint}_{used}.csv"   # one merged file per agent
        nrows = collapse_env_id(src_csvs, dst, agent_id=used)
        print(f"  agent {used} <- seed {s} ({run.name}): {nrows} rows")
        used += 1

    if used == 0:
        print("[combine] no agents found; nothing to plot.")
        return

    out_dir = merged_root / "analysis_test"
    # test_viz computes per-agent preferences (one row per agent after the
    # env_id collapse) and writes test_preferences.csv. It also emits the stock
    # mean-bar chart; we additionally draw the Unity-faithful all-agents chart.
    test_viz(merged_root, out_dir, chick_experiment=a.chick)
    pref_csv = out_dir / "test_preferences.csv"
    unity_png = out_dir / f"all_agents_test_{a.imprint}.png"
    _plot_unity_all_agents(pref_csv, unity_png, a.imprint, a.chick, used)
    print(f"\n[combine] {used} agents plotted.")
    print(f"[combine] Unity-style all-agents chart: {unity_png}")
    print(f"[combine] stock mean chart: {out_dir / f'test_preference_{a.imprint}.png'}")
    print(f"[combine] preferences csv: {pref_csv}")


# Unity all_imprinting_conds_test condition order (rest shown separately).
_UNITY_ORDER = ["1color", "1shape", "1shape&color", "2color", "2shape",
                "2shape&color", "binding"]
_BAR_COLORS = {
    "1color": "#3a8fbf", "1shape": "#f2e34c", "1shape&color": "#5b54a4",
    "2color": "#4aa45a", "2shape": "#c0407a", "2shape&color": "#2b50a0",
    "binding": "#b9c0e0", "rest": "darkgrey",
}


def _chick_band(chick: str, imprint: str):
    """{cond_lower: (avg, avg_dev)} from ChickData/<chick>.csv."""
    from nett_skrl.analysis.api import _load_chick_data
    return _load_chick_data(chick, imprint)


def _plot_unity_all_agents(pref_csv: Path, png: Path, imprint: str,
                           chick: str, n_agents: int, include_rest: bool = False):
    """Faithful Unity all_imprinting_conds_test chart: colored mean bars, one
    black dot per agent (jittered), SE error bars, pink chick band (avg±avg_dev),
    50% chance line, percent y-axis, significance stars (one-sample t vs 0.5)."""
    import csv as _csv
    import math
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    by_cond: dict[str, list[float]] = {}
    with open(pref_csv) as fh:
        for row in _csv.DictReader(fh):
            v = float(row["correct_pct"])
            if v > 1.5:
                v /= 100.0
            by_cond.setdefault(row["test_condition"], []).append(v)

    order = list(_UNITY_ORDER) + (["rest"] if include_rest else [])
    labels = [c for c in order if c in by_cond]
    chick = _chick_band(chick, imprint)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(range(len(labels)))
    means, ses = [], []
    for c in labels:
        xs = by_cond[c]
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((u - m) ** 2 for u in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
        means.append(m)
        ses.append(sd / math.sqrt(len(xs)) if xs else 0.0)

    ax.bar(x, [m * 100 for m in means], yerr=[s * 100 for s in ses],
           color=[_BAR_COLORS.get(c, "grey") for c in labels],
           capsize=8, width=0.7, linewidth=0, zorder=1,
           error_kw=dict(ecolor="black", elinewidth=1.5, zorder=4))

    # chick reference band (avg ± avg_dev) + center line, per condition.
    for i, c in enumerate(labels):
        ref = chick.get(c.lower())
        if not ref:
            continue
        avg, dev = ref
        ax.add_patch(plt.Rectangle((i - 0.35, (avg - dev) * 100), 0.7, 2 * dev * 100,
                                   facecolor="pink", alpha=0.55, edgecolor="none", zorder=2))
        ax.plot([i - 0.35, i + 0.35], [avg * 100, avg * 100],
                color="crimson", linewidth=1.8, zorder=3)

    # individual agent dots (jittered).
    import random
    rng = random.Random(0)
    for i, c in enumerate(labels):
        for v in by_cond[c]:
            ax.plot(i + rng.uniform(-0.18, 0.18), v * 100, "o",
                    color="black", markersize=5, zorder=5)

    # significance stars: one-sample t-test of agent prefs vs 0.5.
    try:
        from scipy import stats
        for i, c in enumerate(labels):
            xs = by_cond[c]
            if len(xs) > 1:
                t, p = stats.ttest_1samp(xs, 0.5)
                star = "**" if p <= 0.01 else ("*" if p <= 0.05 else "")
                if star:
                    ax.text(i, 2, star, ha="center", va="bottom", fontsize=14)
    except Exception:
        pass

    ax.axhline(50, ls="--", color="grey", linewidth=1.2, zorder=0)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Percent Correct", fontsize=13, fontweight="bold")
    ax.set_xlabel("Test Condition", fontsize=13, fontweight="bold")
    ax.set_title(f"All {n_agents} agents — imprint {imprint}", fontsize=12)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
