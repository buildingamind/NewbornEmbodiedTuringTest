"""Score an Isaac run with the UNITY pipeline's estimator, for cross-engine comparison.

★ WHY THIS EXISTS. The Isaac port's headline results have to be comparable to the published
Unity/SB3 numbers, and the two pipelines did not compute the same statistic. Comparing
Isaac's ``correct_pct_mean`` against Unity's ``avgs`` is an apples-to-oranges error that
looks like a result. This module computes UNITY's statistic -- on either engine's logs --
and refuses to be trusted until it has reproduced Unity's own published table.

THE STATISTIC. Of the test steps where the agent stands in the outer third of the chamber on
one side, the fraction spent on the side showing the *correct* monitor, averaged:

    within an episode  ->  over that agent's episodes  ->  over agents

⚠ **THE NESTING IS LOAD-BEARING.** Pooling flat over steps instead moves ``2color`` by
0.004, which is the same order as effects under active discussion -- so the wrong nesting is
invisible and wrong. The nesting was recovered by fitting Unity's published table, not
assumed: flat pooling agrees to 0.0036, the nesting above to 0.0011.

GEOMETRY IS NOT DUPLICATED HERE. The outer-third rule and the correct-side test come from
:func:`~nett_skrl.analysis.api.in_correct_chamber_third`, the single definition already used
by ``side_preference`` and the per-brain verdicts. A second copy of ``half_x / 3`` is exactly
how the two engines would drift apart again. The result is insensitive to it anyway: a
``half_x`` of 30.15 (the agent's clamped range) or 36.0 reproduces Unity's table just as well
and moves no condition by more than 0.003.

WHAT THIS DOES NOT DO. It reports a preference, not a mechanism, and it inherits every limit
of the run it reads -- including the fixed test start pose, which holds the effective sample
size per brain near 1 (see ``SIDE_LOCK_INVESTIGATION.md``). It is a comparison instrument.

Usage (validates against Unity first, then scores each Isaac CSV given)::

    python -m nett_skrl.analysis.unity_parity <run>/Object1/logs/test_Object1_0.csv
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .api import DEFAULT_CHAMBER_HALF_X, in_correct_chamber_third, side_preference

#: Unity's ``jan22`` CNN bundle. Outside both repos -- callers pass an explicit root, and
#: the tests skip when it is absent, the same contract as the repoA source-text tests.
DEFAULT_UNITY_BUNDLE = Path.home() / "code" / "jan22_cnntests_light_bundle"

_UNITY_RESULTS = "results/_0000_CNNTests/cnn_light/Object1"
_UNITY_STATS = "analysis/_0000_CNNTests/cnn_light/stats_across_all_agents.csv"

#: Presentation order: positive controls, then colour, then the shape conditions at issue.
CONDITION_ORDER = (
    "rest", "1color", "2color", "1shape&color", "2shape&color", "1shape", "2shape", "binding",
)


# --------------------------------------------------------------------- Student's t
# scipy is not in the nett-private venv (it carries Isaac Sim, not SciPy), so the two-sided
# one-sample t p-value is computed from the regularized incomplete beta directly. Validated
# against Unity's published (tval, pval) pairs by ``validate_against_unity``.
def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-30, 3e-16
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = tiny if abs(d) < tiny else d
    d = h = 1.0 / d
    for m in range(1, 300):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
                    -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))):
            d = 1.0 + num * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + num / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_pref = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_pref) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_pref) * _betacf(b, a, 1.0 - x) / b


def t_test_1samp(values: Sequence[float], mu: float = 0.5) -> tuple[float, float]:
    """Two-sided one-sample t against ``mu``; ``(nan, nan)`` when undefined."""
    n = len(values)
    if n < 2:
        return math.nan, math.nan
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    if var <= 0.0:
        return math.nan, math.nan
    t = (mean - mu) / math.sqrt(var / n)
    df = n - 1
    return t, _betainc(df / 2.0, 0.5, df / (df + t * t))


# --------------------------------------------------------------------- the estimator
def agent_stats(
    rows: Iterable[Mapping[str, str]],
    *,
    x_field: str,
    episode_fields: Sequence[str],
    chamber_half_x: float = DEFAULT_CHAMBER_HALF_X,
) -> dict[str, dict[str, float]]:
    """Per-condition ``{"score", "side_preference"}`` for ONE agent.

    ``score`` is Unity's statistic: the mean over episodes of the within-episode correct-side
    fraction. ``side_preference`` is ``(R - L) / (R + L)`` over the same outer-third steps,
    pooled across the agent's episodes.

    ★ THE TWO TRAVEL TOGETHER BY DESIGN. On a counterbalanced two-alternative design a
    PERFECTLY side-locked agent scores EXACTLY 0.5 -- indistinguishable from indifference in
    ``score`` alone. Mining ``~/code/analysis`` on 2026-07-30 found the two highest historical
    binding numbers were both artifacts of exactly that: ``archive/bind0_Manju2`` at 0.923 with
    ``|sp|max = 1.000``, and ``archive/bind3/small_ccc51`` at 0.799 with ``|sp|avg = 0.997``
    (ten fully locked agents). Computing the screen here, in the same pass, is what stops a
    mean from being reported without it.

    ``episode_fields`` names the columns that jointly identify an episode -- Unity logs one
    agent per file and numbers episodes with ``Episode``; Isaac interleaves parallel envs in
    one file, so an episode there is ``(env_id, episode)``. Getting this wrong on the Isaac
    side would average over env-interleaved fragments, not episodes.
    """
    per_episode: dict[str, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))
    sides: dict[str, list[int]] = defaultdict(lambda: [0, 0])       # cond -> [right, left]
    for row in rows:
        try:
            agent_x = float(row[x_field])
        except (TypeError, ValueError):
            continue                      # a torn row biases nothing if dropped outright
        outer, correct = in_correct_chamber_third(
            agent_x, row["correct.monitor"], chamber_half_x)
        if not outer:
            continue
        cond = row["test.cond"]
        per_episode[cond][tuple(row[f] for f in episode_fields)].append(int(correct))
        # Which SIDE, independent of where the target is. The outer-third test above is the
        # shared one; only the sign is read here, so there is no second copy of the rule.
        sides[cond][0 if agent_x > 0 else 1] += 1
    out: dict[str, dict[str, float]] = {}
    for cond, episodes in per_episode.items():
        right, left = sides[cond]
        out[cond] = {
            "score": math.fsum(math.fsum(v) / len(v) for v in episodes.values()) / len(episodes),
            # Never None here: a condition only appears once it has an outer-third step.
            "side_preference": side_preference(left, right),
        }
    return out


def agent_scores(rows: Iterable[Mapping[str, str]], **kw) -> dict[str, float]:
    """The scores-only view of :func:`agent_stats`, for the nesting contract."""
    return {cond: s["score"] for cond, s in agent_stats(rows, **kw).items()}


def summarize(
    agents: Sequence[Mapping[str, Mapping[str, float]]],
) -> list[dict[str, float | int | str]]:
    """Aggregate per-agent :func:`agent_stats` across agents, one record per condition.

    Emits ``sp_abs_mean`` and ``sp_abs_max`` -- the mean and worst per-agent ``|side
    preference|`` -- next to every mean, so :func:`format_table` can print the side-lock
    screen unconditionally. ``|sp|`` is aggregated as an ABSOLUTE value: two agents locked on
    opposite walls have signed preferences that cancel to ~0 while both are fully locked.

    ⚠ ``n_immobile`` IS PART OF THE RESULT, NOT A DIAGNOSTIC. An agent that never leaves the
    centre third scores nothing in any condition, and dropping it quietly turns n=4 into n=3
    with no visible trace -- which is how a 4-agent arm gets reported as a 3-agent one. It
    happens for real: a policy can train to a normal reward and still, under deterministic
    evaluation from a fixed start pose, reverse into the back wall and sit there for every
    test step (observed 2026-07-29, arm A offset 1, reward tail 162.5, ``max|x| = 1e-9``).
    Callers must surface it; :func:`format_table` prints it.
    """
    immobile = sum(1 for a in agents if not a)
    out = []
    for cond in CONDITION_ORDER:
        scored = [a[cond] for a in agents if cond in a]
        if not scored:
            continue
        vals = [s["score"] for s in scored]
        sp = [abs(s["side_preference"]) for s in scored]
        mean = math.fsum(vals) / len(vals)
        sd = (math.sqrt(math.fsum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
              if len(vals) > 1 else math.nan)
        t, p = t_test_1samp(vals)
        out.append({"condition": cond, "mean": mean, "sd": sd, "n": len(vals),
                    "tval": t, "pval": p, "n_immobile": immobile,
                    "sp_abs_mean": math.fsum(sp) / len(sp), "sp_abs_max": max(sp)})
    return out


# --------------------------------------------------------------------- readers
def read_unity_log(path: Path) -> list[dict[str, str]]:
    """Unity's logger pads every field with a space and interleaves bare trace lines.

    The header is ``"Episode, Step, agent.x, ..."`` and stray ``OnActionReceived`` lines sit
    between data rows, so ``csv.DictReader`` alone yields whitespace-prefixed keys and short
    rows. Rows whose field count does not match the header are dropped.
    """
    with open(path, newline="") as fh:
        header = [h.strip() for h in fh.readline().rstrip("\n").split(",")]
        rows = []
        for line in fh:
            parts = [v.strip() for v in line.rstrip("\n").split(",")]
            if len(parts) == len(header):
                rows.append(dict(zip(header, parts)))
    return rows


def read_isaac_log(path: Path) -> list[dict[str, str]]:
    """Test-phase rows of an Isaac ``test_<imprint>_<n>.csv``."""
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("experiment.phase") == "test"]


# ------------------------------------------------- Unity's AGGREGATED per-episode results
# ★ A SECOND UNITY LAYOUT, AND THE ONE THAT MADE THE ARCHIVE USABLE. Alongside the raw
# per-step logs above, the Unity pipeline wrote a per-episode ``test_results.csv``:
#
#   Episode,left.monitor,right.monitor,correct.monitor,experiment.phase,imprint.cond,
#   test.cond,left_steps,right_steps,middle_steps,filename,agent
#
# ~/code/analysis holds 1279 variant directories with this file and NONE with ``agent.x``, so
# it is the only route to the historical results. The step counts are enough for BOTH the
# nested estimator and ``|sp|`` -- which is what let the 2026-07-30 mining screen the
# headline binding numbers for side-lock and disqualify them. This reader lives in the module
# so that mining is reproducible rather than a scratch script.
#
# ⚠ THE GEOMETRY IS UNITY'S, NOT OURS. ``left_steps``/``right_steps`` were already reduced by
# Unity using Unity's own outer-third rule, so ``chamber_half_x`` cannot be varied here and is
# deliberately not accepted. Numbers from this reader are therefore comparable to Unity's
# published table but NOT threshold-robustness-testable the way :func:`score_unity` is.
def read_unity_results(path: Path) -> list[dict[str, str]]:
    """Test-phase rows of a Unity ``test_results.csv`` (one row per episode per agent)."""
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if (r.get("experiment.phase") or "").strip() == "test"]


def unity_results_agents(path: Path) -> list[dict[str, dict[str, float]]]:
    """Per-agent :func:`agent_stats`-shaped records from an aggregated ``test_results.csv``.

    ⚠ AN AGENT IS ``(imprint.cond, agent)``, NOT ``agent``. One file holds every imprint
    condition, and a brain imprinted on Object1 is a different subject from one imprinted on
    Object2 -- they merely share a brain index. Keying on ``agent`` alone would merge the two
    and halve the reported n while averaging across imprinting conditions (in
    ``archive/Compendium1.2.15`` that is 10 agents reported as 5).

    Episodes with no outer-third steps are skipped: the within-episode fraction has an empty
    denominator there, and counting them as 0.5 would invent data.
    """
    per_episode: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sides: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in read_unity_results(path):
        try:
            left, right = int(row["left_steps"]), int(row["right_steps"])
        except (KeyError, TypeError, ValueError):
            continue                      # a torn row biases nothing if dropped outright
        if left + right == 0:
            continue
        agent = (row["imprint.cond"], row["agent"])
        cond = row["test.cond"]
        correct = right if row["correct.monitor"].strip() == "right" else left
        per_episode[agent][cond].append(correct / (left + right))
        sides[agent][cond][0] += right
        sides[agent][cond][1] += left
    out = []
    for agent in sorted(per_episode):
        stats = {}
        for cond, fractions in per_episode[agent].items():
            r, l = sides[agent][cond]
            stats[cond] = {"score": math.fsum(fractions) / len(fractions),
                           "side_preference": side_preference(l, r)}
        out.append(stats)
    return out


def score_unity_results(path: Path) -> list[dict]:
    """Score one aggregated Unity ``test_results.csv`` -- an archived experiment variant."""
    return summarize(unity_results_agents(path))


def score_unity(bundle: Path = DEFAULT_UNITY_BUNDLE, **kw) -> list[dict]:
    """Recompute Unity's per-condition table from Unity's own raw per-agent logs."""
    return summarize([
        agent_stats(read_unity_log(f), x_field="agent.x", episode_fields=("Episode",), **kw)
        for f in sorted((bundle / _UNITY_RESULTS).glob("brain_*/logs/test_*.csv"))
    ])


def isaac_agents(csv_path: Path, **kw) -> list[dict[str, dict[str, float]]]:
    """Per-agent :func:`agent_stats` from one Isaac test CSV, one entry per brain."""
    by_brain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_isaac_log(csv_path):
        by_brain[row["brain_id"]].append(row)
    return [
        agent_stats(rows, x_field="agent.x", episode_fields=("env_id", "episode"), **kw)
        for _, rows in sorted(by_brain.items(), key=lambda kv: int(kv[0]))
    ]


def score_isaac(csv_path: Path, **kw) -> list[dict]:
    """Score one Isaac run, whose single CSV holds every brain."""
    return summarize(isaac_agents(csv_path, **kw))


def score_isaac_runs(csv_paths: Sequence[Path], **kw) -> list[dict]:
    """Pool SEPARATE single-brain runs into one arm, each file contributing its agents.

    ⚠ NEEDED BECAUSE ``brain_id`` IS FILE-LOCAL. A one-brain-per-GPU fan-out
    (``NETT_BRAINS=1`` + ``NETT_BRAIN_OFFSET``) writes ``brain_id=0`` in every run's CSV --
    the offset lands in the run directory name, not the column. Concatenating the rows and
    grouping by ``brain_id`` therefore collapses a 4-agent arm into ONE agent, which shrinks
    the reported sd toward 0 and inflates |t|: a spurious significant result from a purely
    clerical merge. Each path's agents are kept distinct here instead.
    """
    agents: list[Mapping[str, Mapping[str, float]]] = []
    for path in csv_paths:
        agents.extend(isaac_agents(path, **kw))
    return summarize(agents)


def published_unity_table(bundle: Path = DEFAULT_UNITY_BUNDLE) -> dict[str, dict[str, str]]:
    """Unity's shipped ``stats_across_all_agents.csv``, keyed by condition.

    Note it omits ``rest`` -- Unity's own pipeline drops the positive control from the
    across-agents table, so no validation is possible for that row.
    """
    with open(bundle / _UNITY_STATS, newline="") as fh:
        return {r["test.cond"]: r for r in csv.DictReader(fh)}


def validate_against_unity(bundle: Path = DEFAULT_UNITY_BUNDLE, **kw) -> dict[str, float]:
    """Largest disagreement between this estimator and Unity's published table.

    Returns ``{"mean": .., "tval": .., "pval": ..}`` -- the max absolute difference in each.
    A caller that has not checked this has not established that its Isaac column and its
    Unity column mean the same thing.
    """
    published = published_unity_table(bundle)
    worst = {"mean": 0.0, "tval": 0.0, "pval": 0.0}
    for rec in score_unity(bundle, **kw):
        ref = published.get(str(rec["condition"]))
        if ref is None:
            continue                                  # 'rest' is absent upstream
        for key, col in (("mean", "avgs"), ("tval", "tval"), ("pval", "pval")):
            worst[key] = max(worst[key], abs(float(rec[key]) - float(ref[col])))
    return worst


# --------------------------------------------------------------------- CLI
def format_table(title: str, records: Sequence[Mapping]) -> str:
    if not records:
        return (f"== {title} ==\nNO AGENTS SCORED -- every agent stayed in the centre third, "
                f"or no test rows were read. This is not a null result.\n")
    # ★ |sp| IS PRINTED UNCONDITIONALLY, next to the mean it qualifies. The rule "no mean
    # without its side-preference" was previously a matter of discipline in prose; a
    # side-locked agent scores exactly 0.5 on a counterbalanced design, so a table without
    # this column cannot be read correctly no matter how careful the reader is.
    lines = [f"== {title} ==",
             f"{'condition':<14}{'mean':>8}{'sd':>8}{'n':>4}{'t':>9}{'p':>10}"
             f" {'|sp|avg':>8}{'|sp|max':>8}"]
    locked = False
    for r in records:
        p = float(r["pval"])
        star = "*" if (not math.isnan(p) and p < 0.05) else " "
        sp_avg, sp_max = float(r["sp_abs_mean"]), float(r["sp_abs_max"])
        locked = locked or sp_avg > 0.5
        lines.append(f"{r['condition']:<14}{float(r['mean']):>8.4f}{float(r['sd']):>8.4f}"
                     f"{int(r['n']):>4d}{float(r['tval']):>9.3f}{p:>10.4f}{star}"
                     f"{sp_avg:>8.3f}{sp_max:>8.3f}")
    immobile = int(records[0].get("n_immobile", 0) or 0)
    if immobile:
        lines.append(f"⚠ {immobile} agent(s) EXCLUDED as immobile (never left the centre "
                     f"third); n above is the scored count, not the arm size.")
    if locked:
        lines.append("⚠ SIDE-LOCK: |sp|avg > 0.5 in a condition above. A locked agent scores "
                     "~0.5 on a counterbalanced design, so those means are not preferences.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="*", type=Path, help="Isaac test_<imprint>_<n>.csv files")
    ap.add_argument("--unity-bundle", type=Path, default=DEFAULT_UNITY_BUNDLE)
    ap.add_argument("--chamber-half-x", type=float, default=DEFAULT_CHAMBER_HALF_X)
    ap.add_argument("--pool", metavar="LABEL",
                    help="treat every CSV given as agents of ONE arm and print a single "
                         "table (for a one-brain-per-GPU fan-out); default is one table "
                         "per file")
    ap.add_argument("--unity-results", type=Path, action="append", default=[],
                    metavar="test_results.csv",
                    help="score an ARCHIVED Unity variant from its aggregated per-episode "
                         "test_results.csv (~/code/analysis/...); repeatable. Needs no "
                         "bundle -- these are self-contained and --chamber-half-x is inert "
                         "because the step counts were already reduced by Unity")
    args = ap.parse_args(argv)
    kw = {"chamber_half_x": args.chamber_half_x}

    # The archive path is self-contained: it scores Unity against Unity, so there is nothing
    # cross-engine to validate and the bundle is not required.
    for path in args.unity_results:
        print(format_table(f"UNITY archive {path.parent.name}", score_unity_results(path)))
    if args.unity_results and not args.csv:
        return 0

    if not args.unity_bundle.exists():
        print(f"Unity bundle not found: {args.unity_bundle}\n"
              f"Cross-engine numbers are not comparable without it; refusing to print a "
              f"table that looks like one.")
        return 2

    worst = validate_against_unity(args.unity_bundle, **kw)
    print(f"estimator vs Unity's published table: max |d| mean={worst['mean']:.5f} "
          f"t={worst['tval']:.5f} p={worst['pval']:.5f}\n")
    print(format_table("UNITY jan22 CNN (SB3/SAC)", score_unity(args.unity_bundle, **kw)))
    if args.pool:
        print(format_table(f"ISAAC {args.pool} (pooled, {len(args.csv)} runs)",
                           score_isaac_runs(args.csv, **kw)))
        return 0
    for path in args.csv:
        run = next((p.name for p in path.parents
                    if p.name.startswith(("CNN", "ViT", "ViViT", "SimCLR"))), path.name)
        print(format_table(f"ISAAC {run}", score_isaac(path, **kw)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
