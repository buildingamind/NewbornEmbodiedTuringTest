"""Pool the emissive-curve replication runs and report the preregistered endpoint.

Reads the ``*_result.json`` files written by ``replicate_emissive_curve.py`` and
reports, per brightness point:

* how many independent runs and brains contributed,
* the fraction of brains classified ``LEARN`` (the PRIMARY endpoint),
* the SIDE-LOCK / chance / n-a breakdown,
* how many brains were immobile (empty denominator),
* ``correct_pct_mean`` for reference only.

``correct_pct_mean`` is NOT the endpoint. A side-locked brain and a wandering
brain both average to ~0.5, so the scalar cannot express the claim under test.

Runs whose ``episode_integrity_ok`` is not true are EXCLUDED and listed
separately: an under-sized eval budget truncates episodes unevenly across
co-hosted brains, which fabricates between-brain differences.

    python examples/aggregate_emissive_replication.py ~/nett_emissive_replication

Pass ``--include-originals`` with one or more ``BRIGHTNESS=RUN_DIR`` pairs to
fold in earlier runs; they are analysed on the fly. Note that the 2026-07-24
originals record their chamber variant ONLY in their output directory name, so
that mapping is an assertion by the caller, not something the artifacts prove.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

VERDICTS = ("LEARN", "SIDE-LOCK", "chance", "n/a")


def integrity(run_dir: Path) -> tuple[bool | None, bool]:
    """Return ``(no_truncated_episodes, episodes_per_env_even)`` for a run tree.

    Recomputed here rather than read from the result file so that the gate has
    ONE definition and old result files cannot lock in a stale verdict.

    Only truncation gates. Uneven episodes-per-env is a caveat: 260 test episodes
    do not divide over 16 envs, so 4 envs always run one extra. Every run of this
    recipe has it, published ones included, so gating on it would discard the
    whole dataset instead of discriminating within it.
    """
    from nett_skrl.analysis.episode_integrity import verify_run

    checks = verify_run(run_dir)
    if not checks:
        return None, False
    return (
        all(res.ok for _, res in checks),
        all(len(set(res.episodes_per_env.values())) <= 1 for _, res in checks),
    )


def _scores_from_result(path: Path) -> tuple[int, dict, bool | None, bool, str, str]:
    with path.open() as fh:
        result = json.load(fh)
    rest = next(iter(result.get("scores", {}).values()), {})
    run_dir = Path(result["run_dir"])
    ok, even = integrity(run_dir)
    return (
        int(result["emissive_intensity"]),
        rest,
        ok,
        even,
        result.get("locomotion") or _locomotion_from_config(run_dir),
        result.get("chamber_style") or _style_from_variant(result.get("chamber_variant")),
    )


def _style_from_variant(variant: str | None) -> str:
    """Chamber FAMILY the run happened in, from its variant name if not recorded.

    Pooled alongside physics and for the same reason: the 2026-07-25 restyle moves
    monitor/wall contrast by ~+11 LSB at EVERY brightness (Michelson 0.070 -> 0.107
    at e1000), so a flat point and a restyled point at the same emissive value are
    different stimuli, not different seeds. Runs predating the style axis recorded
    nothing, hence the fallback and the explicit unknown marker.
    """
    if not variant:
        return "(unrecorded)"
    return "realistic" if variant.startswith("chamber_re") else "flat"


def _locomotion_from_config(run_dir: Path) -> str:
    """Physics the run used, or a marker that it followed the runtime default.

    A run that records neither is NOT comparable to one that pinned the mode:
    the runtime default flipped kinematic -> wheeled on 2026-07-24, so "whatever
    was default" means different physics before and after that date.
    """
    try:
        import yaml

        with (run_dir / "config.yaml").open() as fh:
            cfg = yaml.safe_load(fh)
        loco = (cfg.get("environment") or {}).get("locomotion")
        if loco:
            return str(loco)
    except Exception:  # noqa: BLE001 - provenance reporting must not break scoring
        pass
    return "(unpinned: runtime default of the day)"


def _scores_from_run(run_dir: Path) -> dict:
    """Analyse an earlier run so it can be scored on the same endpoint.

    Writes to a SEPARATE ``analysis_rescored/`` tree. The run's own
    ``analysis/`` is the artifact that backed whatever was already published;
    regenerating in place would overwrite it.
    """
    from nett_skrl.analysis import analyze

    out = Path(analyze(run_dir, run_dir / "analysis_rescored"))
    with (out / "summary.json").open() as fh:
        summary = json.load(fh)
    for by_cond in summary.get("test", {}).values():
        if "rest" in by_cond:
            return by_cond["rest"]
    return {}


def _accumulate(bucket: dict, rest: dict, locomotion: str = "", style: str = "") -> None:
    counts = rest.get("verdict_counts") or {}
    for v in VERDICTS:
        bucket["verdicts"][v] += counts.get(v, 0)
    bucket["runs"] += 1
    bucket["immobile"] += rest.get("n_brains_immobile") or 0
    if locomotion:
        bucket["locomotion"].add(locomotion)
    if style:
        bucket["style"].add(style)
    if rest.get("correct_pct_mean") is not None:
        bucket["pct"].append(rest["correct_pct_mean"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_root", type=Path, help="dir holding the *_result.json files")
    ap.add_argument("--include-originals", nargs="*", default=[], metavar="EMISSIVE=DIR",
                    help="fold in earlier runs, e.g. 300=~/nett_kg_e300_nature_cnn/<run>")
    args = ap.parse_args()

    by_point: dict[int, dict] = defaultdict(
        lambda: {"verdicts": dict.fromkeys(VERDICTS, 0), "runs": 0,
                 "immobile": 0, "pct": [], "locomotion": set(), "style": set()}
    )
    excluded: list[tuple[str, str]] = []
    uneven: list[str] = []

    for res in sorted(Path(args.out_root).expanduser().glob("*_result.json")):
        emissive, rest, integrity_ok, even, locomotion, style = _scores_from_result(res)
        if integrity_ok is not True:
            excluded.append((res.name, "truncated test episodes"))
            continue
        if not rest.get("verdict_counts"):
            excluded.append((res.name, "no verdict counts — analysis predates them?"))
            continue
        if not even:
            uneven.append(res.name)
        _accumulate(by_point[emissive], rest, locomotion, style)

    for spec in args.include_originals:
        emissive, _, run_dir = spec.partition("=")
        path = Path(run_dir).expanduser()
        integrity_ok, even = integrity(path)
        if integrity_ok is not True:
            excluded.append((run_dir, "truncated test episodes"))
            continue
        rest = _scores_from_run(path)
        if rest.get("verdict_counts"):
            if not even:
                uneven.append(run_dir)
            _accumulate(by_point[int(emissive)], rest)
        else:
            excluded.append((run_dir, "no verdict counts after analysis"))

    print(f"{'emissive':>9} {'runs':>5} {'brains':>7} {'LEARN':>6} {'S-LOCK':>7} "
          f"{'chance':>7} {'n/a':>5} {'immob':>6} {'learn_frac':>11} {'pct_mean':>9}  "
          f"{'physics':<12} chamber")
    print("-" * 112)
    for emissive in sorted(by_point):
        b = by_point[emissive]
        v = b["verdicts"]
        n = sum(v.values())
        frac = f"{v['LEARN'] / n:.3f}" if n else "n/a"
        pct = f"{sum(b['pct']) / len(b['pct']):.4f}" if b["pct"] else "n/a"
        loco = ",".join(sorted(b["locomotion"])) or "?"
        style = ",".join(sorted(b["style"])) or "?"
        print(f"{emissive:9d} {b['runs']:5d} {n:7d} {v['LEARN']:6d} {v['SIDE-LOCK']:7d} "
              f"{v['chance']:7d} {v['n/a']:5d} {b['immobile']:6d} {frac:>11} {pct:>9}  "
              f"{loco:<12} {style}")

    mixed = {loco for b in by_point.values() for loco in b["locomotion"]}
    if len(mixed) > 1:
        print(f"\n⚠ MIXED PHYSICS across the pooled runs ({sorted(mixed)}). Points are "
              "NOT comparable — locomotion changes the agent's dynamics, so a "
              "difference cannot be attributed to brightness.")

    mixed_style = {st for b in by_point.values() for st in b["style"]}
    if len(mixed_style) > 1:
        print(f"\n⚠ MIXED CHAMBER STYLES across the pooled runs ({sorted(mixed_style)}). "
              "Points are NOT comparable — the restyle shifts monitor/wall contrast by "
              "~+11 LSB at every brightness, so this pools different STIMULI, not "
              "different seeds.")

    if excluded:
        print("\nEXCLUDED:")
        for name, why in excluded:
            print(f"  {name}: {why}")
    if uneven:
        print(f"\nCAVEAT — episodes-per-env uneven (4 envs run one extra; a fixed "
              f"property of the eval budget, present in every run) in {len(uneven)} run(s).")

    # The falsifiable claim the replication was preregistered to test.
    if 300 in by_point and 1000 in by_point:
        def frac(point: int) -> float:
            v = by_point[point]["verdicts"]
            n = sum(v.values())
            return v["LEARN"] / n if n else float("nan")

        f300, f1000 = frac(300), frac(1000)
        verdict = "REPRODUCED" if f300 < f1000 else "NOT REPRODUCED"
        print(f"\nPREREGISTERED CLAIM 'e300 learns strictly less often than e1000': "
              f"{verdict} ({f300:.3f} vs {f1000:.3f})")
        if verdict != "REPRODUCED":
            print("  => downgrade the curve in blueprint.md; do not explain it away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
