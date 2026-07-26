"""Replicate ONE point of the monitor-brightness (emissive) binding curve.

WHY THIS EXISTS. The published curve (e300 2/8 brains learned, e1000 7/8,
e2000 6/8) was produced by hand-editing ``train_binding_8brain.py`` three ways
per point -- wandb ``online`` -> ``disabled``, ``episodes.train`` 2000 -> 1000,
and a per-point ``OUTPUT`` -- with ``NETT_CHAMBER_VARIANT`` exported in the
shell. Nothing tracked recorded that recipe, so the numbers could not be
re-derived from any entry point in either repo. This driver IS that recipe: it
imports the canonical config and applies the deltas, so the curve has a
reproducible source.

It is NOT ``isaac_lab/scripts/run_emissive_binding_sweep.py``. That is a
different, smaller experiment (2 brains, res128) and it did not produce the
curve.

THE SWEPT VARIABLE is the baked ``inputs:emissive_intensity`` of the chamber's
monitors, selected by pointing the env at a pre-baked chamber variant. The
variants differ in NOTHING else (verified by
``isaac_lab/scripts/verify/verify_emissive_variants.py``).

THE SEED KNOB is ``brain_id_offset``. ``TaskConfig`` derives the base seed from
the CONDITION NAME alone, so re-running with the SAME offset is a determinism
replay and running with a DIFFERENT offset is an independent sample. Do not
conflate the two: the published points are all offset 0, so replications must
use offsets that have not been used at that brightness.

⚠ EVERYTHING ELSE MUST BE PINNED, NOT INHERITED. The first replication wave
(2026-07-25) accidentally changed the PHYSICS as well as the seed: the canonical
config omits ``locomotion``, and the runtime default flipped kinematic ->
wheeled hours after the curve was measured, so the wave ran on GPU PhysX while
the published curve ran on CPU PhysX. Its results are therefore not a clean
replication of the published numbers. ``--locomotion`` now pins it, defaulting
to what the published curve actually ran under.

    # one point, one GPU, ~6 h
    cd src_isaac
    PYTHONPATH=.:../../NewbornEmbodiedTuringTest_Private/isaac_lab/source \\
    CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES \\
    python examples/replicate_emissive_curve.py --emissive 1000 --brain-id-offset 8 \\
        --out-root ~/nett_emissive_replication

Scoring uses ``verdict`` / ``learn_fraction`` from the analysis summary, not
``correct_pct_mean``: a side-locked brain and a wandering brain both average to
~0.5, so the scalar cannot express the endpoint this curve is about.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# Brightness points with a published measurement. Replications must target these.
MEASURED_POINTS = (300, 650, 1000, 2000)

#: Episodes used for the published curve. Binding needs >= 1000 episodes to bind
#: at all -- shorter budgets produce chance/side-lock at EVERY brightness and say
#: nothing about the curve.
CURVE_EPISODES_TRAIN = 1000

#: Physics the PUBLISHED curve actually ran under. It is pinned rather than
#: inherited because the runtime default flipped to "wheeled" a few hours after
#: the curve was measured -- see build_config().
PUBLISHED_LOCOMOTION = "kinematic"


#: Wall/ceiling appearance of the chamber the run happens in. "flat" is what the
#: PUBLISHED curve ran in (albedo 1.0, roughness 1.0, nothing casts a shadow);
#: "realistic" is the 2026-07-25 restyle (albedo 0.85, roughness 0.5, micro-normal
#: map, upper bezels cast). They are separate baked asset families, so this only
#: changes which file is referenced.
#:
#: ⚠ IT IS NOT A COSMETIC KNOB. Measured on the agent's own centre-eye camera
#: (isaac_lab/scripts/capture_emissive_sweep.py, monitor vs wall patch), the
#: restyle leaves MONITOR luma unchanged to within 1 LSB at every brightness but
#: drops WALL luma by a constant ~11 LSB, so the monitor/wall gap widens by ~11
#: LSB and Michelson contrast by ~+0.036 EVERYWHERE. At the operative default
#: e1000 that is 0.070 -> 0.107, i.e. +52% relative separability. A restyled run
#: is therefore NOT comparable to a flat run at the same emissive value.
PUBLISHED_STYLE = "flat"


def variant_name(emissive: int, style: str = PUBLISHED_STYLE) -> str:
    prefix = "chamber_re" if style == "realistic" else "chamber_e"
    return f"{prefix}{int(emissive)}.usdc"


def build_config(emissive: int, brain_id_offset: int, episodes_train: int,
                 locomotion: str = PUBLISHED_LOCOMOTION,
                 style: str = PUBLISHED_STYLE,
                 resolution: int | None = None,
                 auto_envs: bool = False) -> dict:
    """The canonical 8-brain binding config with the curve's deltas applied."""
    import train_binding_8brain as canonical

    # The run directory NAMES the chamber family. The 2026-07-24 curve runs
    # recorded their variant NOWHERE in their artifacts -- not config.yaml (the
    # env var overrides cfg), not hparams.json, not the launch logs -- so the
    # only provenance was the directory name. Keeping the style in the name means
    # a restyled run can never be mistaken for a flat one at the same brightness.
    style_tag = "" if style == PUBLISHED_STYLE else f"{style[0]}"
    cfg = copy.deepcopy(canonical.CONFIG)
    # Delta 0: PIN the physics. The canonical config omits `locomotion`, so it
    # inherits the RUNTIME DEFAULT -- and that default flipped kinematic ->
    # wheeled on 2026-07-24 14:46, AFTER the curve was measured at 07:39. A
    # replication that inherits the default therefore silently changes the
    # physics as well as the seed, and cannot attribute a difference to either.
    # Measured: the published runs logged "PhysX placement: cpu (kinematic ...)",
    # the first replication wave logged "cuda:0 (wheeled articulation ...)".
    cfg["environment"] = dict(cfg["environment"], locomotion=locomotion)
    cfg["name"] = (
        f"binding_{style_tag}e{int(emissive)}_off{brain_id_offset}"
        f"_{datetime.now():%Y%m%d_%H%M%S}"
    )
    # Delta 1: no wandb. The curve runs were offline; leaving it online would add
    # a network dependency to a 6 h run and change nothing scientific.
    cfg["brain"]["wandb"] = dict(cfg["brain"]["wandb"], mode="disabled")
    # Delta 2: the curve's training budget (the canonical example ships 2000).
    cfg["episodes"] = dict(cfg["episodes"], train=episodes_train)
    # Delta 3: the seed knob. Absent from the canonical config, which is offset 0.
    cfg["brain_id_offset"] = brain_id_offset
    # Delta 4 (optional): observation resolution. The canonical recipe is 256. 128
    # is ~4x less render work per env, which matters because throughput here is
    # bound by the per-GPU RTX submission floor, not by compute (GPU util ~25-38%
    # with the CPU idle). ⚠ IT CHANGES THE STIMULUS -- a res128 point is not
    # comparable to the published res256 curve.
    if resolution is not None:
        cfg["environment"] = dict(cfg["environment"], input_resolution=int(resolution))
    # Delta 5 (optional): let measured VRAM size the env count. Per the schema,
    # "auto" VERIFIES training's count and only ever backs OFF (training num_envs is
    # load-bearing for learning), and separately measures how wide the TEST phase can
    # run -- test only replays a fixed schedule, so width there is free speed.
    # Requires task_memory "auto": the ceiling comes from the dry-run measurement.
    if auto_envs:
        cfg["max_parallel_envs"] = "auto"
        cfg["task_memory"] = "auto"
    return cfg


def score(analysis_dir: Path) -> dict:
    """Per-brain verdicts + the primary endpoint, from the analysis summary."""
    with (analysis_dir / "summary.json").open() as fh:
        summary = json.load(fh)
    out: dict = {}
    for imprint, by_cond in summary.get("test", {}).items():
        rest = by_cond.get("rest")
        if rest:
            out[imprint] = {
                "learn_fraction": rest.get("learn_fraction"),
                "verdict_counts": rest.get("verdict_counts"),
                "n_brains": rest.get("n_brains"),
                "n_brains_immobile": rest.get("n_brains_immobile"),
                "correct_pct_mean": rest.get("correct_pct_mean"),
                "side_preference_abs_mean": rest.get("side_preference_abs_mean"),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emissive", type=int, required=True,
                    help=f"baked monitor brightness; measured points: {MEASURED_POINTS}")
    ap.add_argument("--brain-id-offset", type=int, required=True,
                    help="seed knob; must differ from any offset already run at "
                         "this brightness or the run is a determinism replay, "
                         "not an independent sample")
    ap.add_argument("--episodes-train", type=int, default=CURVE_EPISODES_TRAIN)
    ap.add_argument("--locomotion", default=PUBLISHED_LOCOMOTION,
                    choices=("kinematic", "wheeled"),
                    help="pinned, NOT inherited from the runtime default, which "
                         "flipped after the curve was measured (default: the "
                         "mode the published curve ran under)")
    ap.add_argument("--chamber-style", default=PUBLISHED_STYLE,
                    choices=("flat", "realistic"),
                    help="which baked chamber family to run in; pinned, not "
                         "inherited (default: what the published curve ran in). "
                         "See PUBLISHED_STYLE -- 'realistic' shifts monitor/wall "
                         "contrast by ~+11 LSB at EVERY brightness, so its points "
                         "are not comparable to flat points at the same value")
    ap.add_argument("--device", type=int, default=0,
                    help="index WITHIN the visible devices (pin with CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--resolution", type=int, default=None,
                    help="observation resolution; default = the canonical 256. 128 is "
                         "~4x less render work but CHANGES THE STIMULUS, so res128 "
                         "points are not comparable to the published curve.")
    ap.add_argument("--auto-envs", action="store_true",
                    help="max_parallel_envs/task_memory = auto: verify training's env "
                         "count against measured VRAM (backs off only) and size the "
                         "TEST phase as wide as VRAM allows")
    ap.add_argument("--devices", type=int, nargs="+", default=None,
                    help="spread this run's per-brain tasks over SEVERAL GPUs instead "
                         "of packing all of them onto one. NETT dispatches one task "
                         "per brain (nett.py::_max_concurrent_tasks) and places each on "
                         "the most ledger-free GPU, so 8 brains over 8 GPUs means 2 envs "
                         "per GPU instead of 16 — the per-GPU RTX submission floor is "
                         "what limits throughput here (GPU util only ~25-38%, CPU idle), "
                         "so this cuts WALL time per run at the cost of running one "
                         "brightness point at a time.")
    ap.add_argument("--out-root", default="~/nett_emissive_replication")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.emissive_replication")

    # Must be set BEFORE the env is constructed in the task subprocesses, which
    # inherit this environment.
    os.environ["NETT_CHAMBER_VARIANT"] = variant_name(args.emissive, args.chamber_style)

    from nett_skrl import NETT
    from nett_skrl.analysis import analyze
    from nett_skrl.analysis.episode_integrity import verify_run

    cfg = build_config(args.emissive, args.brain_id_offset, args.episodes_train,
                       args.locomotion, args.chamber_style,
                       resolution=args.resolution, auto_envs=args.auto_envs)
    out_root = Path(os.path.expanduser(args.out_root))
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("emissive=%d variant=%s style=%s brain_id_offset=%d episodes_train=%d "
             "locomotion=%s", args.emissive, os.environ["NETT_CHAMBER_VARIANT"],
             args.chamber_style, args.brain_id_offset, args.episodes_train,
             args.locomotion)

    t0 = time.time()
    NETT(cfg).run(output_path=str(out_root),
                  devices=args.devices if args.devices else [args.device],
                  verbose=True)
    train_s = time.time() - t0

    run_dir = out_root / cfg["name"]
    analysis = Path(analyze(run_dir))

    result = {
        "emissive_intensity": args.emissive,
        "chamber_variant": os.environ["NETT_CHAMBER_VARIANT"],
        "chamber_style": args.chamber_style,
        "brain_id_offset": args.brain_id_offset,
        "episodes_train": args.episodes_train,
        "locomotion": args.locomotion,
        "devices": args.devices if args.devices else [args.device],
        "resolution": args.resolution or "canonical(256)",
        "auto_envs": bool(args.auto_envs),
        "wall_clock_s": {"run": round(train_s, 1),
                         "run_plus_analysis": round(time.time() - t0, 1)},
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis),
        "scores": score(analysis),
    }
    # Gate: an under-sized eval budget silently truncates the last episode of each
    # env, unevenly across co-hosted brains, which FABRICATES between-brain
    # differences. A run that fails this must be discarded, not scored.
    try:
        checks = verify_run(run_dir)
        # verify_run returns [(path, result)] and never raises -- an empty list
        # means it could not check anything, which is NOT a pass.
        #
        # THE GATE IS TRUNCATION (``res.ok``), and only that. A short episode
        # drops steps unevenly across co-hosted brains, which fabricates
        # between-brain differences in the preference metric.
        result["episode_integrity_ok"] = bool(checks) and all(
            res.ok for _, res in checks
        )
        # NOT a gate: 260 test episodes do not divide evenly over 16 envs, so 4
        # envs always run one extra. That is a fixed property of the configured
        # eval budget -- every run of this recipe has it, including the published
        # ones -- so gating on it would discard every run rather than discriminate
        # between them. Recorded as a comparability caveat instead: correct_pct is
        # a ratio of sums over pooled steps, so the extra episode shifts a brain's
        # weight by at most ~1/16.
        result["episodes_per_env_even"] = all(
            len(set(res.episodes_per_env.values())) <= 1 for _, res in checks
        )
        result["episode_integrity_detail"] = {
            path.name: res.summary() for path, res in checks
        }
    except Exception as exc:  # noqa: BLE001 - the gate must not lose the result file
        result["episode_integrity_ok"] = None
        result["episode_integrity_error"] = f"{type(exc).__name__}: {exc}"
    if result.get("episode_integrity_ok") is not True:
        log.warning("EPISODE INTEGRITY NOT CLEAN — truncated episodes found. "
                    "Discard this run rather than scoring it (see "
                    "episode_integrity_detail in the result file)")

    res_path = out_root / f"{cfg['name']}_result.json"
    with res_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    log.info("RESULT %s", json.dumps(result["scores"]))
    log.info("wrote %s", res_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
