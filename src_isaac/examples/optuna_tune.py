"""Optuna HP-tuning engine for the Isaac/skrl NETT stack.

See ``/home/zlaborde/code/isaac/optuna_tune/SPEC.md`` for the full spec. The
engine searches RL + encoder/policy hyperparameters for the **best AND most
stable** learning trajectories on the binding/Object1/closeness task.

API (Tester targets these):
    sample_config(trial, encoder) -> dict
    run_training(config, gpu, out_dir) -> Path           # brain_1 tfevents path
    read_trajectory(tb_path) -> dict
    score_trajectory(traj) -> tuple[float, list[float]]   # PURE, no Isaac
    objective(trial) -> float                             # never raises
    main()                                                # CLI entry point

``score_trajectory`` and ``read_trajectory`` (parsing helpers) are import-safe
without Isaac so they can be unit-tested directly. ``run_training`` and
``objective`` import NETT lazily.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

# Optuna is import-safe without Isaac.
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

logger = logging.getLogger("optuna_tune")

# --------------------------------------------------------------------------- #
# Constants / fixed task definition (SPEC §TRIAL design)
# --------------------------------------------------------------------------- #

OPTUNA_DIR = Path("/home/zlaborde/code/isaac/optuna_tune")
DESIGN_SHEET = "/home/zlaborde/code/isaac/videos/binding/DesignSheet_Binding.csv"
MEDIA_ROOT = "/home/zlaborde/code/isaac/videos/binding/videos"
CONDITION = "Object1"

# Trial seed (fixed for reproducible stability measurement, per SPEC).
TRIAL_SEED = 12345

# Objective knobs (SPEC §OBJECTIVE; Critic may tune the 0.25/0.5 weights).
IMPROVEMENT_WEIGHT = 0.5
INSTABILITY_WEIGHT = 0.5
HARD_FAIL_SCORE = -1.0
MIN_UPDATES = 8
SCORE_CLIP = (-1.0, 2.0)

# Pruning: report a running-mean intermediate every this-many updates.
REPORT_EVERY = 5

# Trial length target: TOTAL TIMESTEPS a trial trains for (rollouts-independent).
# 400k (~50 updates at rollouts=8000) reveals the learning slope; the diagnostic
# known-good run showed clear learning (EV>0.7, reward +50%) well within this.
DEFAULT_TARGET_TIMESTEPS = 400_000
SMOKE_TARGET_TIMESTEPS = 40_000

REWARD_TAG = "Reward / Instantaneous reward (mean)"
LOSS_TAGS = (
    "Loss / Policy loss",
    "Loss / Value loss",
    "Loss / KL divergence",
)


# --------------------------------------------------------------------------- #
# 1. Search space -> config  (SPEC §SEARCH SPACE)
# --------------------------------------------------------------------------- #


def sample_config(trial: optuna.Trial, encoder: str, *, target_timesteps: int = DEFAULT_TARGET_TIMESTEPS) -> dict:
    """Sample the HP search space and build a valid NETT config dict.

    ``encoder`` is ``"nature_cnn"`` (Phase 1) or ``"guess_what_moves"`` (Phase 2).
    The GWM hook adds the framestack wrapper, ``num_frames=2``, and a ``conv_dim``.
    ``target_timesteps`` controls trial length in env-steps (small for smoke runs).
    """
    if encoder not in ("nature_cnn", "guess_what_moves"):
        raise ValueError(f"unknown encoder: {encoder!r}")

    # --- RL algorithm search space (shared across encoders) ----------------
    # Iter-3 (CORRECTED regime): iters 1-2 tuned in a BROKEN regime — rollouts=800
    # gave ~10x noisier gradients so the value function never fit (EV stuck
    # negative -> entropy collapse -> chance), and the ranges (fit to that noise)
    # EXCLUDED the known-good anchor (rollouts=8000, ent=0.01, lr=3e-4, steps=500,
    # ~1M timesteps) that demonstrably trains to >90% rest. This space CONTAINS
    # that anchor and searches around it. The Tester guards that the anchor is
    # representable. Trial length is set by TOTAL TIMESTEPS (not update count).
    rollouts = trial.suggest_categorical("rollouts", [4000, 8000])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)   # incl 3e-4
    mini_batches = trial.suggest_categorical("mini_batches", [8, 16, 32])         # batch = rollouts/mb
    learning_epochs = trial.suggest_int("learning_epochs", 5, 10)                 # known-good 10
    entropy_loss_scale = trial.suggest_float("entropy_loss_scale", 5e-3, 3e-2, log=True)  # incl 0.01
    value_loss_scale = trial.suggest_float("value_loss_scale", 0.25, 0.75)
    grad_norm_clip = trial.suggest_float("grad_norm_clip", 0.3, 1.0)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.97)
    discount_factor = trial.suggest_float("discount_factor", 0.95, 0.999)
    ratio_clip = trial.suggest_float("ratio_clip", 0.1, 0.3)
    kl_threshold = trial.suggest_categorical("kl_threshold", [0.05, 0.2, 0.5])    # loose like known-good

    # --- Encoder / policy architecture -------------------------------------
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    features_dim = trial.suggest_categorical("features_dim", [128, 256, 512])

    # Fixed task definition (SPEC §TRIAL design). steps_per_episode=500 matches
    # the known-good config (100 was too short for the chick to navigate/sustain).
    steps_per_episode = 500

    encoder_cfg: dict[str, Any] = {
        "trainable": True,
        "features_dim": features_dim,
    }
    body: dict[str, Any] = {"wrappers": []}
    input_resolution = 64

    if encoder == "guess_what_moves":
        # Phase-2 hook: dual-stream needs framestacked frames + a conv width.
        conv_dim = trial.suggest_categorical("conv_dim", [64, 80, 96, 128])
        encoder_cfg["num_frames"] = 2
        encoder_cfg["conv_dim"] = conv_dim
        body["wrappers"] = ["framestack"]

    # Trial length is set by TOTAL TIMESTEPS, independent of rollouts (the iter-1
    # confound was that length scaled with rollouts). total_timesteps =
    # episodes.train * steps_per_episode; actual PPO updates = total_timesteps /
    # rollouts; envs_per_brain = rollouts // steps_per_episode (set internally).
    episodes_train = max(1, round(target_timesteps / steps_per_episode))

    config: dict[str, Any] = {
        "name": f"trial_{trial.number}_{datetime.now():%Y%m%d_%H%M%S}",
        "environment": {
            "design_sheet": DESIGN_SHEET,
            "media_root": MEDIA_ROOT,
            "conditions": [CONDITION],
            "headless": True,
            "input_resolution": input_resolution,
            "camera_fov": 150.0,
            "reward_types": ["closeness"],
            "enable_neck_flexion": False,
            "enable_lateral_bending": False,
        },
        "body": body,
        "brain": {
            "algorithm": "PPO",
            "encoder": encoder,
            "encoder_cfg": encoder_cfg,
            "algorithm_cfg": {
                "learning_rate": learning_rate,
                "rollouts": rollouts,
                "mini_batches": mini_batches,
                "learning_epochs": learning_epochs,
                "entropy_loss_scale": entropy_loss_scale,
                "value_loss_scale": value_loss_scale,
                "grad_norm_clip": grad_norm_clip,
                "lambda": gae_lambda,
                "discount_factor": discount_factor,
                "ratio_clip": ratio_clip,
                "kl_threshold": kl_threshold,
            },
            "model": {
                "shared_encoder": True,
                "value_bound": None,
                "hidden_sizes": [hidden_size, hidden_size],
                "clip_actions": False,
            },
            "wandb": {"mode": "disabled"},
        },
        "num_brains": 1,
        "episodes": {"train": episodes_train, "test": 2},
        "steps_per_episode": steps_per_episode,
        "eval_freq": 10_000_000,
        "task_memory": "auto",
    }
    # Stash derived bookkeeping for logging / pruning (not part of NETT config).
    trial.set_user_attr("target_timesteps", target_timesteps)
    trial.set_user_attr("episodes_train", episodes_train)
    return config


# --------------------------------------------------------------------------- #
# 2. Run one training trial -> tfevents path
# --------------------------------------------------------------------------- #


def run_training(config: dict, gpu: int, out_dir: Path) -> Path:
    """Run one training trial pinned to ``gpu`` and return the brain_1 tfevents.

    Each trial runs NETT in its OWN child process (this script re-invoked in
    ``--worker`` mode) with ``CUDA_VISIBLE_DEVICES=<gpu>`` set in that child's
    env only. This is the only way to keep concurrent trials truly isolated:

      * ``CUDA_VISIBLE_DEVICES`` hides every other GPU from Isaac/PhysX, which
        otherwise creates contexts (and a multi-GB allocation) on ALL visible
        GPUs — including GPU0's pre-existing process — even when ``devices=[N]``
        pins the primary compute. Inside the child only one GPU is visible, so
        it re-indexes to ``cuda:0`` and we pass ``devices=[0]``.
      * Setting it per-child (a fresh env dict per ``Popen``) is RACE-FREE,
        unlike mutating the shared parent ``os.environ`` from N threads.
      * A dedicated child also guarantees full Isaac/Kit teardown + VRAM
        release when the trial ends (no orphan contexts in the tuner process).
    """
    import subprocess
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config))

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--worker-config", str(config_path),
        "--worker-out", str(out_dir),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-25:])
        raise RuntimeError(
            f"training worker (gpu {gpu}) exited {proc.returncode}:\n{tail}"
        )

    return _find_tfevents(out_dir, config["name"])


def _worker_main(config_path: str, out_dir: str) -> None:
    """Child-process entry: run NETT on the single visible GPU (cuda:0)."""
    from nett_skrl import NETT  # lazy: needs Isaac

    with open(config_path) as f:
        config = json.load(f)
    NETT(config).run(output_path=str(out_dir), devices=[0], verbose=False)


def _find_tfevents(out_dir: Path, name: str) -> Path:
    """Locate the *training* tfevents file for ``name``/brain_1.

    Several tfevents files can live in the same ``brain_1`` dir: the dry-run
    memory probes and the test phase each write one in addition to the real
    training run. We pick the file with the most ``REWARD_TAG`` points (the
    training run logs one per PPO update, far more than the probe/test files),
    falling back to file size if none have reward scalars.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    brain_dir = out_dir / name / CONDITION / "wandb_runs" / "brain_1"
    events = sorted(brain_dir.glob("events.out.tfevents.*"))
    if not events:
        # Fall back to any brain dir / any condition (robustness).
        events = sorted((out_dir / name).glob("*/wandb_runs/brain_*/events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"no tfevents under {out_dir / name}")
    if len(events) == 1:
        return events[0]

    def reward_points(p: Path) -> int:
        try:
            ea = EventAccumulator(str(p), size_guidance={"scalars": 1_000_000})
            ea.Reload()
            if REWARD_TAG not in set(ea.Tags().get("scalars", [])):
                return 0
            return len(ea.Scalars(REWARD_TAG))
        except Exception:
            return 0

    scored = [(reward_points(p), p.stat().st_size, p) for p in events]
    best = max(scored, key=lambda t: (t[0], t[1]))
    return best[2]


# --------------------------------------------------------------------------- #
# 3. Parse a tfevents file -> trajectory dict
# --------------------------------------------------------------------------- #


def read_trajectory(tb_path: str | Path) -> dict:
    """Read reward + loss series from a tfevents file.

    Returns a dict with:
        reward:   list[float]  (REWARD_TAG series, step-ordered)
        losses:   dict[tag -> list[float]]
        has_nan:  bool         (NaN/Inf anywhere in reward or losses)
        crashed:  bool         (file unreadable / empty reward series)
        n:        int          (len(reward))
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_path = str(tb_path)
    result: dict[str, Any] = {
        "reward": [],
        "losses": {},
        "has_nan": False,
        "crashed": False,
        "n": 0,
    }
    try:
        ea = EventAccumulator(tb_path, size_guidance={"scalars": 1_000_000})
        ea.Reload()
        tags = set(ea.Tags().get("scalars", []))
    except Exception as exc:  # corrupt / missing file
        logger.warning("read_trajectory: failed to load %s: %s", tb_path, exc)
        result["crashed"] = True
        return result

    def series(tag: str) -> list[float]:
        if tag not in tags:
            return []
        return [float(s.value) for s in sorted(ea.Scalars(tag), key=lambda s: s.step)]

    reward = series(REWARD_TAG)
    losses = {tag: series(tag) for tag in LOSS_TAGS if tag in tags}

    result["reward"] = reward
    result["losses"] = losses
    result["n"] = len(reward)

    def _bad(xs: list[float]) -> bool:
        return any(math.isnan(x) or math.isinf(x) for x in xs)

    result["has_nan"] = _bad(reward) or any(_bad(v) for v in losses.values())
    result["crashed"] = len(reward) == 0
    return result


# --------------------------------------------------------------------------- #
# 4. Score a trajectory  (PURE — no Isaac, unit-testable)  (SPEC §OBJECTIVE)
# --------------------------------------------------------------------------- #


def _running_mean(xs: list[float]) -> list[float]:
    out, acc = [], 0.0
    for i, x in enumerate(xs, 1):
        acc += x
        out.append(acc / i)
    return out


def _max_drawdown(xs: list[float]) -> float:
    """Largest peak->trough drop (>= 0)."""
    if not xs:
        return 0.0
    peak = xs[0]
    worst = 0.0
    for x in xs:
        if x > peak:
            peak = x
        worst = max(worst, peak - x)
    return worst


def _std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def score_trajectory(traj: dict) -> tuple[float, list[float]]:
    """The PURE objective (SPEC §OBJECTIVE). Returns ``(score, intermediates)``.

    score = clip( P + 0.5*max(I,0) - 0.5*U , [-1, 2] )
      P = mean(reward over last quarter)
      I = late_mean - early_mean (improvement)
      U = std(diff(reward_late)) + 0.5*max_drawdown(reward_late)
    Hard fail (-1.0) on: NaN/Inf, crash, or T < MIN_UPDATES.
    ``intermediates`` = running-mean reward, for trial.report() pruning.
    No exceptions raised; depends only on the plain ``traj`` dict.
    """
    reward = list(traj.get("reward", []))
    intermediates = _running_mean(reward)

    if traj.get("crashed") or traj.get("has_nan"):
        return HARD_FAIL_SCORE, intermediates
    T = len(reward)
    if T < MIN_UPDATES:
        return HARD_FAIL_SCORE, intermediates
    if any(math.isnan(x) or math.isinf(x) for x in reward):
        return HARD_FAIL_SCORE, intermediates

    q = max(1, T // 4)
    early = reward[:q]
    late = reward[-q:]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)

    P = late_mean
    I = late_mean - early_mean

    diffs = [late[i + 1] - late[i] for i in range(len(late) - 1)]
    jitter = _std(diffs)
    drawdown = _max_drawdown(late)
    U = jitter + 0.5 * drawdown

    score = P + IMPROVEMENT_WEIGHT * max(I, 0.0) - INSTABILITY_WEIGHT * U
    score = max(SCORE_CLIP[0], min(SCORE_CLIP[1], score))
    return float(score), intermediates


# --------------------------------------------------------------------------- #
# 5. Objective (never raises)
# --------------------------------------------------------------------------- #

# GPU assignment for the running trial (set by the worker via thread-local).
_thread_local = threading.local()


def _current_gpu(default: int = 0) -> int:
    return getattr(_thread_local, "gpu", default)


def _report_intermediates(trial: optuna.Trial, intermediates: list[float]) -> bool:
    """Report running-mean reward every REPORT_EVERY updates; return should_prune."""
    pruned = False
    for step, val in enumerate(intermediates):
        if step % REPORT_EVERY == 0 or step == len(intermediates) - 1:
            try:
                trial.report(val, step)
                if trial.should_prune():
                    pruned = True
            except Exception:  # reporting must never break the trial
                logger.debug("trial.report failed at step %d", step, exc_info=True)
    return pruned


def objective(trial: optuna.Trial) -> float:
    """sample -> run -> read -> score. NEVER raises; any failure => -1.0."""
    encoder = trial.study.user_attrs.get("encoder", "nature_cnn")
    target_timesteps = trial.study.user_attrs.get("target_timesteps", DEFAULT_TARGET_TIMESTEPS)
    study_name = trial.study.study_name
    gpu = _current_gpu()

    out_dir = OPTUNA_DIR / "runs" / study_name / f"trial_{trial.number}"

    try:
        config = sample_config(trial, encoder, target_timesteps=target_timesteps)
    except Exception:
        logger.exception("trial %d: sample_config failed", trial.number)
        return HARD_FAIL_SCORE

    logger.info(
        "trial %d on GPU %d: lr=%.2e rollouts=%s mb=%s epochs=%s ent=%.2e H=%s fdim=%s",
        trial.number, gpu,
        config["brain"]["algorithm_cfg"]["learning_rate"],
        config["brain"]["algorithm_cfg"]["rollouts"],
        config["brain"]["algorithm_cfg"]["mini_batches"],
        config["brain"]["algorithm_cfg"]["learning_epochs"],
        config["brain"]["algorithm_cfg"]["entropy_loss_scale"],
        config["brain"]["model"]["hidden_sizes"][0],
        config["brain"]["encoder_cfg"]["features_dim"],
    )

    try:
        tb_path = run_training(config, gpu, out_dir)
    except Exception:
        logger.exception("trial %d: training crashed", trial.number)
        trial.set_user_attr("failure", "training_crash")
        return HARD_FAIL_SCORE

    try:
        traj = read_trajectory(tb_path)
    except Exception:
        logger.exception("trial %d: read_trajectory failed", trial.number)
        trial.set_user_attr("failure", "read_failure")
        return HARD_FAIL_SCORE

    score, intermediates = score_trajectory(traj)

    # Pruning (best-effort; can't actually short-circuit a finished run, but
    # reporting populates MedianPruner state for the next trials and lets a
    # genuinely-bad run be marked pruned).
    pruned = _report_intermediates(trial, intermediates)

    # Diagnostics for the progress file.
    trial.set_user_attr("n_updates", traj.get("n", 0))
    trial.set_user_attr("has_nan", bool(traj.get("has_nan")))
    trial.set_user_attr("crashed", bool(traj.get("crashed")))
    if traj.get("reward"):
        r = traj["reward"]
        trial.set_user_attr("reward_first", round(float(r[0]), 4))
        trial.set_user_attr("reward_last", round(float(r[-1]), 4))
        trial.set_user_attr("reward_max", round(float(max(r)), 4))

    if score <= HARD_FAIL_SCORE:
        reason = (
            "crashed" if traj.get("crashed")
            else "nan/inf" if traj.get("has_nan")
            else f"too_short(n={traj.get('n')})" if traj.get("n", 0) < MIN_UPDATES
            else "low_score"
        )
        trial.set_user_attr("failure", reason)
        logger.warning("trial %d hard-failed: %s", trial.number, reason)

    if pruned:
        # Surface to Optuna's pruning bookkeeping but still return the score so
        # the run we already paid for contributes to the study.
        logger.info("trial %d flagged should_prune (score=%.4f)", trial.number, score)

    logger.info("trial %d score=%.4f (n=%d)", trial.number, score, traj.get("n", 0))
    return score


# --------------------------------------------------------------------------- #
# 6. Progress reporting
# --------------------------------------------------------------------------- #


def write_progress(study: optuna.Study, study_name: str) -> None:
    """Write best-params + study stats to ``<study>_progress.md``."""
    path = OPTUNA_DIR / f"{study_name}_progress.md"
    trials = study.get_trials(deepcopy=False)
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

    lines = [
        f"# Optuna study `{study_name}` — progress",
        "",
        f"_updated {datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"- encoder: `{study.user_attrs.get('encoder', '?')}`",
        f"- total trials: {len(trials)}  (complete: {len(completed)})",
        f"- storage: `{OPTUNA_DIR / (study_name + '.db')}`",
        "",
    ]

    best = None
    try:
        if completed:
            best = study.best_trial
    except Exception:
        best = None

    if best is not None:
        lines += [
            "## Best trial",
            "",
            f"- number: {best.number}",
            f"- value (score): {best.value:.4f}",
            f"- reward last/max: {best.user_attrs.get('reward_last')}/{best.user_attrs.get('reward_max')}",
            "",
            "### Best params",
            "",
            "| param | value |",
            "|---|---|",
        ]
        for k, v in sorted(best.params.items()):
            vs = f"{v:.3e}" if isinstance(v, float) and abs(v) < 1e-2 else v
            lines.append(f"| {k} | {vs} |")
        lines.append("")

    lines += [
        "## All trials",
        "",
        "| # | state | score | n_upd | r_last | r_max | failure |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in trials:
        score = f"{t.value:.4f}" if t.value is not None else "-"
        lines.append(
            f"| {t.number} | {t.state.name} | {score} "
            f"| {t.user_attrs.get('n_updates', '-')} "
            f"| {t.user_attrs.get('reward_last', '-')} "
            f"| {t.user_attrs.get('reward_max', '-')} "
            f"| {t.user_attrs.get('failure', '')} |"
        )
    lines.append("")

    path.write_text("\n".join(lines))
    logger.info("wrote progress -> %s", path)


# --------------------------------------------------------------------------- #
# 7. main / CLI
# --------------------------------------------------------------------------- #


def _run_with_gpu_pool(study, n_trials, n_jobs, gpus):
    """Run ``n_trials`` of ``objective`` across ``n_jobs`` threads, each thread
    holding a distinct GPU from ``gpus`` (round-robin via a queue)."""
    study_name = study.study_name
    gpu_q: queue.Queue[int] = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    progress_lock = threading.Lock()

    def run_one(_idx):
        gpu = gpu_q.get()
        _thread_local.gpu = gpu
        try:
            study.optimize(objective, n_trials=1, catch=(Exception,))
        finally:
            gpu_q.put(gpu)
            with progress_lock:
                try:
                    write_progress(study, study_name)
                except Exception:
                    logger.exception("write_progress failed")

    if n_jobs <= 1:
        for i in range(n_trials):
            run_one(i)
    else:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(gpus))) as ex:
            list(ex.map(run_one, range(n_trials)))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Optuna HP tuning for Isaac/skrl NETT")
    p.add_argument("--encoder", choices=["nature_cnn", "guess_what_moves"], default="nature_cnn")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--gpus", type=str, default="1,2,3,4,5,6,7",
                   help="comma-separated GPU ids for trial pinning")
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--target-timesteps", type=int, default=None,
                   help="total env-steps per trial (default 400k; use ~40k for smoke)")
    # Internal: per-trial training worker (one Isaac process, single visible GPU).
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-config", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-out", type=str, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s",
    )

    if args.worker:
        _worker_main(args.worker_config, args.worker_out)
        return

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        raise SystemExit("--gpus must list at least one GPU id")

    study_name = args.study_name or f"{args.encoder}_study"
    target_timesteps = args.target_timesteps or DEFAULT_TARGET_TIMESTEPS

    storage = f"sqlite:///{OPTUNA_DIR / (study_name + '.db')}"
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=0),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    # Stash run-wide config so objective() (which only gets `trial`) can read it.
    study.set_user_attr("encoder", args.encoder)
    study.set_user_attr("target_timesteps", target_timesteps)

    logger.info(
        "study=%s encoder=%s trials=%d n_jobs=%d gpus=%s target_timesteps=%d storage=%s",
        study_name, args.encoder, args.trials, args.n_jobs, gpus, target_timesteps, storage,
    )

    _run_with_gpu_pool(study, args.trials, args.n_jobs, gpus)

    write_progress(study, study_name)
    logger.info("DONE. best value=%s", getattr(study, "best_value", None))


if __name__ == "__main__":
    main()
