"""Run the 100-episode end-to-end training, then run analysis on it.

Usage (PYTHONPATH=src_isaac):

    python src_isaac/examples/run_e2e_and_analyze.py

Edit ``CONFIG`` and ``OUTPUT`` below to tweak the run. The flow:

  1. ``NETT(CONFIG).run(OUTPUT)`` runs the full skrl train + test phases.
     Scalars stream live to W&B via the per-agent mirror, recordings /
     checkpoints / CSVs sync after training, runs are marked ``finished``.
  2. ``analyze(run_dir)`` for each run directory under ``OUTPUT`` emits
     reward curves, test preferences, and a ``summary.json`` under
     ``<run_dir>/analysis/``.

The analysis step only reads on-disk artifacts so it works on a
partially-completed run too — re-run this script with the training call
commented out to refresh plots without retraining.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

OUTPUT = Path("~/nett_e2e_out").expanduser()

# Mirrors the config that successfully completed a full run on 2026-05-24
# (200,000 timesteps, ~88 min, 1,997 wandb scalar flushes, 15 keys).
#
# Single brain, monocular vision, no mid-training eval (eval would spawn
# a second Isaac Sim concurrently with the train Isaac Sim and OOM a
# 32 GB box — keep eval_freq above total step count so only the post-train
# test phase runs).
CONFIG: dict = {
    "name": f"e2e_1kep_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": "/home/zlaborde/code/isaac/videos/binding/DesignSheet_Binding.csv",
        "media_root": "/home/zlaborde/code/isaac/videos/binding/videos",
        "conditions": ["Object1"],
        "headless": True,
        "input_resolution": 64,
        "camera_fov": 60.0,
        "reward_types": ["closeness"],
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "small",
        "encoder_cfg": {"trainable": True},
        "algorithm_cfg": {
            "rollouts": 2000,
            "mini_batches": 8,
            "learning_rate": 3e-4,
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
        },
        "model": {
            "value_bound": None,
            "shared_encoder": True,
        },
        "wandb": {
            "mode": "online",
            "project": "nett-skrl",
        },
    },
    "num_brains": 1,
    "episodes": {"train": 1200, "test": 1},
    "steps_per_episode": 200,
    "eval_freq": 10_000_000,
    "task_memory": 0.1,
    "max_parallel_envs": 52,
}


def find_run_dirs(output_root: Path) -> list[Path]:
    """Every NETT run directory under ``output_root``.

    A run dir contains a ``config.yaml`` (saved by NETT for reproducibility)
    and at least one per-condition subtree (``logs/`` or ``wandb_runs/``).
    """
    if not output_root.exists():
        return []
    runs = []
    for child in sorted(output_root.iterdir()):
        if not (child.is_dir() and (child / "config.yaml").exists()):
            continue
        if any((sub / "logs").exists() or (sub / "wandb_runs").exists()
               for sub in child.iterdir() if sub.is_dir()):
            runs.append(child)
    return runs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.e2e")

    log.info("training: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    for run_dir in find_run_dirs(OUTPUT):
        log.info("analyzing: %s", run_dir)
        out = analyze(run_dir)
        log.info("  → %s", out)
        log.info("uploading analysis graphs to W&B: %s", run_dir)
        log_analysis_to_wandb(run_dir, out)
