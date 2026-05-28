"""Run the 1000-episode end-to-end training, then run analysis on it.

Usage (PYTHONPATH=src_isaac):

    /home/zach/nett_private/bin/python src_isaac/examples/run_e2e_and_analyze.py

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
from nett_skrl.analysis import analyze

OUTPUT = Path("/tmp/nett_e2e_out")

# Mirrors the config that successfully completed a full run on 2026-05-24
# (200,000 timesteps, ~88 min, 1,997 wandb scalar flushes, 15 keys).
#
# Single brain, monocular vision, no mid-training eval (eval would spawn
# a second Isaac Sim concurrently with the train Isaac Sim and OOM a
# 32 GB box — keep eval_freq above total step count so only the post-train
# test phase runs).
CONFIG: dict = {
    "name": f"e2e_1000ep_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": "/home/zach/Code/NewbornEmbodiedTuringTest_Private/isaac_lab/assets/design_sheets/binding.csv",
        "media_root": "/home/zach/Code/NewbornEmbodiedTuringTest_Private/isaac_lab/assets/videos",
        "conditions": ["Object1"],
        "headless": True,
        "binocular_vision": False,
        "input_resolution": 64,
        "reward_types": ["closeness"],
        # Recording: per camera, per phase, with the legacy NETT selector
        # format. Selector forms: slice string ("0:5:1" = episodes 0..4
        # step 1), integer N (single episode index N), list of indices,
        # or null (= every episode in that phase). Omit a phase to skip it.
        "recording": {
            "fps": 24,
            "egocentric": {"train": [0], "test": [0]},
            "chamber":    {"train": [0], "test": [0]},
        },
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "small",
        "batch_size": 500,
        "buffer_size": 4000,
        "learning_rate": 1.0e-5,
        "train_encoder": True,
        "wandb": {
            "mode": "online",
            "project": "nett-skrl",
        },
    },
    "num_brains": 1,
    "episodes": {"train": 1000, "test": 1},
    "steps_per_episode": 200,
    "eval_freq": 25000000,   # past total (200k) → no mid-train eval; avoids 2-sim OOM
    "task_memory": 4,
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
