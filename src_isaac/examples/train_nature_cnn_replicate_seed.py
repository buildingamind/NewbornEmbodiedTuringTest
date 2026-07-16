"""Seed-variant of train_nature_cnn_replicate.py.

Same config as the validated replication run, but reads NETT_SEED_OFFSET from the
environment to perturb the seed (via brain_id_offset). Used to characterize the
color-condition distribution across seeds (offset 0 = the original run's seed).
"""
from __future__ import annotations
from _paths import BINDING_DESIGN_SHEET, BINDING_MEDIA_ROOT

import logging
import os
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

from _train_common import assert_target_preferences

OFFSET = int(os.environ.get("NETT_SEED_OFFSET", "1"))
OUTPUT = Path("~/nett_nature_cnn_replicate_out").expanduser()

CONFIG: dict = {
    "name": f"nature_cnn_replicate_seed{OFFSET}_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": BINDING_DESIGN_SHEET,
        "media_root": BINDING_MEDIA_ROOT,
        "conditions": ["Object1"],
        "headless": True,
        "input_resolution": 64,
        "camera_fov": 150.0,
        "reward_types": ["closeness"],
        "enable_neck_flexion": False,
        "enable_lateral_bending": False,
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "nature_cnn",
        "encoder_cfg": {"trainable": True, "features_dim": 512},
        "algorithm_cfg": {
            "rollouts": 8000,
            "mini_batches": 16,
            "learning_rate": 3e-4,
            "learning_epochs": 10,
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
            "entropy_loss_scale": 0.01,
            "kl_threshold": 0.5,
            "value_clip": 0,
        },
        "model": {
            "value_bound": None,
            "shared_encoder": True,
            "hidden_sizes": [],
            "clip_actions": False,
        },
        "wandb": {
            "mode": "online",
            "project": "nett-compact-models",
            "tags": [
                "nature_cnn", "ppo", "replicate-baseline", f"seed-offset={OFFSET}",
                "ent=0.01", "lr=3e-4", "rollouts=8000", "steps=500", "2000ep",
                "closeness-only", "fov=150", "2d-action", "seed-variance-study",
            ],
        },
    },
    "num_brains": 1,
    "brain_id_offset": OFFSET,
    "episodes": {"train": 2000, "test": 20},
    "steps_per_episode": 500,
    "eval_freq": 10_000_000,
    "task_memory": "auto",
    "max_parallel_envs": 32,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.nature_cnn_replicate_seed")
    log.info("training seed-offset=%d: output=%s", OFFSET, OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")
    run_dir = OUTPUT / CONFIG["name"]
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    try:
        assert_target_preferences(out)
    except Exception as exc:
        log.warning("target preference check: %s", exc)
    log.info("  -> %s", out)
