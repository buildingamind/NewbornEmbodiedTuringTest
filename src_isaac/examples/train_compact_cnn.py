"""Train a Compact CNN encoder with PPO on NETT object-binding tasks.

Model: CompactCNN (~187 K encoder params, ~212 K total with PPO heads)
Architecture: 3-layer CNN with adaptive 4×4 spatial pool → Linear(1024→128)

Target: >=60% preference on 1color, 2color, 2shape&color test conditions
        after 4000 training episodes with PPO.

Usage (PYTHONPATH=src_isaac):

    python src_isaac/examples/train_compact_cnn.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

from _train_common import assert_target_preferences

OUTPUT = Path("~/nett_compact_cnn_out").expanduser()

CONFIG: dict = {
    "name": f"compact_cnn_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": "/home/zlaborde/code/isaac/videos/binding/DesignSheet_Binding.csv",
        "media_root": "/home/zlaborde/code/isaac/videos/binding/videos",
        "conditions": ["Object1"],
        "headless": True,
        "binocular_vision": False,
        "input_resolution": 128,
        "camera_fov": 150.0,
        "reward_types": ["closeness", "completeness"],
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "compact_cnn",
        "encoder_cfg": {
            "trainable": True,
            "features_dim": 256,      # 2x larger than 128 — more representational capacity
        },
        "algorithm_cfg": {
            "rollouts": 2000,         # 10 envs (2000//200=10); buffer=2000×obs (manageable)
            "mini_batches": 4,        # 4 mini-batches of 500 each; proven stable config
            "learning_rate": 1e-4,    # proven stable — lr=2e-4 diverges at chunk 3 (step ~9600)
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
            "entropy_loss_scale": 0.3,  # raised from 0.1 — both prior 0.1 runs (CNN, ViT, 3DCNN) collapsed to exact 50% side-bias
        },
        "checkpoint_freq": 10_000_000,  # effectively disabled — single continuous training run
        "model": {
            "value_bound": None,
            "shared_encoder": True,
            "hidden_sizes": [64, 64],
        },
        
        "wandb": {
            "mode": "online",
            "project": "nett-compact-models",
            "tags": ["compact_cnn", "ppo", "4000ep"],
        },
    },
    "num_brains": 1,
    "episodes": {"train": 5000, "test": 4},
    "steps_per_episode": 200,
    "eval_freq": 10_000_000,
    "task_memory": 0.1,
    "max_parallel_envs": 52,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.compact_cnn")

    log.info("training compact_cnn: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    run_dir = OUTPUT / CONFIG["name"]
    log.info("analyzing: %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    assert_target_preferences(out)
    log.info("  -> %s", out)
