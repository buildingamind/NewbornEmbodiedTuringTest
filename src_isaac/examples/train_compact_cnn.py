"""Train a Compact CNN encoder with PPO on NETT object-binding tasks.

Model: CompactCNN (~187 K encoder params, ~212 K total with PPO heads)
Architecture: 3-layer CNN with adaptive 4×4 spatial pool → Linear(1024→128)

Target: ≥60% preference on 1color, 2color, 2shape&color test conditions
        after 4000 training episodes with PPO.

Usage (PYTHONPATH=src_isaac):

    /home/zach/nett_private/bin/python src_isaac/examples/train_compact_cnn.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

OUTPUT = Path("~/nett_compact_cnn_out").expanduser()

CONFIG: dict = {
    "name": f"compact_cnn_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": "/home/zach/Code/NewbornEmbodiedTuringTest_Private/isaac_lab/assets/design_sheets/binding.csv",
        "media_root": "/home/zach/Code/NewbornEmbodiedTuringTest_Private/isaac_lab/assets/videos",
        "conditions": ["Object1"],
        "headless": True,
        "binocular_vision": False,
        "input_resolution": 64,
        "camera_fov": 60.0,
        "reward_types": ["closeness"],
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "compact_cnn",
        "encoder_cfg": {
            "trainable": True,
            "features_dim": 128,
        },
        "algorithm_cfg": {
            "rollouts": 2000,         # memory recommendation: 10 envs, 2x more PPO updates
            "mini_batches": 4,        # 200 scaled rollouts / 4 = 50 per mini-batch (stable)
            "learning_rate": 2e-4,    # faster than 1e-4 but below diverging 3e-4
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
        },
        "checkpoint_freq": 200000,    # save every 200K env-steps for recovery
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
    "episodes": {"train": 4000, "test": 1},
    "steps_per_episode": 200,
    "eval_freq": 10_000_000,
    "task_memory": 0.1,
    "max_parallel_envs": 52,
}


def find_run_dirs(output_root: Path) -> list[Path]:
    if not output_root.exists():
        return []
    runs = []
    for child in sorted(output_root.iterdir()):
        if not (child.is_dir() and (child / "config.yaml").exists()):
            continue
        if any(
            (sub / "logs").exists() or (sub / "wandb_runs").exists()
            for sub in child.iterdir()
            if sub.is_dir()
        ):
            runs.append(child)
    return runs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.compact_cnn")

    log.info("training compact_cnn: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    for run_dir in find_run_dirs(OUTPUT):
        log.info("analyzing: %s", run_dir)
        out = analyze(run_dir)
        log.info("  → %s", out)
        log_analysis_to_wandb(run_dir, out)
