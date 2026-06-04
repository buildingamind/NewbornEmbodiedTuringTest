"""Train a SimCLR-CLTT encoder with PPO + CLTT temporal contrastive reward.

Model: SimCLRCLTT (~204 K encoder params, ~229 K total with PPO heads)
Architecture:
    Backbone: SimCLR-style CNN (Chen et al., 2020) with BatchNorm
        Conv(3→32→BN→ReLU) → Conv(32→32→BN→ReLU) → Conv(32→64→BN→ReLU)
        → AdaptiveAvgPool(4,4) → Flatten → Linear(1024→128)
    Projector: Linear(128→64) → ReLU → Linear(64→64)  [auxiliary, not in RL path]

Reward: CLTT temporal contrastive reward (weight=0.05)
    Treats (obs_t, obs_{t+1}) as positive pairs in NT-Xent objective.
    Only the projection head is updated by CLTT; the backbone is updated
    jointly by the PPO actor-critic gradient.

Target: ≥60% preference on 1color, 2color, 2shape&color test conditions
        after 4000 training episodes with PPO + CLTT.

Usage (PYTHONPATH=src_isaac):

    /home/zach/nett_private/bin/python src_isaac/examples/train_simclr_cltt.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

OUTPUT = Path("~/nett_simclr_cltt_out").expanduser()

CONFIG: dict = {
    "name": f"simclr_cltt_{datetime.now():%Y%m%d_%H%M%S}",
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
        "encoder": "simclr_cltt",
        "encoder_cfg": {
            "trainable": True,
            "features_dim": 128,
        },
        # CLTT temporal contrastive auxiliary reward (small weight so it acts as
        # a regulariser rather than overriding the closeness extrinsic reward)
        "reward": "CLTT",
        "reward_cfg": {
            "weight": 0.05,
            "temperature": 0.1,
            "proj_lr": 1e-3,
            "beta": 0.1,
        },
        "algorithm_cfg": {
            "rollouts": 4000,
            "mini_batches": 4,        # 50 per mini-batch (stable)
            "learning_rate": 1e-4,
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
        },
        "checkpoint_freq": 200000,
        "model": {
            "value_bound": None,
            "shared_encoder": True,
            "hidden_sizes": [64, 64],
        },
        "wandb": {
            "mode": "online",
            "project": "nett-compact-models",
            "tags": ["simclr_cltt", "ppo", "4000ep", "contrastive"],
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
    log = logging.getLogger("nett.simclr_cltt")

    log.info("training simclr_cltt: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    for run_dir in find_run_dirs(OUTPUT):
        log.info("analyzing: %s", run_dir)
        out = analyze(run_dir)
        log.info("  → %s", out)
        log_analysis_to_wandb(run_dir, out)
