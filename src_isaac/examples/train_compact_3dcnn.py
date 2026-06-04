"""Train a Compact 3D CNN encoder with PPO on NETT object-binding tasks.

Model: Compact3DCNN (~188 K encoder params, ~213 K total with PPO heads)
Architecture: Ji et al. (2013) 3D CNN adapted for compact RL
    Input: 2-frame FrameStack → (6, 64, 64) CHW
    Conv3d(3→32, kernel=(2,3,3), stride=(1,2,2)) collapses temporal dim
    → Conv2d(32→64) → Conv2d(64→64) → AdaptiveAvgPool(4,4) → Linear(→128)

Body wrappers: framestack (n_stack=2) stacks consecutive frames before
    ChannelsFirst conversion. The 3DCNN encoder reshapes (B,6,H,W)→(B,3,2,H,W).

Target: ≥60% preference on 1color, 2color, 2shape&color test conditions
        after 4000 training episodes with PPO.

Usage (PYTHONPATH=src_isaac):

    /home/zach/nett_private/bin/python src_isaac/examples/train_compact_3dcnn.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb
from nett_skrl.body.wrappers.framestack import FrameStack

OUTPUT = Path("~/nett_compact_3dcnn_out").expanduser()

# FrameStack is configured as a body wrapper class (n_stack=2 is the default)
CONFIG: dict = {
    "name": f"compact_3dcnn_{datetime.now():%Y%m%d_%H%M%S}",
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
    "body": {
        "wrappers": [FrameStack],   # stacks last 2 frames on channel axis (6-ch input)
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "compact_3dcnn",
        "encoder_cfg": {
            "trainable": True,
            "features_dim": 128,
            "num_frames": 2,
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
            "tags": ["compact_3dcnn", "ppo", "4000ep", "2frame"],
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
    log = logging.getLogger("nett.compact_3dcnn")

    log.info("training compact_3dcnn: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    for run_dir in find_run_dirs(OUTPUT):
        log.info("analyzing: %s", run_dir)
        out = analyze(run_dir)
        log.info("  → %s", out)
        log_analysis_to_wandb(run_dir, out)
