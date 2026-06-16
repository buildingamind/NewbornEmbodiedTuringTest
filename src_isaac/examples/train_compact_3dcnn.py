"""Train a Compact 3D CNN encoder with PPO on NETT object-binding tasks.

Model: Compact3DCNN (~188 K encoder params, ~213 K total with PPO heads)
Architecture: Ji et al. (2013) 3D CNN adapted for compact RL
    Input: 2-frame FrameStack → (6, 64, 64) CHW
    Conv3d(3→32, kernel=(2,3,3), stride=(1,2,2)) collapses temporal dim
    → Conv2d(32→64) → Conv2d(64→64) → AdaptiveAvgPool(4,4) → Linear(→128)

Body wrappers: framestack (n_stack=2) stacks consecutive frames before
    ChannelsFirst conversion. The 3DCNN encoder reshapes (B,6,H,W)→(B,3,2,H,W).

Target: >=60% preference on 1color, 2color, 2shape&color test conditions
        after 4000 training episodes with PPO.

Usage (PYTHONPATH=src_isaac):

    python src_isaac/examples/train_compact_3dcnn.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

from _train_common import assert_target_preferences

OUTPUT = Path("~/nett_compact_3dcnn_out").expanduser()

# FrameStack is configured through the wrapper registry (n_stack=2 is the default).
CONFIG: dict = {
    "name": f"compact_3dcnn_{datetime.now():%Y%m%d_%H%M%S}",
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
    "body": {
        "wrappers": ["framestack"],   # stacks last 2 frames on channel axis (6-ch input)
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
            "rollouts": 1000,         # 5 envs (1000//200=5); 2-frame obs doubles buffer - keep safe
            "mini_batches": 4,        # 4 mini-batches of 500 each; proven stable config
            "learning_rate": 1e-4,
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
            "entropy_loss_scale": 0.3,  # raised from 0.1 — both prior 0.1 runs (CNN, ViT, 3DCNN) collapsed to exact 50% side-bias
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
    "episodes": {"train": 5000, "test": 4},
    "steps_per_episode": 200,
    "eval_freq": 10_000_000,
    "task_memory": 0.1,
    "max_parallel_envs": 52,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.compact_3dcnn")

    log.info("training compact_3dcnn: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    run_dir = OUTPUT / CONFIG["name"]
    log.info("analyzing: %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    assert_target_preferences(out)
    log.info("  -> %s", out)
