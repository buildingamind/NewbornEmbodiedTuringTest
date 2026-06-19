"""Train a Compact ViT encoder with PPO on NETT object-binding tasks.

Model: CompactViT (~313 K encoder params, ~338 K total with PPO heads)
Architecture: ViT-style (Dosovitskiy et al., 2021)
    patch_size=8, embed_dim=128, depth=2, num_heads=4, mlp_ratio=2.0
    64×64 input → 64 patches + CLS → 2 Transformer blocks → features_dim=128

Target: >=60% preference on 1color, 2color, 2shape&color test conditions
        after 4000 training episodes with PPO.

Usage (PYTHONPATH=src_isaac):

    python src_isaac/examples/train_compact_vit.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

from _train_common import assert_target_preferences

OUTPUT = Path("~/nett_compact_vit_out").expanduser()

CONFIG: dict = {
    "name": f"compact_vit_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": "/home/zlaborde/code/isaac/videos/binding/DesignSheet_Binding.csv",
        "media_root": "/home/zlaborde/code/isaac/videos/binding/videos",
        "conditions": ["Object1"],
        "headless": True,
        "input_resolution": 128,
        "camera_fov": 150.0,
        "reward_types": ["closeness", "completeness"],
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "compact_vit",
        "encoder_cfg": {
            "trainable": True,
            "features_dim": 128,
            "patch_size": 16,
            "embed_dim": 128,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
        },
        "algorithm_cfg": {
            "rollouts": 2000,         # 10 envs (2000//200=10); buffer=2000×obs (manageable)
            "mini_batches": 4,        # 4 mini-batches of 500 each; proven stable config
            "learning_rate": 5e-5,    # ViTs benefit from lower LR; conservative for stability
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
            "tags": ["compact_vit", "ppo", "4000ep"],
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
    log = logging.getLogger("nett.compact_vit")

    log.info("training compact_vit: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    run_dir = OUTPUT / CONFIG["name"]
    log.info("analyzing: %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    assert_target_preferences(out)
    log.info("  -> %s", out)
