"""Train a Compact ViViT encoder with PPO on NETT object-binding tasks.

Model: CompactViViT (~252 K encoder params, ~272 K total with PPO heads)
Architecture: ViViT Model 1 with factored position embeddings
    (Arnab et al., 2021 "ViViT: A Video Vision Transformer")
    Input: 2-frame FrameStack → (6, 64, 64) CHW
    Tubelet embed: Conv3d(3, 96, (1,8,8), (1,8,8)) → 128 spatiotemporal tokens
    + CLS token → 3 Transformer blocks (dim=96, heads=3, mlp_ratio=2.0)

Body wrappers: framestack (n_stack=2) stacks consecutive frames before
    ChannelsFirst conversion. CompactViViT reshapes (B,6,H,W)→(B,3,2,H,W).

Target: >=60% preference on 1color, 2color, 2shape&color test conditions
        after 4000 training episodes with PPO.

Usage (PYTHONPATH=src_isaac):

    python src_isaac/examples/train_compact_vivit.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

from _train_common import assert_target_preferences

OUTPUT = Path("~/nett_compact_vivit_out").expanduser()

CONFIG: dict = {
    "name": f"compact_vivit_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": "/home/zlaborde/code/isaac/videos/binding/DesignSheet_Binding.csv",
        "media_root": "/home/zlaborde/code/isaac/videos/binding/videos",
        "conditions": ["Object1"],
        "headless": True,
        "input_resolution": 128,
        "camera_fov": 150.0,
        "reward_types": ["closeness", "completeness"],
    },
    "body": {
        "wrappers": ["framestack"],   # stacks last 2 frames -> 6-channel input
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "compact_vivit",
        "encoder_cfg": {
            "trainable": True,
            "features_dim": 96,
            "patch_size": 16,
            "embed_dim": 96,
            "depth": 3,
            "num_heads": 3,
            "mlp_ratio": 2.0,
            "num_frames": 2,
        },
        "algorithm_cfg": {
            "rollouts": 1000,         # 5 envs (1000//200=5); 2-frame obs doubles buffer - keep safe
            "mini_batches": 4,        # 4 mini-batches of 500 each; proven stable config
            "learning_rate": 5e-5,    # ViViT: conservative LR
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
            "tags": ["compact_vivit", "ppo", "4000ep", "2frame"],
        },
    },
    "num_brains": 1,
    "episodes": {"train": 5000, "test": 4},
    "steps_per_episode": 200,
    "eval_freq": 10_000_000,
    "task_memory": 0.1,
    "max_parallel_envs": 49,  # 7x7: a square tile grid. 52 tiles 8x7 -> distorted fisheye (#488)
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.compact_vivit")

    log.info("training compact_vivit: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    run_dir = OUTPUT / CONFIG["name"]
    log.info("analyzing: %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    assert_target_preferences(out)
    log.info("  -> %s", out)
