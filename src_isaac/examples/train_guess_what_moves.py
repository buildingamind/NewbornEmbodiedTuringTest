"""Train a GuessWhatMoves dual-pathway encoder with PPO on NETT object-binding tasks.

Model: GuessWhatMoves (~146 K encoder params, ~171 K total with PPO heads)
Architecture: Two-stream visual encoder
    (Simonyan & Zisserman, 2014; Pathak et al., 2019)

    What pathway  (appearance/content):
        2D CNN on most-recent frame:
        Conv(3→32, k=3, s=2) → Conv(32→32, k=3, s=2) → AvgPool(4,4)
        → Flatten → 512-dim appearance features

    Moves pathway (temporal/motion):
        Conv3d(3→16, k=(2,3,3), s=(1,2,2)) on both frames, collapses time
        → Conv2d(16→32, k=3, s=2) → AvgPool(4,4)
        → Flatten → 512-dim motion features

    Fusion: Concat(what, moves) → Linear(1024→128) → ReLU

Body wrappers: framestack (n_stack=2) stacks consecutive frames before
    ChannelsFirst. GuessWhatMoves splits the 6-channel input internally.

Target: >=60% preference on 1color, 2color, 2shape&color test conditions
        after 4000 training episodes with PPO.

Usage (PYTHONPATH=src_isaac):

    python src_isaac/examples/train_guess_what_moves.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

from _train_common import assert_target_preferences

OUTPUT = Path("~/nett_guess_what_moves_out").expanduser()

CONFIG: dict = {
    "name": f"guess_what_moves_{datetime.now():%Y%m%d_%H%M%S}",
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
        "wrappers": ["framestack"],   # stacks last 2 frames -> 6-channel CHW input
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "guess_what_moves",
        "encoder_cfg": {
            "trainable": True,
            "features_dim": 512,      # matches larger what_cnn (NatureCNN-style, no pooling)
            "num_frames": 2,
        },
        "algorithm_cfg": {
            "rollouts": 2000,
            "mini_batches": 4,
            "learning_rate": 1e-4,
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
            "entropy_loss_scale": 0.01,  # 0.3 accelerated NaN divergence; 0.01 is the empirically-best value
            "learning_epochs": 4,        # SKRL default=8 caused mid-training NaN divergence
            "kl_threshold": 0.01,        # early-stop each epoch if policy drifts too much
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
            "tags": ["guess_what_moves", "ppo", "2000ep", "2frame", "dual_stream", "no-pooling", "closeness+completeness"],
        },
    },
    "num_brains": 1,
    "episodes": {"train": 2000, "test": 4},
    "steps_per_episode": 200,
    "eval_freq": 10_000_000,
    "task_memory": 0.1,
    "max_parallel_envs": 49,  # 7x7: a square tile grid. 52 tiles 8x7 -> distorted fisheye (#488)
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.guess_what_moves")

    log.info("training guess_what_moves: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    run_dir = OUTPUT / CONFIG["name"]
    log.info("analyzing: %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    # GWM goal: rest >= 90%, 2shape >= 60% (different targets from NatureCNN)
    assert_target_preferences(out, threshold=0.60, rest_threshold=0.90,
                              conditions=("2shape",))
    log.info("  -> %s", out)
