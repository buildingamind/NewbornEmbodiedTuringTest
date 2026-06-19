"""Fast 2-brain unified-wandb smoke: confirms NETT_UNIFIED_WANDB=1 yields ONE
wandb run with per-brain namespaced metrics (brain_1/..., brain_2/...).
Expect exactly ONE 'View run at' line in the log (brains 2..N reuse brain 1's run).
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from nett_skrl import NETT

CONFIG = {
    "name": f"unified_smoke_{datetime.now():%H%M%S}",
    "environment": {
        "design_sheet": "/home/zlaborde/code/isaac/videos/binding/DesignSheet_Binding.csv",
        "media_root": "/home/zlaborde/code/isaac/videos/binding/videos",
        "conditions": ["Object1"],
        "headless": True, "input_resolution": 64,
        "camera_fov": 150.0, "reward_types": ["closeness"],
        "enable_neck_flexion": False, "enable_lateral_bending": False,
    },
    "brain": {
        "algorithm": "PPO", "encoder": "nature_cnn",
        "encoder_cfg": {"trainable": True, "features_dim": 512},
        "algorithm_cfg": {"rollouts": 200, "mini_batches": 2, "learning_rate": 3e-4},
        "model": {"shared_encoder": True, "hidden_sizes": [], "clip_actions": False},
        "wandb": {"mode": "online", "project": "nett-binding-replication",
                  "tags": ["unified-smoke"]},
    },
    "num_brains": 2,
    "episodes": {"train": 4, "test": 1},
    "steps_per_episode": 50,
    "eval_freq": 10_000_000,
    "task_memory": 12,
    "max_parallel_envs": 4,
}

if __name__ == "__main__":
    NETT(CONFIG).run(output_path="/tmp/unified_smoke_out", devices=[0], verbose=True)
    print("UNIFIED_SMOKE_DONE_OK")
