"""Encoder comparison on the validated replication config.

Holds ALL training hyperparameters identical to the working nature_cnn
replication (ent=0.01, lr=3e-4, rollouts=8000, mini_batches=16, hidden_sizes=[],
clip_actions=False, steps=500, 2000ep, 8 envs, FOV=150, 2D action,
closeness-only, input_resolution=64, seed offset=0 = original seed 57160185).
ONLY the encoder (and 2-frame framestacking for temporal encoders) varies, so
any test-performance difference is attributable to the encoder.

Select via NETT_ENCODER in {compact_vit, compact_vivit, compact_3dcnn}.
ViViT / 3D-CNN are 2-frame: body framestack (n_stack=2) + encoder num_frames=2.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb
from _train_common import assert_target_preferences

ENC = os.environ.get("NETT_ENCODER", "compact_vit")

# encoder-specific architecture cfg (from the existing compact_* example scripts)
# and whether the encoder needs 2-frame temporal stacking.
_ENC_SPECS = {
    "compact_vit": {
        "encoder_cfg": {"trainable": True, "features_dim": 128, "patch_size": 8,
                         "embed_dim": 128, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0},
        "temporal": False,
    },
    "compact_vivit": {
        "encoder_cfg": {"trainable": True, "features_dim": 96, "patch_size": 8,
                         "embed_dim": 96, "depth": 3, "num_heads": 3, "mlp_ratio": 2.0,
                         "num_frames": 2},
        "temporal": True,
    },
    "compact_3dcnn": {
        "encoder_cfg": {"trainable": True, "features_dim": 128, "num_frames": 2},
        "temporal": True,
    },
}
spec = _ENC_SPECS[ENC]
OUTPUT = Path("~/nett_nature_cnn_replicate_out").expanduser()

CONFIG: dict = {
    "name": f"replicate_{ENC}_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": "/home/zach/Code/NewbornEmbodiedTuringTest_Private/isaac_lab/assets/design_sheets/binding.csv",
        "media_root": "/home/zach/Code/NewbornEmbodiedTuringTest_Private/isaac_lab/assets/videos",
        "conditions": ["Object1"],
        "headless": True,
        "binocular_vision": False,
        "input_resolution": 64,
        "camera_fov": 150.0,
        "reward_types": ["closeness"],
        "enable_neck_flexion": False,
        "enable_lateral_bending": False,
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": ENC,
        "encoder_cfg": spec["encoder_cfg"],
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
            "tags": [ENC, "ppo", "replicate-baseline", "encoder-study",
                     "ent=0.01", "lr=3e-4", "rollouts=8000", "steps=500",
                     "2000ep", "closeness-only", "fov=150", "2d-action"]
                    + (["2frame"] if spec["temporal"] else []),
        },
    },
    "num_brains": 1,
    "episodes": {"train": 2000, "test": 5},
    "steps_per_episode": 500,
    "eval_freq": 10_000_000,
    "task_memory": "auto",
    "max_parallel_envs": 32,
}
if spec["temporal"]:
    CONFIG["body"] = {"wrappers": ["framestack"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.replicate_encoders")
    log.info("encoder=%s temporal=%s output=%s", ENC, spec["temporal"], OUTPUT)
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
