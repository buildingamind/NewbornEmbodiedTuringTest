"""8-brain binding replication for CNN, ViT, and GuessWhatMoves encoders.

Matches the validated replication hyperparameters from train_nature_cnn_replicate.py
(ent=0.01, lr=3e-4, rollouts=8000, batch=500, steps=500, 2000ep, closeness-only,
FOV=150, 2D action, hidden_sizes=[], clip_actions=False).

Select encoder via NETT_ENCODER env var: nature_cnn | compact_vit | guess_what_moves
Select GPU via NETT_DEVICE env var (default: 0).
"""
from __future__ import annotations
from _paths import BINDING_DESIGN_SHEET, BINDING_MEDIA_ROOT

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb
from _train_common import assert_target_preferences

ENC = os.environ.get("NETT_ENCODER", "nature_cnn")
DEVICE = int(os.environ.get("NETT_DEVICE", "0"))

# res=256, features_dim=512 for all (encoders pool to keep <700k; see encoder
# classes: nature_cnn 4x4, guess_what_moves what-pathway 3x3).
_ENC_SPECS = {
    "nature_cnn": {
        "encoder_cfg": {"trainable": True, "features_dim": 512},
        "temporal": False,
    },
    "compact_vit": {
        # depth=3 -> ~621k total, matching nature_cnn (602k) / GWM (640k).
        "encoder_cfg": {
            "trainable": True, "features_dim": 512, "patch_size": 8,
            "embed_dim": 128, "depth": 3, "num_heads": 4, "mlp_ratio": 2.0,
        },
        "temporal": False,
    },
    "guess_what_moves": {
        "encoder_cfg": {"trainable": True, "features_dim": 512, "num_frames": 2},
        "temporal": True,
    },
}

if ENC not in _ENC_SPECS:
    print(f"Unknown encoder: {ENC}. Choose from: {list(_ENC_SPECS)}")
    sys.exit(1)

spec = _ENC_SPECS[ENC]
_DOCKER = Path("/data/DesignSheet_Binding.csv").exists()
_DESIGN = "/data/DesignSheet_Binding.csv" if _DOCKER else BINDING_DESIGN_SHEET
_MEDIA = "/data/videos" if _DOCKER else BINDING_MEDIA_ROOT
OUTPUT = Path("/output") if _DOCKER else Path(f"~/nett_binding_8brain_{ENC}").expanduser()

CONFIG: dict = {
    "name": f"binding_8brain_{ENC}_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": _DESIGN,
        "media_root": _MEDIA,
        "conditions": ["Object1"],
        "headless": True,
        "input_resolution": 256,
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
            "project": "nett-binding-replication",
            "tags": [
                ENC, "ppo", "8brain", "binding", "Object1",
                "ent=0.01", "lr=3e-4", "rollouts=8000", "batch=500",
                "steps=500", "2000ep", "closeness-only", "fov=150",
                "2d-action",
            ] + (["2frame"] if spec["temporal"] else []),
        },
    },
    "num_brains": 8,
    "episodes": {"train": 2000, "test": 5},
    "steps_per_episode": 500,
    "eval_freq": 10_000_000,
    "task_memory": 20,
    "max_parallel_envs": 16,   # res=256: 2 envs/brain (res256 obs is 16x bigger)
}
if spec["temporal"]:
    CONFIG["body"] = {"wrappers": ["framestack"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger(f"nett.binding_8brain_{ENC}")

    log.info("encoder=%s device=%d output=%s", ENC, DEVICE, OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[DEVICE], verbose=True)
    log.info("training complete")

    run_dir = OUTPUT / CONFIG["name"]
    log.info("analyzing: %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    try:
        assert_target_preferences(out)
    except Exception as exc:
        log.warning("target preference check: %s", exc)
    log.info("  -> %s", out)
