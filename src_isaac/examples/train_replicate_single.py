"""Single-brain replication run, pinned to ONE GPU as cuda:0.

Launch one of these PER GPU with CUDA_VISIBLE_DEVICES=<phys> so the physical GPU
appears as cuda:0 (Isaac USD only supports cuda:0). devices=[0] makes NETT
assign config.device=0 -> cuda:0 = that single visible GPU.

Goal constraints baked in:
  - binding / Object1, FOV=150, 2D action (no neck flexion / lateral bending),
    reward = closeness only, input_resolution=64, test=5 episodes, wandb online.

Env vars:
  NETT_ENCODER     : nature_cnn (default) | compact_vit | guess_what_moves
  NETT_SEED_OFFSET : integer added to brain seeding (distinct seeds per run)
  NETT_ENVS        : parallel envs (default 16). rollouts/mini_batches are set so
                     batch stays 500 and each env runs exactly 1 episode/rollout:
                     rollouts = NETT_ENVS * steps_per_episode (500),
                     mini_batches = rollouts // 500 (= NETT_ENVS).
  NETT_NAME        : run name suffix (default = encoder + timestamp)
  NETT_OUTPUT      : output dir (default ~/nett_replicate_out)
"""
from __future__ import annotations
from _paths import BINDING_DESIGN_SHEET, BINDING_MEDIA_ROOT

import logging
import os
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb
from _train_common import assert_target_preferences

ENC = os.environ.get("NETT_ENCODER", "nature_cnn")
SEED_OFFSET = int(os.environ.get("NETT_SEED_OFFSET", "0"))
ENVS = int(os.environ.get("NETT_ENVS", "16"))
STEPS = 500
ROLLOUTS = ENVS * STEPS          # exactly 1 episode/env/rollout (no mid-ep bootstrap)
MINI_BATCHES = max(1, ROLLOUTS // 500)   # batch size = 500 (SB3 reference)

OUTPUT = Path(os.environ.get("NETT_OUTPUT", "~/nett_replicate_out")).expanduser()
NAME = os.environ.get("NETT_NAME", f"{ENC}_s{SEED_OFFSET}_{datetime.now():%Y%m%d_%H%M%S}")

# Per-encoder spec. input_resolution=256, features_dim=512 for ALL encoders.
# To stay under the 700k hard limit at res=256, the no-pool CNNs use an
# AdaptiveAvgPool before flatten (added in the encoder classes): nature_cnn
# pools to 4x4 (-> ~602k), guess_what_moves' what-pathway pools to 3x3
# (-> ~640k). compact_vit (embed_dim=128, fd=512) is ~489k.
_ENC_SPECS = {
    "nature_cnn": {
        "encoder_cfg": {"trainable": True, "features_dim": 512},
        "temporal": False,
        "hidden_sizes": [],
    },
    "compact_vit": {
        # depth=3 -> ~621k total, matching nature_cnn (602k) / GWM (640k).
        "encoder_cfg": {"trainable": True, "features_dim": 512, "patch_size": 8,
                        "embed_dim": 128, "depth": 3, "num_heads": 4, "mlp_ratio": 2.0},
        "temporal": False,
        "hidden_sizes": [],
    },
    "guess_what_moves": {
        "encoder_cfg": {"trainable": True, "features_dim": 512, "num_frames": 2},
        "temporal": True,
        "hidden_sizes": [],
    },
}
spec = _ENC_SPECS[ENC]

env_cfg = {
    "design_sheet": BINDING_DESIGN_SHEET,
    "media_root": BINDING_MEDIA_ROOT,
    "conditions": ["Object1"],
    "headless": True,
    "input_resolution": 256,
    "camera_fov": 150.0,
    "reward_types": ["closeness"],
    "enable_neck_flexion": False,
    "enable_lateral_bending": False,
}

brain_cfg = {
    "algorithm": "PPO",
    "encoder": ENC,
    "encoder_cfg": spec["encoder_cfg"],
    "algorithm_cfg": {
        "rollouts": ROLLOUTS,
        "mini_batches": MINI_BATCHES,
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
        "hidden_sizes": spec["hidden_sizes"],
        "clip_actions": False,
    },
    "wandb": {
        "mode": "online",
        "project": "nett-compact-models",
        "tags": [ENC, "ppo", "replicate", "Object1", "fov=150", "2d-action",
                 "closeness-only", f"envs={ENVS}", f"batch=500", f"seed_off={SEED_OFFSET}"],
    },
}

CONFIG: dict = {
    "name": NAME,
    "environment": env_cfg,
    "brain": brain_cfg,
    "num_brains": 1,
    "episodes": {"train": 2000, "test": 5},
    "steps_per_episode": STEPS,
    "eval_freq": 10_000_000,
    # task_memory: "auto" runs a VRAM dry-run estimator (boots Isaac ~2 extra
    # times). When envs is fixed/capped we can skip it by passing a fixed GB via
    # NETT_TASK_MEMORY (e.g. 12) -> no dry-run, faster startup.
    "task_memory": (float(os.environ["NETT_TASK_MEMORY"])
                    if os.environ.get("NETT_TASK_MEMORY") else "auto"),
    "max_parallel_envs": ENVS,
    "brain_id_offset": SEED_OFFSET,   # shifts brain_id -> distinct seed per run
}
if spec["temporal"]:
    CONFIG["body"] = {"wrappers": ["framestack"]}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.replicate_single")
    log.info("encoder=%s envs=%s rollouts=%s mini_batches=%s seed_off=%s out=%s/%s",
             ENC, ENVS, ROLLOUTS, MINI_BATCHES, SEED_OFFSET, OUTPUT, NAME)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    run_dir = OUTPUT / NAME
    log.info("analyzing: %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    try:
        assert_target_preferences(out)
    except Exception as exc:
        log.warning("target preference check: %s", exc)
    log.info("RESULT %s -> %s", NAME, out)
    print("REPLICATE_SINGLE_DONE_OK")
