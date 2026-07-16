"""Replicate the SB3/Unity binding_vit_small baseline in Isaac Sim / skrl.

Reference (bug-free) run: ~/run_CompendiumGoodCheck39_bundle
  encoder=ViT, ent_coef=0.01, lr=3e-4, buffer_size=8000, batch_size=500,
  learning_rate=3e-4, steps_per_episode=500, train=2000 episodes,
  reward=closeness, custom_policy_arch=[].

Reference Object1 test results (mean over 4 brains) — the replication target:
  1color=0.674  1shape&color=0.660  2color=0.805  2shape&color=0.807
  binding=0.622 ; 1shape/2shape ~0.50 (chance, expected).

This script matches the reference on EVERY hyperparameter axis at once
(the principled baseline), with three deliberate, justified deviations:

  1. encoder: ViT -> nature_cnn. GPU is 8 GB; nature_cnn is a smaller no-pool
     CNN. (Pooling encoders collapse the left-right signal; established.)
  2. FOV = 150 (goal mandate: both monitors visible at test-episode start).
  3. 2D action space (turn+forward only): enable_neck_flexion /
     enable_lateral_bending set False explicitly (goal mandate).

vs run 16 (which got ZERO learning), this restores the reference values that
run 16 had wrong:
  entropy_loss_scale 0.0  -> 0.01   (ent_coef from the bug-free reference)
  rollouts           4000 -> 8000   (buffer_size)
  mini_batches       8    -> 16     (8000/16 = 500 = reference batch_size)
  steps_per_episode  200  -> 500    (reference)
  train episodes     4000 -> 2000   (reference)

Parallelism (the one big deliberate Isaac change, kept): 16 envs x 500 steps =
exactly one full episode per env per update, 16 complete episodes per 8000-step
rollout. No mid-episode bootstrap.

Architecture fixes retained (they align WITH the SB3 reference, not against it):
  hidden_sizes=[]  (CnnPolicy custom_policy_arch=[] -> no trunk)
  clip_actions=False (SB3 stores unclipped Normal samples for the PPO ratio)
"""

from __future__ import annotations
from _paths import BINDING_DESIGN_SHEET, BINDING_MEDIA_ROOT

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

from _train_common import assert_target_preferences

OUTPUT = Path("~/nett_nature_cnn_replicate_out").expanduser()

CONFIG: dict = {
    "name": f"nature_cnn_replicate_{datetime.now():%Y%m%d_%H%M%S}",
    "environment": {
        "design_sheet": BINDING_DESIGN_SHEET,
        "media_root": BINDING_MEDIA_ROOT,
        "conditions": ["Object1"],
        "headless": True,
        "input_resolution": 64,            # NETT legacy default; fits 8 GB w/ 8000 buffer
        "camera_fov": 150.0,
        "reward_types": ["closeness"],
        "enable_neck_flexion": False,      # 2D action space (goal mandate)
        "enable_lateral_bending": False,   # 2D action space (goal mandate)
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "nature_cnn",
        "encoder_cfg": {
            "trainable": True,
            "features_dim": 512,
        },
        "algorithm_cfg": {
            "rollouts": 8000,              # = reference buffer_size
            "mini_batches": 16,            # 8000/16 = 500 = reference batch_size
            "learning_rate": 3e-4,
            "learning_epochs": 10,
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
            "entropy_loss_scale": 0.01,    # = reference ent_coef (run 16 had 0.0)
            "kl_threshold": 0.5,           # loose (~SB3 target_kl=None)
            "value_clip": 0,
        },
        "model": {
            "value_bound": None,
            "shared_encoder": True,
            "hidden_sizes": [],            # = SB3 custom_policy_arch=[]
            "clip_actions": False,         # SB3 stores unclipped samples
        },
        "wandb": {
            "mode": "online",
            "project": "nett-compact-models",
            "tags": [
                "nature_cnn", "ppo", "replicate-baseline",
                "ent=0.01", "lr=3e-4", "rollouts=8000", "batch=500",
                "steps=500", "2000ep", "closeness-only", "fov=150",
                "2d-action", "16-envs",
            ],
        },
    },
    "num_brains": 1,
    "episodes": {"train": 2000, "test": 20},
    "steps_per_episode": 500,
    "eval_freq": 10_000_000,               # disable mid-training eval
    "task_memory": "auto",
    "max_parallel_envs": 8,                # 8000/8 = 1000 steps/env = 2 whole episodes/env
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.nature_cnn_replicate")

    log.info("training nature_cnn_replicate: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    run_dir = OUTPUT / CONFIG["name"]
    log.info("analyzing: %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    try:
        assert_target_preferences(out)
    except Exception as exc:  # report but don't crash before we inspect summary
        log.warning("target preference check: %s", exc)
    log.info("  -> %s", out)
