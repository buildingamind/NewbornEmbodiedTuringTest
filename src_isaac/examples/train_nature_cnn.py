"""Train NatureCNN (no spatial pooling) with PPO on NETT object-binding tasks.

Follow-up to train_nature_cnn_debug.py, which broke the universal exact-50%
collapse seen across 6 prior architectures (CompactCNN/SmallCNN's final
AdaptiveAvgPool2d((4,4)) was destroying the left-right positional signal
needed for visual discrimination -- see feedback_compact_model_training memory).

Two 2000-episode runs at lr=1e-4 (the "proven baseline" rate) produced wildly
different outcomes:
    run 1 (seed A): 2color=81.3%, 2shape&color=60.4% PASS; 1color=50.0% FAIL
    run 2 (seed B): 2shape&color=62.8% PASS; 1color=25.1%, 2color=50.0% FAIL

High seed-to-seed variance -- which conditions clear 60% differs run to run,
and per-brain scores for the harder conditions are often *perfectly bimodal*
(each brain commits to a fixed left/right bias and averages to ~50%), vs.
genuine partial/intermediate scores when the agent is actually discriminating
visually. That's the signature of "still consolidating" rather than a hard
ceiling -- but lr=1e-4 only stays numerically stable for ~2000 episodes
(a 4000-episode run at that rate diverged to NaN at step ~40600, right past
where the successful run completed).

This run halves the learning rate to 5e-5 and extends training to 3000
episodes (60K steps/env) -- enough headroom past the 1e-4 divergence point
to test whether a slower, more stable optimization lets the agent consolidate
genuine discrimination across ALL three target conditions consistently,
rather than landing on a per-seed mix of real-discrimination and fixed-bias
fallback. Everything else matches the successful debug config
(entropy_loss_scale=0.01, features_dim=512, fov=150, input_resolution=128).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb

from _train_common import assert_target_preferences

OUTPUT = Path("~/nett_nature_cnn_out").expanduser()

CONFIG: dict = {
    "name": f"nature_cnn_{datetime.now():%Y%m%d_%H%M%S}",
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
    "brain": {
        "algorithm": "PPO",
        "encoder": "nature_cnn",
        "encoder_cfg": {
            "trainable": True,
            "features_dim": 512,
        },
        "algorithm_cfg": {
            "rollouts": 2000,
            "mini_batches": 4,
            "learning_rate": 5e-5,       # halved from proven 1e-4 -- that rate diverged to
                                          # NaN at step ~40600 in a 4000ep run; lower rate
                                          # should stay stable longer, giving the agent more
                                          # time to consolidate genuine discrimination instead
                                          # of falling back to per-brain fixed-bias on harder
                                          # conditions (high seed-to-seed variance observed
                                          # at 2000ep: passing conditions differ run to run)
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
            "entropy_loss_scale": 0.01,
        },
        "checkpoint_freq": 10_000_000,
        "model": {
            "value_bound": None,
            "shared_encoder": True,
            "hidden_sizes": [64, 64],
        },
        "wandb": {
            "mode": "online",
            "project": "nett-compact-models",
            "tags": ["nature_cnn", "ppo", "no-pooling", "lr5e-5", "3000ep"],
        },
    },
    "num_brains": 1,
    "episodes": {"train": 3000, "test": 4},
    "steps_per_episode": 200,
    "eval_freq": 10_000_000,
    "task_memory": 0.2,
    "max_parallel_envs": 52,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.nature_cnn")

    log.info("training nature_cnn: output=%s", OUTPUT)
    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)
    log.info("training complete")

    run_dir = OUTPUT / CONFIG["name"]
    log.info("analyzing: %s", run_dir)
    out = analyze(run_dir)
    log_analysis_to_wandb(run_dir, out)
    assert_target_preferences(out)
    log.info("  -> %s", out)
