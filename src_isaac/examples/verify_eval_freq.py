"""Minimal eval_freq verification run.

Trains for 4 short episodes, triggers 1 mid-training eval, then finishes.
Confirms: eval runs without error, bar-chart values appear in W&B, training
resumes, memory is cleaned up between subprocesses.

Run with:
    python src_isaac/examples/verify_eval_freq.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from nett_skrl import NETT

OUTPUT = Path("~/nett_eval_verify").expanduser()

# 4 episodes × 200 steps = 800 total env interactions.
# eval_freq = 400 → 1 eval after the first 2 episodes, then training finishes.
# rollouts = 200 → envs_per_brain = 200/200 = 1 → very light on memory.
# max_parallel_envs = 4 → eval uses 4 envs (52 test tasks / 4 = 13 tasks/env).
CONFIG = {
    "name": "eval_freq_verify",
    "environment": {
        "design_sheet": "/home/zlaborde/code/isaac/videos/binding/DesignSheet_Binding.csv",
        "media_root": "/home/zlaborde/code/isaac/videos/binding/videos",
        "conditions": ["Object1"],
        "headless": True,
        "input_resolution": 64,
        "camera_fov": 60.0,
        "reward_types": ["closeness"],
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "small",
        "encoder_cfg": {"trainable": True},
        "algorithm_cfg": {
            # rollouts=200, steps_per_episode=200 → 1 env per brain.
            "rollouts": 200,
            "mini_batches": 4,
            "learning_rate": 3e-4,
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
        },
        "model": {
            "value_bound": None,
            "shared_encoder": True,
        },
        "wandb": {
            "mode": "online",
            "project": "nett-skrl-eval-verify",
        },
    },
    "num_brains": 1,
    # 4 training episodes × 200 steps = 800 total env interactions.
    # eval_freq = 400 → eval fires at step 400, then training finishes at 800.
    # episodes_test = 1: 52 test tasks × 1 ep each = 52 total test episodes.
    # max_parallel_envs = 52 → eval_num_envs = 52 (52 divides 52, ≤ 52).
    # With eval_timesteps fix: ceil(52/52) × 200 = 200 eval steps total (very fast).
    "episodes": {"train": 4, "test": 1},
    "steps_per_episode": 200,
    "eval_freq": 400,
    "task_memory": 0.1,
    "max_parallel_envs": 52,
}

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s"
    )
    log = logging.getLogger("nett.verify")

    log.info("Starting eval_freq verification run")
    log.info("Output: %s", OUTPUT)
    log.info(
        "Expected: 2 training chunks (400 steps each) + 1 mid-training eval (200 steps "
        "with 52 parallel envs) + 1 final test phase"
    )

    NETT(CONFIG).run(output_path=str(OUTPUT), devices=[0], verbose=True)

    log.info("Run complete — check W&B project 'nett-skrl-eval-verify' for:")
    log.info("  eval/*/correct_pct metrics at global_step=40 (400//10 envs_per_brain)")
    log.info(
        "  Continuous training reward curve (no global_step reset between chunks)"
    )

    run_dir = next(OUTPUT.iterdir()) if OUTPUT.exists() else None
    if run_dir:
        eval_csvs = list((run_dir / "Object1" / "logs").glob("test_*_400.csv"))
        log.info(
            "Eval CSV files at step 400: %s",
            eval_csvs if eval_csvs else "none found (check logs/)",
        )
        metrics_csv = run_dir / "Object1" / "logs" / "eval_metrics.csv"
        if metrics_csv.exists():
            log.info("eval_metrics.csv contents:\n%s", metrics_csv.read_text())
