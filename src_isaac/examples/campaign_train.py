"""Campaign driver: ONE (model x experiment/imprint) training+test+analysis run.

Drives the 9-model x 3-condition sweep requested in the project goal:
  models : CNN, 3DCNN, SimCLR-CLTT, ViT, ViT-CLTT, ViT+VICReg, ViViT,
           ViViT+VICReg, GuessWhatMoves
  conds  : binding/Object2, parsing/ship-1, viewinvariance/Ship_Front

All training hyperparameters are held identical to the VALIDATED SB3/Unity
replication protocol (train_binding_8brain_targets.py): PPO, ent=0.01, lr=3e-4,
rollouts=8000, mini_batches=16, learning_epochs=10, steps=500, 2000 train / 5
test episodes, closeness-only extrinsic reward, FOV=150, 2D action space
(neck flexion + lateral bending OFF), input_resolution=256, hidden_sizes=[],
shared_encoder, clip_actions=False, 8 brains. ONLY the encoder (+ its temporal
framestacking + its self-supervised objective) varies between models.

Self-supervised objectives:
  * CLTT  (SimCLR-CLTT, ViT-CLTT) -> reward="CLTT": temporal contrastive INTRINSIC
    reward (obs_t, obs_{t+1} positive pairs); trains only the projector head, does
    NOT alter the closeness extrinsic reward.
  * VICReg (ViT+VICReg, ViViT+VICReg) -> NETT_AUX_LOSS=vicreg aux loss folded into
    the PPO update on the shared encoder (weight 1.0, user directive).

Select via env:
  NETT_MODEL       one of the 9 labels above (required)
  NETT_EXPERIMENT  binding | parsing | viewinvariance (default binding)
  NETT_IMPRINT     imprint condition override (default = goal condition per exp)
  NETT_DEVICE      GPU index (default 0)
  NETT_BRAINS      brains/seeds (default 8)
  NETT_TRAIN_EPS   training episodes (default 2000)
  NETT_MAX_ENVS    max parallel envs (default 32)
  NETT_RES         input resolution (default 256)
  NETT_AUX_WEIGHT  unused for VICReg (driver forces 1.0 per user directive)
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

VIDEOS = "/home/zlaborde/code/isaac/videos"
CLTT_REWARD_CFG = {"weight": 0.05, "temperature": 0.1, "proj_lr": 1e-3, "beta": 0.1}

# Encoder capacity tuned so each model is ~700K total params AT res=128 (measured
# via examples/count_params.py). features_dim is FIXED at 512 for EVERY model so
# the PPO policy/value head (Linear(512->action)) is IDENTICAL across models — a
# fair comparison where only the encoder differs. Encoders scale via internals:
# ViT/ViViT via embed_dim (kept divisible by num_heads); the conv/SimCLR/GWM
# models via conv_dim (final conv channels -> flatten size -> Linear params).
VIT_CFG = {
    "trainable": True, "features_dim": 512, "patch_size": 16,
    "embed_dim": 144, "depth": 3, "num_heads": 4, "mlp_ratio": 2.0,   # 144 -> ~699K
    "pool": "cls", "stem": "linear",
}
VIVIT_CFG = {
    "trainable": True, "features_dim": 512, "patch_size": 16,
    "embed_dim": 144, "depth": 3, "num_heads": 3, "mlp_ratio": 2.0,   # 144 -> ~699K
    "num_frames": 2, "temporal_mode": "joint", "pool": "cls",
}

# model label -> (encoder name, encoder_cfg, framestack?, reward(None|"CLTT"), aux(None|"vicreg"))
MODELS: dict[str, dict] = {
    "CNN":            dict(encoder="nature_cnn",      cfg={"trainable": True, "features_dim": 512, "conv_dim": 75},                  framestack=False),  # ~699K
    "3DCNN":          dict(encoder="compact_3dcnn",   cfg={"trainable": True, "features_dim": 512, "conv_dim": 77, "num_frames": 2}, framestack=True),   # ~698K
    "SimCLR-CLTT":    dict(encoder="simclr_cltt",     cfg={"trainable": True, "features_dim": 512, "conv_dim": 77},                  framestack=False, reward="CLTT"),  # ~702K
    "ViT":            dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=False),
    "ViT-CLTT":       dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=False, reward="CLTT"),
    "ViT+VICReg":     dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=False, aux="vicreg"),
    "ViViT":          dict(encoder="compact_vivit",   cfg=dict(VIVIT_CFG),                                                           framestack=True),
    "ViViT+VICReg":   dict(encoder="compact_vivit",   cfg=dict(VIVIT_CFG),                                                           framestack=True,  aux="vicreg"),
    "GuessWhatMoves": dict(encoder="guess_what_moves", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75, "num_frames": 2}, framestack=True),  # ~698K
}

# experiment -> (design sheet rel path, media rel path, default imprint per goal)
EXPERIMENTS: dict[str, tuple[str, str, str]] = {
    "binding":        ("binding/DesignSheet_Binding.csv",               "binding/videos",        "Object2"),
    "parsing":        ("parsing/DesignSheet_Parsing.csv",               "parsing/videos",        "ship-1"),
    "viewinvariance": ("viewinvariance/DesignSheet_ViewInvariance.csv", "viewinvariance/videos", "Ship_Front"),
}


def _slug(s: str) -> str:
    return s.replace("+", "").replace("-", "").replace(" ", "")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("nett.campaign")

    model = os.environ.get("NETT_MODEL", "")
    if model not in MODELS:
        log.error("NETT_MODEL=%r invalid; choose from %s", model, list(MODELS))
        return 2
    exp = os.environ.get("NETT_EXPERIMENT", "binding")
    if exp not in EXPERIMENTS:
        log.error("NETT_EXPERIMENT=%r invalid; choose from %s", exp, list(EXPERIMENTS))
        return 2

    spec = MODELS[model]
    sheet_rel, media_rel, default_imprint = EXPERIMENTS[exp]
    imprint = os.environ.get("NETT_IMPRINT", default_imprint)
    # NETT runs all `num_brains` brains as skrl agents inside ONE process on ONE
    # GPU (build_tasks makes one task per condition; _run_single_mode runs every
    # brain together). So a job is pinned to a single GPU, and the campaign runs
    # one job per GPU concurrently (campaign_run.py). At res=128 a job is only a
    # few GB, so 8 concurrent jobs (one per GPU) fit comfortably.
    device = int(os.environ.get("NETT_DEVICE", "0"))
    brains = int(os.environ.get("NETT_BRAINS", "8"))
    train_eps = int(os.environ.get("NETT_TRAIN_EPS", "2000"))
    max_envs = int(os.environ.get("NETT_MAX_ENVS", "32"))
    res = int(os.environ.get("NETT_RES", "256"))

    # VICReg aux is read from the environment by agent_factory; set it here so the
    # whole process tree (incl. brain subprocesses) inherits it. CLTT is a reward
    # set directly in the brain config below.
    if spec.get("aux") == "vicreg":
        os.environ["NETT_AUX_LOSS"] = "vicreg"
        # User directive: VICReg aux weight = 1.0 (ViT+VICReg, ViViT+VICReg only).
        # Force it here so an inherited campaign-wide default cannot override it.
        os.environ["NETT_AUX_WEIGHT"] = "1.0"
    else:
        # ensure no stray aux leaks in from a shared shell
        os.environ.pop("NETT_AUX_LOSS", None)

    # Rollout-buffer placement (read by agent_factory). Single-frame models keep
    # the buffer ON GPU as uint8 (~6 GB) to eliminate per-step obs GPU->CPU copies
    # and per-update CPU->GPU minibatch copies -> ~60% less CPU. Framestack models
    # carry a 2-frame buffer (~13 GB) that cannot share 23 GB VRAM with the
    # renderer (OOMs even at 32 envs), so they keep the buffer on CPU.
    # An explicit NETT_MEMORY_DEVICE (e.g. for A/B benchmarks) overrides the
    # per-model default below.
    if "NETT_MEMORY_DEVICE" not in os.environ:
        if spec["framestack"]:
            os.environ["NETT_MEMORY_DEVICE"] = "cpu"
            os.environ.pop("NETT_UINT8_BUFFER", None)
        else:
            os.environ["NETT_MEMORY_DEVICE"] = "cuda"
            os.environ["NETT_UINT8_BUFFER"] = "1"

    # Import AFTER setting aux/buffer env so any import-time reads see it.
    from nett_skrl import NETT
    from nett_skrl.analysis import analyze, log_analysis_to_wandb

    name = f"{_slug(model)}_{exp}_{imprint}_{datetime.now():%m%d_%H%M%S}"[:63]
    out = Path(f"~/nett_campaign/{exp}_{_slug(model)}").expanduser()

    brain: dict = {
        "algorithm": "PPO",
        "encoder": spec["encoder"],
        "encoder_cfg": spec["cfg"],
        "algorithm_cfg": {
            "rollouts": 8192,   # 8192 = 256 steps x 32 envs/brain -> 1 episode/env/rollout
            # batch = 8192 / mini_batches. 16 -> 512. The scheduler escalates
            # mini_batches on OOM to shrink the update batch.
            "mini_batches": int(os.environ.get("NETT_MINIBATCHES", "16")),
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
            "project": f"nett-{exp}-replication",
            "tags": [model, spec["encoder"], "ppo", "8brain", exp, imprint,
                     "rollouts=8000", "steps=500", f"{train_eps}ep",
                     "closeness-only", "fov=150", "2d-action", f"res={res}",
                     *( ["cltt"] if spec.get("reward") == "CLTT" else [] ),
                     *( ["vicreg"] if spec.get("aux") == "vicreg" else [] )],
        },
    }
    if spec.get("reward") == "CLTT":
        brain["reward"] = "CLTT"
        brain["reward_cfg"] = dict(CLTT_REWARD_CFG)

    config: dict = {
        "name": name,
        "environment": {
            "design_sheet": f"{VIDEOS}/{sheet_rel}",
            "media_root": f"{VIDEOS}/{media_rel}",
            "conditions": [imprint],
            "headless": True,
            "binocular_vision": False,
            "input_resolution": res,
            "camera_fov": 150.0,
            "reward_types": ["closeness"],
            "enable_neck_flexion": False,
            "enable_lateral_bending": False,
        },
        "brain": brain,
        "body": {"wrappers": (["framestack"] if spec["framestack"] else [])},
        "num_brains": brains,
        "episodes": {"train": train_eps, "test": 5},
        "steps_per_episode": int(os.environ.get("NETT_STEPS", "256")),
        "eval_freq": 10_000_000,
        "task_memory": float(os.environ.get("NETT_TASK_MEMORY", "1")),
        "max_parallel_envs": max_envs,
    }

    log.info("MODEL=%s EXP=%s IMPRINT=%s device=%d brains=%d res=%d envs=%d mb=%s aux=%s reward=%s -> %s",
             model, exp, imprint, device, brains, res, max_envs,
             os.environ.get("NETT_MINIBATCHES", "16"), spec.get("aux"), spec.get("reward"), out)
    NETT(config).run(output_path=str(out), devices=[device], verbose=True)
    log.info("training complete")

    run_dir = out / name
    log.info("analyzing: %s", run_dir)
    result = analyze(run_dir)
    log_analysis_to_wandb(run_dir, result)
    log.info("DONE %s -> %s", name, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
