"""Campaign driver: ONE (model x experiment/imprint) training+test+analysis run.

Drives the 9-model x 3-condition sweep requested in the project goal:
  models : CNN, 3DCNN, SimCLR-CLTT, ViT, ViT-CLTT, ViT+VICReg, ViViT,
           ViViT+VICReg, GuessWhatMoves
  conds  : binding/Object1, parsing/fork-1, viewinvariance/Fork_Front

All training hyperparameters are held identical to the VALIDATED SB3/Unity
replication protocol (train_binding_8brain_targets.py): PPO, ent=0.01, lr=3e-4,
rollouts=8000, mini_batches=16, learning_epochs=10, steps=500, 2000 train / 10
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
  NETT_BRAINS      brains_per_process (default 8); 1 = single-brain topology
  NETT_BRAIN_OFFSET brain/seed index offset (default 0); a single-brain fleet packs
                   offsets 0..N-1 as N processes (folds the former orch_run_single.py)
  NETT_TRAIN_EPS   training episodes (default 2000)
  NETT_MAX_ENVS    max parallel envs (default 32)
  NETT_RES         input resolution (default 256)
  NETT_REWARD_TYPES comma-sep (default closeness); e.g. closeness,completeness
  NETT_DESIGN_SHEET / NETT_MEDIA_ROOT  override the per-experiment sheet/media defaults
  NETT_AUX_WEIGHT  unused for VICReg (driver forces 1.0 per user directive)
"""
from __future__ import annotations
from _paths import VIDEOS_ROOT

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

VIDEOS = VIDEOS_ROOT
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
    "binding":        ("binding/DesignSheet_Binding.csv",               "binding/videos",        "Object1"),
    "parsing":        ("parsing/DesignSheet_Parsing.csv",               "parsing/videos",        "fork-1"),
    "viewinvariance": ("viewinvariance/DesignSheet_ViewInvariance.csv", "viewinvariance/videos", "Fork_Front"),
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
    # `brains_per_process` (NETT_BRAINS) brains run as skrl agents inside ONE process on
    # ONE GPU, sharing one env (disjoint slices — no crossover). NETT_BRAIN_OFFSET shifts
    # the brain/seed index, so a SINGLE-BRAIN topology (NETT_BRAINS=1 with offsets 0..N-1
    # across N processes) reproduces the N-brain run, and campaign_run.py can PACK several
    # such processes per GPU. This folds the former orch_run_single.py.
    # NOTE brain_id_offset hash-mixes the whole task seed (runtime/task.py), so a
    # single-brain-offset-b run is NOT yet bit-identical to brain b of a multi-brain run;
    # the unified global-brain-id seeding is the determinism-plan's job (do it there).
    device = int(os.environ.get("NETT_DEVICE", "0"))
    brains = int(os.environ.get("NETT_BRAINS", "8"))
    offset = int(os.environ.get("NETT_BRAIN_OFFSET", "0"))
    train_eps = int(os.environ.get("NETT_TRAIN_EPS", "2000"))
    max_envs = int(os.environ.get("NETT_MAX_ENVS", "32"))
    res = int(os.environ.get("NETT_RES", "256"))
    # Overridable (defaults preserve the existing campaign behavior; single-brain runs
    # that reproduce orch_run_single set these). reward_types default = closeness only.
    reward_types = [r.strip() for r in
                    os.environ.get("NETT_REWARD_TYPES", "closeness").split(",") if r.strip()]

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
        # Single-frame: leave NETT_MEMORY_DEVICE UNSET so the buffer follows the
        # compute device (resolve_memory_device), which is already the default.
        #
        # BUG FIXED 2026-07-21: this used to set the literal "cuda". agent_factory
        # compares torch.device(mem_device) != torch.device(device), and
        # torch.device("cuda") != torch.device("cuda:0") is TRUE -- so setting a bare
        # "cuda" selected the CPU hybrid path, the exact opposite of the intent, and
        # silently disabled the uint8 on-GPU buffer this branch was written to enable.
        # Never name a CUDA device without its index in a value that gets compared.

    # Import AFTER setting aux/buffer env so any import-time reads see it.
    from nett_skrl import NETT
    from nett_skrl.analysis import analyze, log_analysis_to_wandb

    # off{offset} keeps single-brain-packed runs (same model/exp, different offsets on the
    # same GPU/out dir) from colliding; off0 for the multi-brain case is harmless.
    name = f"{_slug(model)}_{exp}_{imprint}_off{offset}_{datetime.now():%m%d_%H%M%S}"[:63]
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
            "design_sheet": os.environ.get("NETT_DESIGN_SHEET", f"{VIDEOS}/{sheet_rel}"),
            "media_root": os.environ.get("NETT_MEDIA_ROOT", f"{VIDEOS}/{media_rel}"),
            "conditions": [imprint],
            "headless": True,
            "input_resolution": res,
            "camera_fov": 150.0,
            "reward_types": reward_types,
            "enable_neck_flexion": False,
            "enable_lateral_bending": False,
        },
        "brain": brain,
        "body": {"wrappers": (["framestack"] if spec["framestack"] else [])},
        "num_brains": brains,
        "brain_id_offset": offset,
        "episodes": {"train": train_eps, "test": int(os.environ.get("NETT_TEST_EPS", "10"))},
        "steps_per_episode": int(os.environ.get("NETT_STEPS", "256")),
        "eval_freq": 10_000_000,
        "task_memory": float(os.environ.get("NETT_TASK_MEMORY", "1")),
        "max_parallel_envs": max_envs,
    }

    log.info("MODEL=%s EXP=%s IMPRINT=%s device=%d brains=%d off=%d res=%d envs=%d mb=%s "
             "rewards=%s aux=%s ssl=%s -> %s",
             model, exp, imprint, device, brains, offset, res, max_envs,
             os.environ.get("NETT_MINIBATCHES", "16"), reward_types, spec.get("aux"),
             spec.get("reward"), out)
    t0 = time.time()
    NETT(config).run(output_path=str(out), devices=[device], verbose=True)
    train_secs = time.time() - t0
    log.info("training complete in %.1fs", train_secs)

    run_dir = out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Per-run timing for campaign_score.py's wall-time aggregation.
    (run_dir / "campaign_timing.json").write_text(json.dumps(
        {"name": name, "model": model, "experiment": exp, "imprint": imprint,
         "brain_offset": offset, "num_brains": brains, "device": device,
         "num_envs": max_envs, "res": res, "train_eps": train_eps,
         "train_secs": round(train_secs, 1), "finished": datetime.now().isoformat()}, indent=2))
    log.info("analyzing: %s", run_dir)
    result = analyze(run_dir)
    log_analysis_to_wandb(run_dir, result)
    log.info("DONE %s -> %s", name, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
