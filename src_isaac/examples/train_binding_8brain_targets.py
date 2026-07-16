"""8-brain binding/Object1 replication for the THREE target encoders:
3DCNN (2-frame), ViViT (2-frame), SimCLR-CLTT (2-frame).

Uses the VALIDATED SB3/Unity replication protocol (train_nature_cnn_replicate.py /
train_binding_8brain.py): ent=0.01, lr=3e-4, rollouts=8000, mini_batches=16,
learning_epochs=10, steps=500, 2000 train / 5 test episodes, closeness-only
extrinsic reward, FOV=150, 2D action (neck flexion + lateral bending OFF),
input_resolution=256, hidden_sizes=[], shared_encoder, clip_actions=False.

All three encoders: features_dim=512 (embedding dimension, user directive),
2-frame framestack, param-equalized and < 700k:
  compact_3dcnn  582k | compact_vivit 535k (embed_dim=120) | simclr_cltt 592k

SimCLR-CLTT keeps its CLTT temporal-contrastive INTRINSIC reward (does not count
as extrinsic; small weight, trains only the detached projection head). Extrinsic
reward remains closeness-only per the goal.

Select encoder via NETT_ENCODER: compact_3dcnn | compact_vivit | simclr_cltt
Select GPU via NETT_DEVICE (default: 0).
"""
from __future__ import annotations
from _paths import VIDEOS_ROOT

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from nett_skrl import NETT
from nett_skrl.analysis import analyze, log_analysis_to_wandb
from _train_common import assert_target_preferences

ENC = os.environ.get("NETT_ENCODER", "compact_3dcnn")
DEVICE = int(os.environ.get("NETT_DEVICE", "0"))
# scope = MAX_PARALLEL_ENVS // num_brains must keep (rollouts//scope) % steps == 0.
# rollouts=8000, steps=500, num_brains=8 -> scope in {1,2,4,8} (caps 8,16,32,64).
MAX_PARALLEL_ENVS = int(os.environ.get("NETT_MAX_ENVS", "32"))  # scope=4
# Diagnostic overrides (default to the goal values).
RES = int(os.environ.get("NETT_RES", "256"))
NUM_BRAINS = int(os.environ.get("NETT_BRAINS", "8"))
TRAIN_EPS = int(os.environ.get("NETT_TRAIN_EPS", "2000"))

_ENC_SPECS = {
    # Single-frame "normal CNN" — diagnostic baseline (no temporal, pair with NETT_FRAMESTACK=0).
    "nature_cnn": {
        "encoder_cfg": {"trainable": True, "features_dim": 512},
        "extra_brain": {},
    },
    "compact_3dcnn": {
        "encoder_cfg": {"trainable": True, "features_dim": 512, "num_frames": 2},
        "extra_brain": {},
    },
    "compact_vivit": {
        "encoder_cfg": {
            "trainable": True, "features_dim": 512, "patch_size": 16,
            "embed_dim": 120, "depth": 3, "num_heads": 3, "mlp_ratio": 2.0,
            "num_frames": 2,
            # conv-free frame-fusion sweep: temporal_mode in {joint,early,factored,late}, pool in {cls,mean}
            "temporal_mode": os.environ.get("NETT_VIVIT_TEMPORAL", "joint"),
            "pool": os.environ.get("NETT_VIVIT_POOL", "cls"),
        },
        "extra_brain": {},
    },
    "simclr_cltt": {
        "encoder_cfg": {"trainable": True, "features_dim": 512},
        # CLTT temporal-contrastive auxiliary (intrinsic) reward — defining
        # mechanism of SimCLR-CLTT; small weight, trains only the projector head.
        "extra_brain": {
            "reward": "CLTT",
            "reward_cfg": {"weight": 0.05, "temperature": 0.1,
                           "proj_lr": 1e-3, "beta": 0.1},
        },
    },
    # --- Second encoder batch (CNN / ViT / GuessWhatMoves), 2-frame, <700k,
    # param-matched (NatureCNN 607k | CompactViT 561k | GuessWhatMoves 639k) at
    # res256/features_dim=512. Run with NETT_FRAMESTACK=1 (2-frame).
    "compact_vit": {
        # Single-frame spatial ViT (CLS pooling). patch_size=16 -> 256 patches at
        # res256 (keeps attention light, matches the ViViT res256 patch count).
        # depth=3/embed_dim=128 -> 595k params (1-frame), param-matched to
        # NatureCNN 601k / GuessWhatMoves 639k; embed_dim>=144 or depth>=4 exceeds 700k.
        "encoder_cfg": {
            "trainable": True, "features_dim": 512, "patch_size": 16,
            # width is env-overridable for the ~1M "wider model" arm
            # (NETT_VIT_EMBED=168 NETT_VIT_HEADS=8 -> ~990k).
            "embed_dim": int(os.environ.get("NETT_VIT_EMBED", "128")),
            "depth": 3,
            "num_heads": int(os.environ.get("NETT_VIT_HEADS", "4")),
            "mlp_ratio": 2.0,
            # pool="cls" (default ViT) collapses here like ViViT; pivot to
            # pool="spatial" (+ NETT_AUX_LOSS=simclr) preserves object location.
            "pool": os.environ.get("NETT_VIT_POOL", "cls"),
            # stem="conv" (NETT_VIT_STEM) = conv-stem patch embed for plain-ViT
            # trainability (no aux); "linear" = standard single patch conv.
            "stem": os.environ.get("NETT_VIT_STEM", "linear"),
        },
        "extra_brain": {},
    },
    "guess_what_moves": {
        # Dual-pathway (what: current-frame 2D CNN | moves: Conv3d motion) 2-frame.
        "encoder_cfg": {"trainable": True, "features_dim": 512, "num_frames": 2},
        "extra_brain": {},
    },
}

if ENC not in _ENC_SPECS:
    print(f"Unknown encoder: {ENC}. Choose from: {list(_ENC_SPECS)}")
    sys.exit(1)

spec = _ENC_SPECS[ENC]

# Optional LR warmup (NETT_LR_WARMUP=<scheduler-steps>): LinearLR ramps the LR
# from 10% to 100% of peak over the first N scheduler steps (skrl steps the
# scheduler once per learning epoch). Targets the transformer's early-training
# instability (the transient-rise-then-collapse of plain ViT). Passes through
# NETT's algorithm-cfg `extra` -> setattr on the skrl cfg (same path as value_preprocessor).
_WARMUP: dict = {}
if os.environ.get("NETT_LR_WARMUP"):
    import torch.optim.lr_scheduler as _lr
    _WARMUP = {
        "learning_rate_scheduler": _lr.LinearLR,
        "learning_rate_scheduler_kwargs": {
            "start_factor": 0.1, "end_factor": 1.0,
            "total_iters": int(os.environ["NETT_LR_WARMUP"]),
        },
    }

_TAG = os.environ.get("NETT_TAG", "")

# Experiment selector (NETT_EXPERIMENT): binding/Object1 (default), parsing/fork-1,
# viewinvariance/Fork_Front. Switches the design sheet, media, imprint condition,
# output dir and wandb naming. Imprint overridable via NETT_IMPRINT.
EXP = os.environ.get("NETT_EXPERIMENT", "binding")
_EXP_MAP = {
    "binding":        ("binding/DesignSheet_Binding.csv",               "binding/videos",        "Object1"),
    "parsing":        ("parsing/DesignSheet_Parsing.csv",               "parsing/videos",        "fork-1"),
    "viewinvariance": ("viewinvariance/DesignSheet_ViewInvariance.csv", "viewinvariance/videos", "Fork_Front"),
}
if EXP not in _EXP_MAP:
    print(f"Unknown NETT_EXPERIMENT: {EXP}. Choose from: {list(_EXP_MAP)}"); sys.exit(1)
_VIDEOS = VIDEOS_ROOT
DESIGN_SHEET = f"{_VIDEOS}/{_EXP_MAP[EXP][0]}"
MEDIA_ROOT = f"{_VIDEOS}/{_EXP_MAP[EXP][1]}"
IMPRINT = os.environ.get("NETT_IMPRINT", _EXP_MAP[EXP][2])

OUTPUT = Path(f"~/nett_{EXP}_8brain_{ENC}{_TAG}").expanduser()

_brain: dict = {
    "algorithm": "PPO",
    "encoder": ENC,
    "encoder_cfg": spec["encoder_cfg"],
    "algorithm_cfg": {
        "rollouts": 8000,
        "mini_batches": int(os.environ.get("NETT_MINIBATCHES", "16")),
        "learning_rate": float(os.environ.get("NETT_LR", "3e-4")),
        "learning_epochs": 10,
        "value_loss_scale": 0.5,
        "grad_norm_clip": 0.5,
        "entropy_loss_scale": float(os.environ.get("NETT_ENT", "0.01")),
        "kl_threshold": 0.5,
        "value_clip": 0,
        **_WARMUP,
    },
    "model": {
        "value_bound": None,
        "shared_encoder": True,
        "hidden_sizes": [],
        "clip_actions": False,
    },
    "wandb": {
        # Env-controllable so unattended runs can avoid the wandb-login prompt
        # (set NETT_WANDB_MODE=offline when no API key is configured).
        "mode": os.environ.get("NETT_WANDB_MODE", "online"),
        "project": f"nett-{EXP}-replication",
        "tags": [
            ENC, "ppo", "8brain", EXP, IMPRINT,
            "rollouts=8000", "steps=500", "2000ep", "closeness-only",
            "fov=150", "2d-action", "feat512", f"res={RES}",
        ],
    },
}
_brain.update(spec["extra_brain"])

CONFIG: dict = {
    # Include the per-process TAG + brain offset so each split process gets a
    # UNIQUE run name -> unique wandb run id (id = sha256(name:cond:brain_id);
    # brain_id is always 1 for single-brain runs, and second-resolution
    # timestamps collide when 8 procs launch in one loop -> all share one wandb run).
    "name": f"{ENC}{_TAG}_off{os.environ.get('NETT_BRAIN_OFFSET','0')}_{datetime.now():%m%d_%H%M%S}"[:63],
    "environment": {
        "design_sheet": DESIGN_SHEET,
        "media_root": MEDIA_ROOT,
        "conditions": [IMPRINT],
        "headless": True,
        "input_resolution": RES,
        "camera_fov": 150.0,
        "reward_types": ["closeness"],
        "enable_neck_flexion": False,
        "enable_lateral_bending": False,
    },
    "brain": _brain,
    # 2-frame framestack for all three encoders; NETT_FRAMESTACK=0 disables (diagnostic).
    "body": {"wrappers": (["framestack"] if os.environ.get("NETT_FRAMESTACK", "1") == "1" else [])},
    "num_brains": NUM_BRAINS,
    # Shifts the whole run's seed so split processes (subsets of the 8 brains on
    # separate GPUs) get genuinely distinct, independent brains. Must differ per split.
    "brain_id_offset": int(os.environ.get("NETT_BRAIN_OFFSET", "0")),
    "episodes": {"train": TRAIN_EPS, "test": 5},
    "steps_per_episode": 500,
    "eval_freq": 10_000_000,
    # Scheduler reserves this many GB on the (NVML-indexed) GPU before launch.
    # NVML ignores CUDA_VISIBLE_DEVICES, so concurrent runs all check physical
    # GPU0; keep this <= free headroom there. Real footprint (CPU rollout buffer)
    # is only ~3-12 GB depending on env count.
    "task_memory": float(os.environ.get("NETT_TASK_MEMORY", "1")),
    "max_parallel_envs": MAX_PARALLEL_ENVS,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger(f"nett.binding_8brain_{ENC}")

    log.info("encoder=%s device=%d max_envs=%d output=%s",
             ENC, DEVICE, MAX_PARALLEL_ENVS, OUTPUT)
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
