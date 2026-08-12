"""Campaign driver: ONE (model x experiment/imprint) training+test+analysis run.

Drives the 9-model x 3-condition sweep requested in the project goal:
  models : CNN, 3DCNN, SimCLR-CLTT, ViT, ViT-CLTT, ViT+VICReg, ViViT,
           ViViT+VICReg, GuessWhatMoves
  conds  : binding/Object1, parsing/fork-1, viewinvariance/Fork_Front

All training hyperparameters are held identical to the VALIDATED SB3/Unity
replication protocol (train_binding_8brain_targets.py): PPO, ent=0.01, lr=3e-4,
rollouts=8000, mini_batches=16, learning_epochs=10, steps=500, 2000 train / 20
test episodes, closeness-only extrinsic reward, FOV=150, 2D action space
(neck flexion + lateral bending OFF), input_resolution=128, hidden_sizes=[],
shared_encoder, clip_actions=False, 7 brains. ONLY the encoder (+ its temporal
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
  NETT_BRAINS      brains_per_process (default 7 -> 112 envs, a square grid)
  NETT_BRAIN_OFFSET brain/seed index offset (default 0); a single-brain fleet packs
                   offsets 0..N-1 as N processes (folds the former orch_run_single.py)
  NETT_TRAIN_EPS   training episodes (default 2000)
  NETT_MAX_ENVS    max parallel envs, TOTAL across brains (default 112)
  NETT_RES         input resolution (default 128)
  NETT_ROLLOUTS    per-brain transitions between PPO updates (default 8000)
  NETT_MINIBATCHES minibatches per update (default 16 -> batch 500 = 1 episode)
  NETT_STEPS       steps_per_episode (default 500)
  NETT_TEST_EPS    test episodes (default 20)
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
_HERE = Path(__file__).resolve().parent
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
# ── The QK-ablation arms (SIDE_LOCK_INVESTIGATION.md Phases 10-12). `attn_mode` swaps the
# token-mixing operator only; embed_dim is re-matched to the ViT arm's 697,184 in each,
# because an ablation that also deletes a third of the parameters proves nothing.
VIT_NOQK_CFG = {**VIT_CFG, "embed_dim": 164, "attn_mode": "uniform"}   # 706,368 at 128x80 (+1.8%)
VIT_MIXER_CFG = {**VIT_CFG, "embed_dim": 160, "attn_mode": "mixer"}    # 682,675 at 128x80 (-1.6%)
# ⚠ THOSE TWO ARE CLS-POOLED AND ARE NOT VALID CONTROLS. They are kept because they still
# BUILD and still match on parameters at the current eye -- but note the justification that
# used to be here ("kept to reproduce the 2026-08-01/02 arms") IS NOW VOID: those arms ran at
# a square 128x128, and at 128x80 these are a different model on a different input, so they
# reproduce nothing. Treat any run of them as a new experiment, and do not read binding off
# one: with pool="cls" the readout is ONE token, which under uniform/static
# mixing is a global average, so a sub-patch object is diluted ~64x: both fell to EXACT
# chance on a supervised small-stimulus probe while qk and CNN scored 1.000. Removing QK
# therefore also removed small-object detection -- upstream of every condition, including
# the positive control, which all three RL arms duly failed.
#
# ★ pool="spatial" FIXED IT (probe 2026-08-03: both ablations 0.497 -> 1.000 at sz8/n10, and
# 0.998-1.000 at sz2/n30). It reduces each patch token, folds them back to the patch grid and
# pools to a small map, so per-patch evidence survives the readout via the residual stream.
# Applied to the qk arm TOO, or the comparison confounds pooling with routing. Those were the
# valid ablation arms, and the result of record is that qk - mixer on binding is +0.053 at
# p = 0.163 (n=56), i.e. NOT resolvable at that n. See NEXT_STEPS.md §G3.
#
# ⚠⚠ THE THREE SPATIAL ARMS BELOW ARE SQUARE-EYE ARCHIVE CONFIGS, NOT LAUNCHABLE ARMS.
# Scoped 2026-08-12, when the eye became permanently non-square. They are calibrated for a
# SQUARE 128x128 -> 8x8 tokens -> 4x4 pool, 65 tokens, which is what the 2026-08-03/05 n=56
# arms trained at. The current 128x80 eye gives a 5x8 token grid with no square divisor but 1,
# and `mixer`'s NxN matrix changes size, so 136/152/156 hold nothing at ~697K any more.
#
# ★ THEY ARE KEPT BECAUSE OFFLINE RE-ANALYSIS STILL NEEDS THEM. The n=56 checkpoints are on
# disk (~/nett_vit_sp_20260803, ~/nett_vit_mixer_sp_20260803) and examples/probe_frozen_features.py
# rebuilds each arm's encoder through MODELS[...] at a SQUARE RES=128 to load them. That probe
# is how Phases 14/15 were produced, and re-asking the QK question with the per-agent COUPLING
# as the endpoint -- which separates at p ~ 0.006 on the n already on disk -- runs through it.
# Deleting these would have broken that, and it is the cheapest open experiment there is.
#
# ⚠ DO NOT LAUNCH THEM AS TRAINING ARMS. At 128x80 CompactViT raises at CONSTRUCTION, so the
# failure is immediate and loud rather than a silently incomparable run -- but do not rely on
# that as the guard; the point is that a new spatial arm is a NEW EXPERIMENT that does not pool
# with the n=56 results, because it is a different model on a different input.
#
# TO REVIVE THE LINE at the current eye: CompactViT now accepts a RECTANGULAR spatial_grid,
# so (5, 4) works (20 cells, the closest analogue to the old 4x4 = 16). Re-match embed_dim
# with examples/count_params.py against what the ViT arm costs at THAT resolution, and treat
# the result as a fresh baseline.
VIT_SP = {**VIT_CFG, "pool": "spatial", "spatial_grid": 4, "spatial_reduce_dim": 16}
VIT_SP_CFG = {**VIT_SP, "embed_dim": 136}                              # 696,000 at 128x128
VIT_MIXER_SP_CFG = {**VIT_SP, "embed_dim": 152, "attn_mode": "mixer"}  # 693,907 at 128x128
VIT_NOQK_SP_CFG = {**VIT_SP, "embed_dim": 156, "attn_mode": "uniform"} # 706,928 at 128x128
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
    "ViT-NoQK":       dict(encoder="compact_vit",     cfg=dict(VIT_NOQK_CFG),                                                        framestack=False),  # ~706K at 128x80
    "ViT-Mixer":      dict(encoder="compact_vit",     cfg=dict(VIT_MIXER_CFG),                                                       framestack=False),  # ~683K at 128x80
    # ⚠ SQUARE-EYE ARCHIVE ONLY -- for probe_frozen_features.py to rebuild the 2026-08 n=56
    # encoders at RES=128 square. NOT launchable at the 128x80 eye (raises at construction).
    "ViT-Sp":         dict(encoder="compact_vit",     cfg=dict(VIT_SP_CFG),                                                          framestack=False),  # 696K at 128x128
    "ViT-Mixer-Sp":   dict(encoder="compact_vit",     cfg=dict(VIT_MIXER_SP_CFG),                                                    framestack=False),  # 694K at 128x128
    "ViT-NoQK-Sp":    dict(encoder="compact_vit",     cfg=dict(VIT_NOQK_SP_CFG),                                                     framestack=False),  # 707K at 128x128
    "ViT-CLTT":       dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=False, reward="CLTT"),
    "ViT+VICReg":     dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=False, aux="vicreg"),
    "ViViT":          dict(encoder="compact_vivit",   cfg=dict(VIVIT_CFG),                                                           framestack=True),
    "ViViT+VICReg":   dict(encoder="compact_vivit",   cfg=dict(VIVIT_CFG),                                                           framestack=True,  aux="vicreg"),
    "GuessWhatMoves": dict(encoder="guess_what_moves", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75, "num_frames": 2}, framestack=True),  # ~698K
}

# experiment -> (design sheet, media dir, default imprint per goal). parsing and
# viewinvariance use the .mov sheet variants: the base .webm sheets reference clips
# that DO NOT EXIST on disk, so the env disables video (blank monitors) and the
# imprinting stimulus is destroyed (root cause of the June campaign's ~chance
# parsing/viewinvariance scores). viewinvariance's .mov sheet lives under
# examples/orch_sheets/ (generated to reference the on-disk .mov clips), not the media root.
EXPERIMENTS: dict[str, tuple[str, str, str]] = {
    "binding":        (f"{VIDEOS}/binding/DesignSheet_Binding.csv",     f"{VIDEOS}/binding/videos",        "Object1"),
    "parsing":        (f"{VIDEOS}/parsing/DesignSheet_Parsing_mov.csv", f"{VIDEOS}/parsing/videos",        "fork-1"),
    "viewinvariance": (f"{_HERE}/orch_sheets/DesignSheet_ViewInvariance_mov.csv",
                       f"{VIDEOS}/viewinvariance/videos", "Fork_Front"),
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
    sheet, media, default_imprint = EXPERIMENTS[exp]
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
    # 7, not 8: 7 x 16 envs = 112 = an 11x11 SQUARE tile grid. 8 x 16 = 128
    # tiles 12x11 and distorts the fisheye (Isaac Sim #488).
    brains = int(os.environ.get("NETT_BRAINS", "7"))
    offset = int(os.environ.get("NETT_BRAIN_OFFSET", "0"))
    train_eps = int(os.environ.get("NETT_TRAIN_EPS", "2000"))
    max_envs = int(os.environ.get("NETT_MAX_ENVS", "112"))
    # 128 stands by decision (see nett_env_cfg.ObservationCfg.input_resolution).
    res = int(os.environ.get("NETT_RES", "128"))
    # Overridable (defaults preserve the existing campaign behavior; single-brain runs
    # that reproduce orch_run_single set these). reward_types default = closeness only.
    reward_types = [r.strip() for r in
                    os.environ.get("NETT_REWARD_TYPES", "closeness").split(",") if r.strip()]

    # ── Arm knobs. campaign/launch_arm.sh sets these per arm; without them the launcher
    # would set an env var this driver silently ignores, which is strictly worse than one
    # it never sets -- the name says one thing and the data is another. (That exact class
    # of bug ran Object1 into a directory named `object2` on 2026-07-31.)
    #
    # NETT_HIDDEN_SIZES: comma-separated policy-head MLP; "" = LINEAR, which is the campaign
    # default and identical across encoders (why the 9-model sweep chose it).
    # ⚠ [64,64] was defaulted 2026-07-30 and REVERTED the same day: the Unity archive's head
    # effect (linear 0.516 -> 0.724, n=10) did NOT transfer -- matched Isaac arms gave 0.804
    # vs 0.742, Welch p = 0.51. n=8 (sd~0.19) could not have resolved it either way; ~152
    # agents/arm would be needed. See NEXT_STEPS.md §G5 before running a head ladder.
    _hs = os.environ.get("NETT_HIDDEN_SIZES", "")
    hidden_sizes = [int(x) for x in _hs.split(",") if x.strip()] if _hs.strip() else []
    # NETT_ENTROPY: PPO entropy_loss_scale. 0.01 is the validated SB3/Unity value.
    # ⚠ 0.03 was tested at n=56: it moves the LOCK but not binding (`learn_frac` 12/56 in
    # both arms, p = 1.00), loosening |sp| exactly where the cue is unusable. A lever on the
    # measurability of the shape conditions, not on binding. SIDE_LOCK Phase 8.
    entropy = float(os.environ.get("NETT_ENTROPY", "0.01"))
    # ★ RECORD THE EVALUATION ACTION MODE WITH THE RUN. NETT_EVAL_STOCHASTIC (read in
    # brain/trainer.py) decides whether test-time actions are the Gaussian policy's MEAN or
    # a SAMPLE, and with a fixed test start pose the mean makes every episode of a condition
    # a bit-identical replay -- a binary readout instead of a graded preference. So it
    # changes what the scores MEAN, not just their noise. On 2026-07-29 a stochastic retest
    # was compared against a deterministic summary with nothing on disk saying which
    # protocol produced which number; this driver must never leave that ambiguity again.
    from nett_skrl.brain.trainer import eval_stochastic_enabled
    eval_stochastic = eval_stochastic_enabled()

    _rollouts_for_budget = int(os.environ.get("NETT_ROLLOUTS", "8000"))

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
    # ★ ROLLOUT-AWARE (2026-07-28). The on-GPU buffer is rollouts x obs PER BRAIN, so
    # doubling rollouts to 16000 (for whole-episode rollouts at 32 envs/brain) doubled
    # it to 8 x 16000 x 128x128x3 = 6.3 GB. Added to the ~14.5 GB the renderer needs at
    # 256 envs that exceeds the A10's 23 GB: MEASURED as a hard Vulkan
    # ERROR_OUT_OF_DEVICE_MEMORY on the camera projection texture, with the card pinned
    # at 22.4 GB. The old policy ("single-frame -> GPU buffer") was written when the
    # buffer was 3.2 GB and is no longer safe on size alone, so size is now what decides.
    _obs_bytes = res * res * 3
    _buf_gb = brains * _rollouts_for_budget * _obs_bytes / 1e9
    _budget_gb = float(os.environ.get("NETT_GPU_BUFFER_BUDGET_GB", "4.0"))
    # ⚠ SIZE IS NECESSARY BUT NOT SUFFICIENT. The buffer fitting says nothing about the
    # ENCODER's activation peak. Measured previously: the on-GPU buffer is reliable only
    # for the lean nature_cnn; ViT/ViViT attention creeps to ~22 GB and OOMs LATE -- about
    # 72% through training, i.e. after hours of compute. So the GPU buffer is opt-IN by
    # model, and every other encoder keeps it on the host.
    _GPU_BUFFER_MODELS = {"nature_cnn"}
    _lean = spec["encoder"] in _GPU_BUFFER_MODELS
    if "NETT_MEMORY_DEVICE" not in os.environ:
        if spec["framestack"] or _buf_gb > _budget_gb or not _lean:
            os.environ["NETT_MEMORY_DEVICE"] = "cpu"
            os.environ.pop("NETT_UINT8_BUFFER", None)
            why = ("framestack" if spec["framestack"]
                   else f"> {_budget_gb:.1f} GB GPU budget" if _buf_gb > _budget_gb
                   else f"encoder {spec['encoder']} not in the GPU-buffer allowlist "
                        "(attention peak OOMs late)")
            log.info("rollout buffer -> CPU (est %.1f GB; %s)", _buf_gb, why)
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
    # ★ SEPARATE ROOT PER BASELINE. The June runs live under ~/nett_campaign; nothing
    # produced before 2026-07-28 is poolable with anything after (dt 1/60->1/24,
    # FXAA->DLAA, 256->500 step episodes, and the 128-env non-square render). Writing
    # the new baseline into the same tree would make the two trivially confusable by
    # any glob that walks it. NETT_OUT_ROOT keeps them apart.
    _out_root = os.environ.get("NETT_OUT_ROOT", "~/nett_campaign")
    out = Path(f"{_out_root}/{exp}_{_slug(model)}").expanduser()

    # Resolved once: the tag list below cannot reference `brain` while `brain` is
    # still being constructed (self-reference -> UnboundLocalError).
    _rollouts = int(os.environ.get("NETT_ROLLOUTS", "8000"))
    brain: dict = {
        "algorithm": "PPO",
        "encoder": spec["encoder"],
        "encoder_cfg": spec["cfg"],
        "algorithm_cfg": {
            # ★ 8000 over 16 envs/brain -> EXACTLY 1 FULL EPISODE per env per rollout
            # (user directive: "envs should always record only full episodes"), with
            # 7 BRAINS x 16 = 112 envs, which tiles 11x11 -- SQUARE, so the fisheye is
            # undistorted. This is why the brain count is 7 and not 8: at 8 brains the
            # same shape gives 128 envs, which tiles 12x11 and hits the NON-SQUARE
            # distortion of Isaac Sim #488. Several jobs of the June campaign trained
            # through exactly that. 256 envs was tried and OOMs the A10 outright at
            # res128 (Vulkan ERROR_OUT_OF_DEVICE_MEMORY at 22.4/23 GB, renderer alone).
            "rollouts": _rollouts,
            # batch = rollouts / mini_batches = 8000 / 16 = 500 = ONE EPISODE.
            "mini_batches": int(os.environ.get("NETT_MINIBATCHES", "16")),
            "learning_rate": 3e-4,
            "learning_epochs": 10,
            "value_loss_scale": 0.5,
            "grad_norm_clip": 0.5,
            "entropy_loss_scale": entropy,
            "kl_threshold": 0.5,
            "value_clip": 0,
        },
        "model": {
            "value_bound": None,
            "shared_encoder": True,
            "hidden_sizes": hidden_sizes,
            "clip_actions": False,
        },
        "wandb": {
            "mode": "online",
            "project": f"nett-{exp}-replication",
            "tags": [model, spec["encoder"], "ppo", f"{brains}brain", exp, imprint,
                     f"rollouts={_rollouts}", f"steps={os.environ.get('NETT_STEPS', '500')}",
                     f"{train_eps}ep",
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
            "design_sheet": os.environ.get("NETT_DESIGN_SHEET", sheet),
            "media_root": os.environ.get("NETT_MEDIA_ROOT", media),
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
        "episodes": {"train": train_eps, "test": int(os.environ.get("NETT_TEST_EPS", "20"))},
        "steps_per_episode": int(os.environ.get("NETT_STEPS", "500")),
        "eval_freq": 10_000_000,
        "task_memory": float(os.environ.get("NETT_TASK_MEMORY", "1")),
        "max_parallel_envs": max_envs,
    }

    # ★ REFUSE A DISTORTED RENDER. max_parallel_envs is passed through verbatim (an
    # explicit value is deliberately never overridden -- the recipe's num_envs is
    # load-bearing for learning), so nothing downstream re-checks it. The June campaign
    # ran many jobs at 128 envs, which tiles 12x11: Isaac Sim #488 applies the fisheye
    # lens distortion over the full non-square canvas, so every training frame was
    # rendered at the wrong aspect. Fitting in VRAM says nothing about whether the
    # render is correct. Fail loudly rather than train a whole job through it.
    from nett_skrl.runtime.parallel_envs import is_valid_num_envs, largest_valid_num_envs
    _envs = config["max_parallel_envs"]
    if not is_valid_num_envs(_envs, brains):
        _alt = largest_valid_num_envs(_envs, brains)
        raise SystemExit(
            f"NETT_MAX_ENVS={_envs} is not a valid env count for {brains} brains: it "
            f"must be a multiple of num_brains AND tile into a SQUARE camera grid "
            f"(Isaac Sim #488 -- a non-square grid distorts the fisheye). "
            f"Largest valid value at or below it: {_alt}."
        )
    # And the rollout must cover WHOLE episodes per env (user directive): every env
    # records only complete episodes, so rollouts/scope must be a multiple of steps.
    _scope = _envs // brains
    _roll = brain["algorithm_cfg"]["rollouts"]
    _steps = config["steps_per_episode"]
    if (_roll // _scope) % _steps != 0:
        raise SystemExit(
            f"rollouts={_roll} over scope={_scope} envs/brain gives {_roll // _scope} "
            f"steps/env, which is not a whole number of {_steps}-step episodes. "
            f"Envs must record only full episodes."
        )

    log.info("MODEL=%s EXP=%s IMPRINT=%s device=%d brains=%d off=%d res=%d envs=%d mb=%s "
             "rewards=%s aux=%s ssl=%s -> %s",
             model, exp, imprint, device, brains, offset, res, max_envs,
             os.environ.get("NETT_MINIBATCHES", "16"), reward_types, spec.get("aux"),
             spec.get("reward"), out)
    log.info("hidden_sizes=%s entropy=%s eval=%s", hidden_sizes, entropy,
             "stochastic" if eval_stochastic else "mean")
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
         # ★ The three arm knobs, recorded so a score is never ambiguous about the protocol
         # that produced it. eval_stochastic especially: nothing evaluated under one value
         # pools with anything under the other.
         "hidden_sizes": hidden_sizes, "entropy_loss_scale": entropy,
         "eval_stochastic": eval_stochastic,
         "train_secs": round(train_secs, 1), "finished": datetime.now().isoformat()}, indent=2))
    log.info("analyzing: %s", run_dir)
    result = analyze(run_dir)
    log_analysis_to_wandb(run_dir, result)
    log.info("DONE %s -> %s", name, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
