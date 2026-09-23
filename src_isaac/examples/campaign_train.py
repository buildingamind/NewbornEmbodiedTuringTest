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
  * CLTT  (SimCLR-CLTT, ViT-CLTT) -> aux="cltt": temporal contrastive AUXILIARY
    LOSS on (obs_t, obs_{t+1}) pairs, folded into the PPO backward so it shapes the
    shared encoder. The extrinsic closeness reward is untouched.
    Until 2026-08-28 these two arms declared reward="CLTT" instead, which routed
    them to an INTRINSIC REWARD adapter constructed per env row -- so the NT-Xent
    saw B=1, had no negatives, and was identically 0.0, while a cosine bonus was
    added to the reward every step. No arm in this campaign uses a custom reward.
  * VICReg (ViT+VICReg, ViViT+VICReg) -> NETT_AUX_LOSS=vicreg aux loss folded into
    the PPO update on the shared encoder (weight 1.0, user directive).

Select via env (a CURATED SUBSET -- see the warning below):
⛔ THIS LIST IS PARTIAL AND ALWAYS WILL BE. It names ~21 of the 143 NETT_* variables the
code reads. It is a getting-started list, NOT an index: a name's ABSENCE FROM HERE MEANS
NOTHING. The authoritative answer to "does this knob exist?" is docs/env_vars.md, which is
GENERATED from the code (scripts/gen_env_index.py) and kept in sync by tests/test_env_index.py.
⚠ Why it matters that absence means nothing: a name the code never reads raises nothing and
warns nothing -- the arm trains, scores, and files under its experimental label while running
the CONTROL. A queue row was written naming `NETT_BODY_WRAPPERS (DOES NOT EXIST)` on exactly that
mistake; body wrappers come from the arm's MODELS entry, never from the environment.

  NETT_MODEL       one of the 9 labels above (required)
  NETT_EXPERIMENT  binding | parsing | viewinvariance (default binding)
  NETT_IMPRINT     imprint condition override (default = goal condition per exp)
  NETT_DEVICE      GPU index (default 0)
  NETT_BRAINS      brains_per_process (default 7 -> 112 envs, a square grid)
  NETT_BRAIN_OFFSET brain/seed index offset (default 0); a single-brain fleet packs
                   offsets 0..N-1 as N processes (folds the former orch_run_single.py)
  NETT_TRAIN_EPS   training episodes (default 2000)
  NETT_MAX_ENVS    max parallel envs, TOTAL across brains (default 112)
  NETT_RES         legacy SQUARE input_resolution (default 128). ⛔ It does NOT reach the
                   camera: repo B ObservationCfg.eye_resolution=(128, 80) wins whenever set,
                   so any value but 128 is refused. Use NETT_EYE_RES.
  NETT_EYE_RES     eye camera WIDTHxHEIGHT (default unset = repo B's 128x80). Must keep
                   16:10: the fisheye is isotropic, so the aspect ratio IS the vertical FOV.
  NETT_ROLLOUTS    per-brain transitions between PPO updates (default 8000)
  NETT_MINIBATCHES minibatches per update (default 16 -> batch 500 = 1 episode)
  NETT_CHECKPOINT_FREQ  timesteps between agent_{step}.pt snapshots (default: unset =
                   NO periodic checkpointing; a crash loses the whole run)
  NETT_STEPS       steps_per_episode (default 500)
  NETT_OUT_ROOT    campaign root (default ~/nett_campaign)
  NETT_RUN_NAME    pin the run directory name instead of stamping it with the wall
                   clock; REFUSES an existing directory. Only a resume needs this --
                   see examples/gate_a_resume.py.
  NETT_TEST_EPS    REPEATS PER TEST ROW (default 20) -- NOT episodes, NOT
                   episodes/condition. design.py:14 returns {imprint_cond:
                   num_test_ROWS} and brain.py:161 does {k: v * episodes["test"]},
                   so N per test condition = (rows for that condition) x this.
                   DesignSheet_Parsing_mov_MULTIBG has 56 test rows per imprint
                   condition (IOF 12, NF 12, BothFam 6, BothUnfam 24, Rest 2), so
                   the default 20 gives IOF/NF = 240 each, 1120 episodes/brain.
                   Verified against the Unity corpus: 1 repeat reproduces the
                   56-episode logs/ (12,12,6,24) and 5 reproduces the 280-episode
                   _eval/ (60,60,30,120), cell for cell.
  NETT_REWARD_TYPES comma-sep (default closeness); e.g. closeness,completeness
  NETT_DESIGN_SHEET / NETT_MEDIA_ROOT  override the per-experiment sheet/media defaults
  NETT_AUX_WEIGHT  unused for VICReg (driver forces 1.0 per user directive)
  NETT_AUX_WEIGHT_OVERRIDE  diagnostic only: replaces the model spec's aux_weight (default unset)
  NETT_LEARNING_RATE  PPO Adam learning rate (default 3e-4, the campaign value)
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
# ══════════════════════════════════════════════════════════════════════════════════════════
# WAVE 15 -- IS THE ViT's WEAKNESS ARCHITECTURAL? Six one-factor variants of VIT_CFG.
#
# ViT-CLTT-Ref is the only ViT arm that has ever cleared the incumbent on BU same-bg, and it
# did so INCONCLUSIVELY (0.5696, band edge 0.5170, n=7). These rows ask whether what is
# limiting it is the SHAPE of the encoder rather than the objective, so the aux, the imprint,
# the offsets and the episode budget are all held at ViT-CLTT-Ref's values and exactly one
# architectural knob moves per row.
#
# ⛔ EVERY embed_dim BELOW WAS SOLVED AGAINST A MEASURED BASELINE, NOT AGAINST THE "~699K"
# COMMENT ON VIT_CFG. That annotation is the THREE-channel figure; ViT-CLTT-Ref runs
# framestack=True, so its encoder takes 6 channels and is 804,320 parameters. Matching to
# 699K would have handed all six rows ~13% less capacity than the arm they are compared to --
# a capacity contrast wearing an architecture label. Each row's parameter count and its
# deviation from 804,320 are recorded on its line; the widest is +2.12%, inside the +1.8% /
# -1.6% this file already accepted for the QK ablations.
#
# ⚠ dim/head is NOT held constant and cannot be: with embed_dim re-solved per row, heads=4
# gives 18-40 dims per head across the set. Where a row moves head count (R1) that is the
# factor under test; elsewhere it is a consequence of the capacity match. Read no row as a
# clean test of dim/head.
VIT_CLTT_H8_CFG    = {**VIT_CFG, "num_heads": 8, "embed_dim": 144}   # ENC 804,320  +0.00%  d/head 18
VIT_CLTT_P8_CFG    = {**VIT_CFG, "patch_size": 8, "embed_dim": 160}  # ENC 789,952  -1.79%  d/head 40; 40 -> 160 tokens
VIT_CLTT_SP_CFG    = {**VIT_CFG, "pool": "spatial", "spatial_grid": (5, 4),
                      "spatial_reduce_dim": 16, "embed_dim": 132}    # ENC 797,704  -0.82%  d/head 33
VIT_CLTT_CONV_CFG  = {**VIT_CFG, "stem": "conv", "embed_dim": 156}   # ENC 821,337  +2.12%  d/head 39
VIT_CLTT_D6_CFG    = {**VIT_CFG, "depth": 6, "embed_dim": 108}       # ENC 793,556  -1.34%  d/head 27
VIT_CLTT_MLP4_CFG  = {**VIT_CFG, "mlp_ratio": 4.0, "embed_dim": 124} # ENC 818,416  +1.75%  d/head 31
# Row 8's encoder. dvs_polarity emits 2 channels and framestack doubles that to 4, against the
# RGB stack's 6, so embed_dim is re-solved for the narrower input: 152 -> 800,336 (-0.50%).
VIT_CLTT_DVS_CFG   = {**VIT_CFG, "embed_dim": 152}                   # ENC 800,336  -0.50%  d/head 38  (4 input channels)

VIT_NOQK_CFG = {**VIT_CFG, "embed_dim": 164, "attn_mode": "uniform"}   # ENCODER 706,368 at 128x80 (+1.8% vs ViT enc 693,728); agent 707,909
VIT_MIXER_CFG = {**VIT_CFG, "embed_dim": 160, "attn_mode": "mixer"}    # ENCODER 682,675 at 128x80 (-1.6% vs ViT enc 693,728); agent 684,216
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
# ⛔ ONE SOURCE FOR THE STACK DEPTH. The framestack wrapper's n_stack and every temporal
# encoder's `num_frames` describe THE SAME QUANTITY from two different config surfaces, and
# nothing checked that they agreed. While both were hardwired to 2 they could not disagree;
# NETT_FRAMESTACK_N makes disagreement reachable, and a mismatch silently scrambles time
# into colour WITHOUT changing the parameter count (see encoders/utils/temporal.py).
# Deriving both from this one value is the fix; the encoders still raise if it is defeated.
# ⚠ DEFAULT 2 -- every arm run before 2026-09-02 used 2, and this preserves that exactly.
_FRAMESTACK_N = int(os.environ.get("NETT_FRAMESTACK_N", "2"))

VIT_SP = {**VIT_CFG, "pool": "spatial", "spatial_grid": 4, "spatial_reduce_dim": 16}
VIT_SP_CFG = {**VIT_SP, "embed_dim": 136}                              # 696,000 at 128x128
VIT_MIXER_SP_CFG = {**VIT_SP, "embed_dim": 152, "attn_mode": "mixer"}  # 693,907 at 128x128
VIT_NOQK_SP_CFG = {**VIT_SP, "embed_dim": 156, "attn_mode": "uniform"} # 706,928 at 128x128

# ⭐ THE SPATIAL LINE AT THE LIVE EYE -- ADDED 2026-09-13, owner-authorised, ADDITIVE ON PURPOSE.
# ⛔ THE THREE CONFIGS ABOVE ARE NOT DEAD AND MUST NOT BE EDITED. They are how
# examples/probe_frozen_features.py rebuilds the archived SQUARE-eye arms at RES=128 to load the
# n=56 checkpoints that are on disk; Phases 14/15 came through that path, and the open QK question
# (per-agent coupling, p ~ 0.006 on the n already saved) still runs through it. Changing them in
# place revives training and silently closes that experiment --
# tests/test_vit_attention_ablation.py::test_square_eye_archive_arms_still_rebuild_for_offline_reanalysis
# exists to catch exactly that, and it caught it. ⇒ New eye, new labels; the archive keeps its own.
#
# WHY BOTH VALUES MOVE, and why either alone is wrong:
#   1. GRID. The pool divides the TOKEN grid, not the image. 80x128 at patch 16 gives 5x8 tokens
#      and `spatial_grid=4` does not divide 5, so construction raises. ⛔ NO SQUARE GRID EXISTS
#      HERE: 5 and 8 are coprime, so the only square divisor is 1 -- a global average, which is
#      exactly the CLS collapse this readout exists to avoid. (5, 4) is forced, and is the nearest
#      analogue to the archive's 4x4: 20 cells against 16.
#   2. WIDTHS. The archive embed_dims were matched to parameter count at 128x128. Carried to
#      80x128 unchanged they give 725,504 / 715,395 / 735,952 against the ViT baseline of 693,728
#      -- +4.58% / +3.12% / +6.09%. ⛔ A GRID-ONLY FIX WOULD RUN AND BE SILENTLY NON-COMPARABLE,
#      which is worse than not running: the matched-parameter design is the whole basis for
#      attributing a difference to the ABLATION rather than to capacity.
# Measured at the live 80x128 eye (insect, reproduced independently before the edit):
VIT_SP_RECT = {**VIT_CFG, "pool": "spatial", "spatial_grid": (5, 4), "spatial_reduce_dim": 16}
VIT_SP_RECT_CFG = {**VIT_SP_RECT, "embed_dim": 132}                             # 696,328 (+0.37%)
VIT_MIXER_SP_RECT_CFG = {**VIT_SP_RECT, "embed_dim": 148, "attn_mode": "mixer"} # 690,371 (-0.48%)
VIT_NOQK_SP_RECT_CFG = {**VIT_SP_RECT, "embed_dim": 148, "attn_mode": "uniform"}# 685,328 (-1.21%)
VIVIT_CFG = {
    "trainable": True, "features_dim": 512, "patch_size": 16,
    "embed_dim": 144, "depth": 3, "num_heads": 3, "mlp_ratio": 2.0,   # 144 -> ~699K
    "num_frames": _FRAMESTACK_N, "temporal_mode": "joint", "pool": "cls",
}

# ── OWNER-REQUESTED DIRECTIONS, 2026-09-03. Queued, NOT launched.
#
# SIZE SERIES on the ViT2F body. ONLY `embed_dim` varies; every other field is ViT2F's and
# framestack stays True, so the arms differ in PARAMETER COUNT ALONE. Counts MEASURED by
# constructing the encoder at the LIVE eye (128x80) with the real channel count:
#     embed_dim 108, 2 frames ->   511,597   (~500K target, +2.32%)
#     embed_dim 144, 2 frames ->   805,861   (ViT2F exactly as it stands)
#     embed_dim 164, 2 frames ->   996,221   (~1M target, -0.38%)
# ⚠ embed_dim must be divisible by num_heads (4) or MultiheadAttention asserts.
# ⚠ ViT2F IS NOT PARAMETER-MATCHED TO ViT: framestack doubles input channels, +110,592
#   params (+15.9%) at identical cfg. This series is matched to ViT2F, not to ViT.
# ⚠ examples/count_params.py will NOT reproduce these: it builds a SQUARE 128x128 obs and
#   hardcodes 2 frames, a second copy of the stack depth that cannot track NETT_FRAMESTACK_N.
#
# ⛔⛔ TWO DEFINITIONS OF "params" LIVE IN THIS FILE. Verified against 4 saved checkpoints
#   (final_agent.pt) on 2026-09-03 -- the built encoder reproduces the SAVED encoder exactly:
#       ENCODER  = the compact_vit tower alone
#       AGENT    = ENCODER + mean_layer(1024+2) + log_std(2) + value_head(512+1)
#                = ENCODER + 1,541   (constant while features_dim=512 and the action dim is 2)
#   The ViT-NoQK / ViT-Mixer / ViT-Sp comments below are ENCODER counts. Every count added
#   for the size and 3F series is written as ENCODER/AGENT so the two are never compared.
VIT_500K_CFG = {**VIT_CFG, "embed_dim": 108}   # enc 510,056 / agent 511,597 @ 128x80, 2 frames
VIT_1M_CFG   = {**VIT_CFG, "embed_dim": 164}   # enc 994,680 / agent 996,221 @ 128x80, 2 frames
# ── ViViT SIZE SERIES, owner-requested 2026-09-06. The 500K/1M ladder existed only on the
#    ViT2F body; ViViT was never given one. Counts MEASURED by constructing CompactViViT at the
#    LIVE eye (128x80) at the DEFAULT _FRAMESTACK_N=2, the same method and the same frame count
#    as the two lines above. Reproduced independently on a second node to the digit.
#    ⚠ embed_dim must stay divisible by num_heads=3 (ViT uses 4, ViViT uses 3).
#        embed_dim 144 ->   694,016   (ViViT exactly as it stands, already run)
#        embed_dim 177 ->   993,128   (-0.69%)  <- chosen
#        embed_dim 180 -> 1,022,912   (+2.29%)
#    ⛔ FRAME COUNT IS PROCESS-GLOBAL AND ENV-OVERRIDABLE HERE. At the default this arm is
#      frame-matched to ViT2F-1M, so the matched-budget contrast falls straight out of the
#      ladder -- but any launch that sets NETT_FRAMESTACK_N=3 for a 3F arm ALSO makes this one
#      3-frame (+embed_dim params, one temporal pos-emb) with no raise. Do not co-launch.
#    ⚠ A PEER NODE HARDCODES num_frames=2 with no _FRAMESTACK_N at all. Same value today, by a
#      DIFFERENT MECHANISM -- so agreement now is not agreement under a launch that sets the var.
VIVIT_1M_CFG = {**VIVIT_CFG, "embed_dim": 177}  # enc 993,128 @ 128x80, 2 frames -- size series
#
# THIRD-FRAME QUESTION. Adding a frame ALSO adds parameters, so "ViT3F" alone confounds the
# two. Both arms are provided and the parameter-matched one is the one that answers it:
#     embed_dim 144, 3 frames -> enc 914,912 / agent 916,453  frames AND +110,592 (CONFOUNDED)
#     embed_dim 132, 3 frames -> enc 800,696 / agent 802,237  frames alone, size at ViT2F -0.45%
VIT_3F_PM_CFG = {**VIT_CFG, "embed_dim": 132}  # enc 800,696 / agent 802,237 @ 128x80, 3 frames
#
# ── ⭐ CAPACITY LADDER, owner-requested 2026-09-21: "is there a sweet spot in parameter counts
#    for performance and learnability?" Pre-registered at NETT_Global_Workspace
#    `notes/commander/capacity-sweep-preregistration.md` BEFORE any row.
#
#    THE EXISTING SIZE SERIES IS NOT AN ANSWER TO IT. ViT2F 500K/800K/1M is a 1.95x span and
#    published a NULL; the `CNNcap*` arms span 12.7%, which is a NON-MEASUREMENT rather than a
#    null. A capacity question needs a span wide enough for capacity to bind, so this ladder is
#    built on the 3DCNN -- the family whose NF readings are near-exchangeable across arms
#    (§4dh.41e), i.e. the one where an arm-level contrast is cheapest to resolve.
#
#    ⚠ ONLY `conv_dim` varies. Every other field is "3DCNN"'s and framestack stays True, so the
#    rungs differ in PARAMETER COUNT ALONE. Counts MEASURED at the LIVE eye (128x80) at
#    _FRAMESTACK_N=2 by constructing the encoder through the arm's real body wrappers -- the
#    same method and the same tool that reproduces the three ViT2F lines above TO THE DIGIT.
#        conv_dim  26 -> enc   248,762 / agent   250,303   LOW
#        conv_dim  77 -> enc   695,981 / agent   697,522   BASE == "3DCNN" exactly as it stands
#        conv_dim 230 -> enc 2,037,638 / agent 2,039,179   HIGH      ⇒ 8.2x span LOW->HIGH
#
#    ⛔ OOM IS ACTIVATION-DRIVEN, NOT PARAMETER-DRIVEN (FINDINGS.md:17280): a 0.24M-param
#    UnityViT once OOM'd a 24 GiB card, and at ~0.79M `depth 6` ran healthy while `patch 8`
#    OOM'd. conv_dim 230 is ~3x BASE's channel width at every 3D conv, so HIGH's activation
#    footprint is NOT predicted by its parameter count. It is FIT-probed with
#    `mem_probe_condition.sh` on the node that will run it, flat across >=2 depths, BEFORE a
#    row leaves `proposed`. A parameter count is not a memory budget.
#
#    ⛔ EVERY CAPACITY ROW LAUNCHES WITH NETT_CHECKPOINT_FREQ=6250. The score-vs-step curve
#    this question really wants was never logged anywhere in the fleet (`eval_freq` is
#    10000000 = off on every arm, and it CHUNKS TRAINING so it must not be switched on here --
#    it would buy the curve at the price of making these arms non-comparable with the corpus
#    they are placed against). The curve therefore comes from OFFLINE retests of retained
#    checkpoints, which touch the training run not at all. Retention costs ~6 MB per
#    checkpoint per brain and CANNOT BE ADDED AFTERWARDS AT ANY PRICE.
_3DCNN_BASE_CFG = {"trainable": True, "features_dim": 512, "conv_dim": 77, "num_frames": _FRAMESTACK_N}
CNN3D_LOW_CFG   = {**_3DCNN_BASE_CFG, "conv_dim":  26}  # enc   248,762 / agent   250,303 @ 128x80, 2 frames
CNN3D_HIGH_CFG  = {**_3DCNN_BASE_CFG, "conv_dim": 230}  # enc 2,037,638 / agent 2,039,179 @ 128x80, 2 frames
#    ⭐ THE SAME TOTAL, IN A DIFFERENT PLACE. HIGH above is 92.5% ONE nn.Linear: `conv_dim`
#    scales the last conv AND the flatten->features head together (flatten = conv_dim*16), so
#    it cannot say whether a capacity effect is about the COUNT or about WHERE the parameters
#    are. This rung holds conv_dim at the BASE value, so flatten (1,232) and the head
#    (631,296) are BYTE-IDENTICAL to BASE, and widens the stack instead: conv share 7.5% ->
#    68.5% at a total 1.56% BELOW HIGH. HIGH vs this rung is a matched-parameter 2x2 on
#    placement. Counted from the edited encoder, not from a formula (prereg Amendment 7).
#    ⚠ ITS MEMORY IS NOT HIGH'S: +2,796 MiB vs +526, and 95% activations rather than
#    optimiser state, because mid_dim widens the largest feature map (32x20) while conv_dim
#    only touches the one just before a (4,4) pool. IT NEEDS ITS OWN FIT FLOOR.
CNN3D_CONV2M_CFG = {**_3DCNN_BASE_CFG, "stem_dim": 160, "mid_dim": 640}  # enc 2,005,933 / agent 2,007,474 @ 128x80, 2 frames
#
#    THE CROSS-ARCHITECTURE LEG. The owner asked whether a sweet spot "spans across
#    architectures". ViT2F already has 500K/800K/1M, but those arms are Sep-4-era shas, so
#    pairing a fresh HIGH against them is a sha-confounded contrast -- the exact confound
#    L181/L182 are on lion to separate. ⇒ this entry exists so a ViT2F leg can be run BASE+HIGH
#    AT ONE SHA if the 3DCNN ladder separates. It is NOT a fourth rung of the old ladder.
#    ⚠ embed_dim must stay divisible by num_heads (4).
VIT_2M_CFG   = {**VIT_CFG, "embed_dim": 256}   # enc 2,117,632 / agent 2,119,173 @ 128x80, 2 frames

# model label -> (encoder name, encoder_cfg, framestack?, reward(None|"CLTT"), aux(None|"vicreg"))
MODELS: dict[str, dict] = {
    "CNN":            dict(encoder="nature_cnn",      cfg={"trainable": True, "features_dim": 512, "conv_dim": 75},                  framestack=False),  # ~699K
    "3DCNN":          dict(encoder="compact_3dcnn",   cfg={"trainable": True, "features_dim": 512, "conv_dim": 77, "num_frames": _FRAMESTACK_N}, framestack=True),   # ~698K
    "SimCLR-CLTT":    dict(encoder="simclr_cltt",     cfg={"trainable": True, "features_dim": 512, "conv_dim": 77},                  framestack=True,  aux="cltt", aux_weight=1.0),  # ~702K
    # ⛔ framestack FLIPPED False->True 2026-09-02. CLTT's positive pair IS the temporal
    # pair (obs_t, obs_t+1), and the ONLY place a pair exists is the stacked channel axis --
    # there is no next_observations in the skrl PPO sample tuple. With framestack=False the
    # loss had no second frame to contrast, so the arm was incoherent as declared. It could
    # never have run either way: "cltt" was not registered in AUX_LOSSES until today.
    "ViT":            dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=False),
    "ViT2F":          dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=True),   # framestack-matched no-aux CONTROL for ViT-CLTT; same split as CNN/CNN2F
    "ViT-NoQK":       dict(encoder="compact_vit",     cfg=dict(VIT_NOQK_CFG),                                                        framestack=False),  # ENCODER ~706,368 at 128x80
    "ViT-Mixer":      dict(encoder="compact_vit",     cfg=dict(VIT_MIXER_CFG),                                                       framestack=False),  # ENCODER ~682,675 at 128x80
    # ⚠ SQUARE-EYE ARCHIVE ONLY -- for probe_frozen_features.py to rebuild the 2026-08 n=56
    # encoders at RES=128 square. NOT launchable at the 128x80 eye (raises at construction).
    "ViT-Sp":         dict(encoder="compact_vit",     cfg=dict(VIT_SP_CFG),                                                          framestack=False),  # 696K at 128x128
    "ViT-Mixer-Sp":   dict(encoder="compact_vit",     cfg=dict(VIT_MIXER_SP_CFG),                                                    framestack=False),  # 694K at 128x128
    "ViT-NoQK-Sp":    dict(encoder="compact_vit",     cfg=dict(VIT_NOQK_SP_CFG),                                                     framestack=False),  # 707K at 128x128
    # ⚠ THE THREE BELOW ARE THE ONLY -Sp MODELS THAT TRAIN AT THE LIVE 80x128 EYE. The three above
    # are ARCHIVE-ONLY: they exist so the saved square-eye checkpoints keep rebuilding, and they
    # raise at construction on a non-square eye. Do not "fix" them -- see the block above.
    "ViT-Sp-Rect":      dict(encoder="compact_vit", cfg=dict(VIT_SP_RECT_CFG),       framestack=False),  # 696K at 80x128
    "ViT-Mixer-Sp-Rect":dict(encoder="compact_vit", cfg=dict(VIT_MIXER_SP_RECT_CFG), framestack=False),  # 690K at 80x128
    "ViT-NoQK-Sp-Rect": dict(encoder="compact_vit", cfg=dict(VIT_NOQK_SP_RECT_CFG),  framestack=False),  # 685K at 80x128
    "ViT-CLTT":       dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=True,  aux="cltt", aux_weight=1.0),  # framestack True for the same reason as SimCLR-CLTT
    # Reference-faithful rebuild; cltt is retained unchanged as their paired incumbent.
    "SimCLR-CLTT-Ref": dict(encoder="simclr_cltt", cfg={"trainable": True, "features_dim": 512, "conv_dim": 77}, framestack=True, aux="cltt_ref", aux_weight=1.0),
    "ViT-CLTT-Ref":    dict(encoder="compact_vit", cfg=dict(VIT_CFG),                                            framestack=True, aux="cltt_ref", aux_weight=1.0),

    # ── WAVE 15: eight ViT rows, all holding ViT-CLTT-Ref's aux/imprint/offsets/budget fixed.
    # ⚠ Rows 1-6 move ONE architectural knob each; row 7 swaps the whole encoder for repo A's
    # Unity default; row 8 changes the INPUT rather than the encoder. Read them as one family
    # only for the shared question ("is the shape the limit?"), never as one comparison.
    #
    # R1: more heads at fixed width. Splits the SAME 144 dims into 8 subspaces of 18 instead of
    # 4 of 36. This is the only row whose parameter count is exactly the incumbent's, because
    # MultiheadAttention's projections do not depend on head count. More heads = more
    # simultaneous binding relations at lower per-relation resolution.
    "ViT-CLTT-Ref-H8":   dict(encoder="compact_vit", cfg=dict(VIT_CLTT_H8_CFG),   framestack=True, aux="cltt_ref", aux_weight=1.0),
    # R2: finer patches. 5x8=40 tokens -> 10x16=160. ⭐ THE ROW WITH THE CLEAREST PRIOR: at
    # patch 16 a chick-scale object can sit inside ONE token, so "parsing" has nothing to route
    # between. 4x the tokens is 16x the attention work -- budget wall-clock, not parameters.
    "ViT-CLTT-Ref-P8":   dict(encoder="compact_vit", cfg=dict(VIT_CLTT_P8_CFG),   framestack=True, aux="cltt_ref", aux_weight=1.0),
    # R3: spatial pooling instead of CLS. ⛔ CLS pooling reads ONE token, which is why the
    # ViViT arms scored EXACT chance on a supervised small-stimulus probe: a sub-patch object is
    # averaged ~64x into the global summary. This is the row that tests whether the ViT's
    # weakness is in the READOUT rather than the trunk -- if so, the trunk rows should all fail
    # and this one should not.
    "ViT-CLTT-Ref-Sp":   dict(encoder="compact_vit", cfg=dict(VIT_CLTT_SP_CFG),   framestack=True, aux="cltt_ref", aux_weight=1.0),
    # R4: convolutional stem (Xiao et al. 2021, "Early Convolutions Help Transformers See
    # Better"). A linear patch projection has no local inductive bias at all; the conv stem
    # gives the first layer edges and corners without giving the trunk convolution.
    "ViT-CLTT-Ref-Conv": dict(encoder="compact_vit", cfg=dict(VIT_CLTT_CONV_CFG), framestack=True, aux="cltt_ref", aux_weight=1.0),
    # R5: depth over width -- 6 blocks of 108 instead of 3 of 144. Binding is compositional, so
    # if the limit is the NUMBER of routing steps rather than their width this is the row that
    # moves. ⚠ Its 3-of-4 counterpart is R6: they trade the same budget in opposite directions.
    "ViT-CLTT-Ref-D6":   dict(encoder="compact_vit", cfg=dict(VIT_CLTT_D6_CFG),   framestack=True, aux="cltt_ref", aux_weight=1.0),
    # R6: width over depth -- mlp_ratio 2.0 -> 4.0 (the ViT paper's own ratio) at 3 blocks of
    # 124. Per-token capacity up, routing steps unchanged.
    "ViT-CLTT-Ref-MLP4": dict(encoder="compact_vit", cfg=dict(VIT_CLTT_MLP4_CFG), framestack=True, aux="cltt_ref", aux_weight=1.0),
    # R7: repo A's ORIGINAL Unity ViT -- vit.py -> vit_contrastive.LitClassifier -> SimpleViT --
    # vendored at nett_skrl/brain/encoders/unity_vit.py because vit_pytorch and lightning are
    # not installed and this venv is shared with running arms. Its defaults are kept verbatim,
    # including three upstream quirks that look like bugs (no Xavier init, a no-op fc=Identity,
    # a never-read projection head); see that file's docstring for each and for the two forced
    # deviations. Differs from every row above in FOUR ways at once (patch 4, dim 64, sincos
    # position, mean pooling) plus size.
    # ⛔ THE ONE ROW THAT IS NOT CAPACITY-MATCHED, DELIBERATELY: 237,888 encoder parameters,
    # 30% of the 804,320 the others carry. Matching it would stop it being the default, which
    # is the only thing it is for. A loss here is therefore NOT evidence about architecture --
    # it is confounded with capacity by construction. Only a WIN is interpretable.
    # ⚠ 640 tokens at patch 4 vs 40 at patch 16: ~256x the attention work at 1/3 the params.
    "UnityViT-CLTT-Ref": dict(encoder="unity_vit",   cfg={"trainable": True, "features_dim": 512}, framestack=True, aux="cltt_ref", aux_weight=1.0),
    # R8: the same encoder fed EVENTS instead of pixels. dvs_polarity replaces each frame with
    # a 2-channel (ON, OFF) brightness-change map, so framestack hands the encoder 4 channels
    # rather than 6 and embed_dim is re-solved to 152 to hold capacity.
    # ⛔ `pre=` IS LOAD-BEARING: the wrapper must sit INNERMOST, before framestack. It is itself
    # temporal -- it holds the previous frame -- so it must consume RAW frames and emit events
    # that framestack then stacks. Reversed it would difference two already-stacked tensors.
    # ⛔ THE ENV BLOCK IS NOT OPTIONAL. cltt_ref_aux derives its stack depth as
    # `channels // channels-per-frame`, which defaults to 3 for RGB; at 4 channels that is
    # 4 % 3 != 0 and the arm DIES on its first aux step. NETT_AUX_CLTT_CHANNELS_PER_FRAME=2 is
    # how the arm states the truth. The failure is loud, which is the only reason a default is
    # tolerable here -- see cltt_views.resolve_channels_per_frame for the residual hazard.
    # ⚠ WHAT THIS ARM CANNOT SEE: a stationary object against a stationary agent produces zero
    # events in both channels. Colour is gone by construction -- that is the point, since the
    # backgrounds are cued by brightness and colour while the objects are matched for them --
    # but so is every static cue, and the falsifier must be able to tell "motion is enough" from
    # "the encoder learned nothing".
    "ViT-CLTT-Ref-DVS":  dict(encoder="compact_vit", cfg=dict(VIT_CLTT_DVS_CFG),  framestack=True, pre=["dvs_polarity"], aux="cltt_ref", aux_weight=1.0),

    # ══════════════════════════════════════════════════════════════════════════════════════════
    # WAVE 17 -- OBJECT-CENTRIC PRESSURE ON THE TOKENS (owner request 2026-09-17).
    #
    # ⭐ THE ENCODER IS THE SAME OBJECT IN ALL SIX ROWS. Every entry below is
    # `cfg=dict(VIT_CFG)` -- byte-identical to "ViT-CLTT-Ref", the concurrent control (row 00) --
    # and framestack=True, so all six build the SAME 804,320-parameter encoder at 6 input
    # channels and the same 5x8=40 token grid at patch 16. ⛔ DO NOT RE-SOLVE embed_dim HERE.
    # Wave 15 re-matched embed_dim per row because it MOVED the encoder; this wave does not
    # touch it, and only the aux head differs. Resizing the trunk to "match capacity" would
    # introduce the very confound wave 15 had to work around. Head sizes (examples/count_params.py,
    # measured at the live 128x80 eye, 2-frame stack) are on each line: they are what the aux
    # optimizer group carries ON TOP of the shared encoder, and they are NOT matched to each
    # other -- an aux head is the method, not a capacity knob, and a row's head is as big as its
    # objective needs. cltt_ref's own 329,216-parameter projector is inside every one of them,
    # because each aux here is `cltt_ref + ONE new term` (brain/aux/with_cltt_ref.py).
    #
    # ⚠ ALL FIVE DIFFER FROM ROW 00 BY EXACTLY ONE THING: the `aux` key (and, for ViT-SlotsFG-Ego,
    # one env flag). Same experiment, imprint, offsets, budget, framestack, patch size and
    # aux_weight=1.0.
    #
    # R1 (row 01): CLTT on the PATCH TOKENS instead of only the CLS readout. DenseCL argmax
    # correspondence on a stop-grad EMA teacher, VICRegL top-gamma filter, NT-Xent at tau 0.5 --
    # the same contrastive family as cltt_ref, moved from one pooled vector to the token grid.
    # ⛔ THIS ROW CANNOT RUN UNDER NETT_AUX_STRICT=1: the anchors are selected with `gather`,
    # whose backward is index_add-shaped and has no deterministic CUDA kernel. The shipped path
    # (the relaxed aux backward in ppo_aux) is fine; the strict control is not available for it.
    # ⚠ It holds two extra token-grad views at NETT_AUX_PATCH_BATCH=512 -- the memory row to watch.
    "ViT-CLTT-Patch":    dict(encoder="compact_vit", cfg=dict(VIT_CFG), framestack=True, aux="cltt_patch",     aux_weight=1.0),  # aux head 470,016 (cltt_ref 329,216 + patch projector 140,800)
    # R2 (row 02): a VideoSAUR-style TEMPORAL TOKEN-AFFINITY target. The EMA teacher's
    # softmax(cos(u_i^t, u_j^{t+k})/tau) is the distribution; a per-token MLP on the student's
    # frame-t tokens supplies the log-probabilities. No slots, so nothing forces grouping -- the
    # honest scope is "VideoSAUR's target, not VideoSAUR's bottleneck".
    "ViT-PatchAffinity": dict(encoder="compact_vit", cfg=dict(VIT_CFG), framestack=True, aux="patch_affinity", aux_weight=1.0),  # aux head 382,536 (cltt_ref 329,216 + affinity MLP 53,320)
    # R3 (row 03): C3. An action-conditioned, CONTENT-INDEPENDENT routing A(a_bar) transports the
    # student's tokens; the residual against the EMA teacher's t+k tokens is what a global
    # ego-motion transport cannot explain. Content-independent on purpose: the stimulus video
    # loops deterministically, so a content-aware predictor would learn the object's own motion
    # and drive exactly the residual we want to zero.
    "ViT-EgoResidual":   dict(encoder="compact_vit", cfg=dict(VIT_CFG), framestack=True, aux="ego_residual",   aux_weight=1.0),  # aux head 516,784 (cltt_ref 329,216 + routing 104,192 + predictor 83,376)
    # R4 (row 04): slots over the 40 ViT tokens, SlotContrast's temporal slot contrast, a pixel
    # reconstruction and the separation entropy. ⛔ THE I77 FIX IS IN THIS TERM: the slot init at t
    # is a FIXED LEARNED vector and the init at t+k is predictor(slots_t), so "slot k at t" and
    # "slot k at t+k" are a correspondence by construction rather than two independent draws.
    # ⛔⛔ THE NAME IS "ViT-Slots" AND NOT "ViT-SlotsFG". DO NOT "RESTORE" IT. This row runs NO
    # foreground indicator: `fg_bce` and `stuff` -- Tian et al.'s fg/bg pair, against the ego
    # term's objectness -- sit inside `if self.ego is not None` in slot_fg_aux._core and never
    # execute at NETT_AUX_SLOTFG_EGO=0. Confirmed on the apparatus: 6 sentinels at fire_rate 0
    # across 7/7 brains over 6,720 aux steps. The separation entropy IS active (it is outside
    # that branch), so the row is "slot contrast + pixel reconstruction + separation entropy" --
    # and the model key travels into results/*.csv and the scoreboards, where this comment does
    # not follow it. A name that promises a mechanism the row does not run is a label that
    # survives every later reading of the numbers.
    # ⚠ The MODULE, the knob family and the aux key stay `slot_fg`: they are SHARED with row 05,
    # where the indicator does exist, and renaming them would break the knob-to-row mapping to
    # fix a label. ⚠ C4 (the background never varies) is why no symmetry breaker exists here, so
    # expect fg_centre_corr at chance as the NULL. L_sep's axis is UNVERIFIED against the paper.
    "ViT-Slots":         dict(encoder="compact_vit", cfg=dict(VIT_CFG), framestack=True, aux="slot_fg",        aux_weight=1.0),  # aux head 675,012 (cltt_ref 329,216 + slot_fg 345,796 at the screened convsbd/pixels/scale-2 default)
    # R5 (row 05): the composition -- the ego residual's per-token objectness as the FOREGROUND
    # TARGET for row 04's slots. This is the symmetry breaker C4 denies row 04.
    # ⛔⛔ THIS ENTRY IS BYTE-IDENTICAL TO "ViT-Slots" AND THAT IS NOT A MISTAKE -- IT IS ALSO
    # NOT SELF-SUFFICIENT. The composition is selected by NETT_AUX_SLOTFG_EGO=1, an ENV knob, and
    # MODELS has no per-model env field (nothing in this file reads one; see `spec.get` uses
    # around agent_factory). Exactly the ViT3F hazard documented below: launching this entry
    # WITHOUT NETT_AUX_SLOTFG_EGO=1 trains row 04 wearing row 05's label, and nothing raises.
    # The queue row carries the knob and the launcher must verify it in-band (the term emits
    # `slot_fg_used_ego` = 1.0 exactly when the ego branch is live -- read that, do not trust
    # the label). The name exists so the two rows are distinguishable in results/*.csv, which
    # keys on the model string.
    # ⚠ Its head is LARGER than row 04's (the ego routing + predictor live inside slot_fg's head
    # so they reach the optimizer), which is a property of the composition, not a capacity knob.
    # ⚠ THIS ONE KEEPS THE "FG": with the knob on, the indicator really runs, and the FG is the
    # whole difference between the two rows.
    "ViT-SlotsFG-Ego":   dict(encoder="compact_vit", cfg=dict(VIT_CFG), framestack=True, aux="slot_fg",        aux_weight=1.0),  # aux head 862,580 WITH NETT_AUX_SLOTFG_EGO=1 (= row 04's 675,012 + ego 187,568); 675,012 WITHOUT IT, i.e. row 04 wearing this label
    # ══════════════════════════════════════════════════════════════════════════════════════════

    # ⭐ 3DCNN-CLTT-Ref: the ONE-FACTOR cross of the two arms that currently top the two
    # chick-referenced parsing conditions. cfg is BYTE-IDENTICAL to "3DCNN" above -- the only
    # difference from that row is aux="cltt_ref", exactly as "ViT-CLTT-Ref" differs from "ViT".
    # ⛔ THE FRAMESTACK QUESTION IS ALREADY ANSWERED IN cltt_ref_aux AND NEEDS NO NEW MECHANISM.
    # The aux is encoder-agnostic: it needs only features_dim, _prepare_image and encode_prepared,
    # and Compact3DCNN inherits the last two from HWCFeatureExtractor -> NETTFeatureExtractor.
    # _prepare_image returns (B, C*T, H, W), so the aux's own `T = shape[1] // 3` discovers T=2
    # correctly, and its guard REFUSES offsets that are not multiples of T -- the default "2,4"
    # are, so the two positive views hold DISJOINT frame sets and no shared-frame shortcut exists.
    # ⚠ WHAT IS GENUINELY DIFFERENT AND MUST NOT BE GLOSSED: Compact3DCNN's first Conv3d has
    # kernel depth = num_frames and then .squeeze(2), so it COLLAPSES time inside the encoder.
    # The contrast therefore pulls together two disjoint MOTION SEGMENTS, [f(t-1),f(t)] against
    # [f(t+1),f(t+2)], where ViT-CLTT-Ref's pulls together two channel-stacked frame pairs through
    # a spatial backbone. Same objective, different quantity being made invariant. That is the
    # hypothesis under test, not an implementation detail -- do not report it as "CLTT on 3DCNN".
    "3DCNN-CLTT-Ref":  dict(encoder="compact_3dcnn",  cfg={"trainable": True, "features_dim": 512, "conv_dim": 77, "num_frames": _FRAMESTACK_N}, framestack=True, aux="cltt_ref", aux_weight=1.0),
    # ⭐ The MOTION half of the pair. Identical to 3DCNN-CLTT-Ref in every field but `aux`: the
    # views are whole temporally-offset stacks instead of a repeated still frame, so the Conv3d
    # gets real temporal gradient. Run BOTH -- one arm cannot separate "a contrastive objective
    # helps this encoder" from "a contrastive objective OVER MOTION helps this encoder".
    "3DCNN-CLTT-Stack": dict(encoder="compact_3dcnn",  cfg={"trainable": True, "features_dim": 512, "conv_dim": 77, "num_frames": _FRAMESTACK_N}, framestack=True, aux="cltt_ref_stack", aux_weight=1.0),
    # Schneider's single-frame temporal positives come from attached rollout memory.
    "SimCLR-CLTT-Schneider": dict(encoder="simclr_cltt", cfg={"trainable": True, "features_dim": 512, "conv_dim": 77}, framestack=False, aux="cltt_schneider", aux_weight=1.0),
    "ViT-CLTT-Schneider":    dict(encoder="compact_vit", cfg=dict(VIT_CFG), framestack=False, aux="cltt_schneider", aux_weight=1.0),
    "ViT+VICReg":     dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=False, aux="vicreg"),
    # Temporal-pair VICReg: identical objective and expander to `vicreg`; only the
    # view construction differs (aug(t) vs aug(t+k), not two augs of one frame).
    "ViT-VICReg-TT":   dict(encoder="compact_vit", cfg=dict(VIT_CFG), framestack=False, aux="vicreg_tt", aux_weight=1.0),
    # ── owner-requested, 2026-09-03. Queued, NOT launched. See the block above the MODELS table.
    "ViT2F-500K":     dict(encoder="compact_vit",     cfg=dict(VIT_500K_CFG),                                                        framestack=True),   # enc 510,056 / agent 511,597 -- size series
    "ViT2F-1M":       dict(encoder="compact_vit",     cfg=dict(VIT_1M_CFG),                                                          framestack=True),   # enc 994,680 / agent 996,221 -- size series
    # ⭐ CAPACITY LADDER 2026-09-21 -- see the CAPACITY LADDER block above. LOW/HIGH only: the
    # BASE rung is "3DCNN" itself, unchanged, so the reference needs no entry and cannot drift
    # from it. Comparisons are ENCODER counts; `agent` adds a constant 1,541.
    "3DCNN-250K":     dict(encoder="compact_3dcnn",   cfg=dict(CNN3D_LOW_CFG),                                                       framestack=True),   # enc   248,762 / agent   250,303 -- capacity LOW
    "3DCNN-2M":       dict(encoder="compact_3dcnn",   cfg=dict(CNN3D_HIGH_CFG),                                                      framestack=True),   # enc 2,037,638 / agent 2,039,179 -- capacity HIGH ⛔ FIT-PROBE BEFORE LAUNCH
    "3DCNN-2M-conv":  dict(encoder="compact_3dcnn",   cfg=dict(CNN3D_CONV2M_CFG),                                                    framestack=True),   # enc 2,005,933 / agent 2,007,474 -- capacity HIGH, CONV-wide at a matched total ⛔ FIT-PROBE SEPARATELY FROM 3DCNN-2M
    "ViT2F-2M":       dict(encoder="compact_vit",     cfg=dict(VIT_2M_CFG),                                                          framestack=True),   # enc 2,117,632 / agent 2,119,173 -- capacity HIGH, cross-arch leg
    # ⛔⛔ THE 3F ARMS ARE NOT SELF-SUFFICIENT. Stack depth is GLOBAL: _FRAMESTACK_N reads
    # NETT_FRAMESTACK_N (default 2) and there is NO per-model field for it. Launching either
    # entry without NETT_FRAMESTACK_N=3 silently yields a 2-FRAME run wearing a 3F label --
    # compact_vit takes frames as CHANNELS and declares no num_frames, so nothing raises.
    # The launcher must set it AND verify the resolved channel count in-band.
    # ⚠ It is process-global: any other framestacked arm in the same launch also becomes 3F.
    "ViT3F":          dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=True),   # enc 914,912 / agent 916,453 @3F -- CONFOUNDED with size
    "ViT3F-PM":       dict(encoder="compact_vit",     cfg=dict(VIT_3F_PM_CFG),                                                       framestack=True),   # enc 800,696 / agent 802,237 @3F -- param-matched to ViT2F
    "ViViT":          dict(encoder="compact_vivit",   cfg=dict(VIVIT_CFG),                                                           framestack=True),
    "ViViT+VICReg":   dict(encoder="compact_vivit",   cfg=dict(VIVIT_CFG),                                                           framestack=True,  aux="vicreg"),
    "ViViT-VICReg-TT": dict(encoder="compact_vivit", cfg=dict(VIVIT_CFG), framestack=True, aux="vicreg_tt", aux_weight=1.0),
    "ViViT-1M":       dict(encoder="compact_vivit",   cfg=dict(VIVIT_1M_CFG),                                                     framestack=True),   # enc 993,128 -- size series
    "GuessWhatMoves": dict(encoder="guess_what_moves", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75, "num_frames": _FRAMESTACK_N}, framestack=True),  # ~698K

    # ── MOTION-LOSS ARMS (added 2026-08-28 under the owner's (encoder, aux loss)
    # framing). Every one declares framestack=True: EoO and GWM need a temporal
    # pair, and there is NO next_observations in the PPO sample tuple, so the pair
    # comes from the stacked channel axis or it does not exist. Both losses RAISE
    # on a non-framestacked observation rather than silently comparing a frame with
    # itself -- which would make the objective identically zero while the run
    # logged aux=eoo.
    #
    # ★ The LOSS is the variable, so each is crossed with more than one encoder.
    # nature_cnn with framestack sees 6 channels; its first conv grows by ~6K params,
    # which is inside the band the campaign already runs (0.89x-1.12x).
    # ⛔ THE CONTROL FOR THE CNN2F FAMILY, added 2026-08-28. Without it the only available
    # contrast is `CNN2F+EoO - CNN`, which differs in TWO things at once: the auxiliary loss AND
    # framestacking (CNN is single-frame). Any gap then credits the loss with what may be the
    # extra frame. This is CNN2F+EoO with the aux removed and NOTHING else changed, so
    # `CNN2F+EoO - CNN2F` isolates the loss. Uses only pieces already proven in this campaign.
    "CNN2F":          dict(encoder="nature_cnn",      cfg={"trainable": True, "features_dim": 512, "conv_dim": 75}, framestack=True),
    "CNN2F+EoO":      dict(encoder="nature_cnn",      cfg={"trainable": True, "features_dim": 512, "conv_dim": 75},                  framestack=True,  aux="eoo", aux_weight=1.0),
    "CNN2F+GWM":      dict(encoder="nature_cnn",      cfg={"trainable": True, "features_dim": 512, "conv_dim": 75},                  framestack=True,  aux="gwm", aux_weight=1.0),
    "3DCNN+EoO":      dict(encoder="compact_3dcnn",   cfg={"trainable": True, "features_dim": 512, "conv_dim": 77, "num_frames": _FRAMESTACK_N}, framestack=True,  aux="eoo", aux_weight=1.0),
    "3DCNN+GWM":      dict(encoder="compact_3dcnn",   cfg={"trainable": True, "features_dim": 512, "conv_dim": 77, "num_frames": _FRAMESTACK_N}, framestack=True,  aux="gwm", aux_weight=1.0),
    "GWM+EoO":        dict(encoder="guess_what_moves", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75, "num_frames": _FRAMESTACK_N}, framestack=True,  aux="eoo", aux_weight=1.0),
    "GWM+GWM":        dict(encoder="guess_what_moves", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75, "num_frames": _FRAMESTACK_N}, framestack=True,  aux="gwm", aux_weight=1.0),

    # Policy sees the last RGB frame; the stack supplies the aux-only video pair.
    # Objectness shapes the host encoder, so CNN is the single-frame control.
    # Combined budget exceeds ~700K at the real eye; see docs/dual_stream_aux.md.
    "CNN+EoO-Dual": dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75, "input_frames": 1}, framestack=True, aux="eoo_dual", aux_weight=1.0),
    "CNN+GWM-Dual": dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75, "input_frames": 1}, framestack=True, aux="gwm_dual", aux_weight=1.0),

    # ── THE FAITHFUL MoTok ARM. ⛔ NOT an (encoder, aux) arm and it cannot be made
    # into one. In Unity MoTok holds its OWN model and OWN AdamW and its mask
    # MULTIPLIES THE OBSERVATION (seg_wrappers.py:213) -- it hands the policy a
    # SEGMENTED IMAGE rather than shaping the policy's representation. Those are
    # different hypotheses, and the 8 chick-level Unity brains in
    # /data/zlaborde/_gwm_train_rA10_parsing_motok are evidence for THIS one.
    # ⚠ NO aux= KEY ON PURPOSE. Routing MoTok through the aux interface would train a
    # parallel network while the policy encoder received nothing -- a silent no-op
    # that would log aux=motok and score like a scientific negative.
    # ⚠ framestack=False: MoTok is SINGLE-FRAME (get_masks ignores frame_next), and
    # its 55.5%-of-params dorsal stream is untrained. That dead pathway is the
    # physical form of the owner's "MoTok still needs a loss to train its motion
    # pathway" -- it is the finding, not a gap to paper over.
    # ⛔ SCORE THIS ARM PERMUTATION-INVARIANTLY. Nothing in a recon+VQ objective binds
    # slot 1 to the object, so with 2 slots and a symmetric init WHICH SLOT LANDS ON
    # THE OBJECT IS A PER-SEED COIN FLIP. A metric assuming slot 1 measures the coin
    # flip and returns a plausible mean over noise.
    "MoTok-Seg":      dict(encoder="nature_cnn",      cfg={"trainable": True, "features_dim": 512, "conv_dim": 75},                  framestack=False, seg="motok_seg"),
    "MoTok-Seg2F":    dict(encoder="nature_cnn",      cfg={"trainable": True, "features_dim": 512, "conv_dim": 75},                  framestack=True,  seg="motok_seg"),
    # Separate perception optimizer; mask multiplies the observation, no aux.
    # The honest control is CNN2F, which sees the same two-frame stack.
    # ★ THE SLOT-COUNT SERIES. The imprinting scene has 2 regions (chamber + object)
    # but the PARSING TEST has 3 (chamber + two monitors), so a segmenter frozen on a
    # binary habit cannot represent the scene it is tested on. Motion also separates
    # more than 2 groups even during imprinting, since parallax moves near and far
    # surfaces at different rates. At K>2 the mask rule switches to suppressing only
    # the background slot -- keeping ONE slot could delete one of the two test
    # alternatives outright. Read on Novel Familiar and Both Unfamiliar.
    "CNN2F+GWM-Seg":    dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75}, framestack=True, seg="gwm_seg", seg_after=True, seg_queries=2),
    "CNN2F+GWM-Seg-Q3": dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75}, framestack=True, seg="gwm_seg", seg_after=True, seg_queries=3),
    "CNN2F+GWM-Seg-Q5": dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75}, framestack=True, seg="gwm_seg", seg_after=True, seg_queries=5),

    # ── THE 14-CONDITION WAVE (2026-09-15). Both entries hold the encoder at nature_cnn
    # and framestack=True so every arm in that wave differs from `CNN2F` in the AUX LOSS
    # ALONE. The gradient-routing arm is NOT a separate model: NETT_DECOUPLE_ENCODER is an
    # environment flag on these same entries, so "standard" and "decoupled" are byte-identical
    # architectures and the contrast cannot be confounded with a construction difference.
    #
    # ⛔ CNN2F+CLTT-Ref IS THE FIRST CLTT ARM ON A CNN TRUNK. Every existing CLTT entry rides
    # compact_vit or simclr_cltt, so `cltt_ref - CNN2F` did not exist and the family could
    # only ever be compared across a trunk change. cltt_ref's views are the CURRENT RGB frame
    # repeated across the stack slots (cltt_views.current_frame_stack), which is benign for a
    # 2-D CNN over the channel stack and would be degenerate for a motion encoder -- nature_cnn
    # is the former, so this pairing is sound where 3DCNN+CLTT would not be.
    "CNN2F+CLTT-Ref":  dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75}, framestack=True, aux="cltt_ref", aux_weight=1.0),
    # ⚠ ~803K with the SlotContrast head on nature_cnn (trunk 666,272 + 137,025), against a
    # ~700K SOFT budget, and the EMA target doubles trunk ACTIVATION memory on top. Declared
    # here rather than discovered at launch: this arm is the wave's memory outlier and the
    # reason the memory probe exists.
    # Luminance-standardised control. `pre` puts lumnorm INNERMOST -- before framestack --
    # so the policy stacks standardised frames rather than standardising a stack.
    "CNN2F+LumNorm":   dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75}, framestack=True, pre=("lumnorm",)),
    "CNN2F+SlotContrast": dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75}, framestack=True, aux="slot_contrast", aux_weight=1.0),
}

# ⛔ `_3DCNN_BASE_CFG` above is a SECOND COPY of "3DCNN"'s cfg literal, and a second copy is a
# drift hazard wearing a convenience's clothes: edit one, and the capacity ladder quietly stops
# being one-factor while every comment still says it is. This fails the IMPORT -- on every arm,
# before a card is touched -- the moment the two disagree. A comment asserting they match is not
# a check; this is.
assert MODELS["3DCNN"]["cfg"] == _3DCNN_BASE_CFG, (
    "capacity ladder is no longer one-factor: 3DCNN's cfg and _3DCNN_BASE_CFG have drifted.\n"
    f"  MODELS['3DCNN']['cfg'] = {MODELS['3DCNN']['cfg']}\n"
    f"  _3DCNN_BASE_CFG        = {_3DCNN_BASE_CFG}")
# ⛔ EACH RUNG DECLARES THE ONE AXIS IT IS ALLOWED TO MOVE, and the assertion is per-rung
# rather than a single shared set. A shared set would let a HEAD rung silently acquire a CONV
# key (or the reverse) and still pass -- which is the whole point of the 2x2 gone.
_RUNG_AXIS = {
    "3DCNN-250K":    ({"conv_dim"},            CNN3D_LOW_CFG),
    "3DCNN-2M":      ({"conv_dim"},            CNN3D_HIGH_CFG),
    "3DCNN-2M-conv": ({"stem_dim", "mid_dim"}, CNN3D_CONV2M_CFG),
}
for _lbl, (_axis, _cfg) in _RUNG_AXIS.items():
    _diff = {k for k in set(_cfg) | set(_3DCNN_BASE_CFG)
             if _cfg.get(k) != _3DCNN_BASE_CFG.get(k)}
    assert _diff == _axis, (
        f"{_lbl} differs from the 3DCNN base in {sorted(_diff)}, not in {sorted(_axis)} alone -- "
        "the capacity rung would confound width with whatever else changed.")
    assert MODELS[_lbl]["cfg"] == _cfg, (
        f"MODELS[{_lbl!r}] does not carry the cfg its rung declares -- the label and the\n"
        "  arithmetic in the comment above would describe different models.")
# ⛔ A CONV RUNG MUST NOT MOVE THE HEAD. flatten = conv_dim*16, so an equal conv_dim is an
# equal head; asserted rather than trusted, because it is the claim the 2x2 rests on.
assert CNN3D_CONV2M_CFG["conv_dim"] == _3DCNN_BASE_CFG["conv_dim"], (
    "3DCNN-2M-conv moved conv_dim, so its head is NOT byte-identical to BASE and it is no "
    "longer a matched-parameter placement contrast.")
del _lbl, _axis, _cfg, _diff

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


def segmentation_wrappers(spec: dict) -> list[str]:
    """Body order is innermost first; only pair-based segmenters go last.

    ⛔ THIS IS THE ONLY PLACE BODY WRAPPERS ARE CHOSEN, AND THERE IS NO ENVIRONMENT HOOK.
    `NETT_BODY_WRAPPERS` does not exist and never has -- a queue row naming it would launch
    happily, log the experimental label, and run the CONTROL, which is a difference with no
    symptom. A wrapper reaches an arm by being declared in that arm's MODELS entry, so the
    model NAME always says which wrappers ran.

    `pre` holds wrappers that must sit INNERMOST, before framestack and before any
    segmenter -- `lumnorm` is one: it standardises each raw frame, and every downstream
    consumer (segmenters included) must see the standardised version, not the raw one.
    """
    pre = list(spec.get("pre", ()))
    seg = [spec["seg"]] if spec.get("seg") else []
    stack = ["framestack"] if spec["framestack"] else []
    body = stack + seg if spec.get("seg_after") else seg + stack
    return pre + body


def _slug(s: str) -> str:
    return s.replace("+", "").replace("-", "").replace(" ", "")



def _camera_fov_override() -> float | None:
    """The DELIBERATE aperture override, or ``None`` to defer.

    ``None`` is the normal answer. It means the aperture comes from its single
    declaration, ``nett_isaac.nett_env_cfg.ObservationCfg.fov`` (300.0 deg), rather than
    from a number typed into this driver. See the note in ``main`` for why a literal here
    is a trap rather than a convenience.
    """
    raw = os.environ.get("NETT_CAMERA_FOV", "").strip()
    return float(raw) if raw else None


def _resolved_camera_fov() -> float:
    """The aperture that will ACTUALLY run, for labelling and provenance.

    Resolved rather than assumed: an override wins, otherwise the value is read back from
    the one declaration. Never hardcode this into a tag -- a tag carrying the number
    someone typed instead of the number that ran is precisely how a 150 deg corpus came to
    be reported as 300 deg.
    """
    override = _camera_fov_override()
    if override is not None:
        return override

    # KIT-FREE ON PURPOSE. ``nett_env_cfg`` imports ``isaaclab``, which cannot be imported
    # before ``SimulationApp`` exists -- and this runs while the wandb tags are being
    # built, long before the boot. That import barrier is *why* the aperture was hardcoded
    # here in the first place. So read the declaration from its source text instead.
    #
    # It FAILS LOUDLY if the declaration moves. It must never fall back to a literal: a
    # silent fallback is exactly the failure being fixed.
    import re
    from nett_isaac import lens as _lens
    cfg_src = Path(_lens.__file__).with_name("nett_env_cfg.py")
    text = cfg_src.read_text(encoding="utf-8")
    cls = re.search(r"^class ObservationCfg\b.*?(?=^class |\Z)", text, re.S | re.M)
    if cls is None:
        raise SystemExit(f"cannot locate class ObservationCfg in {cfg_src}: the aperture "
                         f"declaration moved. Refusing to guess the field of view.")
    m = re.search(r"^\s*fov\s*:\s*float\s*=\s*([0-9.]+)", cls.group(0), re.M)
    if m is None:
        raise SystemExit(f"cannot read ObservationCfg.fov from {cfg_src}: the aperture "
                         f"declaration changed shape. Refusing to guess the field of view.")
    declared = float(m.group(1))

    # ``lens.DEFAULT_FOV_H_DEG`` is an INDEPENDENT literal -- nett_env_cfg does not import
    # it -- so the two can drift, which is the documented failure mode.
    if float(_lens.DEFAULT_FOV_H_DEG) != declared:
        # ⛔ NOT ``log``: that name is a LOCAL of main(), so this line raised NameError -- and it
        # raised on the ONE path the warning exists to report. Found by seat:insect, who forced the
        # drift rather than trusting a clean default run. The safety check crashed exactly when it
        # was the only thing standing between us and a second silent aperture divergence.
        logging.getLogger("nett.campaign").warning(
            "APERTURE DECLARATIONS DISAGREE: nett_env_cfg.ObservationCfg.fov=%g but "
            "lens.DEFAULT_FOV_H_DEG=%g. These are independent literals; LensSpec uses the "
            "cfg value when present and lens's only as a fallback. Using %g.",
            declared, float(_lens.DEFAULT_FOV_H_DEG), declared)
    return declared



_ENV_TRUE = frozenset(("1", "true", "yes", "on"))
_ENV_FALSE = frozenset(("", "0", "false", "no", "off"))


def _env_flag(name: str, default: bool = False) -> bool:
    """Boolean env var that FAILS LOUD on anything it does not recognise.

    ⛔ THE FIRST VERSION OF THIS WAS FAIL-OPEN -- ``not in ("", "0", "false", "no", "off")``
    -- so ``0.0``, ``disabled``, ``none``, ``O``, ``2`` and the typo ``flase`` all read as
    TRUE. seat:lion caught it in review. A deny-list makes every misspelling mean "on",
    which is the silent-wrong-default failure this fleet has now hit three times in one
    night from one side or the other.
    ⭐ A bare allow-list would only move the silence to the other side: ``flase`` would
    then silently mean OFF. So this raises instead. The variable is new, nothing legacy
    sets it, and a typo'd flag should stop a launch rather than quietly pick either answer.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _ENV_TRUE:
        return True
    if v in _ENV_FALSE:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a recognised boolean. "
        f"Use one of {sorted(_ENV_TRUE)} or {sorted(_ENV_FALSE)}."
    )


def eye_resolution_override(res: int):
    """The eye camera ``(width, height)`` from ``NETT_EYE_RES``, or ``None`` = repo B's default.

    ⛔ WHY THIS EXISTS: ``NETT_RES`` is copied to ``observation.input_resolution`` only. Repo B's
    ``ObservationCfg.eye_resolution`` defaults to ``(128, 80)`` and ``lens.eye_resolution`` returns it
    whenever it is set, so ``input_resolution`` never reaches the camera. A ``NETT_RES=256`` run
    rendered 128x80 while its manifest and log said ``res=256`` -- so any value but 128 is refused.
    """
    raw = os.environ.get("NETT_EYE_RES", "").strip()
    if not raw:
        if res != 128:
            raise SystemExit(
                f"NETT_RES={res} does not reach the camera: repo B's ObservationCfg.eye_resolution "
                "(128x80) wins over input_resolution. Set NETT_EYE_RES=WIDTHxHEIGHT (16:10) instead.")
        return None
    try:
        w, h = (int(v) for v in raw.lower().split("x"))
    except ValueError:
        raise SystemExit(f"NETT_EYE_RES={raw!r}: expected WIDTHxHEIGHT, e.g. 256x160") from None
    if w <= 0 or h <= 0 or w * 10 != h * 16:
        raise SystemExit(
            f"NETT_EYE_RES={raw!r}: must be positive and 16:10 (e.g. 256x160, 448x280). The fisheye "
            "is isotropic, so another aspect ratio changes the vertical field of view, not only "
            "the pixel count.")
    return (w, h)

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
    eye_res = eye_resolution_override(res)
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

    # The aux loss is read from the environment by agent_factory; set it here so the
    # whole process tree (incl. brain subprocesses) inherits it.
    #
    # This dispatch was `== "vicreg"` until 2026-08-28, which meant an arm declared
    # with any other aux fell to the else branch and had NETT_AUX_LOSS popped -- so
    # every aux except vicreg was unreachable through this launcher, and `export
    # NETT_AUX_LOSS=...` could not fix it because defeating a shell-set value is the
    # pop's whole purpose. Generalised so `"aux": <kind>` in a MODELS spec works for
    # any kind agent_factory/ppo_aux know about (simclr, vicreg, cltt, ...).
    #
    # aux_weight defaults to 1.0, preserving the standing VICReg directive; declare
    # `"aux_weight": <float>` in the spec to override per arm. Forced here so an
    # inherited campaign-wide default cannot override the arm's own setting.
    # This line of work tests ENCODERS AND LOSSES, never rewards. Every arm runs the
    # environment's own extrinsic reward, unmodified. A `reward` key in a MODELS spec
    # would route the arm through IntrinsicRewardAdapter and add a per-step bonus to
    # that extrinsic signal, which would make its scores incomparable with every other
    # arm's. Refuse it here rather than let it be discovered in a scoreboard.
    if spec.get("reward"):
        raise ValueError(
            f"arm {model!r} declares reward={spec['reward']!r}. Custom rewards are not "
            f"used in this campaign -- express the objective as an auxiliary loss "
            f'(\'"aux": <kind>\') so it shapes the encoder instead of the reward.'
        )

    # The seg wrapper reads its slot count from the environment, and the launcher
    # builds body wrappers with the env ALONE -- so the arm's choice has to be
    # exported here or every arm silently runs the default. POP when unset: a stale
    # NETT_SEG_QUERIES from a shared shell would otherwise decide the arm.
    if spec.get("seg_queries") is not None:
        os.environ["NETT_SEG_QUERIES"] = str(spec["seg_queries"])
    else:
        os.environ.pop("NETT_SEG_QUERIES", None)

    aux_kind = spec.get("aux")
    if aux_kind:
        os.environ["NETT_AUX_LOSS"] = str(aux_kind)
        # NETT_AUX_WEIGHT_OVERRIDE: replaces the spec's aux_weight for a DIAGNOSTIC arm (2026-09-21
        # owner diagnostic: aux-weight dose on a fixed model). Unset = the spec's value, unchanged.
        # The spec stays the record of the model; the override is recorded by the launcher's environ
        # capture and must be named in the queue row.
        os.environ["NETT_AUX_WEIGHT"] = os.environ.get("NETT_AUX_WEIGHT_OVERRIDE") or str(spec.get("aux_weight", 1.0))
    else:
        # ensure no stray aux leaks in from a shared shell. BOTH keys: popping the
        # kind alone leaves a stale weight behind, which is inert (kind gates first
        # in agent_factory) but makes "weight set, kind absent" indistinguishable
        # from a real misconfiguration for anything that tries to warn about it.
        os.environ.pop("NETT_AUX_LOSS", None)
        os.environ.pop("NETT_AUX_WEIGHT", None)

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
    # res*res*3 over-counts the 128x80 eye (kept as-is: it decides the buffer device for every
    # existing arm). An explicit NETT_EYE_RES is sized by what it renders.
    _obs_bytes = eye_res[0] * eye_res[1] * 3 if eye_res else res * res * 3
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

    # NETT_RUN_NAME: pin the run directory instead of stamping it with the wall clock.
    # ⛔ EXISTS FOR ONE REASON: a resume has to place checkpoints under
    # <out>/<name>/<imprint>/wandb_runs/brain_i/checkpoints BEFORE training starts, and
    # brain.py reads them there. With the name minted from datetime.now() inside this
    # process, the only way to seed them is to guess the second or race the mkdir --
    # and a resume that silently loses the race trains from RANDOM WEIGHTS while
    # reporting success (load_latest_checkpoints logs and continues; nothing raises).
    # See examples/gate_a_resume.py, its only caller.
    # ⚠ Inert unless set, and it REFUSES an existing directory: run_dir.mkdir uses
    # exist_ok=True, so a reused name would merge two runs' logs into one tree with no
    # error and no way to tell them apart afterwards.
    _forced_name = os.environ.get("NETT_RUN_NAME", "").strip()
    if _forced_name:
        name = _forced_name[:63]
        # ⛔ THE PREDICATE IS "HOLDS RUN OUTPUT", NOT "EXISTS", AND THE DIFFERENCE IS THE
        # WHOLE POINT. The first version refused on mere existence and so refused the ONLY
        # caller this knob has: gate_a_resume.py must create this directory to seed
        # checkpoints into it before the run starts, so the guard fired on its own client
        # 2 seconds in. Existence is not the hazard -- merging two runs' LOGS is -- and a
        # guard whose predicate is not the thing it protects both blocks the legitimate case
        # and leaves the illegitimate one reachable by any other route.
        _existing = out / name
        _output = [p for p in (_existing / "campaign_timing.json", *_existing.glob("*/logs"),
                               *_existing.glob("logs")) if p.exists()]
        if _output:
            raise SystemExit(
                f"NETT_RUN_NAME={name!r} already holds run output under {out}: "
                f"{[str(p.relative_to(_existing)) for p in _output]}. Refusing: the run "
                f"directory is created with exist_ok=True, so continuing would merge this "
                f"run's logs into that one silently. Choose another name or move it aside."
            )
        log.warning("NETT_RUN_NAME pins the run directory to %s (no wall-clock stamp)", name)

    # Resolved once: the tag list below cannot reference `brain` while `brain` is
    # still being constructed (self-reference -> UnboundLocalError).
    _rollouts = int(os.environ.get("NETT_ROLLOUTS", "8000"))
    # Resolved BEFORE the wandb tags, which consume camera_fov_resolved.
    camera_fov_override = _camera_fov_override()
    camera_fov_resolved = _resolved_camera_fov()

    brain: dict = {
        "algorithm": "PPO",
        "encoder": spec["encoder"],
        "encoder_cfg": spec["cfg"],
        # ⛔ WITHOUT THIS KEY THERE IS NO PERIODIC CHECKPOINTING AT ALL. Brain.checkpoint_freq
        # defaults to None (schema.json, brain.py:87); experiment.py:49 turns that into
        # checkpoint_interval=0; and skrl gates its in-loop save on `> 0` twice
        # (agents/torch/base.py:233 and :469), so agent_{timestep}.pt is NEVER written.
        # run_recorder.py:97 then becomes the ONLY write, at the very end of training, and a
        # run that dies before it loses 100% of its training. Measured 2026-08-31 on
        # host:chicken: 0 agent_*.pt and 0 best_agent.pt against 35 final_agent.pt, and three
        # truncated arms across two nodes lost ~38.5k+ units for exactly this reason.
        # ⚠ Unit is TIMESTEPS, not episodes. skrl's own "auto" rule is timesteps//10
        # (base.py:232), which gives ~10 snapshots.
        # ⚠ SAFE AS OF 2026-07-31: this does NOT re-introduce the 33-subprocess split, because
        # _training_boundaries(total, *, eval_freq) does not take checkpoint_freq at all
        # (task_runner.py:503-524). Structural, not a promise.
        # ⚠ DEGRADES SILENTLY IF THE RUN IS CHUNKED (seat:lion, verified here 2026-08-31).
        # skrl restarts its timestep at 0 in every chunk (sequential.py:84) and names the file
        # agent_{timestep}.pt into an experiment_dir with NO chunk component, so chunk 2
        # OVERWRITES chunk 1 and you get fewer snapshots than requested, with no error.
        # Chunking comes from eval_freq alone. campaign_train.py:482 pins eval_freq=10_000_000
        # against a 62,500-timestep run, so _training_boundaries returns ONE boundary and the
        # run is a single chunk -- executed, not read: (62500, 1e7)->1, (125000, 1e7)->1,
        # while (62500, 6250)->10 and (62500, 20000)->4 prove the function can chunk.
        # ⛔ If you ever lower eval_freq below the run length, this knob under-delivers.
        # Default None = inert; existing behaviour is unchanged unless the knob is set.
        "checkpoint_freq": (int(os.environ["NETT_CHECKPOINT_FREQ"])
                            if os.environ.get("NETT_CHECKPOINT_FREQ") else None),
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
            # NETT_LEARNING_RATE: PPO Adam learning rate. Default 3e-4 is the campaign value and is
            # unchanged; the knob exists for the 2026-09-21 owner diagnostic (ViT-family policies show
            # 2-4x the KL and clip fraction of the CNNs at the shared 3e-4). Unset = campaign default.
            "learning_rate": float(os.environ.get("NETT_LEARNING_RATE", "3e-4")),
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
            # ⛔ WAS HARDCODED "online". skrl calls wandb.init() unconditionally when
            # experiment.wandb is truthy, and nett_skrl/brain/experiment.py:97 passes this
            # `mode` straight into wandb_kwargs -- which OVERRIDES the WANDB_MODE env var.
            # On a box with no API key that raises UsageError AFTER Kit has booted and the
            # env is built, and the driver surfaces it as the unrelated-looking
            # "RuntimeError: Mode train subprocess failed with exit code 1", with the real
            # cause ~100 lines earlier in the log. Same env hook and same name as
            # campaign_retest.py:71; the DEFAULT is unchanged, so nothing silently moves.
            "mode": os.environ.get("NETT_WANDB_MODE", "online"),
            "project": f"nett-{exp}-replication",
            "tags": [model, spec["encoder"], "ppo", f"{brains}brain", exp, imprint,
                     f"rollouts={_rollouts}", f"steps={os.environ.get('NETT_STEPS', '500')}",
                     f"{train_eps}ep",
                     # ⚠ DERIVED, never a literal. A hardcoded "fov=150" tag is what hid the
                     # override for 19 days: the run announced the number someone typed.
                     "closeness-only", f"fov={camera_fov_resolved:g}", "2d-action", f"res={res}",
                     # Provenance on the run itself, not only in campaign_timing.json:
                     # scores under the two protocols do not pool, and a wandb view that
                     # mixes them silently averages two different measurements.
                     f"eval={'stochastic' if eval_stochastic else 'mean'}",
                     # Tag the actual aux kind, not just vicreg -- otherwise the run
                     # name cannot record WHICH aux ran, and two arms differing only
                     # in their aux loss are indistinguishable in wandb.
                     *( [str(spec["aux"])] if spec.get("aux") else [] )],
        },
    }

    config: dict = {
        "name": name,
        "environment": {
            "design_sheet": os.environ.get("NETT_DESIGN_SHEET", sheet),
            "media_root": os.environ.get("NETT_MEDIA_ROOT", media),
            "conditions": [imprint],
            "headless": True,
            "input_resolution": res,
            # ⛔ camera_fov is DELIBERATELY ABSENT. ``Environment``'s signature carries the
            # reason (environment.py: "None MEANS USE REPO A'S NETTEnvCfg.observation.fov.
            # DO NOT PUT A NUMBER HERE"), because ``_ENV_CFG_FIELDS`` copies any non-None
            # value straight over repo A. A literal 150.0 sat here from d1446a8 (2026-06-16)
            # until 2026-08-31 and silently halved the aperture of every arm this driver ran,
            # while the wandb tag below said "fov=150" because it too was typed rather than
            # derived. A deliberate override now goes through NETT_CAMERA_FOV and is
            # INJECTED AFTER construction, so "absent" keeps meaning "deferred".
            "reward_types": reward_types,
            "enable_neck_flexion": False,
            "enable_lateral_bending": False,
            # TRAIN-PHASE TRAJECTORY LOGGING. repoA gates the per-step CSV on
            # nett_env_cfg.train_step_logging (default False) because _log_step_batch is a
            # per-step GPU->CPU handover; the result is that every train_*.csv on this fleet
            # is HEADER-ONLY (measured: 34 of 34 on chicken, 38 of 40 on lion). That makes
            # the agent's self-generated viewing geometry during imprinting UNMEASURABLE,
            # which is currently the leading hypothesis for the cross-node pose difference:
            # identical media, identical lens coefficients and flat acuity, but each node
            # discriminates the objects at a different pose.
            # ⛔ DEFAULT UNCHANGED. Absent the env var this is False, exactly as before, so
            # no in-flight arm and no existing launcher changes behaviour. Opt in per launch.
            "train_step_logging": _env_flag("NETT_TRAIN_STEP_LOGGING"),
        },
        "brain": brain,
        # ⚠ ORDER IS LOAD-BEARING. body.py:53 applies these in list order, each
        # wrapping the previous, so index 0 is INNERMOST and sees the raw frame.
        # MoTok masks BEFORE framestack; GWM needs the raw pair AFTER framestack.
        "body": {"wrappers": segmentation_wrappers(spec)},
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

    # ⚠ The old `ssl=%s` field logged spec["reward"], the CUSTOM intrinsic reward. Since
    # the guard above raises on any reward key, that field could only ever print None --
    # next to `rewards=`, which reads as "this run has no reward". IT DOES: the extrinsic
    # closeness reward is what the policy is trained on and always was. The custom reward
    # was only ever an ADDITION to it. A field that prints None forever, beside a field
    # that carries the real answer, is worse than no field.
    # aux/aux_weight are read back from the ENVIRONMENT, not from `spec`, because the env
    # is what agent_factory.py actually reads. Logging the spec would report the intent;
    # this reports the effect, and the two differ exactly when something silently disables.
    log.info("MODEL=%s EXP=%s IMPRINT=%s device=%d brains=%d off=%d res=%d envs=%d mb=%s "
             "rewards=%s aux=%s aux_weight=%s -> %s",
             model, exp, imprint, device, brains, offset, res, max_envs,
             os.environ.get("NETT_MINIBATCHES", "16"), reward_types,
             os.environ.get("NETT_AUX_LOSS", "none"),
             os.environ.get("NETT_AUX_WEIGHT", "0"), out)
    log.info("hidden_sizes=%s entropy=%s eval=%s", hidden_sizes, entropy,
             "stochastic" if eval_stochastic else "mean")
    t0 = time.time()
    if eye_res is not None:
        config["environment"]["eye_resolution"] = list(eye_res)
    log.info("eye_resolution=%s (%s)", "x".join(map(str, eye_res)) if eye_res else "128x80",
             "explicit NETT_EYE_RES" if eye_res else "deferred to nett_env_cfg.ObservationCfg.eye_resolution")
    if camera_fov_override is not None:
        config["environment"]["camera_fov"] = camera_fov_override
        log.warning("camera_fov OVERRIDDEN to %g deg, away from the declared aperture "
                    "(nett_env_cfg.ObservationCfg.fov). This is a deliberate control arm "
                    "and does NOT pool with runs at the declared aperture.", camera_fov_override)
    log.info("camera_fov=%g (%s)", camera_fov_resolved,
             "explicit NETT_CAMERA_FOV override" if camera_fov_override is not None
             else "deferred to nett_env_cfg.ObservationCfg.fov")

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
         "eye_resolution": list(eye_res) if eye_res else None,
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
