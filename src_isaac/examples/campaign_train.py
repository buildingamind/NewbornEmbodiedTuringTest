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
  NETT_CHECKPOINT_FREQ  timesteps between agent_{step}.pt snapshots (default: unset =
                   NO periodic checkpointing; a crash loses the whole run)
  NETT_STEPS       steps_per_episode (default 500)
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
    "num_frames": _FRAMESTACK_N, "temporal_mode": "joint", "pool": "cls",
}

# ⛔ ONE SOURCE FOR THE STACK DEPTH. The framestack wrapper's n_stack and every temporal
# encoder's `num_frames` describe THE SAME QUANTITY from two different config surfaces, and
# nothing checked that they agreed. While both were hardwired to 2 they could not disagree;
# NETT_FRAMESTACK_N makes disagreement reachable, and a mismatch silently scrambles time
# into colour WITHOUT changing the parameter count (see encoders/utils/temporal.py).
# Deriving both from this one value is the fix; the encoders still raise if it is defeated.
# ⚠ DEFAULT 2 -- every arm run before 2026-09-02 used 2, and this preserves that exactly.
_FRAMESTACK_N = int(os.environ.get("NETT_FRAMESTACK_N", "2"))

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
    "ViT-NoQK":       dict(encoder="compact_vit",     cfg=dict(VIT_NOQK_CFG),                                                        framestack=False),  # ~706K at 128x80
    "ViT-Mixer":      dict(encoder="compact_vit",     cfg=dict(VIT_MIXER_CFG),                                                       framestack=False),  # ~683K at 128x80
    # ⚠ SQUARE-EYE ARCHIVE ONLY -- for probe_frozen_features.py to rebuild the 2026-08 n=56
    # encoders at RES=128 square. NOT launchable at the 128x80 eye (raises at construction).
    "ViT-Sp":         dict(encoder="compact_vit",     cfg=dict(VIT_SP_CFG),                                                          framestack=False),  # 696K at 128x128
    "ViT-Mixer-Sp":   dict(encoder="compact_vit",     cfg=dict(VIT_MIXER_SP_CFG),                                                    framestack=False),  # 694K at 128x128
    "ViT-NoQK-Sp":    dict(encoder="compact_vit",     cfg=dict(VIT_NOQK_SP_CFG),                                                     framestack=False),  # 707K at 128x128
    "ViT-CLTT":       dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=True,  aux="cltt", aux_weight=1.0),  # framestack True for the same reason as SimCLR-CLTT
    "ViT+VICReg":     dict(encoder="compact_vit",     cfg=dict(VIT_CFG),                                                             framestack=False, aux="vicreg"),
    "ViViT":          dict(encoder="compact_vivit",   cfg=dict(VIVIT_CFG),                                                           framestack=True),
    "ViViT+VICReg":   dict(encoder="compact_vivit",   cfg=dict(VIVIT_CFG),                                                           framestack=True,  aux="vicreg"),
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

    aux_kind = spec.get("aux")
    if aux_kind:
        os.environ["NETT_AUX_LOSS"] = str(aux_kind)
        os.environ["NETT_AUX_WEIGHT"] = str(spec.get("aux_weight", 1.0))
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
        },
        "brain": brain,
        # ⚠ ORDER IS LOAD-BEARING. body.py:53 applies these in list order, each
        # wrapping the previous, so index 0 is INNERMOST and sees the raw frame.
        # A segmentation wrapper must mask BEFORE framestack: MoTok is single-frame,
        # so it masks one frame and framestack then stacks the masked frames. The
        # reverse order would hand a single-frame segmenter a 6-channel tensor.
        "body": {"wrappers": ([spec["seg"]] if spec.get("seg") else [])
                             + (["framestack"] if spec["framestack"] else [])},
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
