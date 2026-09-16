# NETT_* environment variables — GENERATED, do not edit

Regenerate with `python scripts/gen_env_index.py`. `tests/test_env_index.py` fails if
this file drifts from the code, so it cannot quietly go partial.

⛔ **THIS IS THE AUTHORITATIVE ANSWER TO "DOES THIS KNOB EXIST?"** A name absent from
this table is a name the code never reads. Setting it has NO EFFECT and raises nothing —
the run proceeds as the control while its label says otherwise. Before putting a `NETT_*`
name in a queue row, a launcher, or a message, confirm it here or with
`grep -rn NETT_YOUR_NAME src_isaac/`.

⚠ Defaults are the literal second argument at the read site. Where a variable is read in
more than one place the defaults can differ — every site is listed rather than collapsed,
because a knob with two defaults is a real hazard and a single-row summary would hide it.

142 variables.

| variable | default(s) | read at |
|---|---|---|
| `NETT_AMP` | `"bf16"` | `nett_skrl/brain/encoders/nature_cnn.py:69`<br>`examples/probe_frozen_features.py:57` |
| `NETT_AUX_ALLOW_ZERO` | `""` | `nett_skrl/brain/agent_factory.py:51`<br>`nett_skrl/brain/aux/ppo_aux.py:198` |
| `NETT_AUX_BATCH` | `"0"`, `"256"`, `"512"`, `max_samples` | `nett_skrl/brain/agent_factory.py:66`<br>`nett_skrl/brain/aux/cltt_aux.py:74`<br>`nett_skrl/brain/aux/cltt_ref_aux.py:262`<br>`nett_skrl/brain/aux/cltt_schneider_aux.py:90`<br>`nett_skrl/brain/aux/dual_stream.py:224`<br>`nett_skrl/brain/aux/eoo_aux.py:161`<br>*(+7 more)* |
| `NETT_AUX_CLTT_REF_TEMP` | `"0.5"` | `nett_skrl/brain/aux/cltt_ref_aux.py:263` |
| `NETT_AUX_LOSS` | `"none"` | `nett_skrl/brain/agent_factory.py:30`<br>`examples/campaign_train.py:622`<br>`examples/campaign_train.py:900` |
| `NETT_AUX_STRICT` | `""` | `nett_skrl/brain/aux/ppo_aux.py:422` |
| `NETT_AUX_VICREG_TT_OFFSETS` | `"8"` | `nett_skrl/brain/aux/vicreg_tt_aux.py:87` |
| `NETT_AUX_WEIGHT` | `"0"`, `"0.0"` | `nett_skrl/brain/agent_factory.py:31`<br>`examples/campaign_train.py:623`<br>`examples/campaign_train.py:901` |
| `NETT_BRAINS` | `"7"`, `"8"` | `examples/campaign_train.py:545`<br>`examples/train_binding_8brain_targets.py:41` |
| `NETT_BRAIN_OFFSET` | `"0"`, `'0'` | `examples/campaign_train.py:546`<br>`examples/train_binding_8brain_targets.py:190`<br>`examples/train_binding_8brain_targets.py:208` |
| `NETT_CAMERA_FOV` | `""` | `examples/campaign_train.py:436` |
| `NETT_CHECKPOINT_FREQ` | *(required / no literal default)* | `examples/campaign_train.py:757`<br>`examples/campaign_train.py:758` |
| `NETT_DECOUPLE_ENCODER` | `""` | `nett_skrl/brain/models/utils/features.py:23` |
| `NETT_DESIGN_SHEET` | `sheet` | `examples/campaign_run.py:197`<br>`examples/campaign_train.py:816` |
| `NETT_DEVICE` | `"0 (default`, `"0"` | `examples/campaign_retest.py:126`<br>`examples/campaign_train.py:542`<br>`examples/capture_observations.py:358`<br>`examples/capture_observations.py:585`<br>`examples/train_binding_8brain.py:24`<br>`examples/train_binding_8brain_targets.py:35` |
| `NETT_DEVICE_LOST_EXIT_CODE` | `"75"` | `nett_skrl/runtime/reap.py:692` |
| `NETT_DEVICE_LOST_FORENSICS` | `"1"` | `nett_skrl/runtime/crash_guard.py:164` |
| `NETT_DEVICE_LOST_GUARD` | `"1"` | `nett_skrl/runtime/crash_guard.py:175` |
| `NETT_DIAG_DUMP` | *(required / no literal default)* | `nett_skrl/brain/ppo_metrics.py:81` |
| `NETT_DIAG_ENT_START` | *(required / no literal default)* | `nett_skrl/brain/ppo_metrics.py:228` |
| `NETT_DIAG_PEB` | `"off"` | `nett_skrl/brain/ppo_metrics.py:90` |
| `NETT_DRY_RUN_TIMEOUT` | *(required / no literal default)* | `nett_skrl/nett.py:88` |
| `NETT_ENCODER` | `"compact_3dcnn"`, `"compact_vit"`, `"nature_cnn"` | `examples/train_binding_8brain.py:23`<br>`examples/train_binding_8brain_targets.py:34`<br>`examples/train_replicate_encoders.py:25`<br>`examples/train_replicate_single.py:33` |
| `NETT_ENT` | `"0.01"` | `examples/train_binding_8brain_targets.py:160` |
| `NETT_ENTROPY` | `"0.01"` | `examples/campaign_train.py:573` |
| `NETT_ENVS` | `"16"` | `examples/train_replicate_single.py:35` |
| `NETT_EOO_MASK` | `"0.01"` | `nett_skrl/brain/aux/eoo_aux.py:164` |
| `NETT_EOO_PHOTO` | `"1.0"` | `nett_skrl/brain/aux/eoo_aux.py:162` |
| `NETT_EOO_SMOOTH` | `"0.1"` | `nett_skrl/brain/aux/eoo_aux.py:163` |
| `NETT_EVAL_STOCHASTIC` | `"1"` | `nett_skrl/brain/trainer.py:57` |
| `NETT_EXPERIMENT` | `"binding"` | `examples/campaign_train.py:526`<br>`examples/train_binding_8brain_targets.py:134` |
| `NETT_EXPERT_FLOW` | `""` | `nett_skrl/brain/aux/dual_stream.py:181` |
| `NETT_EXPERT_FLOW_PATCH` | `"8"` | `nett_skrl/brain/aux/expert_flow.py:48` |
| `NETT_EXPERT_FLOW_RADIUS` | `"12"` | `nett_skrl/brain/aux/expert_flow.py:46` |
| `NETT_EXPERT_FLOW_STRIDE` | `"2"` | `nett_skrl/brain/aux/expert_flow.py:47` |
| `NETT_EXTRA_KIT_ARGS` | `""` | `nett_skrl/environment/environment.py:315` |
| `NETT_FORCE_RECOMPUTE` | *(required / no literal default)* | `nett_skrl/environment/environment.py:357` |
| `NETT_FRAMESTACK` | `"1"` | `examples/train_binding_8brain_targets.py:204` |
| `NETT_FRAMESTACK_N` | `"2"` | `nett_skrl/body/wrappers/framestack.py:48`<br>`examples/campaign_train.py:147` |
| `NETT_FRAME_FORMAT` | `"bc7"` | `examples/campaign_run.py:181`<br>`scripts/prepare_test_env.py:112` |
| `NETT_GPUS` | `"0,1,2,3,4,5,6,7"` | `examples/campaign_retest_launch.py:51`<br>`examples/campaign_run.py:209` |
| `NETT_GPU_BUFFER_BUDGET_GB` | `"4.0"` | `examples/campaign_train.py:648` |
| `NETT_GWM_BAL` | `"0.1"` | `nett_skrl/brain/aux/gwm_aux.py:95` |
| `NETT_GWM_COH` | `"1.0"` | `nett_skrl/brain/aux/gwm_aux.py:93` |
| `NETT_GWM_PHOTO` | `"1.0"` | `nett_skrl/brain/aux/gwm_aux.py:92` |
| `NETT_GWM_SLOTS` | `slots` | `nett_skrl/brain/aux/gwm_aux.py:87` |
| `NETT_GWM_SMOOTH` | `"0.1"` | `nett_skrl/brain/aux/gwm_aux.py:94` |
| `NETT_HIDDEN_SIZES` | `""` | `examples/campaign_train.py:567` |
| `NETT_IMPRINT` | `_EXP_MAP[EXP][2]`, `default_imprint` | `examples/campaign_train.py:533`<br>`examples/train_binding_8brain_targets.py:145` |
| `NETT_ISAAC_LAB` | *(required / no literal default)* | `examples/campaign_retest_launch.py:38` |
| `NETT_JOBS_PER_GPU` | `"1"`, `"2"` | `examples/campaign_retest_launch.py:52`<br>`examples/campaign_run.py:210` |
| `NETT_LIFECYCLE_DISABLE` | `"0"` | `nett_skrl/runtime/lifecycle.py:82` |
| `NETT_LIFECYCLE_POLL` | `"0.25"` | `nett_skrl/runtime/lifecycle.py:80` |
| `NETT_LIFECYCLE_TERM_GRACE` | `"25"` | `nett_skrl/runtime/lifecycle.py:78` |
| `NETT_LR` | `"3e-4"` | `examples/train_binding_8brain_targets.py:156` |
| `NETT_LR_WARMUP` | *(required / no literal default)* | `examples/train_binding_8brain_targets.py:119`<br>`examples/train_binding_8brain_targets.py:125` |
| `NETT_LUMNORM_MEAN` | `"0.45"` | `nett_skrl/body/wrappers/lumnorm.py:69` |
| `NETT_LUMNORM_STD` | `"0.25"` | `nett_skrl/body/wrappers/lumnorm.py:70` |
| `NETT_MAX_ENVS` | `"112"`, `"32"` | `examples/campaign_train.py:548`<br>`examples/train_binding_8brain_targets.py:38` |
| `NETT_MEDIA_ROOT` | `media` | `examples/campaign_run.py:198`<br>`examples/campaign_train.py:817` |
| `NETT_MEMORY_DEVICE` | *(required / no literal default)* | `nett_skrl/brain/hybrid_memory.py:128`<br>`examples/campaign_train.py:658` |
| `NETT_MINIBATCHES` | `"16"` | `examples/campaign_train.py:770`<br>`examples/campaign_train.py:899`<br>`examples/train_binding_8brain_targets.py:155` |
| `NETT_MODEL` | `""` | `examples/campaign_train.py:522` |
| `NETT_MOTOK_MODE` | `"faithful"` | `nett_skrl/brain/aux/motok_aux.py:323` |
| `NETT_MOTOK_QUERIES` | `"2"` | `nett_skrl/brain/aux/motok_aux.py:321` |
| `NETT_MOTOK_UPSAMPLE` | `"0"` | `nett_skrl/brain/aux/motok_aux.py:322` |
| `NETT_MOTOK_VQ_COEF` | `"0.1"` | `nett_skrl/brain/aux/motok_aux.py:320` |
| `NETT_NAME` | `f"{ENC}_s{SEED_OFFSET}_{datetime.now(` | `examples/train_replicate_single.py:41` |
| `NETT_ONLY_EXPERIMENTS` | `""` | `examples/campaign_run.py:75` |
| `NETT_ONLY_MODELS` | `""` | `examples/campaign_run.py:74` |
| `NETT_OPTUNA_DIR` | `"/home/zlaborde/code/isaac/optuna_tune"` | `examples/optuna_tune.py:49` |
| `NETT_OUTPUT` | `"~/nett_replicate_out"` | `examples/train_replicate_single.py:40` |
| `NETT_OUT_ROOT` | `"~/nett_campaign"` | `examples/campaign_retest_launch.py:44`<br>`examples/campaign_train.py:687` |
| `NETT_PROBE_DEVICE` | `'0'` | `examples/probe_encoder_binding.py:145`<br>`examples/probe_frozen_features.py:365`<br>`examples/probe_policy_readout.py:140` |
| `NETT_PYTHON` | *(required / no literal default)* | `examples/campaign_retest_launch.py:37`<br>`examples/campaign_run.py:36` |
| `NETT_REAP_CRASH_GRACE` | `"120"` | `nett_skrl/runtime/reap.py:103` |
| `NETT_REAP_DISABLE` | `"0"` | `nett_skrl/runtime/reap.py:113` |
| `NETT_REAP_LINEAGE_POLL` | `"5"` | `nett_skrl/runtime/reap.py:111` |
| `NETT_REAP_POLL` | `"0.5"` | `nett_skrl/runtime/reap.py:109` |
| `NETT_REAP_TERM_GRACE` | `"10"` | `nett_skrl/runtime/reap.py:105` |
| `NETT_REAP_TIMEOUT` | `"0"` | `nett_skrl/runtime/reap.py:107` |
| `NETT_RES` | `"128"`, `"256"` | `examples/campaign_run.py:180`<br>`examples/campaign_train.py:550`<br>`examples/count_params.py:17`<br>`examples/sweep_params.py:12`<br>`examples/train_binding_8brain_targets.py:40` |
| `NETT_RETEST_DIR` | `str(ROOT / "_retest"` | `examples/campaign_retest_launch.py:45` |
| `NETT_RETEST_EXPECT_ROWS` | `"0"` | `examples/campaign_retest.py:89` |
| `NETT_RETEST_GLOB` | `str(ROOT / "*" / "*_off*"` | `examples/campaign_retest_launch.py:50` |
| `NETT_REWARD_TYPES` | `"closeness"` | `examples/campaign_train.py:554` |
| `NETT_ROLLOUTS` | `"8000"` | `examples/campaign_train.py:584`<br>`examples/campaign_train.py:725` |
| `NETT_RUN_NAME` | `""` | `examples/campaign_train.py:701` |
| `NETT_RUN_ROOT` | `str(Path.home(` | `examples/probe_frozen_features.py:89` |
| `NETT_SEED_OFFSET` | `"0"`, `"1"` | `examples/train_nature_cnn_replicate_seed.py:20`<br>`examples/train_replicate_single.py:34` |
| `NETT_SEG_BACKBONE_LR` | `"1e-5"` | `nett_skrl/body/wrappers/gwm_seg.py:102` |
| `NETT_SEG_BATCH` | `"8"` | `nett_skrl/body/wrappers/segmentation.py:28` |
| `NETT_SEG_BUFFER` | `"256"` | `nett_skrl/body/wrappers/segmentation.py:30` |
| `NETT_SEG_DEVICE` | *(required / no literal default)* | `nett_skrl/body/wrappers/segmentation.py:46` |
| `NETT_SEG_FG_SLOT` | `"auto"` | `nett_skrl/body/wrappers/segmentation.py:32` |
| `NETT_SEG_FLOW_REG` | `"1e-4"` | `nett_skrl/body/wrappers/gwm_seg.py:103` |
| `NETT_SEG_LR` | `"1e-4"` | `nett_skrl/body/wrappers/segmentation.py:26` |
| `NETT_SEG_MASK_RULE` | `"auto"` | `nett_skrl/body/wrappers/segmentation.py:36` |
| `NETT_SEG_MODEL` | `"motok"` | `nett_skrl/body/wrappers/motok_seg.py:118` |
| `NETT_SEG_QUERIES` | `"2"` | `nett_skrl/body/wrappers/gwm_seg.py:99`<br>`nett_skrl/body/wrappers/motok_seg.py:126`<br>`examples/campaign_train.py:616` |
| `NETT_SEG_TRAIN_EVERY` | `"64"` | `nett_skrl/body/wrappers/segmentation.py:29` |
| `NETT_SEG_UPSAMPLE` | `"0"` | `nett_skrl/body/wrappers/motok_seg.py:127` |
| `NETT_SEG_VQ_COEF` | `"0.1"` | `nett_skrl/body/wrappers/motok_seg.py:128` |
| `NETT_SEG_WD` | `"1e-4"` | `nett_skrl/body/wrappers/segmentation.py:27` |
| `NETT_SIM_DEVICE` | *(required / no literal default)* | `nett_skrl/environment/environment.py:414` |
| `NETT_SKIP_VALIDATION` | *(required / no literal default)* | `nett_skrl/nett.py:417` |
| `NETT_SLOTC_DIM` | `slot_dim` | `nett_skrl/brain/aux/slot_contrast_aux.py:382` |
| `NETT_SLOTC_EMA` | `ema` | `nett_skrl/brain/aux/slot_contrast_aux.py:386` |
| `NETT_SLOTC_SLOTS` | `slots` | `nett_skrl/brain/aux/slot_contrast_aux.py:381` |
| `NETT_SLOTC_TEMP` | `temperature` | `nett_skrl/brain/aux/slot_contrast_aux.py:383` |
| `NETT_SLOTC_W_REC` | `w_rec` | `nett_skrl/brain/aux/slot_contrast_aux.py:385` |
| `NETT_SLOTC_W_SS` | `w_ss` | `nett_skrl/brain/aux/slot_contrast_aux.py:384` |
| `NETT_STAGGER_SECS` | `"12"` | `examples/campaign_retest_launch.py:53`<br>`examples/campaign_run.py:211` |
| `NETT_STALL_EXIT_CODE` | `"77"` | `nett_skrl/runtime/reap.py:744` |
| `NETT_STALL_GUARD` | `"1"` | `nett_skrl/runtime/stall_guard.py:133` |
| `NETT_STEPS` | `"500"`, `'500'` | `examples/campaign_train.py:797`<br>`examples/campaign_train.py:852` |
| `NETT_STRICT_DETERMINISM` | `"0"` | `nett_skrl/runtime/task.py:187` |
| `NETT_TAG` | `""` | `examples/train_binding_8brain_targets.py:129` |
| `NETT_TASK_MEMORY` | `"1"` | `examples/campaign_train.py:854`<br>`examples/train_binding_8brain_targets.py:216`<br>`examples/train_replicate_single.py:121`<br>`examples/train_replicate_single.py:122` |
| `NETT_TEARDOWN_EXIT_CODE` | `"78"` | `nett_skrl/runtime/reap.py:759` |
| `NETT_TEST_ENVS` | *(required / no literal default)* | `nett_skrl/nett.py:542`<br>`nett_skrl/runtime/task_runner.py:478` |
| `NETT_TEST_EPS` | `"20"`, `str(config.get("episodes", {}` | `examples/campaign_retest.py:68`<br>`examples/campaign_train.py:851` |
| `NETT_TEST_GROUP_BY_ROW` | *(required / no literal default)* | `nett_skrl/environment/environment.py:377`<br>`examples/capture_observations.py:557`<br>`examples/capture_observations.py:561`<br>`examples/capture_observations.py:563` |
| `NETT_TEXTURE_DEFAULTS` | `"1"` | `nett_skrl/runtime/texture_defaults.py:98` |
| `NETT_TF32` | `"1"` | `nett_skrl/runtime/task.py:203` |
| `NETT_TRAIN_EPS` | `"2000"` | `examples/campaign_train.py:547`<br>`examples/train_binding_8brain_targets.py:42` |
| `NETT_UINT8_BUFFER` | `"1"` | `nett_skrl/brain/agent_factory.py:184` |
| `NETT_UNIFIED_WANDB` | *(required / no literal default)* | `nett_skrl/brain/experiment.py:83`<br>`nett_skrl/recording/wandb.py:296` |
| `NETT_VICREG_COV` | `"10"` | `nett_skrl/brain/aux/vicreg_aux.py:101`<br>`nett_skrl/brain/aux/vicreg_tt_aux.py:101` |
| `NETT_VICREG_INV` | `"3"` | `nett_skrl/brain/aux/vicreg_aux.py:99`<br>`nett_skrl/brain/aux/vicreg_tt_aux.py:99` |
| `NETT_VICREG_VAR` | `"30"` | `nett_skrl/brain/aux/vicreg_aux.py:100`<br>`nett_skrl/brain/aux/vicreg_tt_aux.py:100` |
| `NETT_VIDEOS_ROOT` | `"/home/zlaborde/code/isaac/videos"` | `examples/_paths.py:18` |
| `NETT_VIT_EMBED` | `"128"` | `examples/train_binding_8brain_targets.py:87` |
| `NETT_VIT_HEADS` | `"4"` | `examples/train_binding_8brain_targets.py:89` |
| `NETT_VIT_POOL` | `"cls"` | `examples/train_binding_8brain_targets.py:93` |
| `NETT_VIT_STEM` | `"linear"` | `examples/train_binding_8brain_targets.py:96` |
| `NETT_VIVIT_POOL` | `"cls"` | `examples/train_binding_8brain_targets.py:61` |
| `NETT_VIVIT_TEMPORAL` | `"joint"` | `examples/train_binding_8brain_targets.py:60` |
| `NETT_VRAM_OOM_EXIT_CODE` | `"76"` | `nett_skrl/runtime/reap.py:718` |
| `NETT_WANDB_GROUP` | *(required / no literal default)* | `nett_skrl/brain/experiment.py:91` |
| `NETT_WANDB_MODE` | `"offline"`, `"online"` | `examples/campaign_retest.py:71`<br>`examples/campaign_retest_launch.py:82`<br>`examples/campaign_train.py:794`<br>`examples/train_binding_8brain_targets.py:174` |
| `NETT_WORKSPACE` | `""` | `examples/gate_a_resume.py:81` |
