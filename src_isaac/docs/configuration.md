# NETT-skrl Configuration

`nett_skrl` runs from YAML or JSON configs validated against
`nett_skrl/schema.json`. Only `name`, `environment.design_sheet`, and
`environment.media_root` are required.

## Minimal Config

```yaml
name: smoke

environment:
  design_sheet: /path/to/binding.csv
  media_root: /path/to/videos

episodes:
  train: 1

steps_per_episode: 50
num_brains: 1
max_parallel_envs: null
```

Run it with:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  nett-skrl --config /path/to/config.yaml --output /path/to/output
```

`VIRTUAL_ENV` must point at the virtualenv that has Isaac Sim and Isaac Lab
installed; `--active` tells `uv` to use it rather than resolving its own.

## Environment

`environment` configures the Isaac Lab `NETTEnv`.

Important fields:

- `design_sheet`: CSV with train/test condition rows.
- `media_root`: directory containing monitor videos or PNG sequence assets.
- `conditions`: optional subset of imprint conditions. `null` runs all train
  conditions in the design sheet.
- `headless`: `true` for server runs; `false` to open the Isaac viewport.
- `input_resolution`: square per-eye resolution. Default **64**.
- `camera_fov`: the chick eye's horizontal field of view in degrees. Default **150.0** —
  the fisheye aperture the animal has, and the one the closeness reward is calibrated for.
  **Treat this as an experimental constant, not a tuning knob.** Two things depend on it:
  - The reward *is* a projection through this optic. `closeness` measures the target's
    projected extent, so changing the FOV changes what the reward means, not just how
    much of the chamber is visible. A wider FOV keeps a monitor in view almost always,
    which raises the reward floor — `optuna_tune/RESULTS_nature_cnn.md` and
    `RESULTS_combined.md` record a whole line of investigation that chased encoder
    capacity before identifying the ~0.40 floor at 150° as the cause.
  - Repo A declares the same number independently (`NETTEnvCfg.observation.fov`). Repo B's
    value is copied into the env cfg **unconditionally**, so if the two drift apart repo B
    wins silently. They *did* drift — repo B defaulted 120.0 until 2026-07-31, so any
    config omitting the key ran an optic no experiment used. Runs from before that date
    that omitted `camera_fov` are not comparable to runs after it; runs that set it
    explicitly (every research config does) are unaffected.
    `tests/test_schema_defaults.py` now fails if the two repos disagree.
- `reward_types`: env-side rewards. Use `[]` for no extrinsic reward.
- `decision_period`: Isaac env steps per motor/log/reward decision.
- `random_first_frame`: randomize stimulus start frame.
- `record_mode`: `tSNE` or `spatial` for record-phase sampling.
- `locomotion`: `wheeled` (**default**) or `kinematic`. `wheeled` drives the body
  through PhysX (GPU PhysX); `kinematic` writes the root pose directly and is
  pinned to CPU PhysX. **Changed 2026-07-24**: the runtime default used to be
  `kinematic` even though `schema.json` declared `wheeled`. `jsonschema` does not
  apply schema defaults, so the Python default governed and any config omitting
  this key silently ran the *unvalidated* mode. The two now agree. Set it
  explicitly if you depend on a specific mode.
- The **wheeled action parameterization** is not a config field — it is the env var
  `NETT_WHEEL_TURNMOVE`, read in repo A's `camera_rig.py`. Both parameterizations drive
  the *same* differential-drive body velocity, and both normalize identically (full
  forward = `body_move_speed_limit`, full in-place turn = `body_turn_speed_limit`):
  - **default** (unset or `1`): `[turn, move]` — decoupled, so turn and move can be
    maxed independently.
  - `NETT_WHEEL_TURNMOVE=0`: `[left, right]` wheel commands — coupled diff-drive, where
    turning trades against forward speed. This is the physically faithful control and
    was the default before 2026-07-31.

  ⚠ Flipped as a **package** with `model.actor_distribution` (see below) — the axes
  interact and only the combined arm was measured. Runs from before 2026-07-31 are not
  comparable to runs after it. Read the DEFAULT-FLIP LEDGER in `isaac/blueprint.md`
  before relying on this or changing it again.
- Chamber lighting is NOT configurable. Repo A ships exactly one chamber,
  `assets/chamber/chamber.usdc`: statically baked radiosity lightmaps with the
  monitors measured at **250 cd/m^2** (the Acer V193W EJb panels of the original
  experiment) against NVIDIA's OmniEmissive reference. It is the only configuration
  that is realistic, temporally static and bit-reproducible at the same time; the
  former `lighting_mode` / emissive-brightness knobs and their chamber variants were
  removed 2026-07-26, and the dead `lighting_mode` key was dropped from `schema.json`
  on 2026-07-27 (a config still carrying it now fails validation loudly instead of
  being silently ignored). `chamber_variant` survives in repo A as a diagnostic-only
  field — it is not reachable from a run config. ⚠ 250, not the 300 stated here before
  2026-07-27: the earlier measurement was pegged to a mid-grey seed texture rather than
  a full-white screen. See `isaac/CHAMBER_LIGHTING_STATE.md`.
- `algorithm`: skrl agent name such as `PPO`, `A2C`, `SAC`, `TD3`, or `DDPG`.
  Recurrent and discrete agents are intentionally unsupported until NETT has
  recurrent model state plumbing or a discrete action adapter.
- `encoder`: visual feature extractor.
- `model`: MLP head settings. Defaults to hidden sizes `[64, 64]`, `elu`
  activation, bounded value output, orthogonal init, and clipped actions.
- `model.actor_distribution`: which stochastic actor head a Gaussian-policy
  algorithm builds. **Defaults to `"gaussian"`** — the independent per-component
  diagonal head — since 2026-07-31; set `"multivariate_gaussian"` for the joint
  covariance, which lets the action components be sampled as correlated.
  ⚠ This default has changed **twice**: to multivariate on 2026-07-30 (on a
  construction argument, never a measured win) and back to diagonal on 2026-07-31
  **on measurement**. Runs made on defaults in different windows are not comparable.
  Explicit `null` has always selected the diagonal head, so old configs and
  checkpoints resolve to the head they were built with regardless.
  ⚠ **A pre-flip checkpoint will not load under the new default, and the error is
  cryptic.** The multivariate head owns an extra learnable parameter `tril_offdiag`
  (the Cholesky off-diagonal); the diagonal head does not. skrl loads with
  `strict=True`, so re-running a config that *omitted* `actor_distribution` before
  2026-07-31 now raises
  `RuntimeError: Unexpected key(s) in state_dict: "tril_offdiag"`.
  **Fix:** set `actor_distribution: multivariate_gaussian` explicitly in that config —
  the checkpoint is fine, only the default moved under it. (Verified against
  `B5_mvg_lr_s0/.../agent_6000.pt`, whose policy keys are `log_std` + `tril_offdiag`.)
  ⚠ Note the two heads are *identical at initialisation*: `tril_offdiag` starts at zero
  and the Cholesky diagonal is `exp(log_std)`, matching `GaussianActor` exactly. They
  diverge only as correlation is learned — so this is not a "correlated vs uncorrelated
  from step 0" comparison.
  ⚠ Flipped as a **package** with repo A's wheeled action default (`[turn, move]`):
  the axes interact, and only the combined arm has data. Evidence, caveats and the
  budget dependence are in the DEFAULT-FLIP LEDGER in `isaac/blueprint.md` — read it
  before relying on this or changing it again.
- `reward`: env reward string or intrinsic reward name.
- `checkpoint_freq`: per-brain checkpoint interval, in **trainer timesteps**
  (`episodes * steps_per_episode / envs_per_brain` — the trainer advances one
  timestep per env-batch, so the divisor is real). skrl writes `agent_{timestep}.pt`
  from inside its training loop, so this does **not** interrupt or chunk training.
  ⚠ It used to: before `44ff079` it also added *training* boundaries, splitting a run
  into one subprocess per checkpoint and overwriting each snapshot with
  `final_agent.pt` — a 500-episode run produced 32 byte-identical files. Verify
  snapshots differ before building any learning curve:
  `md5sum agent_*.pt | awk '{print $1}' | sort -u | wc -l` must equal the file count.
- `encoder_cfg`: encoder constructor config. Defaults include
  `features_dim: 512` and `trainable: true`; extra keys pass through to
  custom encoders.
- `algorithm_cfg`: skrl config overrides. PPO/on-policy configs use
  `rollouts` and `mini_batches`; off-policy configs use skrl replay-buffer
  fields such as `memory_size` and `batch_size`. Extra keys pass through to skrl.
- `reward_cfg`: intrinsic reward constructor config. Defaults include
  `beta: 0.2`, `kappa: 0.0`, `gamma: 0.99`, `weight: 1.0`, and
  `trainable: true`; extra keys pass through.

Unknown `encoder_cfg`, `algorithm_cfg`, and `reward_cfg` keys intentionally pass
through to skrl or custom modules. This keeps advanced skrl options available
without requiring schema changes for every lower-level setting.

For PPO, NETT defaults to `learning_rate: 1e-5`, `rollouts: 8000`,
`mini_batches: 16`, `value_loss_scale: 0.25`, and `grad_norm_clip: 0.25`.
Set those fields in `algorithm_cfg` to override them.

Native encoder names:

- `small`
- `medium`, `Resnet10CNN`
- `large`, `Resnet18CNN`

Heavy encoders such as DINO, ViT, SAM, and CNN-LSTM variants are not built-in
placeholder names. Register a concrete `NETTFeatureExtractor` class before
constructing `Brain`, then use the registered name in config/code.

## Rewards

Env-side rewards are computed inside Isaac `NETTEnv`:

- `closeness`
- `completeness`
- `closeness,completeness`
- `unsupervised`

Intrinsic rewards are added through a skrl env reward-shaping wrapper:

- `ICM`
- `E3B`
- `RIDE`
- `PseudoCounts`
- `NGU`

Compatibility-only reward names currently raise explicit errors unless you
register an implementation:

- `Disagreement`
- `Fabric`
- `RE3`
- `RND`

## Body Wrappers

Supported wrappers:

- `dvs`: dynamic vision sensor transform.
- `retina`: foveated retina transform.
- `video`: frame-stack policy observations on the image channel axis.

The chick has a single monocular fisheye eye; there is no camera-layout toggle.
New configs can place wrapper and perception-side settings under `body`:

```yaml
body:
  wrappers: [video]
  input_resolution: 64
```

## Recording

Recording is configured under `environment.recording`.

```yaml
environment:
  recording:
    fps: 24
    egocentric:
      enabled: true
      modes: [train, test, record]
      episodes: [0, 2]
    chamber:
      enabled: true
      modes: [record]
      episodes: "1:"
```

Egocentric recording uses the chick cameras. Chamber recording uses a separate
Isaac Lab camera. PNG sequences are exported to MP4 after a run where possible.
Episode selectors are zero-based. `null` records every episode in the selected
mode, an integer records one episode, a list records those episodes, and a
slice string such as `"1:"` or `"0:10:2"` records matching episode indices.

## Evaluation

Use `episodes.test` for the final recorded Isaac test/evaluation phase.
Optional top-level `eval_freq` adds metrics-only milestone evaluations during
training. It is interpreted as train environment steps and runs through
separate Isaac subprocesses so it does not create a second live `NETTEnv`
inside the active training app.

```yaml
episodes:
  train: 5000
  test: 100
eval_freq: 50000
```
