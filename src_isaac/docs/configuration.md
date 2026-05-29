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
```

Run it with:

```bash
VIRTUAL_ENV=/home/zach/nett_private uv run --active --project src_isaac \
  nett-skrl --config path/to/config.yaml --output /tmp/nett_skrl_run
```

## Environment

`environment` configures the Isaac Lab `NETTEnv`.

Important fields:

- `design_sheet`: CSV with train/test condition rows.
- `media_root`: directory containing monitor videos or PNG sequence assets.
- `conditions`: optional subset of imprint conditions. `null` runs all train
  conditions in the design sheet.
- `headless`: `true` for server runs; `false` to open the Isaac viewport.
- `binocular_vision`: `true` returns concatenated left/right egocentric camera
  observations.
- `input_resolution`: square per-eye resolution.
- `reward_types`: env-side rewards. Use `[]` for no extrinsic reward.
- `decision_period`: Isaac env steps per motor/log/reward decision.
- `random_first_frame`: randomize stimulus start frame.
- `switch_steps`: video target/non-target switch cadence.
- `record_mode`: `tSNE` or `spatial` for record-phase sampling.

Top-level `eval_freq` is optional. When set, it is interpreted as a train-step
interval and runs metrics-only test rollouts in separate Isaac subprocesses at
each milestone. Final test-mode recording is still controlled by
`episodes.test` and `environment.recording`.

## Brain

`brain` configures skrl agents.

Common fields:

- `algorithm`: skrl agent name such as `PPO`, `A2C`, `SAC`, `TD3`, or `DDPG`.
  Recurrent and discrete agents are intentionally unsupported until NETT has
  recurrent model state plumbing or a discrete action adapter.
- `encoder`: visual feature extractor.
- `model`: MLP head settings. Defaults to hidden sizes `[64, 64]`, `elu`
  activation, bounded value output, orthogonal init, and clipped actions.
- `reward`: env reward string or intrinsic reward name.
- `checkpoint_freq`: per-brain checkpoint interval in trainer steps.
- `encoder_cfg`: encoder constructor config. Defaults include
  `features_dim: 512` and `trainable: true`; extra keys pass through to
  custom encoders.
- `algorithm_cfg`: skrl config overrides. PPO/on-policy configs use
  `rollouts` and `mini_batches`; off-policy configs use skrl replay-buffer
  fields such as `memory_size` and `batch_size`. Extra keys pass through to skrl.
- `reward_cfg`: intrinsic reward constructor config. Defaults include
  `beta: 0.2`, `kappa: 0.0`, `gamma: 0.99`, `weight: 1.0`, and
  `trainable: true`; extra keys pass through.

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

Use native Isaac env settings for camera layout. For example,
`environment.binocular_vision: true` controls the binocular egocentric cameras.
New configs can place wrapper and perception-side settings under `body`:

```yaml
body:
  wrappers: [video]
  binocular_vision: true
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
