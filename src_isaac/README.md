# NETT Isaac Lab / skrl Backend

`src_isaac` is the maintained Isaac Lab + skrl backend for the Newborn
Embodied Turing Test. The legacy Unity/SB3 package in `src/` remains in this
repository as a historical reference; new runtime, training, recording, and
analysis work should target `src_isaac`.

## What This Package Provides

- `nett_skrl.NETT`: config-driven run orchestration.
- `nett_skrl.body.Body`: observation wrapper and perception-setting bridge.
- `nett_skrl.brain.Brain`: skrl agent/model setup, training, testing, recording,
  checkpointing, and intrinsic reward wiring.
- `nett_skrl.body.wrappers`: optional observation transforms.
- `nett_skrl.environment.Environment`: configuration bridge into `nett_isaac`
  / Isaac Lab `NETTEnv`.
- `nett_skrl.analysis`: Isaac-output normalization plus legacy-compatible
  analysis entrypoints.

## Environment Assumptions

This backend assumes:

- Python `3.11`
- Isaac Sim 5.1 / Isaac Lab installed into the active environment
- skrl `>=2.0.0`
- a CUDA-capable GPU for practical Isaac Sim runs
- the private `nett_isaac` Isaac Lab package available as configured in
  `pyproject.toml`

Activate that virtualenv before running anything:

```bash
source /path/to/venv/bin/activate
```

Every `/path/to/...` below is a placeholder — substitute your own. Paths are
deliberately not hardcoded here: this repo is checked out on more than one
machine, and a stale absolute path is worse than no path at all.

`omni` is not imported directly by `nett_skrl`; Isaac Lab owns Isaac Sim app
startup.

## Install

From the repository root, use the Isaac Sim environment:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  python -m pip install -e src_isaac
```

For local development without installing:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  pytest src_isaac/tests
```

## Quick Start

Run the smoke example:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  python src_isaac/examples/run_smoke.py \
  --config src_isaac/examples/smoke.yaml \
  --output /path/to/output
```

Or use the installed CLI:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  nett-skrl --config src_isaac/examples/smoke.yaml --output /path/to/output
```

The config file controls the run. See:

- [`docs/configuration.md`](docs/configuration.md)
- [`examples/full_config.yaml`](examples/full_config.yaml)

## Output Layout

Runs write Isaac-native, vectorization-safe artifacts:

```text
<output>/<run-name>/
  <condition>/
    wandb_runs/brain_1/checkpoints/*.pt
    wandb_runs/brain_2/checkpoints/*.pt
    logs/hparams.json
    logs/eval_metrics.csv
    logs/eval_metrics.jsonl
    recordings/
      egocentric/<mode>/...
      chamber/<mode>/...
```

Analysis adapters can mirror selected artifacts into the legacy-shaped layout:

```python
from nett_skrl.analysis import normalize_isaac_output

legacy_view = normalize_isaac_output("/path/to/output/smoke")
```

## Supported Feature Names

```python
import nett_skrl

nett_skrl.list_algorithms()
nett_skrl.list_encoders()
nett_skrl.list_rewards()
nett_skrl.list_wrappers()
```

Native encoders:

- `small`
- `medium`, `Resnet10CNN`
- `large`, `Resnet18CNN`

Intrinsic rewards:

- `ICM`
- `E3B`
- `RIDE`
- `PseudoCounts`
- `NGU`

Observation wrappers:

- `dvs`
- `retina`
- `video`

Heavy/custom encoders are supported through `register_encoder(name, cls)`.
Register a concrete skrl-compatible implementation before constructing `Brain`
if you need one.

## Development Checks

Fast unit suite:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  pytest src_isaac/tests
```

Expected result at the time this documentation was written:

```text
155 passed, 24 deselected
```

Isaac Sim smoke tests are intentionally separate because they start Isaac Sim
and require a working GPU/driver stack.
