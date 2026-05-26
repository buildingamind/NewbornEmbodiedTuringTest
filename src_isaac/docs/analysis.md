# Analysis And Artifacts

Isaac runs use an Isaac-native artifact layout. Analysis helpers provide a
legacy-shaped view when old analysis scripts expect Unity-era paths.

## Native Output Layout

Typical run output:

```text
<output>/<run-name>/
  <condition>/
    wandb_runs/brain_1/checkpoints/*.pt
    wandb_runs/brain_2/checkpoints/*.pt
    logs/
      hparams.json
      eval_metrics.csv
      eval_metrics.jsonl
    recordings/
      egocentric/<mode>/...
      chamber/<mode>/...
```

Artifact paths should include enough metadata to be safe under vectorized
Isaac runs: env id, brain id, condition, mode, episode, camera, and step where
available.

## Normalize For Legacy Analysis

```python
from nett_skrl.analysis import normalize_isaac_output

legacy_view = normalize_isaac_output("/tmp/nett_skrl_run/smoke")
```

This mirrors:

- `logs/`
- `recordings/egocentric/` to `recordings/agent/`
- `recordings/chamber/`

The original Isaac-native files remain in place.

## Maintained Entry Points

The Isaac package maintains the analysis stack used by current runs:

- `analyze(run_path)`: run train + test analysis and write a summary.
- `merge(paths, output_path)`: combine analysis outputs across runs.
- `train_viz(run_path)`: plot reward curves from skrl/TensorBoard event files.
- `test_viz(run_path)`: compute correct-monitor preference from test CSV logs.

Other Unity-era analysis helpers are intentionally not part of the maintained
Isaac/skrl surface.
