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

legacy_view = normalize_isaac_output("/path/to/output/smoke")
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

## Reading The Preference Numbers

`test_preferences.csv` has one row per (imprint condition, test condition,
brain):

| column | meaning |
| --- | --- |
| `n_steps` | steps in either outer third — the denominator |
| `correct_pct` | steps in the *correct* outer third / `n_steps` |
| `side_preference` | `(R - L) / (R + L)` over outer-third steps, target-agnostic |
| `pct_target_left` | `correct_pct` restricted to LEFT-target episodes |
| `pct_target_right` | `correct_pct` restricted to RIGHT-target episodes |
| `verdict` | `LEARN` / `SIDE-LOCK` / `chance` / `n/a` |

**Do not report `correct_pct` on its own.** It cannot distinguish a side-locked
policy from a wandering one: an agent that walks to the same wall every episode
scores 1.0 whenever the target happens to be that wall and 0.0 otherwise, which
averages to ~0.5 and reads as chance. `side_preference` catches that case with
no target bookkeeping (`|value|` near 1 = parked on one wall); the per-target
split and `verdict` confirm it. See `side_bias_verdict()` for the thresholds.

`n_steps == 0` means the agent never left the centre third, so every ratio has
an empty denominator. `correct_pct` reports 0.5 there — a deliberate fallback,
not a measurement (0.5 is at chance, 0 would read as a novel-stimulus
preference). The other columns are left BLANK, and `analyze()`'s summary counts
those brains under `n_brains_immobile` and excludes them from
`correct_pct_mean_defined` and the `side_preference_*` means.

For binding-style comparisons the endpoint is `learn_fraction` in
`summary.json` — the fraction of brains whose verdict is `LEARN`.
