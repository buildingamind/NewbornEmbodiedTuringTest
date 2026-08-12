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

## Which Evaluation Protocol Produced The Number

Test-time actions are **sampled** from the policy by default
(`NETT_EVAL_STOCHASTIC`, `nett_skrl/brain/trainer.py::eval_stochastic_enabled`);
`0` takes the Gaussian mean instead. This is not a variance knob — it changes what
a score *means*. With the fixed test start pose (`motor.reset_deterministic`) and a
deterministic env, the mean action makes every episode of a condition a **bit-identical
replay**, so the readout is binary (which wall) rather than graded, and a
weak-but-real preference is reported as exactly chance.

★ The default changed to sampling on 2026-07-30. **Scores from before and after that date
are not poolable.** The value is recorded per run in `campaign_timing.json` (and
`campaign_retest_timing.json`) and appears as a wandb tag, so no run is ambiguous about
which protocol produced it — read it before comparing two runs. (The policy head was
defaulted to `[64, 64]` the same day and reverted after the matched control; the campaign
driver's `NETT_HIDDEN_SIZES` default is `""`, a linear head, as it always was.)

A retest of *fixed weights* under both protocols (2026-07-30) measured what sampling
does: it loosens a side-lock (`|sp|avg` −0.05 to −0.11 in every shape condition) but
does not release one, and does not make an insignificant condition significant. It is
the protocol that *can* express a graded preference, not one that manufactures one.

## Comparing Against The Published Unity Numbers

⚠ **Do not compare `correct_pct_mean` against Unity's `avgs`.** They are different
statistics, and the comparison silently looks like a result. Unity averages the
outer-third correct-side fraction **within an episode, then over episodes, then
over agents**; flat pooling over steps shifts `2color` by 0.004, which is the
size of the effects usually under discussion.

`nett_skrl.analysis.unity_parity` computes Unity's statistic on either engine's
logs, reusing `in_correct_chamber_third()` so the geometry cannot drift from the
rest of this page. It refuses to be trusted on assertion: it recomputes Unity's
published table from Unity's own raw per-agent logs first, and reports the worst
disagreement (currently `max |Δmean| = 0.0011` across all seven conditions).

```bash
python -m nett_skrl.analysis.unity_parity <run>/Object1/logs/test_Object1_0.csv
```

The Unity bundle path defaults to `~/code/jan22_cnntests_light_bundle` and is
outside both repos, so `--unity-bundle` overrides it and `tests/test_unity_parity.py`
skips the bundle-dependent checks when it is absent. Without the bundle the CLI
exits 2 rather than printing an Isaac-only table that reads as a comparison.

Every row carries `|sp|avg` and `|sp|max` — the mean and worst per-agent
`|side_preference|` — and the table flags `SIDE-LOCK` when `|sp|avg > 0.5`. This is
the same rule as above, moved into the printer: a fully locked agent scores
*exactly* 0.5 on a counterbalanced design, so a table without the column cannot be
read correctly however careful the reader is. `|sp|` is aggregated as an absolute
value, since two agents locked on opposite walls have signed preferences that cancel
to ~0 while both are fully locked.

### Scoring The Archived Unity Experiments

`~/code/analysis` holds 315 Unity-era experiments in a **second** log layout: a
per-episode `test_results.csv` with `left_steps / right_steps / middle_steps` and
`correct.monitor` per episode per agent. 1279 variant directories have that file and
none have `agent.x`, so it is the only route to those results — and the step counts
support both the nested estimator and `|sp|`, which is what makes the historical
numbers screenable for side-lock.

```bash
python -m nett_skrl.analysis.unity_parity \
  --unity-results ~/code/analysis/archive/Compendium1.2.15/1/test_results.csv
```

Two things to know about this path:

- An agent is `(imprint.cond, agent)`, not `agent`. One file holds every imprint
  condition, and a brain imprinted on Object1 is a different subject from one
  imprinted on Object2.
- ⚠ The geometry is **Unity's**. The step counts were already reduced with Unity's own
  outer-third rule, so `--chamber-half-x` is inert here and these numbers are not
  threshold-robustness-testable the way `score_unity()` is.
