# Development Notes

## Test Commands

Run the fast `src_isaac` unit suite:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  pytest src_isaac/tests
```

Some tests assert on the *source text* of the sibling repoA (`nett_isaac`)
checkout, because the in-Isaac env wiring lives in the other repository. They
resolve it from `$NETT_REPO_A`, falling back to
`<workspace>/NewbornEmbodiedTuringTest_Private`, and **skip** when neither
exists — which is what happens when you run from a worktree. Nothing
cross-checks that path against the `nett_isaac` on `PYTHONPATH`, so set
`NETT_REPO_A` to match it, or run from the primary checkouts.

`optuna` is an optional dependency; its test module skips when it is absent.
No `--ignore` flags are needed. ⚠ That single SKIPPED line stands for **22 tests**
(`importorskip` at module scope collapses the whole file into one), and all 22 pass when
optuna is installed — verified 2026-07-27. One skip line is not one test.

Current expected result from the primary checkouts:

```text
542 passed, 1 skipped, 23 deselected
```

## The 23 deselected tests — read this before trusting a green run

`addopts` deselects everything marked `e2e_isaac` / `e2e_perf`, so a plain `pytest` run
does **not** cover `tests/e2e/`. Run it with:

```bash
pytest src_isaac/tests/e2e -m "e2e_isaac or e2e_perf"
```

⚠ **They will SKIP unless you point them at real stimuli.** `tests/e2e/conftest.py`
resolves its design sheet and media root under repoA's `assets/`, and neither has ever
existed there — the sheets and `.mov` files are experiment data living at
`<workspace>/videos/binding/`. Override per asset:

```bash
NETT_DESIGN_SHEET_MINIMAL=<workspace>/videos/binding/DesignSheet_Binding.csv \
NETT_MEDIA_ROOT=<workspace>/videos/binding/videos \
  pytest src_isaac/tests/e2e -m "e2e_isaac"
```

The skip is honest — it names the path it wanted — but it is terminal, so this tree had
**never once run** until 2026-07-28. `binding_minimal.csv` is now committed under repoA
`assets/design_sheets/`, so only `NETT_MEDIA_ROOT` is still needed. Running it for the
first time surfaced six defects in three classes:

* **interface drift** — `Task(wrappers=...)` had become `Task(body=Body(...))`, and the
  `_FakeProcess` double had fallen behind the real `Process` twice (no `.pid` for
  `reaper.adopt`, no `timeout=` on `join`);
* **stale expectations** — the train CSV assertion predated the `train_step_logging`
  default flipping to False, and the CSV header assertion predated the appended
  `brain_id` column;
* **a real product defect** — the emitted `config.yaml` carried `resolved_test_num_envs`,
  which `schema.json` rejected under `additionalProperties: false`. NETT was writing
  configs it could not re-ingest, breaking the "re-runs identically on another machine"
  property `nett.py` explicitly claims. Fixed in the schema, not the test.

★★ **THE PIN IS THE ONE THAT MATTERS.** Every `NETT(...).run()` in this tree passes
`devices=[0]`. Without it `nett.py` `set_device(most_free_gpu)` picks the most-free
PHYSICAL gpu — pynvml ignores `CUDA_VISIBLE_DEVICES` — and Kit's usdrt scenegraph supports
**only cuda:0**, so the run HANGS at the Fabric XFormPrimView with no error. Measured:
26-71 min wedged unpinned, ~110s pinned. ⚠ It is HOST-STATE DEPENDENT — the picker only
strays off GPU0 when GPU0 is busier — which is why this tree could pass for months and
then wedge. If you add a fixture that calls `NETT` directly, it needs the pin too; that is
exactly how `test_outputs.py`'s own fixture was missed. This mirrors what the rest of the
repo already does (`campaign_train.py` passes `devices=[device]`, and
`examples/_smoke_pin_launch.py` calls `CUDA_VISIBLE_DEVICES=<phys>` + `devices=[0]`
"USD-safe"). To aim the tests at a specific card, set `CUDA_VISIBLE_DEVICES` in the shell.

Current: **20 passed** (the 3 `e2e_perf` benchmarks are separate — they write
`benchmarks/golden.json`). Expect minutes, not seconds.

## Pre-push hook

There is no CI. `src_isaac/scripts/install_git_hooks.sh` points `core.hooksPath` at
`src_isaac/scripts/hooks/`, whose `pre-push` runs the unit suite and the e2e tree and
blocks the push on failure (~43 s today, since e2e skips). It sets `PYTHONPATH` and
`NETT_REPO_A` correctly for you — note `NETT_REPO_A` must be the repo **root**, the
directory *containing* `isaac_lab/`; one level too deep produces 15 "expected source file
missing" failures that look like broken wiring rather than a bad path.
Bypass: `NETT_SKIP_HOOKS=1 git push`; skip only e2e: `NETT_SKIP_E2E=1 git push`.

Run a smoke training job:

```bash
VIRTUAL_ENV=/path/to/venv uv run --active --project src_isaac \
  python src_isaac/examples/run_smoke.py \
  --config src_isaac/examples/smoke.yaml \
  --output /path/to/output
```

## Design Rules

- Prefer Isaac Lab / Isaac Sim for simulation, cameras, vectorization, and
  recording.
- Prefer skrl trainer hooks over SB3 callback emulation.
- Keep heavy model dependencies optional and lazy.
- Remove Unity-only surfaces instead of carrying confusing compatibility shims.
- Use `register_encoder` and `register_reward` for user-provided extensions.
- Keep output paths vectorization-safe with explicit env/brain metadata.

## Useful Entry Points

- `nett_skrl/nett.py`: top-level config orchestration.
- `nett_skrl/runtime/task.py`: immutable task/run config.
- `nett_skrl/runtime/isaac_mode_runner.py`: Isaac Sim per-mode subprocess runner.
- `nett_skrl/runtime/memory.py`: NVML-backed GPU memory accounting.
- `nett_skrl/environment/environment.py`: config bridge into Isaac Lab.
- `nett_skrl/brain/brain.py`: thin public Brain facade.
- `nett_skrl/brain/agent_factory.py`: skrl agent/model/memory construction.
- `nett_skrl/brain/trainer.py`: NETT wrapper around skrl trainers.
- `nett_skrl/brain/env_adapter.py`: env observation/action bridge.
- `nett_skrl/brain/env_wrappers.py`: skrl rollout wrappers such as intrinsic reward injection.
- `nett_skrl/recording/export.py`: PNG sequence to MP4 export.

## Adding A Native Encoder

1. Implement a `nett_skrl.brain.encoders.NETTFeatureExtractor` subclass.
2. Accept Isaac HWC image observations and flattened skrl observations.
3. Register it with `register_encoder(...)` if it is not built in.
4. Add tests for HWC and flat observations.

External encoders can be registered without editing package code:

```python
from nett_skrl.brain.registry import register_encoder

register_encoder("MyEncoder", MyEncoder)
```

## Adding An Intrinsic Reward

Implement the adapter-compatible interface:

```python
class MyReward:
    def watch(self, observations, actions, rewards, terminated, truncated, next_observations):
        ...

    def compute(self, **samples):
        ...

    def update(self, samples=None):
        ...
```

Then register it:

```python
from nett_skrl.brain.registry import register_reward

register_reward("MyReward", MyReward)
```

Intrinsic rewards are added to extrinsic env rewards before skrl transition
storage.
