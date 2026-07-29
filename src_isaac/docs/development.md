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

The suite needs two packages the runtime does not — `pytest` and `psutil` — declared as the
`test` extra: `pip install -e 'src_isaac[test]'`. `psutil` is the one worth knowing about:
`tests/fault_injection.py` imports it at module scope and `tests/conftest.py` imports that,
so a venv without it fails **collection**, and every test errors with `ModuleNotFoundError`
rather than anything that points at the cause. (`tensorboard` is a *main* dependency, not a
test one — `analysis/api.py` reads tfevents to recover reward curves, so an install without
it can train but cannot analyse.)

`optuna` is an optional dependency; its test module skips when it is absent.
No `--ignore` flags are needed. ⚠ That single SKIPPED line stands for **22 tests**
(`importorskip` at module scope collapses the whole file into one), and all 22 pass when
optuna is installed — verified 2026-07-27. One skip line is not one test.

Current expected result from the primary checkouts (measured 2026-07-29, with `optuna`
installed — without it, `581 passed, 1 skipped`):

```text
582 passed, 23 deselected
```

(582 is the committed count: `test_campaign_retest.py`, which contributes 5 of them, was
uncommitted when this figure was first recorded and has since been committed.)

## The 23 deselected tests — read this before trusting a green run

`addopts` deselects everything marked `e2e_isaac` / `e2e_perf`, so a plain `pytest` run
does **not** cover `tests/e2e/`. Run it with:

```bash
pytest src_isaac/tests/e2e -m "e2e_isaac or e2e_perf"
```

**The stimuli now resolve without env vars.** `tests/e2e/conftest.py` picks a media root by
*checking it against the design sheet* — `_resolve_media_root` returns the first candidate
holding every clip the sheet names — rather than pinning one directory. Candidates are
repoA's vendored fixture clips (`isaac_lab/assets/videos/`) and then the workspace stimulus
library (`<workspace>/videos/binding/videos/`). Override with `NETT_DESIGN_SHEET_MINIMAL` /
`NETT_DESIGN_SHEET_FULL` / `NETT_MEDIA_ROOT`; an explicit `NETT_MEDIA_ROOT` is honoured
verbatim and not validated.

⚠ **Pinning is what broke this twice, in opposite directions.** First it was pinned to a
path that never existed and the whole tree auto-skipped — honest, but terminal, so it had
never once run until 2026-07-28. Then it was pinned to repoA's shipped pair
`{O1_imprint.mov, White.mov}` on the belief that this "is exactly the pair
`binding_minimal.csv` names" — it is not, the sheet's `1color` row also names
`O1_1Ca_1.mov`. That was invisible while a missing clip was a `logger.warning` (the tests
passed against **blank monitors**), and became a hard failure of every test that reaches
the test phase the moment repoA `400683cd` made unresolvable media an error. Same pin,
green then red. A media root is only correct *relative to a sheet*, so it is resolved, not
declared — and `test_full_run` now swaps `MEDIA_ROOT_FULL` whenever it swaps in the full
sheet, which it previously did not.

Running this tree for the first time surfaced six defects in three classes:

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

Current: **23 passed in 17m27s** — the whole tree, `e2e_isaac` and the 3 `e2e_perf`
benchmarks together (measured 2026-07-29 on host `lion`, 8×A10, warm frame cache; the perf
three write `benchmarks/golden.json`). Expect tens of minutes, not seconds.

⚠ A **cold** frame cache costs far more than the run itself, and it is the difference
between 17 minutes and not finishing: the same tree was still going at the 50-minute mark
with a cold cache and a mis-resolved media root. Run `scripts/prepare_test_env.py` first —
the pre-push hook does.

The `golden.json` baseline is stamped `host: insect`, and the assertions do not check the
host, so a machine-to-machine difference could in principle read as a regression. Measured:
all three gates pass unchanged on `lion`. Re-record deliberately with `--update-golden` if
the config genuinely changes — not to silence a red gate.

## Pre-push hook

There is no CI. `src_isaac/scripts/install_git_hooks.sh` points `core.hooksPath` at
`src_isaac/scripts/hooks/`, whose `pre-push` runs the unit suite and the e2e tree and
blocks the push on failure. It sets `PYTHONPATH` and `NETT_REPO_A` correctly for you —
note `NETT_REPO_A` must be the repo **root**, the directory *containing* `isaac_lab/`; one
level too deep produces 15 "expected source file missing" failures that look like broken
wiring rather than a bad path.
Bypass: `NETT_SKIP_HOOKS=1 git push`; skip only e2e: `NETT_SKIP_E2E=1 git push`.

**Cost, measured 2026-07-29 on `lion` (8×A10, 64 cores): `4m32s` total** — 34 s unit suite,
a few seconds of prerequisites, `3m48s` for the e2e tree. Serialised on one GPU that tree is
**17m19s**, so the parallel form is ~4.6× faster. (It used to appear to take ~43 s only
because the tree skipped itself for want of stimuli.) `NETT_SKIP_E2E=1` remains the escape
hatch.

### Why the hook is not slow any more

* **The e2e tree spreads over every GPU.** Each xdist worker takes its own card
  (`PYTEST_XDIST_WORKER` → device, in `tests/e2e/conftest.py::_e2e_device`), so the wall
  clock is roughly `tests / GPUs` rounds instead of the sum. `NETT_E2E_JOBS` overrides the
  auto-detected GPU count. ⚠ **One worker per GPU, no more** — two Kit processes on a card
  contend for the same Vulkan/RTX queue and VRAM does not scale linearly.
* **`NETT_KIT_THREADS` is set automatically to `cores / jobs`.** This is not optional:
  Kit's carb pool, torch's intra-op pool and the process pool each size themselves to the
  *whole host per cell*, so N parallel cells oversubscribe the box N-fold and run **slower
  than serial**. One budget governs all three.
* **`--dist loadgroup`, not `loadfile`.** `loadfile` pins a whole file to one worker, and
  `test_full_run.py` alone holds 5 heavy training runs — it became the critical path:
  measured **6m41 with `loadfile` vs 3m48 with `loadgroup`**, same tests, same 8 GPUs.
  `loadgroup` distributes per test but honours
  `@pytest.mark.xdist_group`, which `test_outputs.py` uses to keep its 4 tests together
  because they share one module-scoped training run. Scattering those would rebuild that
  fixture four times, which is worse than serial. **If you add a module- or session-scoped
  fixture to an e2e file, mark that file the same way.**

### ⚠ `e2e_perf` is deliberately NOT in the push gate

It asserts wall-clock throughput and peak VRAM against `benchmarks/golden.json`, which
makes it a *measurement of the machine*, not a test of the diff — so it fails for reasons
that have nothing to do with what you are pushing. Measured 2026-07-29: a push was blocked
at **26.80 vs a 27.02 steps/s threshold**, i.e. 15.2% below a baseline recorded on a
*different host* against a 15% band. A 0.8% miss — noise — and the same gate passed on the
same commit minutes earlier on an idle machine. A benchmark that blocks pushes only teaches
people `--no-verify`.

It also **refuses to run under `-n`** (module-level skip in `test_perf.py`): the other
workers are training on sibling GPUs, so any number it produced would be contended — and
`--update-golden` would then bake that figure in as the permanent "optimum". Run it
deliberately, serially, on an idle machine:

```bash
pytest src_isaac/tests/e2e -m e2e_perf
```

⚠⚠ **`-u` and `timeout` on every pytest call are load-bearing.** Measured 2026-07-29: an
e2e run without them ran 50 minutes, hit a wall-clock cap, was SIGTERMed, and left a
**zero-byte log** — pytest block-buffers stdout when it is a file, so everything it had
printed died in the buffer. A push-blocking hook that can hang and then say nothing is
worse than no hook. `-u` makes partial progress survive the kill; the bound (`NETT_E2E_TIMEOUT`,
default 5400 s; `NETT_UNIT_TIMEOUT`, 900 s) turns a hang into a failure instead of an
indefinite block. The e2e stage runs `-v`, not `-q`, because with one training run per test
"which test was I on" is the entire diagnostic and the progress-dot line never flushes.

⚠ **A killed e2e run orphans its Isaac workers.** They reparent to PID 1 and keep holding
GPU memory *indefinitely* — the observed one was still in state `R` with 2.4 GB on GPU 0
long after its parent died. The hook prints the check on timeout:
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`. Reap with **SIGTERM first**
and give it a real grace (Kit defers the signal while in its render loop; 10 s is not
always enough). A Kit process that is genuinely *wedged* should be diagnosed, not `-9`'d —
repeated SIGKILL of wedged boots can leave the driver in a bad state across all GPUs.

The hook owns everything the suite needs that `git clone` does not carry, which is where
"passes on my machine, fails on yours" actually comes from:

* **The interpreter is discovered, then probed.** It used to default to one host's venv
  path. On a second host the venv is `nett_private` (underscore) rather than
  `nett-private`, **both exist**, and the hardcoded one was a real, executable, *empty*
  interpreter — so the `[ -x "$PY" ]` guard passed and the suite died at collection on
  `ModuleNotFoundError: No module named 'psutil'`, which reads as a missing dependency
  rather than as the wrong Python. The hook now globs `<workspace>/../.venv*/…`, imports
  `pytest, psutil, yaml, jsonschema, torch, skrl` in each candidate, and takes the first
  that succeeds; on failure it prints every candidate and *why* each was rejected.
  `NETT_PYTHON` still wins outright. **Do not put a single hardcoded path back.**
* **`LD_LIBRARY_PATH` for `libGLU.so.1`**, but only when the loader cannot already find it
  (`NETT_GLU_DIR` overrides the `~/glu_libs/lib` default). Without it libneuray fails to
  load, RTX dies, and the tiled camera throws a masquerading "CUDA illegal memory access"
  ~15–20 s into init. repoA's hook does the same; keep them in step.
* **External prerequisites**, via `scripts/prepare_test_env.py` — see below.

### `scripts/prepare_test_env.py`

Builds and verifies the state that lives outside both repositories. Idempotent and
Kit-free, so the hook calls it unconditionally; `--check` verifies without building.

* **Built assets** (`chamber.usdc`, `chick.usdc`) are **verified, never rebuilt.** They are
  versioned — repo policy is that generated USD is source — so a checkout has them, and a
  missing one means a broken checkout. The chamber in particular is a baked experimental
  constant (emissive intensity is compiled in, with no per-run knob), so runs trained
  against different bakes are not comparable: re-baking must be a deliberate commit, never
  a side effect of pushing.
* **Stimulus clips** cannot be built — they are experiment data. Resolved and reported.
* **The compiled frame cache** (`~/.cache/nett_frame_cache`, `$NETT_FRAME_CACHE`) *is*
  built. This is the one piece a fresh machine genuinely lacks, and prebuilding it is not
  just a speed-up: the training path builds it lazily under a cross-process flock, and a
  cold cache plus concurrent Kit boots is a known deadlock.

It reads its sheets, media roots and resolution from `tests/e2e/conftest.py` rather than
restating them, so the hook and the tests cannot end up with different opinions about what
they are running against.

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
