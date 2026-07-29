# CLAUDE.md

Context for AI assistants working in this repository.

## What this repo is

This is **Repo B** of the two-repo Isaac Sim / skrl stack driving the Newborn
Embodied Turing Test. The maintained code is the **`nett_skrl`** package under
[`src_isaac/`](src_isaac/README.md) — orchestration, device placement, the skrl
PPO scaffold, the observation body, the environment bridge, and analysis.

The Isaac Lab environment, robot USD, camera rig, and rewards live in the sibling
**Repo A**, `NewbornEmbodiedTuringTest_Private/isaac_lab` (the `nett_isaac`
package). The two are wired through a single `DirectRLEnv` the skrl side wraps.

> The legacy Unity/SB3/mlagents package (`nett-benchmarks`, `src/nett`) has been
> removed from this branch. Do not resurrect references to it.

## Start here

**`../blueprint.md`** (project root) is the ground truth: the consolidated,
conflict-free technical reality of the whole stack — the venv and how to launch
Python, the per-step execution loop, PhysX device strategy, the fisheye/reward
geometry, determinism, env sizing, and the dead-ends list. Read it before making
changes.

Package-local orientation: [`src_isaac/README.md`](src_isaac/README.md) and
`src_isaac/docs/`.

## Running tests

The pure-Python tests run without booting Isaac Kit:

```bash
cd src_isaac
PYTHONPATH=.:../../NewbornEmbodiedTuringTest_Private/isaac_lab/source \
  /path/to/venv/bin/python -m pytest tests -q
```

(Use the real venv path from `../blueprint.md`.) Tests that import `pxr` require a
Kit-booting driver — that is a harness constraint, not a broken test; see the
blueprint.

Three packages the runtime does not need are the `test` extra —
`pip install -e 'src_isaac[test]'`: `pytest`, `psutil` (without it the suite fails
*collection*, not a test), and `pytest-xdist` (what lets the e2e tree run one worker per
GPU).

⚠ A plain `pytest tests` reports `582 passed, 23 deselected` (`581 passed, 1 skipped`
without `optuna` — that one skip line stands for 22 tests). The deselected 23 are the whole
`tests/e2e/` tree, which had **never run on any host** until 2026-07-27 because its design
sheet and media root pointed at paths that did not exist. They resolve on their own now:
the conftest picks the first media root that actually holds every clip the design sheet
names.

`src_isaac/scripts/install_git_hooks.sh` installs a `pre-push` hook that runs both groups
and blocks the push on failure. It discovers and *probes* the interpreter rather than
hardcoding one (both `nett_private` and `nett-private` venvs exist across the hosts, and
the wrong one is empty but executable), exports `LD_LIBRARY_PATH` for `libGLU.so.1`, and
runs `src_isaac/scripts/prepare_test_env.py` to build the external prerequisites — chiefly
the compiled frame cache under `~/.cache/nett_frame_cache`.

**4m32s total** (measured 2026-07-29, 8×A10): the e2e tree runs one worker per GPU, with
`NETT_KIT_THREADS` set to `cores/jobs` — without that budget N parallel cells oversubscribe
the host and run slower than serial. Serialised on one card the same tree is 17m19s.
`e2e_perf` is excluded from the gate on purpose: it benchmarks the *machine*, not the diff,
and blocked a push once on a 0.8% miss against another host's baseline. Run it deliberately
with `pytest src_isaac/tests/e2e -m e2e_perf`. `NETT_SKIP_E2E=1 git push` skips e2e
entirely. Full detail: `src_isaac/docs/development.md`.
