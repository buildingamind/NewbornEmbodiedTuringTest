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

⚠ A plain `pytest tests` reports `542 passed, 1 skipped, 23 deselected` — and the
deselected 23 are the whole `tests/e2e/` tree, which had **never run on any host** until
2026-07-27 because its design sheet and media root resolve to paths that do not exist in
repoA. Point `NETT_DESIGN_SHEET_MINIMAL` / `NETT_MEDIA_ROOT` at `<workspace>/videos/binding/`
to actually run them. The single remaining skip line is `optuna`, and it stands for 22
tests, all of which pass when optuna is installed.

`src_isaac/scripts/install_git_hooks.sh` installs a `pre-push` hook that runs both groups
and blocks the push on failure. Full detail: `src_isaac/docs/development.md`.
