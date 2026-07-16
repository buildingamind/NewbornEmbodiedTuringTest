# NewbornEmbodiedTuringTest

The maintained backend for the Newborn Embodied Turing Test (NETT) is the
**Isaac Lab + skrl** package under [`src_isaac/`](src_isaac/README.md).

> The legacy Unity/stable-baselines3 package (`nett-benchmarks`, formerly under
> `src/nett`) has been removed from this branch. See git history if you need it.

## Where things are

- [`src_isaac/`](src_isaac/README.md) — the `nett_skrl` package: run
  orchestration, skrl PPO brain, observation body, environment bridge into the
  Isaac Lab `NETTEnv`, and analysis. Has its own `pyproject.toml` (`nett-skrl`),
  docs (`src_isaac/docs/`), examples (`src_isaac/examples/`), and tests
  (`src_isaac/tests/`).
- The Isaac Lab environment / robot / render side lives in the sibling repo
  `NewbornEmbodiedTuringTest_Private/isaac_lab`.

## Ground truth

The consolidated, verified technical state of the two-repo Isaac stack — how to
run it, the venv, the execution loop, known issues, and the determinism story —
is **`../blueprint.md`** at the project root. Start there.
