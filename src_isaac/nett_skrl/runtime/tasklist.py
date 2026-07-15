"""Task construction helpers — one Task per imprint condition."""

from __future__ import annotations

from pathlib import Path

from .task import Task, set_seeds


def build_tasks(
    brain,
    body,
    env,
    num_brains: int,
    num_envs: int,
    conditions: list[str],
    output_dir: Path,
    modes: list[str],
    episodes: dict[str, int] | None,
    memory: float,
    brain_id_offset: int = 0,
    eval_freq: int | None = None,
    max_parallel_envs: int | None = None,
    max_test_envs: int | None = None,
) -> list[Task]:
    return [
        Task(
            brain, body, env, condition, output_dir, modes,
            episodes=episodes,
            memory=memory,
            num_brains=num_brains,
            num_envs=num_envs,
            brain_id_offset=brain_id_offset,
            eval_freq=eval_freq,
            max_parallel_envs=max_parallel_envs,
            max_test_envs=max_test_envs,
        )
        for condition in conditions
    ]


def validate_tasklist(tasks: list[Task]) -> None:
    """Smoke-validate by loading each task's env once through its body."""
    for task in tasks:
        config = task.config
        # Match the real run path (task_runner): seed before building the env so
        # the smoke-validation load is deterministic too (Critic W2).
        set_seeds(config.seed)
        run_config = config.for_mode("train")
        log_path = config.path / "logs"
        log_path.mkdir(exist_ok=True, parents=True)
        task.agent.body.adjust_to_agent(
            task.agent.env,
            num_brains=config.num_brains,
            num_envs=config.num_envs,
        )
        loaded = task.agent.env.load(run_config)
        try:
            loaded = task.agent.body.wrap(loaded)
            obs, _ = loaded.reset()
            if not hasattr(obs, "shape") and not isinstance(obs, dict):
                raise TypeError(f"Unexpected obs type from wrapped env: {type(obs)}")
        finally:
            if hasattr(loaded, "close"):
                loaded.close()
