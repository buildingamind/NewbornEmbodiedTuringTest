"""Task construction helpers — one Task per imprint condition."""

from __future__ import annotations

from pathlib import Path

from .task import Task


def build_tasks(
    brain,
    wrappers,
    env,
    num_brains: int,
    conditions: list[str],
    output_dir: Path,
    modes: list[str],
    episodes: dict[str, int] | None,
    memory: float,
    brain_id_offset: int = 0,
    eval_freq: int | None = None,
) -> list[Task]:
    return [
        Task(
            brain, wrappers, env, condition, output_dir, modes,
            episodes=episodes,
            memory=memory,
            num_brains=num_brains,
            brain_id_offset=brain_id_offset,
            eval_freq=eval_freq,
        )
        for condition in conditions
    ]


def validate_tasklist(tasks: list[Task]) -> None:
    """Smoke-validate by loading each task's env once and folding its wrappers."""
    for task in tasks:
        config = task.config
        run_config = config.for_mode("train")
        log_path = config.path / "logs"
        log_path.mkdir(exist_ok=True, parents=True)
        loaded = task.agent.env.load(run_config)
        try:
            for wrapper in task.agent.wrappers:
                loaded = wrapper(loaded)
            obs, _ = loaded.reset()
            if not hasattr(obs, "shape") and not isinstance(obs, dict):
                raise TypeError(f"Unexpected obs type from wrapped env: {type(obs)}")
        finally:
            if hasattr(loaded, "close"):
                loaded.close()
