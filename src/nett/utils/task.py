"""Task class for holding information for each task to run"""

import logging
from pathlib import Path
from typing import Optional


class TaskConfig:
    """TaskConfig class for holding and creating tasks"""

    device: int
    current_mode: str
    brain_id: int
    condition: str
    modes: list[str]
    n_parallel_envs: int
    memory: Optional[float]
    path: Path
    logger: logging.Logger

    def __init__(
        self,
        brain_id: int,
        condition: str,
        output_dir: Path,
        modes: list[str],
        n_parallel_envs: int,
        memory: Optional[float] = None,
    ):
        self.brain_id = brain_id
        self.condition = condition
        self.modes = modes
        self.n_parallel_envs = n_parallel_envs
        self.memory = memory
        self.path = output_dir.joinpath(condition, f"brain_{brain_id}")
        self.logger = logging.getLogger(f"nett.task-{condition}-{brain_id}")


class Agent:
    """Agent class for running tasks in parallel"""

    brain: "Brain"
    body: "Body"
    env: "Environment"

    def __init__(
        self,
        brain: "Brain",
        body: "Body",
        env: "Environment",
    ):
        self.brain = brain
        self.body = body
        self.env = env


class Task:
    """Holds information for a task"""

    config: TaskConfig
    agent: Agent

    def __init__(
        self,
        brain: "Brain",
        body: "Body",
        env: "Environment",
        brain_id: int,
        condition: str,
        output_dir: Path,
        modes: list[str],
        memory: Optional[float] = None,
    ) -> None:
        """initialize task"""
        n_parallel_envs = getattr(brain, "n_parallel_envs", 1)
        self.config = TaskConfig(
            brain_id, condition, output_dir, modes, n_parallel_envs, memory
        )
        self.agent = Agent(brain, body, env)

    def set_device(self, device: int) -> None:
        self.config.device = device


# Split up task into BBE and else
def run_task(task: Task) -> None:
    config = task.config
    agent = task.agent

    for mode in config.modes:
        config.current_mode = mode

        # create log path
        log_path = config.path / "env_logs"
        log_path.mkdir(exist_ok=True, parents=True)

        with agent.body.embed(agent.env, config) as body_interface:
            # brain.train() or brain.test()
            getattr(agent.brain, mode)(body_interface, config)
            config.logger.info(f"Closing Environment...")

    config.logger.info("Environments Closed")
