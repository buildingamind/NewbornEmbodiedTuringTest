"""Task class for holding information for each task to run"""

import logging
from pathlib import Path
from typing import Optional


class Task:
    """Holds information for a task

    Args:
        brain_id (int): id for the brain
        condition (str): condition for the task
        estimate_memory (bool, optional): whether to estimate memory usage. Defaults to False.
    """

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
        self.config = TaskConfig(brain_id, condition, output_dir, modes, memory)
        self.agent = Agent(brain, body, env)

    def set_device(self, device: int) -> None:
        self.config.device = device


class TaskConfig:
    """TaskConfig class for holding and creating tasks"""
    device: int
    current_mode: str

    def __init__(
        self,
        brain_id: int,
        condition: str,
        output_dir: Path,
        modes: list[str],
        memory: Optional[float] = None,
    ):
        self.brain_id = brain_id
        self.condition = condition
        self.modes = modes
        self.memory = memory
        self.path: Path = output_dir.joinpath(condition, f"brain_{brain_id}")
        self.logger: logging.Logger = logging.getLogger(
            f"nett.task-{condition}-{brain_id}"
        )


class Agent:
    """Agent class for running tasks in parallel"""

    def __init__(
        self,
        brain: "Brain",
        body: "Body",
        env: "Environment",
    ):
        self.brain = brain
        self.body = body
        self.env = env


# Split up task into BBE and else
def run_task(task: Task) -> None:
    config = task.config
    agent = task.agent
    config.logger.info(f"Running {config.current_mode} task")

    for mode in config.modes:
        config.current_mode = mode
        try:
            # create log path
            log_path = config.path / "env_logs"
            log_path.mkdir(exist_ok=True, parents=True)

            with agent.body.embed(agent.env, config) as body_interface:
                # brain.train() or brain.test()
                getattr(agent.brain, mode)(body_interface, config)

        except Exception as e:
            config.logger.exception(f"{mode} env failed: {str(e)}")
            raise e

    config.logger.info("Environments Closed")
