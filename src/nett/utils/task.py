"""Task class for holding information for each task to run"""

import logging
from pathlib import Path

from nett.body.body import Body
from nett.brain.brain import Brain
from nett.environment.environment import Environment


class Task:
    """Holds information for a task

    Args:
        brain_id (int): id for the brain
        condition (str): condition for the task
        estimate_memory (bool, optional): whether to estimate memory usage. Defaults to False.
    """

    device: int

    def __init__(
        self,
        brain: Brain,
        body: Body,
        env: Environment,
        mode: str,
        brain_id: int,
        condition: str,
        output_dir: Path,
        estimate_memory: bool = False,
    ) -> None:
        """initialize task"""
        self.mode: str = mode
        self.brain_id: int = brain_id
        self.condition: str = condition
        self.estimate_memory = estimate_memory
        self.path: Path = output_dir / self.condition / f"brain_{self.brain_id}"
        self.logger: logging.Logger = logging.getLogger(
            f"nett.task-{self.condition}-{self.brain_id}-{self.mode}"
        )
        self.brain = brain
        self.body = body
        self.env = env

    def run(self):
        try:
            # create log path
            log_path = self.path / "env_logs"
            log_path.mkdir(exist_ok=True, parents=True)

            with self.body.embed(self.env, self) as body_interface:
                getattr(self.brain, self.mode)(body_interface, self) # brain.train() or brain.test()

        except Exception as e:
            self.logger.exception(f"{self.mode} env failed: {str(e)}")
            raise e

        self.logger.info("Environments Closed")