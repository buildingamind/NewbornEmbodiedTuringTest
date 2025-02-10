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

    device: int

    def __init__(
        self,
        brain: "Brain",
        body: "Body",
        env: "Environment",
        modes: list[str],
        brain_id: int,
        condition: str,
        output_dir: Path,
        memory: Optional[float] = None,
    ) -> None:
        """initialize task"""
        self.modes: list[str] = modes
        self.brain_id: int = brain_id
        self.condition: str = condition
        self.memory = memory
        self.path: Path = output_dir / self.condition / f"brain_{self.brain_id}"
        self.logger: logging.Logger = logging.getLogger(
            f"nett.task-{self.condition}-{self.brain_id}"
        )
        self.brain = brain
        self.body = body
        self.env = env

    def run(self):
        for mode in self.modes:
            self.current_mode = mode
            try:
                # create log path
                log_path = self.path / "env_logs"
                log_path.mkdir(exist_ok=True, parents=True)

                with self.body.embed(self) as body_interface:
                    # brain.train() or brain.test()
                    getattr(self.brain, mode)(
                        body_interface, self
                    )

            except Exception as e:
                self.logger.exception(f"{mode} env failed: {str(e)}")
                raise e

        self.logger.info("Environments Closed")
