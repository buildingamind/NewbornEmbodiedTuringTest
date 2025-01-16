"""Task class for holding information for each task to run"""

from pathlib import Path


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
