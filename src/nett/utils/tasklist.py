"""TaskList class for holding and creating tasks"""

from itertools import product
from pathlib import Path
from . import Task


class TaskList:
    def __init__(self, num_brains: int, conditions: list[str], output_dir: Path):
        self.output_dir = output_dir
        self.conditions = conditions
        self.conditions = conditions
        self.brain_env_combinations = product(conditions, range(1, num_brains + 1))
        self.n_tasks = len(conditions) * num_brains

    def __call__(self, mode: str, episodes: int):
        self.mode = mode

        self.current = 0

        self.tasks = [
            Task(mode, brain, condition, self.output_dir)
            for brain, condition in self.brain_env_combinations
        ]
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.n_tasks:
            raise StopIteration
        current = self.current
        self.current += 1
        return self.tasks[current]
