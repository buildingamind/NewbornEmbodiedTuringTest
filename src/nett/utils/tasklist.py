"""TaskList class for holding and creating tasks"""

from itertools import product
from pathlib import Path
from .task import Task


class TaskList:
    @classmethod
    def initialize(cls, num_brains: int, conditions: list[str], output_dir: Path):
        cls.output_dir = output_dir
        cls.conditions = conditions
        cls.brain_env_combinations = product(range(1, num_brains + 1), conditions)
        cls.n_tasks = len(conditions) * num_brains

    def __init__(self, mode: str):
        self.mode = mode

        self.current = 0

        self.tasks = [
            Task(mode, brain, condition, self.output_dir)
            for brain, condition in self.brain_env_combinations
        ]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.n_tasks:
            raise StopIteration
        current = self.current
        self.current += 1
        return self.tasks[current]

    @classmethod
    def example(cls) -> Task:
        return Task("train", 0, cls.conditions[0], cls.output_dir, estimate_memory=True)
