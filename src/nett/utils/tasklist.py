"""TaskList class for holding and creating tasks"""

from itertools import product
from pathlib import Path

from nett.body.body import Body
from nett.brain.brain import Brain
from nett.environment.environment import Environment
from .task import Task


class TaskList:

    def __init__(
        self,
        brain: Brain,
        body: Body,
        env: Environment,
        num_brains: int,
        conditions: list[str],
        output_dir: Path,
        modes: list[str],
        memory: float,
    ):
        self.brain_env_combinations = product(range(1, num_brains + 1), conditions)
        self.n_tasks = len(conditions) * num_brains

        self.current = 0

        self.tasks = [
            Task(brain, body, env, brain_id, condition, output_dir, modes, memory)
            for brain_id, condition in self.brain_env_combinations
        ]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.n_tasks:
            raise StopIteration
        current = self.current
        self.current += 1
        return self.tasks[current]
