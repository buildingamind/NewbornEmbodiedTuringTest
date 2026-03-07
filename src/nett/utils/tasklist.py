"""TaskList class for holding and creating tasks"""

from multiprocessing import SimpleQueue
import time
from itertools import product
from pathlib import Path

from .task import Task


class TaskList:
    n_tasks: int
    current: int
    tasks: list[Task]

    def __init__(
        self,
        brain: "Brain",
        body: "Body",
        env: "Environment",
        num_brains: int,
        conditions: list[str],
        output_dir: Path,
        modes: list[str],
        queue: SimpleQueue,
        memory: float,
    ):
        brain_env_combinations = list(product(range(1, num_brains + 1), conditions))
        self.n_tasks = len(brain_env_combinations)

        self.current = 0

        self.tasks = [
            Task(
                brain, body, env, brain_id, condition, output_dir, modes, queue, memory
            )
            for brain_id, condition in brain_env_combinations
        ]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= self.n_tasks:
            raise StopIteration
        current = self.current
        self.current += 1
        return self.tasks[current]


def validate_tasklist(tasklist: TaskList) -> None:
    """Validate the tasklist"""
    for task in tasklist.tasks:
        config = task.config
        agent = task.agent

        # run only once per condition
        if config.brain_id != 1:
            continue

        time.sleep(1)
        # validation only for train
        task.config.current_mode = "train"

        # create log path
        log_path = config.path / "logs"
        log_path.mkdir(exist_ok=True, parents=True)

        agent.body.validate_env(agent.env, config)
