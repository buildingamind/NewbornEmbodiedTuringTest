from concurrent.futures import ProcessPoolExecutor, Future
import os
import sys

from nett.utils.task import Task, run_task


class Executor:

    def __init__(self, verbose: bool) -> None:
        # mute stdout if not verbose
        mute = lambda: setattr(sys, "stdout", open(os.devnull, "w"))
        initializer = mute if not verbose else None

        self.executor = ProcessPoolExecutor(
            initializer=initializer,  # TODO: too many workers
        )

    def submit(self, task: Task) -> Future:
        return self.executor.submit(run_task, task)

    def close(self) -> None:
        self.executor.shutdown()
