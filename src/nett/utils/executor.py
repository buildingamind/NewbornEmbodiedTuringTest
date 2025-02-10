from concurrent.futures import ProcessPoolExecutor, Future
import logging
import os
import sys

from nett.brain.brain import Brain
from nett.environment.environment import Environment
from nett.utils.task import Task


class Executor:

    def __init__(self, verbose: bool) -> None:
        # mute stdout if not verbose
        mute = lambda: setattr(sys, "stdout", open(os.devnull, "w"))
        initializer = mute if not verbose else None

        self.executor = ProcessPoolExecutor(
            initializer=initializer,  # TODO: too many workers
        )

    def submit(self, func: callable) -> Future:
        return self.executor.submit(func)

    def close(self) -> None:
        self.executor.shutdown()
