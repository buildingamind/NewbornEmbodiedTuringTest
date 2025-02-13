from concurrent.futures import ProcessPoolExecutor #, Future
import os
import sys


class Executor(ProcessPoolExecutor):

    def __init__(self, verbose: bool) -> None:
        # mute stdout if not verbose
        mute = lambda: setattr(sys, "stdout", open(os.devnull, "w"))
        initializer = mute if not verbose else None

        super().__init__(
            initializer=initializer,  # TODO: too many workers
        )
