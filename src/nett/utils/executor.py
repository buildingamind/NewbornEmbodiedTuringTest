from concurrent.futures import ProcessPoolExecutor  # , Future
import os
import sys
import threading

from .loading_bar_queue import LoadingBarQueue, updateLoadingBars


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Executor(ProcessPoolExecutor):

    def __init__(self, verbose: bool) -> None:
        # mute stdout if not verbose
        mute = lambda: setattr(sys, "stdout", open(os.devnull, "w"))
        initializer = mute if not verbose else None

        super().__init__(
            initializer=initializer,  # TODO: too many workers
        )

        self.loading_bar = LoadingBarQueue()

        self.loading_bar_thread = threading.Thread(
            target=updateLoadingBars, args=[self.loading_bar]
        )
        self.loading_bar_thread.start()

    def __enter__(self):
        return self, self.loading_bar.queue

    def __exit__(self):
        self.loading_bar.queue.put("close")
        self.loading_bar_thread.join()
        self.loading_bar.close()
        return super().__exit__()
