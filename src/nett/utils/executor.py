from concurrent.futures import ProcessPoolExecutor  # , Future
import os
import sys
import threading

from .loading_bar_queue import LoadingBarQueue, updateLoadingBars


class Executor(ProcessPoolExecutor):

    def __init__(self, verbose: bool) -> None:
        # mute stdout if not verbose
        def mute():
            sys.stdout = open(os.devnull, "w")

        initializer = mute if not verbose else None

        super().__init__(
            max_workers=os.cpu_count(),
            initializer=initializer,
            # max_tasks_per_child=1  # Ensure each task runs in a fresh process
        )

        self.loading_bar = LoadingBarQueue()

        self.loading_bar_thread = threading.Thread(
            target=updateLoadingBars, args=[self.loading_bar]
        )
        self.loading_bar_thread.start()

    def __enter__(self):
        return self, self.loading_bar.queue

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loading_bar.queue.put("close")
        self.loading_bar_thread.join()
        self.loading_bar.close()
        super().__exit__(exc_type, exc_val, exc_tb)
        return False  # Don't suppress exceptions
