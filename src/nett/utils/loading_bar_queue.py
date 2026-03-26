import queue as stdlib_queue
import sys
import time
from multiprocessing import Manager

from tqdm.auto import tqdm


class LoadingBarQueue:
    def __init__(self) -> None:
        self._manager = Manager()
        self.queue = self._manager.Queue()
        self.pbar: dict[str, tqdm] = {}
        self.rows = 0

    def add(self, label: str, num_steps: int) -> None:
        self.pbar[label] = tqdm(
            total=num_steps,
            position=self.rows,
            dynamic_ncols=True,
            desc=f"{label}: ",
            file=sys.stdout,
            leave=True,
            mininterval=1.0,
        )
        self.rows += 1

    def update(self) -> bool:
        while True:
            try:
                pkg = self.queue.get_nowait()
            except stdlib_queue.Empty:
                return False
            if pkg == "close":
                return True
            label, num_steps = pkg
            self.pbar[label].update(num_steps)

    def remove(self, label: str) -> None:
        self.update()
        self.pbar[label].refresh()
        self.pbar[label].close()
        del self.pbar[label]
        self.rows -= 1

    def close(self) -> None:
        self.update()
        for pbar in self.pbar.values():
            pbar.refresh()
            pbar.close()
        self._manager.shutdown()


def updateLoadingBars(loading_bar_queue: LoadingBarQueue) -> None:
    done = False
    while not done:
        done = loading_bar_queue.update()
        time.sleep(0.1)
