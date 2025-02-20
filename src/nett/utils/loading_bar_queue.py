import sys
import time
from multiprocessing import Manager

from tqdm import tqdm


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class LoadingBarQueue:
    def __init__(self) -> None:
        self.manager = Manager()
        self.queue = self.manager.Queue()
        self.pbar: dict[str, int] = {}
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
        # Wait for results and update progress
        while not self.queue.empty():
            pkg = self.queue.get()
            if pkg == "close":
                return True
            else:
                label, num_steps = pkg
                self.pbar[label].update(num_steps)

        return False

    def remove(self, label: str) -> None:
        self.pbar[label].refresh()
        self.pbar[label].close()
        del self.pbar[label]

    def close(self) -> None:
        for pbar in self.pbar.values():
            pbar.refresh()
            pbar.close()
        self.queue.put(None)


def updateLoadingBars(loading_bar_queue: LoadingBarQueue) -> None:
    done = False
    while not done:
        done = loading_bar_queue.update()
        time.sleep(0.1)
