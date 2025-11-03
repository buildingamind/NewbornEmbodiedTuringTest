"""Loading bar queue utility for managing progress bars across multiple processes."""

import sys
import time
from multiprocessing import Manager

from tqdm.auto import tqdm

def singleton(cls):
    """
    Decorator to ensure a class has only one instance.

    Args:
        cls: The class to decorate.

    Returns:
        Function that returns the singleton instance.
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class LoadingBarQueue:
    """
    Manages multiple progress bars across processes using a shared queue.

    Provides methods to add, update, and remove progress bars for tracking
    long-running tasks in multi-process applications.
    """

    def __init__(self) -> None:
        """Initialize the loading bar queue with a multiprocessing manager."""
        self.manager = Manager()
        self.queue = self.manager.Queue()
        self.pbar: dict[str, int] = {}
        self.rows = 0

    def add(self, label: str, num_steps: int) -> None:
        """
        Add a new progress bar.

        Args:
            label: Label for the progress bar.
            num_steps: Total number of steps for the progress bar.
        """
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
        """
        Update all progress bars with pending updates from the queue.

        Returns:
            True if a close signal was received, False otherwise.
        """
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
        """
        Remove a progress bar.

        Args:
            label: Label of the progress bar to remove.
        """
        self.update()
        self.pbar[label].refresh()
        self.pbar[label].close()
        del self.pbar[label]
        self.rows -= 1

    def close(self) -> None:
        """Close all progress bars and clean up resources."""
        self.update()
        for pbar in self.pbar.values():
            pbar.refresh()
            pbar.close()
        self.queue.put(None)


def updateLoadingBars(loading_bar_queue: LoadingBarQueue) -> None:
    """
    Continuously update loading bars until a close signal is received.

    Args:
        loading_bar_queue: LoadingBarQueue instance to update.
    """
    done = False
    while not done:
        done = loading_bar_queue.update()
        time.sleep(0.1)
