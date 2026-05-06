from multiprocessing import SimpleQueue

from stable_baselines3.common.callbacks import BaseCallback


class LoadingBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent using tqdm
    """

    def __init__(self, label: str, queue: SimpleQueue) -> None:
        super().__init__()
        # label to prefix the progress bar
        self.label = label

        # queue to communicate with the loading bar process
        self.bar_queue = queue

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        # self.pbar.update(self.training_env.num_envs)
        self.bar_queue.put((self.label, 1))  # self.training_env.num_envs
        return True
