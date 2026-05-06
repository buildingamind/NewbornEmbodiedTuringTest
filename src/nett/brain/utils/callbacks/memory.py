from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from ....utils.memory import MemoryManager


class MemoryCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    """

    def __init__(self, device: int, save_path: str) -> None:
        super().__init__()
        self.device = device
        self.save_path = save_path
        self.closeEnv = False
        self.memory_manager = MemoryManager()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return not self.closeEnv

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.closeEnv = True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Grab the memory being used by the GPU
        free_memory = self.memory_manager.get_free_memory(self.device)
        # Write the used memory to a file
        with open(Path.joinpath(self.save_path, "mem.txt"), "w") as f:
            f.write(str(free_memory))
