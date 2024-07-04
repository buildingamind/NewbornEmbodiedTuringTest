"""
Callbacks for training the agents.

Classes:
    HParamCallback(BaseCallback)
"""
import os
from typing import Optional

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.logger import HParam

# from nett.utils.train import compute_train_performance

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


# TODO (v0.4): refactor needed, especially logging
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size": self.model.batch_size,
            "n_steps": self.model.n_steps
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class multiBarCallback(ProgressBarCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    def __init__(self, index: Optional[int] = None) -> None:
        super().__init__()
        self.index = index

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.model.n_steps, position=self.index)
        # self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps, position=self.index)

class MemoryCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    """
    def __init__(self, device: int):
        super().__init__()
        self.device = device
        self.close = False

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.close:
            # Create a temporary directory to store the memory usage
            os.makedirs("./.tmp", exist_ok=True)
            # Grab the memory being used by the GPU
            used_memory = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(self.device)).used
            # Write the used memory to a file
            with open("./.tmp/memory_use", "w") as f:
                f.write(str(used_memory))
            # Close the callback
            return False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.close = True
        pass
