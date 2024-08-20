"""
Callbacks for training the agents.

Classes:
    HParamCallback(BaseCallback)
"""
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import HParam

# from nett.utils.train import compute_train_performance

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

def initialize_callbacks(job: "Job") -> CallbackList:
    """
    Initialize the callbacks for training.

    Args:
        job (Job): The job for which to initialize the callbacks.
    
    Returns:
        CallbackList: The list of callbacks for training.
    """
    hparam_callback = HParamCallback() # TODO: Are we using the tensorboard that this creates? See https://www.tensorflow.org/tensorboard Appears to be responsible for logs/events.out.. files

    # creates the parallel progress bars
    loading_bar_callback = multiBarCallback(job.index)

    callback_list = [hparam_callback, loading_bar_callback]

    if job.estimate_memory:
        callback_list.append(MemoryCallback(job.device, save_path=job.paths["base"]))

    if job.save_checkpoints:
        callback_list.append(CheckpointCallback(
            save_freq=job.checkpoint_freq, # defaults to 30_000 steps
            save_path=job.paths["checkpoints"],
            save_replay_buffer=True,
            save_vecnormalize=True))

    return CallbackList(callback_list)

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
        try:
            self.logger.info(f"total timesteps: {self._total_timesteps}")
        except Exception:
            self.logger.info("total timesteps not found")
        try:
            self.logger.info(f"num steps at start: {self._num_timesteps_at_start}")
        except Exception:
            self.logger.info("num steps at start not found")
        try:
            self.logger.info(f"num steps: {self.num_timesteps}")
        except Exception:
            self.logger.info("num steps not found")

        self.pbar = tqdm(total=(self.model.n_steps), position=self.index)
        pass
    def _on_training_end(self) -> None:
        self.pbar.close()
        pass

class MemoryCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    """
    def __init__(self, device: int, save_path: str) -> None:
        super().__init__()
        self.device = device
        self.save_path = save_path
        self.close = False
        nvmlInit()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.close:
            # Create a temporary directory to store the memory usage
            # os.makedirs("./.tmp", exist_ok=True)
            # Grab the memory being used by the GPU
            used_memory = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(self.device)).used
            # Write the used memory to a file
            with open(Path.joinpath(self.save_path, "mem.txt"), "w") as f:
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
