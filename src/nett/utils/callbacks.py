"""
Callbacks for training the agents.

Classes:
    HParamCallback(BaseCallback)
"""
import os
from pathlib import Path
import sys
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

    callback_list = [hparam_callback]

    if job.estimate_memory:
        callback_list.extend([
            # creates the parallel progress bars
            multiBarCallback(job.index, "Estimating Memory Usage"), # TODO: Add progress bars to test aswell
            # creates the memory callback for estimation of memory for a single job
            MemoryCallback(job.device, save_path=job.paths["base"])
            ])
    else:
        # creates the parallel progress bars
        callback_list.append(multiBarCallback(job.index, f"{job.condition}-{job.brain_id}", job.iterations["train"])) # TODO: Add progress bars to test aswell

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

class multiBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    def __init__(self, index: int, label: str, num_steps: int = None) -> None:
        super().__init__()
        # where on the screen the progress bar will be displayed
        self.index = index
        # label to prefix the progress bar
        self.label = label
        # progress bar object
        self.pbar = None
        # number of steps to be done
        self.num_steps = num_steps

    def _on_training_start(self) -> None:
        # if num_steps is None, this means that memory estimation is being done, so the length of a single rollout will be used
        num_steps = self.num_steps if self.num_steps is not None else self.model.n_steps
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=(num_steps), position=self.index, dynamic_ncols=True, desc=self.label, file=sys.stdout)
        pass

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        self.pbar.refresh()
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
