"""
Callbacks for training the agents.

Classes:
    HParamCallback(BaseCallback)
"""

from multiprocessing import SimpleQueue
from pathlib import Path
import sys
import torch as th

# RLlib callback compatibility 
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ...utils.memory import MemoryManager
from ...utils.loading_bar_queue import LoadingBarQueue
import re
import glob
import cv2

# import numpy as np

# from nett.utils.performance import compute_train_performance

def img2video(record_path: Path, expected_length: int, fps: int = 25):
    png_files = glob.glob(str(record_path / "*.png"))
    if not png_files:
        return

    # Group pngs by episode number
    episode_dict = {}
    pattern = re.compile(r"(\d+)_(\d+)\.png$")
    for png in png_files:
        match = pattern.search(png)
        if match:
            ep, frame = int(match.group(1)), int(match.group(2))
            episode_dict.setdefault(ep, []).append((frame, png))

    for ep, frames in episode_dict.items():
        # Sort frames by frame number
        frames_sorted = sorted(frames, key=lambda x: x[0])
        if frames_sorted[-1][0] != expected_length:
            continue
        images = [cv2.imread(f[1]) for f in frames_sorted]
        if not images or images[0] is None:
            continue
        height, width, layers = images[0].shape
        mp4_path = record_path / f"{ep}.mp4"
        out = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for img in images:
            if img is not None:
                out.write(img)
        out.release()
        # Optionally, remove PNGs after conversion
        for _, png_path in frames_sorted:
            try:
                Path(png_path).unlink()
            except Exception:
                pass

# Base callback class for RLlib compatibility
class BaseCallback:
    """Base callback class to mimic SB3 BaseCallback interface."""
    
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.logger = None
        self.locals = {}
        self.globals = {}
        self.model = None
    
    def init_callback(self, model) -> None:
        """Initialize callback with model."""
        self.model = model
    
    def _on_training_start(self) -> None:
        """Called at the start of training."""
        pass
    
    def _on_step(self) -> bool:
        """Called after each step. Return False to stop training."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of rollout."""
        pass
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        pass


# TODO (v0.4): refactor needed, especially logging
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        # For RLlib, hyperparameter logging is handled differently
        # This is a placeholder for compatibility
        if hasattr(self.model, '_algorithm') and self.model._algorithm:
            algorithm = self.model._algorithm
            hparam_dict = {
                "algorithm": algorithm.__class__.__name__,
                "learning_rate": getattr(self.model, 'learning_rate', 'unknown'),
                "batch_size": getattr(self.model, 'batch_size', 'unknown'),
                "n_steps": getattr(self.model, 'n_steps', 'unknown'),
            }
            
            # RLlib logging would be handled through the algorithm's logger
            print(f"Training started with hyperparameters: {hparam_dict}")

    def _on_step(self) -> bool:
        return True


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


class IntrinsicRewardCallback(BaseCallback):
    """
    A unified callback for combining RLeXplore with RLlib algorithms.
    """

    def __init__(self, irs, verbose=0):
        super().__init__(verbose)
        self.irs = irs

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # RLlib callback integration would be different
        # This is a placeholder implementation
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of rollout."""
        # Intrinsic reward computation would be integrated with RLlib's training loop
        pass


class PngToMp4Callback(BaseCallback):
    """
    Callback to convert episode PNG frames to MP4 videos after each rollout or episode.
    PNGs must be named as <episode>_<frame>.png.
    """

    def __init__(self, record_path: Path, expected_size: int, fps: int = 24, verbose: int = 0):
        super().__init__(verbose)
        if not record_path.exists():
            record_path.mkdir(parents=True)

        self.record_path = record_path
        self.fps = fps
        self.expected_size = expected_size

    def _on_rollout_end(self) -> None:
        self._convert_pngs_to_mp4s()

    def _on_step(self) -> bool:
        # Optionally, you can also call conversion here if you want per-episode conversion
        return True

    def _convert_pngs_to_mp4s(self):
        return img2video(self.record_path, self.expected_size, self.fps)
