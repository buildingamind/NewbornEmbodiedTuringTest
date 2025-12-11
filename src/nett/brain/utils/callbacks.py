"""
Callbacks for training the agents.

This module provides custom callback classes for stable-baselines3 agents, including
hyperparameter logging, progress bars, memory monitoring, intrinsic rewards, and
video conversion utilities.

Classes:
    HParamCallback: Logs hyperparameters and metrics to TensorBoard.
    LoadingBarCallback: Displays training progress using a queue-based system.
    MemoryCallback: Monitors GPU memory usage during training.
    IntrinsicRewardWithOnPolicyRL: Integrates intrinsic rewards with on-policy RL algorithms.
    IntrinsicRewardWithOffPolicyRL: Integrates intrinsic rewards with off-policy RL algorithms.
    PngToMp4Callback: Converts episode PNG frames to MP4 videos.

Functions:
    img2video: Converts PNG frames to MP4 video files.
"""

from multiprocessing import SimpleQueue
from pathlib import Path
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.base_class import BaseAlgorithm

from ...utils.memory import MemoryManager
import re
import glob
import cv2

# import numpy as np

# from nett.utils.performance import compute_train_performance

def img2video(record_path: Path, expected_length: int, fps: int = 25):
    """
    Convert PNG frames to MP4 video files.

    Groups PNG files by episode number and converts complete episode sequences
    into MP4 videos. PNG files are expected to be named as <episode>_<frame>.png.
    After successful conversion, the original PNG files are deleted.

    Args:
        record_path: Directory containing PNG files to convert.
        expected_length: Expected number of frames per episode. Only episodes with
            this exact number of frames will be converted.
        fps: Frames per second for the output video. Defaults to 25.

    Note:
        - PNG files that don't match the pattern <episode>_<frame>.png are ignored.
        - Incomplete episodes (fewer frames than expected_length) are skipped.
        - Original PNG files are deleted after successful conversion.
    """
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
        height, width, _ = images[0].shape
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

# TODO (v0.4): refactor needed, especially logging
class HParamCallback(BaseCallback):
    """
    Logs hyperparameters and metrics to TensorBoard at training start.

    This callback saves the model's hyperparameters and associated metrics at the
    beginning of training, making them available in TensorBoard's HPARAMS tab for
    experiment comparison and analysis.
    """

    def _on_training_start(self) -> None:
        """
        Log hyperparameters and metrics to TensorBoard.

        Creates a dictionary of key hyperparameters (algorithm, learning rate, gamma,
        batch_size, n_steps) and metrics (episode length, value loss) and logs them
        to TensorBoard using the HParam logger.
        """
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size": self.model.batch_size,
            "n_steps": self.model.n_steps,
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
        """
        Called after each environment step.

        Returns:
            True to continue training.
        """
        return True


class LoadingBarCallback(BaseCallback):
    """
    Displays training progress using a queue-based communication system.

    This callback updates a progress bar by sending messages through a queue,
    allowing for progress tracking across multiple processes or threads.

    Args:
        label: Prefix label for the progress bar.
        queue: SimpleQueue for inter-process communication with the progress bar.
    """

    def __init__(self, label: str, queue: SimpleQueue) -> None:
        """
        Initialize the loading bar callback.

        Args:
            label: Prefix label for the progress bar.
            queue: SimpleQueue for inter-process communication with the progress bar.
        """
        super().__init__()
        # label to prefix the progress bar
        self.label = label

        # queue to communicate with the loading bar process
        self.bar_queue = queue

    def _on_step(self) -> bool:
        """
        Update the progress bar after each environment step.

        Sends a progress update message through the queue.

        Returns:
            True to continue training.
        """
        # Update progress bar, we do num_envs steps per call to `env.step()`
        # self.pbar.update(self.training_env.num_envs)
        self.bar_queue.put((self.label, 1))  # self.training_env.num_envs
        return True


class MemoryCallback(BaseCallback):
    """
    Monitors and logs GPU memory usage during training.

    This callback tracks GPU memory usage and saves the free memory information
    to a file at the end of training, which is useful for analyzing resource
    utilization and optimizing training configurations.

    Args:
        device: GPU device ID to monitor.
        save_path: Directory path where memory information will be saved.
    """

    def __init__(self, device: int, save_path: str) -> None:
        """
        Initialize the memory monitoring callback.

        Args:
            device: GPU device ID to monitor.
            save_path: Directory path where memory information will be saved.
        """
        super().__init__()
        self.device = device
        self.save_path = save_path
        self.closeEnv = False
        self.memory_manager = MemoryManager()

    def _on_step(self) -> bool:
        """
        Check whether to continue training after each environment step.

        This method is called by the model after each call to env.step().
        For child callbacks (of an EventCallback), this is called when the event is triggered.

        Returns:
            False to abort training early (when closeEnv is True), True otherwise.
        """
        return not self.closeEnv

    def _on_rollout_end(self) -> None:
        """
        Trigger environment closure at the end of rollout.

        This event is triggered before updating the policy.
        """
        self.closeEnv = True

    def _on_training_end(self) -> None:
        """
        Save GPU memory usage information at the end of training.

        This event is triggered before exiting the learn() method.
        Captures free GPU memory and writes it to 'mem.txt' in the save_path directory.
        """
        # Grab the memory being used by the GPU
        free_memory = self.memory_manager.get_free_memory(self.device)
        # Write the used memory to a file
        with open(Path.joinpath(self.save_path, "mem.txt"), "w") as f:
            f.write(str(free_memory))


class IntrinsicRewardWithOnPolicyRL(BaseCallback):
    """
    Integrates intrinsic reward modules with on-policy RL algorithms.

    This callback combines RLeXplore intrinsic reward modules with on-policy
    algorithms from Stable-Baselines3 (e.g., PPO, A2C). It watches agent
    interactions, computes intrinsic rewards, and adds them to the rollout buffer.

    Args:
        irs: Intrinsic reward system (RLeXplore module).
        verbose: Verbosity level. Defaults to 0.

    Note:
        This callback modifies the advantages and returns in the rollout buffer
        by adding computed intrinsic rewards at the end of each rollout.
    """

    def __init__(self, irs, verbose=0):
        """
        Initialize the intrinsic reward callback for on-policy RL.

        Args:
            irs: Intrinsic reward system (RLeXplore module).
            verbose: Verbosity level. Defaults to 0.
        """
        super().__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        """
        Initialize callback with the model's rollout buffer.

        Args:
            model: The RL algorithm model.
        """
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer  #

    def _on_step(self) -> bool:
        """
        Watch agent interactions at each environment step.

        Extracts observations, actions, rewards, dones, and next observations from
        the current step and feeds them to the intrinsic reward system.

        Returns:
            True to continue training.
        """
        observations = self.locals["obs_tensor"]  #
        device = observations.device  #
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)  # ~

        # ===================== watch the interaction ===================== #
        self.irs.watch(
            observations, actions, rewards, dones, dones, next_observations
        )  # ~
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:  ####################################
        """
        Compute intrinsic rewards and update the rollout buffer.

        At the end of each rollout, this method computes intrinsic rewards for all
        collected transitions and adds them to the buffer's advantages and returns.
        This augments the extrinsic rewards with exploration bonuses.
        """
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        # compute the intrinsic rewards
        intrinsic_rewards = self.irs.compute(
            samples=dict(
                observations=obs,
                actions=actions,
                rewards=rewards,
                terminateds=dones,
                truncateds=dones,
                next_observations=new_obs,
            ),
            sync=True,
        )
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #

class IntrinsicRewardWithOffPolicyRL(BaseCallback):
    """
    Integrates intrinsic reward modules with off-policy RL algorithms.

    This callback combines RLeXplore intrinsic reward modules with off-policy
    algorithms from Stable-Baselines3 (e.g., SAC, TD3, DQN). It computes intrinsic
    rewards at each step and updates the intrinsic reward module using samples
    from the replay buffer.

    Args:
        irs: Intrinsic reward system (RLeXplore module).
        verbose: Verbosity level. Defaults to 0.

    Note:
        - Intrinsic rewards are added to extrinsic rewards at each step.
        - The intrinsic reward module is updated using samples from the replay buffer.
        - Updates are performed asynchronously (sync=False for compute).
    """

    def __init__(self, irs, verbose=0):
        """
        Initialize the intrinsic reward callback for off-policy RL.

        Args:
            irs: Intrinsic reward system (RLeXplore module).
            verbose: Verbosity level. Defaults to 0.
        """
        super().__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        """
        Initialize callback with the model's replay buffer.

        Args:
            model: The RL algorithm model.
        """
        super().init_callback(model)
        self.buffer = self.model.replay_buffer  #

    def _on_step(self) -> bool:
        """
        Compute intrinsic rewards and update the intrinsic reward module.

        At each environment step, this method:
        1. Watches the agent interaction with the intrinsic reward system.
        2. Computes intrinsic rewards for the current transition.
        3. Adds intrinsic rewards to the extrinsic rewards.
        4. Updates the intrinsic reward module using replay buffer samples.

        Returns:
            True to continue training.
        """
        device = self.irs.device  #
        obs = th.as_tensor(self.locals["self"]._last_obs, device=device)  #
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_obs = th.as_tensor(self.locals["new_obs"], device=device)  # ~

        # ===================== watch the interaction ===================== #
        self.irs.watch(obs, actions, rewards, dones, dones, next_obs)  # ~
        # ===================== watch the interaction ===================== #
        ####################################
        # ===================== compute the intrinsic rewards ===================== #
        intrinsic_rewards = self.irs.compute(
            samples={
                "observations": obs.unsqueeze(0),
                "actions": actions.unsqueeze(0),
                "rewards": rewards.unsqueeze(0),
                "terminateds": dones.unsqueeze(0),
                "truncateds": dones.unsqueeze(0),
                "next_observations": next_obs.unsqueeze(0),
            },
            sync=False,
        )
        # ===================== compute the intrinsic rewards ===================== #

        try:
            # add the intrinsic rewards to the original rewards
            self.locals["rewards"] += intrinsic_rewards.cpu().numpy().squeeze()
            # update the intrinsic reward module
            replay_data = self.buffer.sample(batch_size=self.irs.batch_size)
            self.irs.update(
                samples={
                    "observations": th.as_tensor(replay_data.observations)
                    .unsqueeze(1)
                    .to(device),  # (n_steps, n_envs, *obs_shape)
                    "actions": th.as_tensor(replay_data.actions)
                    .unsqueeze(1)
                    .to(device),
                    "rewards": th.as_tensor(replay_data.rewards).to(device),
                    "terminateds": th.as_tensor(replay_data.dones).to(device),
                    "truncateds": th.as_tensor(replay_data.dones).to(device),
                    "next_observations": th.as_tensor(replay_data.next_observations)
                    .unsqueeze(1)
                    .to(device),
                }
            )
        except:
            pass
        ####################################
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.

        This method is a no-op for off-policy algorithms since intrinsic reward
        computation and module updates are performed at each step.
        """
        pass


class PngToMp4Callback(BaseCallback):
    """
    Converts episode PNG frames to MP4 videos during training.

    This callback automatically converts PNG frames to MP4 videos at the end of
    each rollout. PNG files must be named following the pattern <episode>_<frame>.png.
    Only complete episodes (with expected_size frames) are converted.

    Args:
        record_path: Directory containing PNG frames and where videos will be saved.
        expected_size: Expected number of frames per complete episode.
        fps: Frames per second for output videos. Defaults to 24.
        verbose: Verbosity level. Defaults to 0.

    Note:
        After successful conversion, the original PNG files are automatically deleted.
    """

    def __init__(self, record_path: Path, expected_size: int, fps: int = 24, verbose: int = 0):
        """
        Initialize the PNG to MP4 conversion callback.

        Args:
            record_path: Directory containing PNG frames and where videos will be saved.
            expected_size: Expected number of frames per complete episode.
            fps: Frames per second for output videos. Defaults to 24.
            verbose: Verbosity level. Defaults to 0.
        """
        super().__init__(verbose)
        if not record_path.exists():
            record_path.mkdir(parents=True)

        self.record_path = record_path
        self.fps = fps
        self.expected_size = expected_size

    def _on_rollout_end(self) -> None:
        """
        Convert PNG frames to MP4 videos at the end of each rollout.
        """
        self._convert_pngs_to_mp4s()

    def _on_step(self) -> bool:
        """
        Called after each environment step.

        Returns:
            True to continue training.

        Note:
            Can be modified to perform per-episode conversion if needed.
        """
        # Optionally, you can also call conversion here if you want per-episode conversion
        return True

    def _convert_pngs_to_mp4s(self):
        """
        Execute the PNG to MP4 conversion process.

        Returns:
            Result of img2video function.
        """
        return img2video(self.record_path, self.expected_size, self.fps)
