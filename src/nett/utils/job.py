"""Job class for training and testing models"""
from pathlib import Path
from typing import Final, Any

from nett import logger

class Job:
  """Holds information for a job

  Args:
    brain_id (int): id for the brain
    condition (str): condition for the job
    device (int): device to run the job on
    dir (Path): directory to store the job 
    index (int): index for the job
    port (int): port for the job
  """

  _MODES: Final = ("train", "test", "full")

  @classmethod
  def initialize(cls, mode: str, steps_per_episode: int, save_checkpoints: bool, checkpoint_freq, batch_mode, output_dir, reward) -> None:

    cls.mode = cls._validate_mode(mode)
    cls.steps_per_episode: int = steps_per_episode
    cls.save_checkpoints: bool = save_checkpoints
    cls.checkpoint_freq: int = checkpoint_freq
    cls.batch_mode: bool = batch_mode
    cls.output_dir: Path = output_dir
    cls.reward: str = reward

  def __init__(self, brain_id: int, condition: str, device: int, index: int, port: int) -> None:
    """initialize job"""
    self.device: int = device
    self.condition: str = condition
    self.brain_id: int = brain_id

    self.paths: dict[str, Path] = self._configure_paths(brain_id, condition)
    self.index: int = index
    self.port: int = port

    # Initialize logger
    self.logger = logger.getChild(__class__.__name__+"."+condition+"."+str(brain_id))

  def _configure_paths(self, brain_id: int, condition: str) -> dict[str, Path]:
    """Configure Paths for the job

    Args:
        brain_id (int): id for the brain
        condition (str): condition for the job

    Returns:
        dict[str, Path]: dictionary of the paths
    """
    SUBDIRS = ["model", "checkpoints", "plots", "logs", "env_recs", "env_logs"]
    job_dir = Path.joinpath(self.output_dir, condition, f"brain_{brain_id}")
    return {subdir: Path.joinpath(job_dir, subdir) for subdir in SUBDIRS}
  
  def env_kwargs(self) -> dict[str, Any]:
    return {
      "rewarded": bool(self.reward == "supervised"),
      "rec_path": str(self.paths["env_recs"]),
      "log_path": str(self.paths["env_logs"]),
      "condition": self.condition,
      "brain_id": self.brain_id,
      "device": self.device,
      "episode_steps": self.steps_per_episode,
      "batch_mode": self.batch_mode
    }

  @staticmethod
  def _validate_mode(mode: str) -> str:
    """Validate the mode

    Args:
      mode (str): mode to validate

    Returns:
      str: mode
    """
    if mode not in Job._MODES:
      raise ValueError(f"Unknown mode type {mode}, should be one of {Job._MODES}")
    return mode