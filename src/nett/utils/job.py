"""Job class for training and testing models"""
from pathlib import Path
from typing import Final, Any

class Job:
  """Holds information for a job

  Args:
    brain_id (int): id for the brain
    condition (str): condition for the job
    device (int): device to run the job on
    index (int): index for the job
    port (int): port for the job
    estimate_memory (bool, optional): whether to estimate memory usage. Defaults to False.
  """

  _MODES: Final = ("train", "test", "full")
  _RECORD: Final = ("agent", "chamber", "state")

  @classmethod
  def initialize(cls, mode: str, output_dir: Path | str, steps_per_episode: int, save_checkpoints: bool, checkpoint_freq: int,  reward: str, batch_mode: bool, iterations: dict[str, int], record: list[str], recording_eps: int) -> None:
    """Initialize the class

    Args:
        mode (str): mode for the job
        output_dir (Path | str): output directory
        save_checkpoints (bool): whether to save checkpoints
        steps_per_episode (int): number of steps per episode
        checkpoint_freq (int): frequency to save checkpoints
        reward (str): reward type
        batch_mode (bool): whether to run in batch mode
        iterations (dict[str, int]): number of iterations for the job with labels "train" and/or "test" to denote the number of iterations for training and testing
        record (list[str]): list of what to record
        recording_eps (int): number of episodes to record
    """
    cls.mode = cls._validate_mode(mode)
    cls.steps_per_episode: int = steps_per_episode
    cls.checkpoint_freq: int = checkpoint_freq
    cls.output_dir: Path = output_dir
    cls.reward: str = reward
    cls.save_checkpoints: bool = save_checkpoints
    cls.batch_mode: bool = batch_mode
    cls.iterations: dict[str, int] = iterations
    cls.record: list[str] = cls._validate_record(record)
    cls.recording_eps: int = recording_eps

  def __init__(self, brain_id: int, condition: str, device: int, index: int, port: int, estimate_memory: bool = False) -> None:
    """initialize job"""
    self.device: int = device
    self.condition: str = condition
    self.brain_id: int = brain_id

    self.paths: dict[str, Path] = self._configure_paths()
    self.index: int = index
    self.port: int = port
    self.estimate_memory: bool = estimate_memory

    # Initialize logger
    from nett import logger

    self.logger = logger.getChild(__class__.__name__+"."+condition+"."+str(brain_id))


  def _configure_paths(self) -> dict[str, Path]:
    """Configure Paths for the job

    Args:
        output_dir (Path): output directory
        brain_id (int): id for the brain
        condition (str): condition for the job

    Returns:
        dict[str, Path]: dictionary of the paths
    """
    paths: dict[str, Path] = {
      "base": Path.joinpath(self.output_dir, self.condition, f"brain_{self.brain_id}")
      }
    SUBDIRS = ["model", "checkpoints", "plots", "logs", "env_recs", "env_logs"]
    for subdir in SUBDIRS:
      paths[subdir] = Path.joinpath(paths["base"], subdir)

    return paths

  def env_kwargs(self) -> dict[str, Any]:
    """Get the environment kwargs
    """
    return {
      "rewarded": bool(self.reward == "supervised"),
      "rec_path": str(self.paths["env_recs"]),
      "log_path": str(self.paths["env_logs"]),
      "condition": self.condition,
      "brain_id": self.brain_id,
      "device": self.device,
      "episode_steps": self.steps_per_episode,
      "batch_mode": self.batch_mode,
      "recording-eps": self.recording_eps,
      "record-chamber": "chamber" in self.record,
      "record-agent": "agent" in self.record
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
  
  @staticmethod
  def _validate_record(record: list[str]) -> list[str]:
    """Validate the record options

    Args:
      record (list[str]): record list to validate

    Returns:
      list[str]: record
    """
    for r in record:
      if r not in Job._RECORD:
        raise ValueError(f"Unknown record option {r}, should be one of {Job._RECORD}")
    return record