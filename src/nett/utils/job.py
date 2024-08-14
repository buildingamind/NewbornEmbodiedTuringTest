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
  """

  _MODES: Final = ("train", "test", "full")

  @classmethod
  def initialize(cls, mode: str, steps_per_episode: int, checkpoint_freq: int, output_dir: Path | str, reward: str, save_checkpoints: bool, batch_mode: bool, iterations: dict[str, int]) -> None:
    """Initialize the class

    Args:
        mode (str): mode for the job
        steps_per_episode (int): number of steps per episode
        save_checkpoints (bool): whether to save checkpoints
        checkpoint_freq (int): frequency to save checkpoints
        batch_mode (bool): whether to run in batch mode
        output_dir (Path | str): output directory
        reward (str): reward type
    """
    cls.mode = cls._validate_mode(mode)
    cls.steps_per_episode: int = steps_per_episode
    cls.checkpoint_freq: int = checkpoint_freq
    cls.output_dir: Path = output_dir
    cls.reward: str = reward
    cls.save_checkpoints: bool = save_checkpoints
    cls.batch_mode: bool = batch_mode
    cls.iterations: dict[str, int] = iterations

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