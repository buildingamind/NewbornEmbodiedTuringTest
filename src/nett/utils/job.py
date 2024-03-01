from nett import Brain, Body, Environment
from copy import deepcopy
from pathlib import Path

class Job:
  def __init__(self, brain_id: int, condition: str, device: int):
    self.device: int = device
    self.condition: str = condition
    self.brain_id: int = brain_id
    self.paths: dict[str, Path] = self._configure_paths(brain_id, condition)
  
  def _configure_paths(self, brain_id: int, condition: str) -> dict[str, Path]:
    SUBDIRS = ["model", "checkpoints", "plots", "logs", "env_recs", "env_logs"]
    job_dir = Path.joinpath(self.output_dir, condition, f"brain_{brain_id}")
    return {subdir: Path.joinpath(job_dir, subdir) for subdir in SUBDIRS}
    