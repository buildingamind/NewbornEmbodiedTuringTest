"""NETT-skrl orchestrator.

Same UX as the legacy `nett.NETT` class — YAML/dict configs in, one (or many)
benchmark run(s) out. The behavioral differences are entirely below the
public API:

  - One process per imprint condition (not per brain x condition).
  - N brains share one vectorized Isaac env via :class:`MultiBrainTrainer`;
    each brain can own multiple parallel env rows.
  - No mlagents port juggling; Isaac Sim doesn't reserve ports.
  - ``task_memory`` declares (or ``"auto"`` measures) per-task VRAM so the
    scheduler can pack multiple tasks per GPU.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, as_completed, wait as future_wait
from pathlib import Path
from typing import Optional

import yaml

from .body import Body
from .brain import Brain
from .environment import Environment
from .runtime import (
    Executor,
    Task,
    TaskConfig,
    build_tasks,
    run_task,
)
from .runtime.memory import MemoryManager
from .utils import validate_config
from .runtime.tasklist import validate_tasklist


# Reserved fallback when dry-run estimation fails for any reason.
_FALLBACK_TASK_MEMORY_GB = 6.0


def _load_schema() -> dict:
    with open(Path(__file__).resolve().parent / "schema.json") as f:
        return json.load(f)


def _modes_from_episodes(episodes: dict[str, int]) -> list[str]:
    return [m for m in ("train", "test", "record") if episodes.get(m, 0) > 0]


def _make_body(body: Optional[dict], wrappers: Optional[list]) -> Body:
    body_config = dict(body or {})
    if wrappers and body_config.get("wrappers"):
        raise ValueError("Specify body.wrappers or top-level wrappers, not both.")
    if wrappers:
        body_config["wrappers"] = wrappers
    return Body(**body_config)


class JobTooBigError(ValueError):
    """No GPU has enough free memory for a single task."""
    def __init__(self):
        super().__init__(
            "No jobs could be scheduled. Task size exceeds the free memory of "
            "every available GPU. Lower 'task_memory' or free up VRAM."
        )


class NETT:
    """YAML/dict-driven benchmark orchestrator (Isaac Lab + skrl backend)."""

    def __init__(self, configs):
        self.logger = logging.getLogger("nett")
        if not isinstance(configs, list):
            configs = [configs]
        try:
            schema = _load_schema()
            self.configs = [validate_config(c, schema) for c in configs]
        except Exception:
            self.logger.exception("Error loading/validating config")
            raise

    # --- Public API --------------------------------------------------------

    def run(
        self,
        output_path: Path | str = ".",
        devices: Optional[list[int]] = None,
        verbose: bool = True,
    ) -> list[Future]:
        """Run all configs. Blocks until every task completes."""
        self.output_path = Path(output_path).resolve()
        self.task_sheet: dict[Future, TaskConfig] = {}
        self.waitlist: list[Task] = []

        with MemoryManager() as self.memory_manager:
            self.devices = self.memory_manager.validate_devices(devices)
            self.logger.info("Devices: %s", self.devices)
            self.free_device_memory = {
                d: self.memory_manager.get_free_memory(d) for d in self.devices
            }
            with Executor(verbose) as self.executor:
                self.logger.info("Launching…")
                for config in self.configs:
                    self.single_run(**config)
                self._task_waiter()
        return list(self.task_sheet.keys())

    def status(self) -> dict[Future, TaskConfig]:
        return self.task_sheet

    # --- Per-config dispatch ----------------------------------------------

    def single_run(
        self,
        name: str,
        environment: dict,
        wrappers: Optional[list] = None,
        body: Optional[dict] = None,
        brain: Optional[dict] = None,
        episodes: Optional[dict[str, int]] = None,
        steps_per_episode: int = 200,
        num_brains: int = 1,
        brain_id_offset: int = 0,
        eval_freq: int | None = None,
        task_memory: str | float = "auto",
        **kwargs,
    ) -> None:
        episodes = episodes or {"train": 5000, "test": 100}
        if not set(episodes).issubset({"train", "test", "record"}) or not episodes:
            raise ValueError("Episodes must use only 'train', 'test', and/or 'record' keys.")

        # Snapshot input for reproducibility.
        input_params = {
            "name": name,
            "environment": environment,
            "body": dict(body or {}),
            "wrappers": list(wrappers or []),
            "brain": brain,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "num_brains": num_brains,
            "brain_id_offset": brain_id_offset,
            "eval_freq": eval_freq,
            "task_memory": task_memory,
        }
        output_dir = self.output_path / name
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as f:
            f.write(yaml.dump(input_params))

        base_brain = Brain(**(brain or {}))
        base_body = _make_body(body, wrappers)
        environment = dict(environment)
        if "reward_types" not in environment:
            inferred_rewards = base_brain.env_reward_types()
            if inferred_rewards:
                environment["reward_types"] = list(inferred_rewards)
        base_env = Environment(**environment)
        self._log_wandb_viewing_instructions(
            name,
            brain or {},
            base_env.conditions,
            num_brains,
        )

        base_brain.calc_iterations(
            num_brains, base_env.iterations_per_test_episode,
            episodes, steps_per_episode,
        )
        base_brain.iterations_per_test_episode = base_env.iterations_per_test_episode
        num_envs = int(num_brains) * int(base_brain.envs_per_agent)

        base_body.adjust_to_agent(
            base_env,
            num_brains=num_brains,
            num_envs=num_envs,
            episode_steps=steps_per_episode,
        )

        modes = _modes_from_episodes(episodes)
        memory = self._resolve_task_memory(
            task_memory, base_brain, base_body, base_env, output_dir,
        )
        tasklist = build_tasks(
            base_brain, base_body, base_env, num_brains, num_envs, base_env.conditions,
            output_dir, modes, episodes, memory, brain_id_offset, eval_freq,
        )

        # Per-task validation runs against the local Isaac Lab build — skip if
        # the env claims it cannot dry-run (e.g. headless device limits).
        self.logger.info("Validating tasks…")
        try:
            fut: Future = self.executor.submit(validate_tasklist, tasklist)
            future_wait([fut], return_when="ALL_COMPLETED")
        except Exception:
            self.logger.exception("Task validation raised; submitting anyway")

        self.logger.info("Assigning tasks…")
        for task in tasklist:
            self._assign_task(task)

    def _log_wandb_viewing_instructions(
        self,
        run_name: str,
        brain_cfg: dict,
        conditions: list[str],
        num_brains: int,
    ) -> None:
        wandb_cfg = (brain_cfg or {}).get("wandb", {}) or {}
        if wandb_cfg.get("mode", "online") != "online":
            return
        project = wandb_cfg.get("project", "nett-skrl")
        entity = wandb_cfg.get("entity") or "<your-default-entity>"
        self.logger.info("W&B online logging is enabled.")
        self.logger.info("Authenticate first with `wandb login` or WANDB_API_KEY.")
        self.logger.info("Open: https://wandb.ai/%s/%s", entity, project)
        self.logger.info("Filter/group by condition and tags from brain.wandb.tags.")
        for condition in conditions:
            for brain_id in range(1, int(num_brains) + 1):
                self.logger.info(
                    "Expected W&B run: %s/%s/brain_%d",
                    run_name,
                    condition,
                    brain_id,
                )

    # --- Memory estimation -------------------------------------------------

    def _resolve_task_memory(
        self,
        task_memory: str | float,
        brain: Brain,
        body: Body,
        env: Environment,
        output_dir: Path,
    ) -> float:
        """Return task VRAM budget in bytes; dry-run estimate when ``"auto"``."""
        if task_memory == "auto":
            return self._estimate_task_memory_via_dry_run(brain, body, env, output_dir)
        return float(task_memory) * (1024**3)

    def _estimate_task_memory_via_dry_run(
        self,
        brain: Brain,
        body: Body,
        env: Environment,
        output_dir: Path,
    ) -> float:
        """Spawn one task on the least-loaded GPU, train past a single update,
        and return ``baseline_free - post_train_free`` for the device."""
        device, baseline_free = self.memory_manager.get_most_free_gpu(self.devices)
        condition = env.conditions[0]

        task = Task(
            brain, 
            body, 
            env, 
            condition, 
            output_dir,
            modes=["train"], 
            episodes={"train": 1}, 
            memory=None,
            num_brains=env.num_brains,
            num_envs=env.num_envs,
        )
        task.set_device(device)
        task.set_dry_run(True)
        # mem.txt lands at the canonical ``config.path / "mem.txt"`` since
        # ``for_mode()`` rewrites ``path`` from ``__post_init__``; validation
        # mode suppresses every other artifact, so this file is the only
        # thing the dry-run leaves behind.
        condition_dir = output_dir / condition
        mem_txt = condition_dir / "mem.txt"
        if mem_txt.exists():
            mem_txt.unlink()

        self.logger.info(
            "Estimating task memory via dry run (device=%d, baseline_free=%.2f GB)",
            device, baseline_free / 1024**3,
        )
        try:
            fut: Future = self.executor.submit(run_task, task)
            future_wait([fut], return_when="ALL_COMPLETED")
            fut.result()  # surface exceptions
            if not mem_txt.exists():
                raise RuntimeError(f"dry-run produced no mem.txt at {mem_txt}")
            post_free = int(mem_txt.read_text().strip())
        except Exception:
            self.logger.exception(
                "Dry-run memory estimation failed; falling back to %.1f GB",
                _FALLBACK_TASK_MEMORY_GB,
            )
            return _FALLBACK_TASK_MEMORY_GB * (1024**3)
        finally:
            if mem_txt.exists():
                mem_txt.unlink()

        consumed = max(0.0, float(baseline_free - post_free))
        self.logger.info(
            "Estimated task memory: %.2f GB (baseline=%.2f, post=%.2f)",
            consumed / 1024**3, baseline_free / 1024**3, post_free / 1024**3,
        )
        # Refresh device-free baseline so the scheduler accounts for any
        # memory the subprocess didn't fully release.
        self.free_device_memory[device] = self.memory_manager.get_free_memory(device)
        return consumed

    # --- Scheduler ---------------------------------------------------------

    def _assign_task(self, task: Task) -> None:
        most_free_gpu, capacity = self.memory_manager.get_most_free_gpu(self.devices)
        # Refresh ledger from live NVML so other-process activity is reflected.
        self.free_device_memory[most_free_gpu] = capacity

        task_memory = float(task.config.memory or 0)
        if task_memory > capacity:
            # Nothing fits right now; queue and retry from _task_waiter when
            # an in-flight task completes and frees memory.
            self.waitlist.append(task)
            return
        task.set_device(most_free_gpu)
        self.free_device_memory[most_free_gpu] -= task_memory
        fut = self.executor.submit(run_task, task)
        self.task_sheet[fut] = task.config
        time.sleep(0.5)  # stagger Isaac Sim startup to avoid contention

    def _task_waiter(self) -> None:
        while self.task_sheet or self.waitlist:
            if not self.task_sheet:
                # All in-flight cleared but waitlist still has items that
                # didn't fit on initial assignment — surface as oversize.
                raise JobTooBigError()
            for done in as_completed(list(self.task_sheet)):
                cfg = self.task_sheet.pop(done)
                # Refresh ledger from live NVML (the in-flight task may have
                # released its allocation when the subprocess exited).
                if cfg.device is not None:
                    self.free_device_memory[cfg.device] = (
                        self.memory_manager.get_free_memory(cfg.device)
                    )
                try:
                    done.result()
                except Exception:
                    self.logger.exception(
                        "Task failed: %s condition=%s", cfg.name, cfg.condition,
                    )
                    raise
                # Promote one waitlisted task if it fits anywhere.
                self._promote_waitlisted()
                break

    def _promote_waitlisted(self) -> None:
        for i, task in enumerate(self.waitlist):
            gpu, capacity = self.memory_manager.get_most_free_gpu(self.devices)
            self.free_device_memory[gpu] = capacity
            if float(task.config.memory or 0) <= capacity:
                self.waitlist.pop(i)
                self._assign_task(task)
                return
