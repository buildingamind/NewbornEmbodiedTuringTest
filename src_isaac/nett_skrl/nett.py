"""NETT-skrl orchestrator.

Same UX as the legacy `nett.NETT` class — YAML/dict configs in, one (or many)
benchmark run(s) out. The behavioral differences are entirely below the
public API:

  - One process per imprint condition (not per brain x condition).
  - N brains share one vectorized Isaac env via :class:`BrainTrainer`;
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
from .environment.design import get_experiment_design
from .runtime import (
    Executor,
    Task,
    TaskConfig,
    build_tasks,
    run_task,
)
from .runtime.memory import MemoryManager
from .runtime.reap import DeviceLostRunError, ReapedTaskError
from .runtime.parallel_envs import capped_num_envs, num_env_candidates
from .validate import validate_config
from .runtime.tasklist import validate_tasklist


# Reserved fallback when dry-run estimation fails for any reason.
_FALLBACK_TASK_MEMORY_GB = 6.0


def _load_schema() -> dict:
    with open(Path(__file__).resolve().parent / "schema.json") as f:
        return json.load(f)


def _modes_from_episodes(episodes: dict[str, int]) -> list[str]:
    return [m for m in ("train", "test", "record") if episodes.get(m, 0) > 0]


def _make_body(body: Optional[dict]) -> Body:
    return Body(**dict(body or {}))


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
        # Device-lost/timeout casualties. Recorded here so the wave can continue,
        # then re-raised in aggregate by _task_waiter -> the run ends nonzero.
        self.failed_tasks: list[tuple[str, BaseException]] = []

        with MemoryManager() as self.memory_manager:
            self.devices = self.memory_manager.validate_devices(devices)
            self.logger.info("Devices: %s", self.devices)
            self.free_device_memory = {
                d: self.memory_manager.get_free_memory(d) for d in self.devices
            }
            with Executor(verbose, max_tasks=self._max_concurrent_tasks()) as self.executor:
                self.logger.info("Launching…")
                for config in self.configs:
                    self.single_run(**config)
                self._task_waiter()
        return list(self.task_sheet.keys())

    def _max_concurrent_tasks(self) -> Optional[int]:
        """Most tasks that can be in flight at once, or None if not provable here.

        One task per brain per condition (see ``build_tasks``), so the count is
        known up front whenever the conditions are -- either stated in the config
        or readable from the design sheet. Returning None (e.g. an ``experiment``
        bundle, or an unreadable sheet) makes Executor keep its historical size:
        a bound we cannot prove must never shrink the pool below the real task
        count, or concurrency would silently drop.
        """
        total = 0
        for config in self.configs:
            env_cfg = config.get("environment") or {}
            conditions = env_cfg.get("conditions")
            if not conditions:
                sheet = env_cfg.get("design_sheet")
                if not sheet:
                    return None
                try:
                    conditions = list(get_experiment_design(sheet))
                except Exception:
                    return None
            total += int(config.get("num_brains", 1) or 1) * len(conditions)
        return total or None

    def status(self) -> dict[Future, TaskConfig]:
        return self.task_sheet

    # --- Per-config dispatch ----------------------------------------------

    def single_run(
        self,
        name: str,
        environment: dict,
        body: Optional[dict] = None,
        brain: Optional[dict] = None,
        episodes: Optional[dict[str, int]] = None,
        steps_per_episode: int = 200,
        num_brains: int = 1,
        brain_id_offset: int = 0,
        eval_freq: int | None = None,
        task_memory: str | float = "auto",
        max_parallel_envs: int | None = None,
        **kwargs,
    ) -> None:
        if "wrappers" in kwargs:
            raise TypeError("Configure observation wrappers under body.wrappers.")
        episodes = episodes or {"train": 5000, "test": 100}
        if not set(episodes).issubset({"train", "test", "record"}) or not episodes:
            raise ValueError("Episodes must use only 'train', 'test', and/or 'record' keys.")

        # Snapshot input for reproducibility.
        input_params = {
            "name": name,
            "environment": environment,
            "body": dict(body or {}),
            "brain": brain,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "num_brains": num_brains,
            "brain_id_offset": brain_id_offset,
            "eval_freq": eval_freq,
            "task_memory": task_memory,
            "max_parallel_envs": max_parallel_envs,
        }
        output_dir = self.output_path / name
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as f:
            f.write(yaml.dump(input_params))

        base_brain = Brain(**(brain or {}))
        base_body = _make_body(body)
        environment = dict(environment)
        if "reward_types" not in environment:
            inferred_rewards = base_brain.env_reward_types()
            if inferred_rewards:
                environment["reward_types"] = list(inferred_rewards)
        base_env = Environment(**environment)
        self._log_wandb_viewing_instructions(
            brain or {},
        )

        base_brain.calc_iterations(
            num_brains, base_env.iterations_per_test_episode,
            episodes, steps_per_episode,
        )
        base_brain.iterations_per_test_episode = base_env.iterations_per_test_episode
        num_envs = capped_num_envs(
            num_brains=num_brains,
            preferred_envs_per_brain=base_brain.envs_per_brain,
            max_parallel_envs=max_parallel_envs,
        )
        self._apply_parallel_env_plan(
            base_brain, base_body, base_env, num_brains, num_envs, steps_per_episode
        )

        modes = _modes_from_episodes(episodes)
        memory, num_envs = self._resolve_task_memory_and_envs(
            task_memory,
            base_brain,
            base_body,
            base_env,
            output_dir,
            num_brains,
            num_envs,
            steps_per_episode,
        )
        tasklist = build_tasks(
            base_brain, base_body, base_env, num_brains, num_envs, base_env.conditions,
            output_dir, modes, episodes, memory, brain_id_offset, eval_freq,
            max_parallel_envs=max_parallel_envs,
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
        brain_cfg: dict,
    ) -> None:
        wandb_cfg = (brain_cfg or {}).get("wandb", {}) or {}
        if wandb_cfg.get("mode", "online") != "online":
            return
        project = wandb_cfg.get("project", "nett-skrl")
        entity = wandb_cfg.get("entity") or "<your-default-entity>"
        self.logger.info(
            f"""W&B online logging is enabled.
                Authenticate first with `wandb login` or WANDB_API_KEY.
                Open: https://wandb.ai/{entity}/{project}"""
        )

    # --- Memory estimation -------------------------------------------------

    def _apply_parallel_env_plan(
        self,
        brain: Brain,
        body: Body,
        env: Environment,
        num_brains: int,
        num_envs: int,
        steps_per_episode: int,
    ) -> None:
        brain.envs_per_brain = max(1, int(num_envs) // max(1, int(num_brains)))
        body.adjust_to_agent(
            env,
            num_brains=num_brains,
            num_envs=num_envs,
            episode_steps=steps_per_episode,
        )

    def _resolve_task_memory_and_envs(
        self,
        task_memory: str | float,
        brain: Brain,
        body: Body,
        env: Environment,
        output_dir: Path,
        num_brains: int,
        num_envs: int,
        steps_per_episode: int,
    ) -> tuple[float, int]:
        """Return task VRAM budget in bytes; dry-run estimate when ``"auto"``."""
        if task_memory != "auto":
            return float(task_memory) * (1024**3), int(num_envs)

        # 2-point linear VRAM model to MAXIMIZE envs in one shot.
        # consumed(n) ~= fixed + n*per_env (rollout buffer scales per-env so the
        # buffer is ~constant => part of `fixed`; per_env is mainly Isaac
        # rendering). Probe two small/safe env counts, fit the line, then solve
        # for the largest env count that fits free*safety. Falls back to the
        # legacy descending dry-run scan on any failure.
        try:
            est = self._estimate_envs_via_linear_model(
                brain, body, env, output_dir, num_brains, num_envs, steps_per_episode
            )
            if est is not None:
                return est
        except Exception:
            self.logger.warning(
                "2-point env estimator failed; falling back to descending scan",
                exc_info=True,
            )

        failures: list[tuple[int, Exception]] = []
        for candidate in num_env_candidates(num_envs, num_brains):
            self._apply_parallel_env_plan(
                brain, body, env, num_brains, candidate, steps_per_episode
            )
            try:
                return (
                    self._estimate_task_memory_via_dry_run(brain, body, env, output_dir),
                    candidate,
                )
            except Exception as exc:
                failures.append((candidate, exc))
                self.logger.warning(
                    "Dry-run memory estimation failed for num_envs=%d; trying smaller plan",
                    candidate,
                    exc_info=True,
                )

        fallback_envs = int(num_brains)
        self._apply_parallel_env_plan(
            brain, body, env, num_brains, fallback_envs, steps_per_episode
        )
        self.logger.error(
            "Dry-run memory estimation failed for all candidates %s; falling back to %.1f GB with num_envs=%d",
            [candidate for candidate, _ in failures],
            _FALLBACK_TASK_MEMORY_GB,
            fallback_envs,
        )
        return _FALLBACK_TASK_MEMORY_GB * (1024**3), fallback_envs

    def _estimate_envs_via_linear_model(
        self,
        brain: Brain,
        body: Body,
        env: Environment,
        output_dir: Path,
        num_brains: int,
        num_envs: int,
        steps_per_episode: int,
    ) -> tuple[float, int] | None:
        """Fit consumed(n)=fixed+n*per_env from two small dry-runs and return
        ``(budget_bytes, max_envs)`` that fits free*SAFETY. Returns ``None`` when
        the env cap is too small to fit two distinct probe points (caller then
        uses the legacy scan). Model-agnostic: both coefficients are MEASURED for
        the actual model via the dry run."""
        nb = max(1, int(num_brains))
        cap = int(num_envs)
        if cap >= 4 * nb:
            n1, n2 = 2 * nb, 4 * nb
        elif cap >= 2 * nb:
            n1, n2 = nb, 2 * nb
        else:
            return None  # too small; let the descending scan handle it

        # Clean free memory on the target device (captured before any dry-run;
        # the dry-run subprocess releases its memory on exit).
        _device, free_bytes = self.memory_manager.get_most_free_gpu(self.devices)

        probes: dict[int, float] = {}
        for n in (n1, n2):
            self._apply_parallel_env_plan(brain, body, env, num_brains, n, steps_per_episode)
            probes[n] = self._estimate_task_memory_via_dry_run(brain, body, env, output_dir)
        c1, c2 = probes[n1], probes[n2]
        per_env = (c2 - c1) / float(n2 - n1)

        if per_env <= 0:
            # Non-monotone (measurement noise): be conservative, use the larger
            # probe's env count + its measured memory.
            self.logger.warning(
                "2-point VRAM model non-monotone (%.2fGB@%d, %.2fGB@%d); using num_envs=%d",
                c1 / 1024**3, n1, c2 / 1024**3, n2, n2,
            )
            self._apply_parallel_env_plan(brain, body, env, num_brains, n2, steps_per_episode)
            return c2, n2

        fixed = c1 - n1 * per_env
        SAFETY = 0.80  # headroom for the dry-run-peak vs sustained-training gap
        budget = free_bytes * SAFETY
        max_fit = int((budget - fixed) // per_env)
        target = max(nb, min(cap, (max_fit // nb) * nb))
        est_consumed = max(c2, fixed + target * per_env)
        self.logger.info(
            "2-point VRAM model: fixed=%.2fGB per_env=%.0fMB free=%.2fGB(@%.0f%%) "
            "-> max_envs=%d (cap=%d, probes %d->%.2fGB, %d->%.2fGB)",
            fixed / 1024**3, per_env / 1024**2, free_bytes / 1024**3, SAFETY * 100,
            target, cap, n1, c1 / 1024**3, n2, c2 / 1024**3,
        )
        self._apply_parallel_env_plan(brain, body, env, num_brains, target, steps_per_episode)
        return est_consumed, target

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
        # mode suppresses every other output, so this file is the only
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
        # Pick the GPU with the most LEDGER-free memory. The ledger (initialised
        # from NVML, decremented on each assign, refreshed on completion) accounts
        # for reservations whose Isaac process has not booted yet — raw NVML lags
        # 1-2 min behind a just-assigned task and would pile many brains onto one
        # GPU before any allocates. Cross-check live NVML on the chosen device so
        # other-process usage still lowers the estimate.
        most_free_gpu = max(self.devices, key=lambda d: self.free_device_memory[d])
        live = self.memory_manager.get_free_memory(most_free_gpu)
        capacity = min(self.free_device_memory[most_free_gpu], live)
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
                # released its allocation when the subprocess exited). The reap
                # has already freed the dead task's VRAM by this point, so this
                # reads the true post-crash free memory.
                if cfg.device is not None:
                    self.free_device_memory[cfg.device] = (
                        self.memory_manager.get_free_memory(cfg.device)
                    )
                try:
                    done.result()
                except ReapedTaskError as exc:
                    # DEVICE_LOST / reap-timeout ONLY. Its processes are already
                    # reaped and its GPU released, so a transient renderer crash
                    # in one task must not cost the other N-1 tasks their hours
                    # of work. Recorded, not swallowed: re-raised in aggregate
                    # below once the wave has drained.
                    self.failed_tasks.append((f"{cfg.name}/{cfg.condition}", exc))
                    self.logger.error(
                        "Task ended by DEVICE_LOST/timeout: %s condition=%s: %s; "
                        "continuing remaining tasks",
                        cfg.name, cfg.condition, exc,
                    )
                except Exception:
                    # Every other failure keeps today's fail-fast re-raise.
                    self.logger.exception(
                        "Task failed: %s condition=%s", cfg.name, cfg.condition,
                    )
                    raise
                # Promote one waitlisted task if it fits anywhere.
                self._promote_waitlisted()
                break
        if self.failed_tasks:
            # The wave completed; the run must still fail loudly. Raised from
            # _task_waiter (not run()) so it propagates out of run() through the
            # Executor/MemoryManager __exit__ and the process ends nonzero.
            raise DeviceLostRunError(self.failed_tasks)

    def _promote_waitlisted(self) -> None:
        for i, task in enumerate(self.waitlist):
            gpu, capacity = self.memory_manager.get_most_free_gpu(self.devices)
            self.free_device_memory[gpu] = capacity
            if float(task.config.memory or 0) <= capacity:
                self.waitlist.pop(i)
                self._assign_task(task)
                return
