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
import math
import os
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
from .runtime.parallel_envs import (
    AUTO,
    capped_num_envs,
    grow_num_envs,
    is_valid_num_envs,
    num_env_candidates,
    select_test_num_envs,
    smallest_valid_num_envs,
    snap_num_envs,
)
from .validate import validate_config
from .runtime.tasklist import validate_tasklist


# Reserved fallback when dry-run estimation fails for any reason.
_FALLBACK_TASK_MEMORY_GB = 6.0

# Absolute cap on a single dry-run probe (seconds), per MODE. A probe is bounded work,
# so a cap is safe here in a way it is not for a real run -- and it is the ONLY thing
# that ends an over-size probe, because the OOM that kills it happens during the scene
# build, before crash_guard can be armed (see task_runner._run_single_mode).
#
# The cap is a tax, not just a safety net: the search FINDS the ceiling by overshooting,
# so every search pays it once. Hence per-mode, sized off measurement rather than one
# conservative number for both:
#   test  = Isaac boot + a few eval steps. Measured ~55s/probe (16..242 envs, res128).
#   train = Isaac boot + ONE ROLLOUT, so it scales with `rollouts`. Measured ~135s at
#           rollouts=800; a rollouts=3200 recipe is ~4x that.
# Both are ~5x their measured cost, so a legitimately slow probe is never mistaken for
# "does not fit" -- a false "too big" silently costs parallelism, which is worse than
# waiting. NETT_DRY_RUN_TIMEOUT overrides both.
_DRY_RUN_TIMEOUT_S = {"test": 300.0, "train": 900.0}


def _dry_run_timeout_for(mode: str) -> float:
    override = os.environ.get("NETT_DRY_RUN_TIMEOUT")
    if override:
        return float(override)
    return _DRY_RUN_TIMEOUT_S.get(mode, 900.0)


def _min_consumed(
    lo: tuple[int, float], hi: tuple[int, float], target: int
) -> float:
    """A LOWER bound on consumed(target), from two measured points below it.

    consumed(n) is CONVEX in num_envs -- measured, res128/1 brain: the marginal cost
    per env climbs 2.48->2.62->3.47->6.23->13.48GB across 2..256 envs, i.e. ~6MB/env
    to ~65MB/env. For a convex function the chord's slope over ``[lo, hi]`` is a lower
    bound on every marginal cost beyond ``hi``, so extending that line past ``hi``
    under-estimates. Hence this is a floor, never an over-estimate.

    NOTE THE DIRECTION. The same arithmetic as a LOWER bound on COST is sound; as an
    UPPER bound on CAPACITY it is exactly the refuted 2-point model that "fitted" 2916
    envs into 23.5GB and died in vkAllocateMemory. Convexity is what makes one valid
    and the other nonsense: a line under a convex curve stays under it.
    """
    (n_lo, c_lo), (n_hi, c_hi) = lo, hi
    if n_hi <= n_lo:
        return c_hi
    slope = (c_hi - c_lo) / float(n_hi - n_lo)
    return c_hi + slope * (int(target) - n_hi)


def _cannot_fit(
    lo: tuple[int, float], hi: tuple[int, float], target: int, budget: float
) -> bool:
    """Is ``target`` PROVABLY over budget, without running it?

    Lets the search stop before the rung that would OOM. That rung is the only one
    that ever does, and it is expensive in a way nothing else here is: the OOM lands
    inside Kit's scene build, where it can WEDGE (no Python exception, and the carb
    hook cannot be armed that early) -- so it costs the probe's whole timeout and
    leaves the GPU held until the reap. Skipping it changes no answer: the search
    stops at the first miss either way.
    """
    return _min_consumed(lo, hi, target) > budget


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
        max_parallel_envs: int | str | None = None,
        **kwargs,
    ) -> None:
        if "wrappers" in kwargs:
            raise TypeError("Configure observation wrappers under body.wrappers.")
        episodes = episodes or {"train": 5000, "test": 100}
        if not set(episodes).issubset({"train", "test", "record"}) or not episodes:
            raise ValueError("Episodes must use only 'train', 'test', and/or 'record' keys.")
        auto_envs = max_parallel_envs == AUTO
        if auto_envs and task_memory != AUTO:
            # "auto" envs means "as many as measured VRAM allows", and the dry run is
            # what measures. A declared task_memory skips it entirely, so there would
            # be nothing to derive the ceiling from.
            raise ValueError(
                'max_parallel_envs: "auto" requires task_memory: "auto" — the env '
                "ceiling is derived from the dry-run VRAM measurement. Either set "
                'task_memory: "auto" or give max_parallel_envs an integer.'
            )

        # Snapshot input for reproducibility. Written now so a crash during
        # resolution still leaves a record, then REWRITTEN below with the resolved
        # values once "auto" has an answer.
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
        self._write_config_snapshot(output_dir, input_params)

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
        if not auto_envs:
            self._warn_if_invalid_num_envs(num_envs, num_brains)
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
            env_cap=None if auto_envs else max_parallel_envs,
        )
        # Test is the phase worth widening (it replays a fixed schedule and learns
        # nothing), while training stays pinned to the recipe -- so measure the test
        # ceiling separately. Opt-in with max_parallel_envs: "auto": it costs a handful
        # of dry runs, so the default path keeps today's behaviour.
        test_ceiling = int(num_envs)
        if auto_envs:
            found = self._resolve_test_env_ceiling(
                base_brain, base_body, base_env, output_dir, num_brains, num_envs,
                steps_per_episode, episodes,
                # ONE task per condition, not per brain x condition: the brains share a
                # single vectorized env inside the task (see build_tasks/BrainTrainer).
                num_tasks=max(1, len(base_env.conditions)),
            )
            if found is not None:
                test_ceiling, test_memory = found
                # The task reserves whatever its WIDEST phase needs. Reserving only
                # train's footprint would let the scheduler pack a GPU that then OOMs
                # when those tasks reach test.
                memory = max(memory, test_memory)
            # The search left brain/body/env planned at the test count; training must
            # start from its own plan.
            self._apply_parallel_env_plan(
                base_brain, base_body, base_env, num_brains, num_envs, steps_per_episode
            )

        # RESOLVED, not requested: "auto" is replaced by the integer actually used, so
        # the emitted config.yaml re-runs identically on another machine. That matters
        # because env0's render depends on num_envs (#4431 brightness, #488 distortion)
        # -- a machine-derived count that was never recorded would make the benchmark
        # machine-dependent.
        input_params["max_parallel_envs"] = int(num_envs)
        input_params["task_memory"] = round(memory / (1024**3), 3)
        input_params["resolved_test_num_envs"] = self._resolved_test_num_envs(
            base_env, num_brains, test_ceiling, episodes
        )
        self._write_config_snapshot(output_dir, input_params)

        tasklist = build_tasks(
            base_brain, base_body, base_env, num_brains, num_envs, base_env.conditions,
            output_dir, modes, episodes, memory, brain_id_offset, eval_freq,
            max_parallel_envs=int(num_envs),
            max_test_envs=int(test_ceiling),
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

    # --- Resolution bookkeeping -------------------------------------------

    @staticmethod
    def _write_config_snapshot(output_dir: Path, params: dict) -> None:
        with open(output_dir / "config.yaml", "w") as f:
            f.write(yaml.dump(params))

    def _warn_if_invalid_num_envs(self, num_envs: int, num_brains: int) -> None:
        """An EXPLICIT env count is never overridden -- the recipe's num_envs is
        load-bearing for learning, and silently changing it would change results. But
        an invalid one renders through a distorted fisheye (#488) or breaks brain
        scoping, so it does not pass quietly."""
        if is_valid_num_envs(num_envs, num_brains):
            return
        suggestion, _ = snap_num_envs(num_envs, num_brains)
        self.logger.warning(
            "num_envs=%d is not a valid count (needs a SQUARE tile grid and a multiple "
            "of num_brains=%d). The fisheye render will be distorted (Isaac Sim #488) "
            "and/or brain scoping will fail. Nearest valid: %d. Not overriding, since "
            "you asked for %d explicitly.",
            num_envs, num_brains, suggestion, num_envs,
        )

    def _resolve_test_env_ceiling(
        self,
        brain: Brain,
        body: Body,
        env: Environment,
        output_dir: Path,
        num_brains: int,
        train_envs: int,
        steps_per_episode: int,
        episodes: dict[str, int],
        num_tasks: int,
    ) -> tuple[int, float] | None:
        """Measure how wide the TEST phase can run. Returns ``(ceiling, bytes)``.

        Test is the phase worth maximizing: it replays a fixed schedule, so unlike
        training its num_envs changes nothing but wall-clock. Training is pinned to the
        recipe (~16 envs, ~2.6GB of a 24GB card), which leaves most of the GPU idle
        during the test phase -- this finds what actually fits there.

        THE BUDGET IS A SHARE, NOT THE WHOLE GPU. Tasks are packed per device by their
        declared memory, so a ceiling measured against all of free VRAM would be a lie
        the moment two tasks land on one GPU and both enter test: each would try to
        allocate the whole card. Dividing by tasks-per-device keeps the reservation
        honest and preserves the wave's 2-cells-per-GPU packing.

        Probes in TEST mode: test allocates no optimizer and no rollout buffer, so a
        train probe would measure the wrong (higher) number -- and would cost a whole
        rollout per rung instead of a few steps.
        """
        if int((episodes or {}).get("test", 0)) <= 0:
            return None
        per_device_tasks = max(1, math.ceil(num_tasks / max(1, len(self.devices))))
        budget = self._vram_budget() / per_device_tasks
        self.logger.info(
            "Measuring test-phase env ceiling (%d task(s) over %d GPU(s) -> %d/GPU, "
            "%.2fGB budget each)",
            num_tasks, len(self.devices), per_device_tasks, budget / 1024**3,
        )
        found = self._search_max_envs_verified(
            brain, body, env, output_dir, num_brains, train_envs, steps_per_episode,
            max_envs=None, budget=budget, mode="test",
        )
        if found is None:
            return None
        memory, ceiling = found
        return ceiling, memory

    def _resolved_test_num_envs(
        self, env: Environment, num_brains: int, ceiling: int, episodes: dict[str, int]
    ) -> dict[str, int] | None:
        """What the TEST phase will pick, per condition, recorded for reproducibility.

        Per-condition because the episode total is (design rows for that condition) x
        episodes_test, and conditions own different row counts. Computed with the same
        pure selector the worker uses, so this is a record, not a second opinion.
        """
        episodes_test = int((episodes or {}).get("test", 0))
        if episodes_test <= 0:
            return None
        forced = os.environ.get("NETT_TEST_ENVS")
        resolved: dict[str, int] = {}
        for condition in env.conditions:
            total = max(1, env.iterations_per_test_episode.get(condition, 1)) * episodes_test
            want = min(int(forced), total) if forced else min(int(ceiling), total)
            resolved[condition] = select_test_num_envs(max(1, want), num_brains, total)[0]
        return resolved

    # --- Memory estimation -------------------------------------------------

    # Headroom for the dry-run-peak vs sustained-training gap.
    _VRAM_SAFETY = 0.80

    def _vram_budget(self) -> float:
        """Usable bytes on the target GPU, read BEFORE any dry run (each dry-run
        subprocess releases its memory on exit, so this stays a clean baseline)."""
        _device, free_bytes = self.memory_manager.get_most_free_gpu(self.devices)
        return free_bytes * self._VRAM_SAFETY

    def _search_max_envs_verified(
        self,
        brain: Brain,
        body: Body,
        env: Environment,
        output_dir: Path,
        num_brains: int,
        seed_envs: int,
        steps_per_episode: int,
        max_envs: int | None = None,
        budget: float | None = None,
        mode: str = "train",
    ) -> tuple[float, int] | None:
        """Largest VALID num_envs whose REAL dry run fits the VRAM budget.

        ``mode`` is the phase to probe in -- train and test have different footprints,
        so each search measures its own.

        ``max_envs`` caps growth. TRAIN passes the recipe's own count: num_envs there is
        load-bearing for learning (``envs_per_brain = rollouts // steps_per_episode``, so
        raising it rewrites the PPO batch composition), and auto's job is to VERIFY that
        count and back off on a smaller GPU -- never to redefine the experiment. TEST
        passes None: it replays a fixed schedule and learns nothing, so there the only
        limit is VRAM.

        ``budget`` overrides the default whole-GPU budget, for callers that must leave
        room for tasks co-scheduled on the same device.

        Every step is MEASURED at the count it reports -- nothing is extrapolated.
        That is not conservatism, it is the measured shape of the curve: Isaac's VRAM
        is badly superlinear in num_envs, so a line fitted at small counts predicts
        capacity that does not exist. Measured here (res128, 1 brain, 23.5GB free)::

            envs      2      4     16     36     64    144     256     400
            GB     2.48   2.49   2.62   2.85   3.47   6.23   13.48    OOM

        Marginal cost per env grows ~6MB -> ~65MB across that range. The old 2-point
        linear fit probed 2 and 4, measured a 6MB/env slope, and concluded ~2900 envs
        fit in 23.5GB; the resulting run died in vkAllocateMemory building the tiled
        camera canvas. Any slope from this data extrapolates to >1400 -- the model was
        not mis-tuned, it was the wrong shape.

        Bidirectional: probe the seed, then double up while each probe still fits, or
        halve down until one does. Returns ``(consumed_bytes, num_envs)``, or None if
        even the smallest valid count fails (caller falls back to the descending scan).
        Doubling stops at the first miss, so the answer is within 2x of the true
        ceiling -- traded deliberately against dry runs that cost ~1-3 min each.
        """
        nb = max(1, int(num_brains))
        budget = self._vram_budget() if budget is None else budget
        floor = smallest_valid_num_envs(nb)
        n, _ = snap_num_envs(max(int(seed_envs), floor), nb)

        def _probe(count: int) -> float | None:
            """Measured bytes at ``count``, or None if it does not fit (a failed dry
            run and an over-budget one mean the same thing here: too many envs).

            An over-size probe does not always die cleanly: a 484-env probe hit
            `vkAllocateMemory ERROR_OUT_OF_DEVICE_MEMORY` and then HUNG, pinning 24GB
            indefinitely, because crash_guard's bounded exit keys on DEVICE_LOST and an
            allocation OOM is not that (a 400-env probe on the same GPU exited cleanly in
            ~20s, so it depends where the OOM lands). That is why the probe carries an
            absolute cap -- see TaskConfig.dry_run_timeout -- and why a reaped probe
            arrives here as an exception, i.e. as "too big", which is exactly right.
            """
            self._apply_parallel_env_plan(brain, body, env, nb, count, steps_per_episode)
            try:
                consumed = self._estimate_task_memory_via_dry_run(
                    brain, body, env, output_dir, mode=mode
                )
            except Exception:
                self.logger.info("VRAM search: num_envs=%d failed to run; too big", count)
                return None
            if consumed > budget:
                self.logger.info(
                    "VRAM search: num_envs=%d needs %.2fGB > %.2fGB budget",
                    count, consumed / 1024**3, budget / 1024**3,
                )
                return None
            return consumed

        best: tuple[int, float] | None = None
        consumed = _probe(n)
        if consumed is None:
            # Seed does not fit -- halve down to the largest count that does.
            while consumed is None and n > floor:
                nxt, _ = snap_num_envs(max(n // 2, floor), nb)
                if nxt >= n:
                    break
                n = nxt
                consumed = _probe(n)
        else:
            # Seed fits -- double up until one does not (or growth is capped).
            previous: tuple[int, float] | None = None
            while consumed is not None:
                best = (n, consumed)
                nxt = grow_num_envs(n, nb)
                if nxt <= n or (max_envs is not None and nxt > max_envs):
                    break
                if previous is not None and _cannot_fit(previous, (n, consumed), nxt, budget):
                    self.logger.info(
                        "VRAM search (%s): num_envs=%d needs at least %.2fGB > %.2fGB "
                        "budget; not probing it",
                        mode, nxt,
                        _min_consumed(previous, (n, consumed), nxt) / 1024**3,
                        budget / 1024**3,
                    )
                    break
                previous = (n, consumed)
                n = nxt
                consumed = _probe(n)
        if consumed is not None:
            best = (n, consumed)
        if best is None:
            return None

        num_envs, memory = best
        self.logger.info(
            "VRAM search (%s): num_envs=%d verified at %.2fGB (budget %.2fGB @%.0f%%)",
            mode, num_envs, memory / 1024**3, budget / 1024**3, self._VRAM_SAFETY * 100,
        )
        self._apply_parallel_env_plan(brain, body, env, nb, num_envs, steps_per_episode)
        return memory, num_envs

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
        env_cap: int | None = None,
    ) -> tuple[float, int]:
        """Return ``(task VRAM budget in bytes, resolved num_envs)``; dry-run estimate
        when ``task_memory="auto"``. ``env_cap=None`` means the env ceiling is "auto"
        (measured VRAM decides)."""
        if task_memory != "auto":
            return float(task_memory) * (1024**3), int(num_envs)

        try:
            if env_cap is None:
                # max_parallel_envs: "auto" -- the ceiling is measured, so it must be
                # VERIFIED at the count it names. Extrapolating here picks thousands of
                # envs and dies in vkAllocateMemory; see _search_max_envs_verified.
                # Bounded by the recipe's own count: auto is a safety net for train, not
                # a licence to change the experiment.
                est = self._search_max_envs_verified(
                    brain, body, env, output_dir, num_brains, num_envs, steps_per_episode,
                    max_envs=num_envs,
                )
            else:
                # An explicit integer cap already bounds the answer, so the cheap
                # 2-point fit stays: it only has to decide whether the CAP fits, and
                # `target = min(cap, max_fit)` means the optimistic slope is harmless
                # in the usual case where the cap binds.
                est = self._estimate_envs_via_linear_model(
                    brain, body, env, output_dir, num_brains, num_envs, steps_per_episode,
                    env_cap=env_cap,
                )
            if est is not None:
                return est
        except Exception:
            self.logger.warning(
                "env estimator failed; falling back to descending scan",
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

        # num_brains itself need not be a legal env count (6 tiles 3x2), so snap.
        fallback_envs = smallest_valid_num_envs(num_brains)
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
        env_cap: int | None = None,
    ) -> tuple[float, int] | None:
        """Fit consumed(n)=fixed+n*per_env from two small dry-runs and return
        ``(budget_bytes, max_envs)`` that fits free*SAFETY. Returns ``None`` when
        the env cap is too small to fit two distinct probe points (caller then
        uses the legacy scan). Both coefficients are MEASURED for the actual model
        via the dry run, so video-size and model-size variation land in ``fixed``
        for free; the envs are only one slice of VRAM.

        ONLY VALID UNDER AN EXPLICIT CAP. ``max_fit`` is not trustworthy on its own:
        real VRAM is superlinear in num_envs, so this line -- fitted at 2 and 4 envs --
        overestimates capacity by ~100x (see _search_max_envs_verified for the measured
        curve). It survives here only because ``target = min(cap, max_fit)``, i.e. the
        cap normally binds and max_fit just has to notice a GPU too small for it.
        ``max_parallel_envs: "auto"`` routes to the verified search instead.

        The result is snapped to a valid num_envs (square tile grid AND a multiple of
        num_brains); an unsnapped count silently distorts every frame (#488)."""
        nb = max(1, int(num_brains))
        # Probe points must not exceed an explicit ceiling. "auto" has none, so it
        # always gets the wider (more accurate) pair -- and must, since its whole job
        # is to find a count the recipe never named.
        if env_cap is None:
            n1, n2 = 2 * nb, 4 * nb
        elif env_cap >= 4 * nb:
            n1, n2 = 2 * nb, 4 * nb
        elif env_cap >= 2 * nb:
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
        SAFETY = self._VRAM_SAFETY
        budget = free_bytes * SAFETY
        max_fit = int((budget - fixed) // per_env)
        want = max_fit if env_cap is None else min(int(env_cap), max_fit)
        # Snap to a VALID count -- square tile grid AND a multiple of num_brains. The
        # old code snapped only to a num_brains multiple, so a fit of e.g. 20 (nb=1)
        # was accepted and trained through a distorted 5x4 fisheye (#488).
        target, went_up = snap_num_envs(want, nb)
        if went_up:
            self.logger.warning(
                "measured VRAM fits only %d envs, but no square-grid multiple of "
                "num_brains=%d is that small; using %d. This may OOM -- a distorted "
                "render or a broken brain scope would be worse. Free VRAM or lower "
                "num_brains.",
                want, nb, target,
            )
        est_consumed = max(c2, fixed + target * per_env)
        self.logger.info(
            "2-point VRAM model: fixed=%.2fGB per_env=%.0fMB free=%.2fGB(@%.0f%%) "
            "-> fits %d -> num_envs=%d (%s, probes %d->%.2fGB, %d->%.2fGB)",
            fixed / 1024**3, per_env / 1024**2, free_bytes / 1024**3, SAFETY * 100,
            max_fit, target,
            "cap=auto" if env_cap is None else f"cap={env_cap}",
            n1, c1 / 1024**3, n2, c2 / 1024**3,
        )
        self._apply_parallel_env_plan(brain, body, env, num_brains, target, steps_per_episode)
        return est_consumed, target

    def _estimate_task_memory_via_dry_run(
        self,
        brain: Brain,
        body: Body,
        env: Environment,
        output_dir: Path,
        mode: str = "train",
    ) -> float:
        """Spawn one task on the least-loaded GPU and return
        ``baseline_free - post_free`` for the device.

        ``mode`` decides what is measured, and the phases differ enough to matter:
        a ``train`` probe runs one rollout so the optimizer state and the update's
        peak are committed, while a ``test`` probe runs a handful of steps because
        test allocates neither -- only env, models and render targets. Measuring test
        with a train probe would be both slower and wrong (high).
        """
        device, baseline_free = self.memory_manager.get_most_free_gpu(self.devices)
        condition = env.conditions[0]

        task = Task(
            brain,
            body,
            env,
            condition,
            output_dir,
            modes=[mode],
            episodes={mode: 1},
            memory=None,
            num_brains=env.num_brains,
            num_envs=env.num_envs,
        )
        task.set_device(device)
        # BOUNDED: a probe deliberately reaches for counts that do not fit, and an
        # over-size one does not reliably die (a 484-env probe hit a Vulkan OOM and
        # then hung, holding 24GB, since crash_guard keys on DEVICE_LOST -- and cannot
        # be armed early enough to see a scene-build OOM without wedging Kit).
        task.set_dry_run(True, timeout=_dry_run_timeout_for(mode))
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
