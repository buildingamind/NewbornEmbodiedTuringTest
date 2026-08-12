"""``BrainTrainer`` — NETT wrapper around skrl's ``SequentialTrainer``."""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from skrl.trainers.torch import SequentialTrainer
from tqdm import tqdm

from ..recording import RecordingCfg, RunRecorder
from .env_wrappers import IntrinsicRewardEnvWrapper

def eval_stochastic_enabled() -> bool:
    """Whether test-time actions are SAMPLED from the policy rather than its MEAN.

    ★ ONE ACCESSOR, so callers cannot disagree with the value that actually produced the
    numbers -- the exact provenance ambiguity that cost the 2026-07-29 stale-summary
    incident, where a stochastic retest was compared against a deterministic summary with
    nothing on disk recording which protocol produced which number.

    ``1`` = SAMPLED (stochastic inference, Unity ML-Agents' default; Unity's
    ``deterministic: False``). ``0`` = the Gaussian MEAN (deterministic inference; Unity's
    ``deterministic: true``). Note the polarity is INVERTED relative to Unity's key name --
    "mean action" and "deterministic action selection" are the SAME option.

    ★ DEFAULT IS ``1`` (SAMPLED) -- decided 2026-08-12, matching the protocol that produced
    every result in SIDE_LOCK_INVESTIGATION.md from Phase 3 onward and every arm launched by
    campaign/launch_arm.sh (which sets it explicitly). Before that date this tree defaulted
    to ``0`` while the isaac1 tree ran at ``1``; the two are now consistent.

    WHY SAMPLED. With the mean action and the fixed test start pose (motor.reset_deterministic)
    the environment is deterministic, so every episode of a condition is a BIT-IDENTICAL
    REPLAY: the readout is binary (which wall) rather than graded, and a weak-but-real
    preference reports as exactly chance. Measured 2026-07-29: the shape conditions came out
    46-50% with |side_preference| = 1.000 for 7/7 brains while `rest` (a strong cue) came out
    100%. Unity ML-Agents samples continuous actions at inference by default, so the original
    runs could express a graded preference.

    ⚠ IT IS NOT FREE, AND THE COST IS MEASURED. Sampling LOOSENS a side-lock without releasing
    it (|sp|avg -0.05 to -0.11 in every shape condition, at fixed weights, Phase 3) and it
    DEGRADES the positive control (ViT `rest` 0.957 -> 0.910; `1color` lost significance,
    p 0.013 -> 0.059). The learned sigma is large relative to the clamped [-1,1] action range
    -- median 1.20 for the ViT arms against 0.79 for CNN -- so this is a substantial
    perturbation, not a dither, and it penalises the higher-sigma architectures more.

    ⚠ NOTHING EVALUATED UNDER ONE VALUE POOLS WITH ANYTHING UNDER THE OTHER. The value is
    recorded per run in campaign_timing.json, so no existing result is ambiguous. Set it
    explicitly in BOTH directions in any harness -- ``0`` is what reproduces a pre-2026-07-30
    run, and code that merely POPPED the variable now gets sampling rather than the mean.
    """
    return os.environ.get("NETT_EVAL_STOCHASTIC", "1").strip().lower() in {"1", "true", "yes"}


# Read once at import: evaluation action selection (see _collect_actions_for_eval).
_EVAL_STOCHASTIC = eval_stochastic_enabled()

logger = logging.getLogger("nett.trainer")


@dataclass
class TrainCfg:
    """Per-run training parameters.

    Checkpoint cadence is delegated to skrl (driven by
    ``Brain.checkpoint_freq`` → ``cfg.experiment.checkpoint_interval``); see
    ``brain/experiment.py::apply_experiment_cfg``. ``hparams_dir`` controls
    only where the JSON hparams backup file is written; skrl captures hparams
    into wandb independently.

    ``output_dir`` / ``condition`` / ``phase`` / ``run_name`` are retained for
    run-layout compatibility; checkpoints, scalars, and videos are written
    through skrl/TensorBoard paths.

    ``initial_timestep`` offsets skrl's internal step counter so that
    ``global_step`` in W&B accumulates continuously across training chunks
    when ``eval_freq`` splits training into multiple subprocesses.
    """

    total_timesteps: int
    initial_timestep: int = 0
    hparams_dir: Path | None = None
    hparams: dict | None = None
    output_dir: Path | None = None
    condition: str | None = None
    phase: str | None = None
    run_name: str | None = None


def brain_scope_sizes(num_envs: int, num_brains: int) -> list[int]:
    """Env-scope size per brain. THE definition of brain<->env ownership.

    Each brain (one model instance) owns ONE CONTIGUOUS scope of the vectorized
    env; skrl's SequentialTrainer slices `agents`/`scopes` in order, so brain b
    owns envs [b*s, (b+1)*s). Brains never share an env -- that isolation is the
    whole point of running N brains in parallel (they differ only by seed, and
    must not exchange information).
    """
    if num_brains < 1:
        raise ValueError("num_brains must be >= 1")
    if num_envs % num_brains != 0:
        raise ValueError(
            f"env.num_envs ({num_envs}) must be divisible by num_brains "
            f"({num_brains}); each brain owns one contiguous env scope."
        )
    return [num_envs // num_brains] * num_brains


def brain_id_per_env(num_envs: int, num_brains: int) -> list[int]:
    """``[env_id] -> owning brain id``, derived from the same scopes as training.

    Single-sourced here so the per-step CSV can record WHICH MODEL produced each
    row instead of downstream analysis guessing from env_id (it used to guess, and
    reported num_envs as "n_brains").
    """
    out: list[int] = []
    for brain, size in enumerate(brain_scope_sizes(num_envs, num_brains)):
        out.extend([brain] * size)
    return out


class BrainTrainer:
    """Thin NETT wrapper over skrl ``SequentialTrainer``.

    One skrl agent owns one vectorized env scope. ``SequentialTrainer`` handles
    the act/step/record/update loop through its native ``agents`` + ``scopes``
    support; :class:`RunRecorder` owns the output writing around that loop.
    """

    def __init__(self, env, agents: list, device: str | torch.device = "cuda"):
        if len(agents) < 1:
            raise ValueError("BrainTrainer requires at least one agent.")
        self.env = env
        self.agents = agents
        self.scopes = brain_scope_sizes(env.num_envs, len(agents))
        self.device = torch.device(device)

    @staticmethod
    def _set_mode(agent, train: bool) -> None:
        agent.enable_training_mode(train, apply_to_models=True)

    def train(
        self,
        cfg: TrainCfg,
        record_cfg: RecordingCfg | None = None,
        intrinsic_reward_adapters: list | None = None,
        dry_run: bool = False,
    ) -> None:
        """Run training through skrl's native sequential trainer.

        Periodic checkpointing is handled inside skrl via
        ``cfg.experiment.checkpoint_interval``; we only handle the *final*
        snapshot here so the absolute last weights are always recorded
        regardless of whether ``total_timesteps`` lands on a checkpoint
        boundary.

        ``dry_run`` short-circuits the output-producing branches: skip
        ``hparams.json``, skip ``train_timing.json``, skip final checkpoints
        and recording exports.
        """
        egocentric_rec = _find_egocentric_recorder(self.env)
        if egocentric_rec is not None and record_cfg is not None and record_cfg.egocentric_enabled:
            egocentric_rec.set_recording(True)
        recorder = RunRecorder(self.agents, self.env.num_envs, egocentric_recorder=egocentric_rec)
        recorder.before_train(cfg, dry_run=dry_run)

        train_env = (
            IntrinsicRewardEnvWrapper(self.env, intrinsic_reward_adapters)
            if intrinsic_reward_adapters else self.env
        )
        # One contiguous skrl run — no chunking needed now that NETT no
        # longer interrupts to write its own checkpoints.
        start = time.perf_counter()
        self._run_skrl_train(train_env, cfg.total_timesteps, initial_timestep=cfg.initial_timestep)
        train_elapsed = time.perf_counter() - start
        recorder.after_train(
            cfg,
            elapsed_s=train_elapsed,
            record_cfg=record_cfg,
            dry_run=dry_run,
        )

    def _run_skrl_train(self, env, timesteps: int, initial_timestep: int = 0) -> None:
        import gc
        # Default GC threshold (700, 10, 10) can allow USD/Gf Python wrapper
        # objects with reference cycles to accumulate for hundreds of steps before
        # gen-0 collection runs. Tighten gen-0 to keep per-step object backlog small.
        gc.set_threshold(200, 5, 5)
        # Store the per-env step offset so the W&B forwarding hook can add it
        # to skrl's local timestep counter (which always starts at 0 in
        # SequentialTrainer 2.x) to make global_step cumulative across chunks.
        if initial_timestep > 0:
            for agent in self.agents:
                agent._nett_timestep_offset = int(initial_timestep)
        trainer = SequentialTrainer(
            env=env,
            agents=self.agents if len(self.agents) > 1 else self.agents[0],
            scopes=list(self.scopes) if len(self.agents) > 1 else None,
            cfg={
                "timesteps": timesteps,
                "headless": True,
            },
        )
        trainer.train()

    def eval(
        self,
        total_timesteps: int,
        *,
        desc: str | None = None,
        show_progress: bool = True,
        policy: str = "agent",
    ) -> dict[int, float]:
        """Deterministic rollout returning mean reward per brain."""
        if total_timesteps <= 0:
            logger.info("eval skipped: total_timesteps=%d", total_timesteps)
            return {i: 0.0 for i in range(len(self.agents))}
        for agent in self.agents:
            self._set_mode(agent, False)
        totals = torch.zeros(len(self.agents), device=self.device)
        observations, _ = self.env.reset()
        steps = range(total_timesteps)
        if show_progress:
            steps = tqdm(
                steps,
                total=total_timesteps,
                desc=desc or "eval",
                unit="timestep",
                file=sys.stdout,
            )
        # ACTION-NOISE STREAM (stochastic eval + grouped ordering only). skrl samples
        # from Normal(mean, std) on the GLOBAL torch RNG, one continuously advanced
        # stream, so which noise a condition receives is decided by its position in the
        # sequence. Re-seeding at each episode boundary from a key that restarts at every
        # new design row gives every condition the SAME noise realizations while repeats
        # within a row still differ -- variance where it is informative, matched where it
        # is a confound. Inert while NETT_EVAL_STOCHASTIC=0 (eval takes the mean, drawing
        # nothing) and while grouping is off (eval_noise_key returns None).
        # ★ SINCE 2026-08-12 THE DEFAULT IS 1, so this path is LIVE out of the box -- it used
        # to be dead unless opted into. Noise realizations now depend on design-row position
        # unless grouping keys them, which is what this block exists to do.
        raw_env = _unwrap_env(self.env) if _EVAL_STOCHASTIC else None
        episode_steps = int(getattr(getattr(raw_env, "cfg", None), "episode_steps", 0) or 0)
        # Save BOTH streams: torch.manual_seed below seeds CPU *and* every CUDA device,
        # so restoring only torch.get_rng_state() would leave training's CUDA generator
        # perturbed by a mid-training eval probe -- silent, and in the one area of this
        # codebase where that matters most.
        rng_state = torch.get_rng_state() if raw_env is not None else None
        cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if raw_env is not None and torch.cuda.is_available()
            else None
        )
        with torch.no_grad():
            for t in steps:
                # Boundary from the step counter, not from the env: reading
                # _step_in_episode would add a device->host sync to every eval step.
                if raw_env is not None and episode_steps > 0 and t % episode_steps == 0:
                    key = raw_env.eval_noise_key()
                    if key is not None:
                        torch.manual_seed(
                            (int(getattr(raw_env, "_resolved_seed", 0)) * 1_000_003) ^ int(key)
                        )
                actions = self._collect_actions_for_eval(
                    observations, t, total_timesteps, policy=policy
                )
                next_observations, rewards, *_ = self.env.step(actions)
                # ``rewards`` may be on the env/sim device (cpu for kinematic
                # PhysX, cuda for wheeled) while ``totals`` is on the policy
                # device. Move rewards to the totals device before accumulating.
                # Metric-only aggregation after the env step: this does not
                # feed back into training transitions or reward shaping.
                reward_rows = rewards.reshape(len(self.agents), self.scopes[0], -1).mean(dim=2)
                totals += reward_rows.to(totals.device, non_blocking=True).mean(dim=1)
                observations = next_observations
        if rng_state is not None:
            # Eval must not perturb training's stream: mid-training probes call this.
            torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        return {i: float(totals[i].item() / total_timesteps) for i in range(len(self.agents))}

    def _collect_actions_for_eval(
        self,
        observations: torch.Tensor,
        timestep: int,
        timesteps: int,
        *,
        policy: str = "agent",
    ) -> torch.Tensor:
        if policy == "target_side_oracle":
            oracle_actions = self._target_side_oracle_actions()
            if oracle_actions is not None:
                return oracle_actions

        states = self.env.state() if hasattr(self.env, "state") else None
        actions = []
        offset = 0
        for agent, scope in zip(self.agents, self.scopes):
            obs_i = observations[offset : offset + scope]
            state_i = states[offset : offset + scope] if states is not None else None
            action_i, outputs = agent.act(obs_i, state_i, timestep=timestep, timesteps=timesteps)
            # ★ MEAN vs SAMPLED ACTION AT TEST -- this decides whether the evaluation can
            # express preference STRENGTH at all (added 2026-07-29; default flipped to
            # SAMPLED 2026-08-12, see eval_stochastic_enabled).
            # NETT_EVAL_STOCHASTIC=0 takes the Gaussian policy's MEAN, discarding the sample.
            # That is the same thing Unity calls `deterministic: true` -- "mean action" and
            # "deterministic action selection" are ONE option, and the flag's polarity is
            # inverted relative to Unity's key name. With
            # the fixed test start pose (motor.reset_deterministic) and a deterministic
            # env, that makes every episode of a condition a BIT-IDENTICAL REPLAY: the
            # readout is binary (which wall), not graded. A weak-but-real preference is
            # then reported as exactly chance -- measured 2026-07-29, shape conditions
            # came out 46-50% with |side_preference| = 1.000 for 7/7 brains, while `rest`
            # (a strong cue) came out 100%.
            # Unity ML-Agents SAMPLES continuous actions at inference by default
            # (deterministic inference is opt-in), so the original runs could express a
            # graded preference. NETT_EVAL_STOCHASTIC=1 restores that behaviour.
            # ⚠ Changing this changes the evaluation protocol and costs a re-baseline.
            if _EVAL_STOCHASTIC:
                actions.append(action_i)
            else:
                actions.append(outputs.get("mean_actions", action_i))
            offset += scope
        return torch.cat(actions, dim=0)

    def _target_side_oracle_actions(self) -> torch.Tensor | None:
        """Closed-loop test policy that steers toward the env's target monitor.

        This policy is opt-in and only used during evaluation. It reads the
        same target side the environment uses for test logging/reward geometry,
        then outputs continuous wheel actions that point the chick at that
        monitor from its current pose.
        """
        raw = _unwrap_env(self.env)
        screens = getattr(raw, "screens", None)
        motor = getattr(raw, "motor", None)
        cfg = getattr(raw, "cfg", None)
        if screens is None or motor is None or cfg is None:
            return None

        action_space = self.env.action_space
        action_dim = int(getattr(action_space, "shape", (2,))[0])
        actions = torch.zeros((self.env.num_envs, action_dim), device=self.device)
        x = motor.x.to(self.device)
        y = motor.y_pos.to(self.device)
        yaw = motor.yaw_deg.to(self.device)
        target_x = torch.empty_like(x)
        target_y = torch.zeros_like(y)
        half_x = float(getattr(cfg, "chamber_half_x", 33.15))

        for env_id in range(self.env.num_envs):
            side = screens.target_side(env_id)
            target_x[env_id] = -half_x if side == "left" else half_x

        dx = target_x - x
        dy = target_y - y
        desired_yaw = torch.rad2deg(torch.atan2(-dx, dy)).remainder(360.0)
        error = (desired_yaw - yaw + 180.0).remainder(360.0) - 180.0
        turn_limit = float(getattr(getattr(cfg, "motor", None), "body_turn_speed_limit", 20.0))
        actions[:, 0] = (error / max(turn_limit, 1e-6)).clamp(-1.0, 1.0)
        actions[:, 1] = torch.where(error.abs() < 45.0, 1.0, 0.15)
        return actions


MultiBrainTrainer = BrainTrainer


def _find_egocentric_recorder(env):
    """Traverse the env wrapper chain to find a ChannelsFirst recorder.

    Uses duck typing (checks for set_recording / drain_completed_episodes) to
    avoid importing from the body package here.
    """
    current = env
    while current is not None:
        if hasattr(current, "set_recording") and hasattr(current, "drain_completed_episodes"):
            return current
        current = getattr(current, "_env", None) or getattr(current, "env", None)
    return None


def _unwrap_env(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "screens") and hasattr(current, "motor"):
            return current
        current = (
            getattr(current, "_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_unwrapped", None)
        )
    return env
