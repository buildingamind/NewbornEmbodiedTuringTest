"""Env-side wrappers used by the skrl training loop.

These wrappers sit between skrl and the Isaac/Gym env. They are deliberately
kept out of :mod:`nett_skrl.body.wrappers`, which is reserved for observation
transforms such as video frame stacking, DVS, and retina preprocessing.
"""

from __future__ import annotations

import torch


class SegmentationStatsEnvWrapper:
    """Forward perception diagnostics through each run's skrl scalar writer.

    Body segmenters sit below the skrl adapter and own their optimization;
    tracking their scalar snapshots here introduces no policy gradient path.
    The body model spans the vectorized environment, so its aggregate stats
    are the same for every agent sharing that environment.
    """

    def __init__(self, env, agents, sources):
        self._env = env
        self._agents = agents
        self._sources = sources

    @classmethod
    def wrap(cls, env, agents):
        from ..body.wrappers.segmentation import SegmentationObservationWrapper

        sources = []
        current, seen = env, set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, SegmentationObservationWrapper):
                sources.append(current)
            current = getattr(current, "_env", None) or getattr(current, "env", None)
        return cls(env, agents, sources) if sources else env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions):
        result = self._env.step(actions)
        for source in self._sources:
            for tag, value in source.last_stats.items():
                for agent in self._agents:
                    agent.track_data(tag, value)
        return result


def step_before_sink(agent):
    """The aux object that takes the env's step_before per recorded row, or None.

    A term declares ``needs_step_before``; a composite (with_cltt_ref) is searched one level down
    through ``.term``. XSP2Term under NETT_XSP2_WINDOWS=hold is the only declarer.
    """
    aux = getattr(agent, "_aux", None)
    for candidate in (aux, getattr(aux, "term", None)):
        if candidate is not None and getattr(candidate, "needs_step_before", False) is True:
            return candidate
    return None


class StepBeforeEnvWrapper:
    """Hand each agent's aux the env's OWN step_before for the transition skrl records next.

    skrl's SequentialTrainer calls env.step, then agent.record_transition, which writes memory row
    ``agent.memory.memory_index``. So right after env.step that index IS the row of this transition,
    and NETTEnv._last_step_for_outputs (set in _apply_action, before the step's increment) is its
    step_before. Brain b owns envs [b*s, (b+1)*s) (trainer.brain_scope_sizes).
    Two guards, both fail loud: at wrap, the env's decision period must equal each sink's
    hold_period; after every step, _step_in_episode must read step_before + 1, or 0 where the step
    ended an episode (one _apply_action per env.step, reset in the same step).
    """

    def __init__(self, env, agents, sinks, raw, sizes):
        self._env = env
        self._agents = agents
        self._sinks = sinks
        self._raw = raw
        self._sizes = sizes

    @classmethod
    def wrap(cls, env, agents, raw, sizes):
        sinks = [step_before_sink(agent) for agent in agents]
        if not any(sink is not None for sink in sinks):
            return env
        for name in ("_last_step_for_outputs", "_step_in_episode", "_decision_period"):
            if not hasattr(raw, name):
                raise RuntimeError(
                    f"StepBeforeEnvWrapper: the unwrapped env {type(raw).__name__} has no {name}; "
                    f"hold windows need NETTEnv's own step counter.")
        period = int(raw._decision_period())
        for agent, sink, size in zip(agents, sinks, sizes):
            if sink is None:
                continue
            if period != int(sink.hold_period):
                raise ValueError(
                    f"StepBeforeEnvWrapper: the training env runs decision_period={period}, but the aux "
                    f"draws hold triples for P={sink.hold_period}.")
            n_env = getattr(agent.memory, "num_envs", size)
            if int(n_env) != int(size):
                raise ValueError(f"StepBeforeEnvWrapper: agent memory holds {n_env} envs, its scope {size}.")
        return cls(env, agents, sinks, raw, list(sizes))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions):
        result = self._env.step(actions)
        terminated, truncated = result[2], result[3]
        state = torch.stack([self._raw._last_step_for_outputs.reshape(-1),
                             self._raw._step_in_episode.reshape(-1)]).cpu()
        step_before, after = state[0], state[1]
        done = (torch.as_tensor(terminated).bool() | torch.as_tensor(truncated).bool()).reshape(-1).cpu()
        expected = torch.where(done, torch.zeros_like(step_before), step_before + 1)
        if not torch.equal(after, expected):
            bad = int((after != expected).nonzero()[0])
            raise RuntimeError(
                f"StepBeforeEnvWrapper: env {bad} step_in_episode {int(after[bad])} after a step from "
                f"step_before {int(step_before[bad])} (done={bool(done[bad])}); expected "
                f"{int(expected[bad])}. One env.step must be one counted step.")
        start = 0
        for agent, sink, size in zip(self._agents, self._sinks, self._sizes):
            if sink is not None:
                sink.record_step_before(int(agent.memory.memory_index), step_before[start:start + size])
            start += size
        return result


class IntrinsicRewardEnvWrapper:
    """Reward-shaping wrapper used before skrl records transitions.

    Intrinsic rewards are intentionally applied one env row at a time. The
    vectorized env is only a synchronous transport for faster stepping; each
    brain-env-condition instance keeps its own reward state and transition.
    """

    def __init__(self, env, adapters: list) -> None:
        self._env = env
        self.adapters = adapters
        self._last_observations = None
        if len(self.adapters) != int(self._env.num_envs):
            raise ValueError(
                f"expected one intrinsic reward adapter per env row "
                f"({self._env.num_envs}); got {len(self.adapters)}"
            )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        observations, infos = self._env.reset()
        self._last_observations = observations
        return observations, infos

    def step(self, actions):
        next_observations, rewards, terminated, truncated, infos = self._env.step(actions)
        if self._last_observations is not None:
            rewards = rewards.clone()
            # Keep env rows behaviorally independent even though the wrapped
            # env steps a batch. Do not combine observations, actions, rewards,
            # or intrinsic-reward state across env rows here.
            for i, adapter in enumerate(self.adapters):
                obs_i = self._last_observations[i : i + 1]
                next_i = next_observations[i : i + 1]
                reward_i = rewards[i : i + 1]
                action_i = actions[i : i + 1]
                term_i = terminated[i : i + 1]
                trunc_i = truncated[i : i + 1]
                if hasattr(adapter, "watch"):
                    adapter.watch(obs_i, action_i, reward_i, term_i, trunc_i, next_i)
                if hasattr(adapter, "compute"):
                    intrinsic = adapter.compute(
                        observations=obs_i,
                        actions=action_i,
                        rewards=reward_i,
                        terminated=term_i,
                        truncated=trunc_i,
                        next_observations=next_i,
                    )
                else:
                    intrinsic = torch.zeros_like(reward_i)
                if hasattr(adapter, "update"):
                    adapter.update()
                rewards[i : i + 1] = reward_i + torch.as_tensor(
                    intrinsic, device=reward_i.device, dtype=reward_i.dtype
                ).view_as(reward_i)
        self._last_observations = next_observations
        return next_observations, rewards, terminated, truncated, infos
