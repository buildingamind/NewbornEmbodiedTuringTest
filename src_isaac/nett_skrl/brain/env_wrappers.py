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


#: Workspace DECISIONS §114: the agent-side decision loop's period. Unset/empty = off (one agent step per tick).
DECISION_LOOP_ENV = "NETT_DECISION_LOOP"


def decision_loop_period() -> int | None:
    """NETT_DECISION_LOOP=P (integer >= 2): one agent decision per P env ticks, train AND test. Unset = None.

    campaign_train passes the same P to the env as ``decision_period`` (every phase), and
    DecisionLoopEnvWrapper asserts the two agree. "1" and anything that is not an integer >= 2 refuse:
    unset is the one way to say "no loop", so a row cannot carry an ambiguous value.
    """
    import os

    raw = os.environ.get(DECISION_LOOP_ENV, "")
    if raw.strip() == "":
        return None
    v = raw.strip()
    if not v.isdigit() or int(v) < 2:
        raise ValueError(f"{DECISION_LOOP_ENV}={raw!r}: expected an integer >= 2 (leave it unset for no loop).")
    return int(v)


class DecisionLoopEnvWrapper:
    """One agent decision per P env ticks: the agent is NOT queried on the P-1 held ticks (DECISIONS §114).

    OWNER, workspace DECISIONS §114 (2026-10-14): "the model does not receive any observations and does
    not return actions until after the hold window is completed. In the case of a 3 step hold window, the
    video would step forward 3 times and 3 observations would be collected, then they would be sent to the
    agent and the agent would return one action. If the agent expects one frame, it would receive only the
    last frame. If the agent expects two frames, it would receive only the last two frames. If the agent
    expects all three frames, as would be the case with XSP, it would receive all three frames." And
    "1. Same 3-step loop": test runs the same loop as train.

    It wraps the skrl adapter (IsaacEnvWrapper), so it covers train, test (BrainTrainer.eval) and record.
    One ``step(action)`` calls the inner env P times with the SAME action. The env (repo B, run at
    ``decision_period = P``) applies it on the tick with step_in_episode % P == 0 and commands zero motion
    on the other P-1 (the wheeled body holds pose); the monitors advance on every tick (advance_all is
    unconditional). Returned:
      * observation -- the LAST tick's. FrameStack sits inside the adapter and sees every tick, so a
        k-frame stack holds the last k ticks (k = NETT_FRAMESTACK_N; k = P gives the whole window).
      * reward      -- the SUM over the P ticks. Repo B zeroes the reward off decision ticks
        (nett_env.py ``reward * _last_decision_mask``), so this equals the decision tick's reward.
      * terminated / truncated -- OR over the P ticks; info -- the last tick's.
    Budgets count AGENT steps, so episodes x NETT_STEPS is the number of DECISIONS and an episode is
    NETT_STEPS decisions = NETT_STEPS x P ticks, as repo B already sizes it: train and test complete the
    same number of episodes as at P = 1, at ~P x the env ticks (wall clock).

    Fail-loud guards: the env's decision period must equal P; after every agent step every env's
    step_in_episode must be a multiple of P (0 after a done), so decisions stay on decision ticks; a done
    may only arrive on the LAST tick (repo B never terminates early and truncates at NETT_STEPS x P).
    """

    def __init__(self, env, period: int, raw) -> None:
        self._env = env
        self.period = int(period)
        self._raw = raw

    @classmethod
    def wrap(cls, wrapped, body_env, period: int | None):
        """``wrapped`` = the skrl adapter, ``body_env`` = the body-wrapped env under it. None = no loop."""
        if period is None:
            return wrapped
        import os

        from ..body.wrappers.acuity import find_acuity

        if find_acuity(body_env):
            raise ValueError(
                f"{DECISION_LOOP_ENV}={period}: an acuity curriculum counts its own step() calls, which "
                f"would be ticks, not decisions, so its schedule would run {period}x too fast. Refusing.")
        if os.environ.get("NETT_XSP2_WINDOWS", "").strip().lower() == "hold":
            raise ValueError(
                f"{DECISION_LOOP_ENV}={period} with NETT_XSP2_WINDOWS=hold: hold windows draw ACROSS stored "
                f"rows by step_before, and under the loop every stored row is the last tick of a window. "
                f"Use the in-observation draw (NETT_XSP2_WINDOWS=frames) instead.")
        raw = _unwrap_nett(body_env)
        for name in ("_step_in_episode", "_decision_period"):
            if not hasattr(raw, name):
                raise RuntimeError(f"DecisionLoopEnvWrapper: the unwrapped env {type(raw).__name__} has no {name}.")
        env_period = int(raw._decision_period())
        if env_period != int(period):
            raise ValueError(
                f"DecisionLoopEnvWrapper: {DECISION_LOOP_ENV}={period} but the env runs decision_period="
                f"{env_period}; the env must gate motors on the same period the agent decides on.")
        print(f"[NETT decision loop] P={period}: one agent decision per {period} env ticks "
              f"(workspace DECISIONS §114)", flush=True)
        return cls(wrapped, period, raw)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, actions):
        reward_sum = terminated_any = truncated_any = None
        for tick in range(self.period):
            observations, reward, terminated, truncated, info = self._env.step(actions)
            done = torch.as_tensor(terminated).bool() | torch.as_tensor(truncated).bool()
            if tick < self.period - 1 and bool(done.any()):
                bad = int(done.reshape(-1).nonzero()[0])
                raise RuntimeError(
                    f"DecisionLoopEnvWrapper: env {bad} ended its episode on tick {tick + 1} of {self.period}; "
                    f"an episode must end on the last tick of a decision (repo B truncates at NETT_STEPS x P).")
            if reward_sum is None:
                reward_sum, terminated_any, truncated_any = reward.clone(), terminated, truncated
            else:
                reward_sum = reward_sum + reward
                terminated_any = terminated_any | terminated
                truncated_any = truncated_any | truncated
        phase = self._raw._step_in_episode.reshape(-1) % self.period
        if bool((phase != 0).any()):
            bad = int(phase.nonzero()[0])
            raise RuntimeError(
                f"DecisionLoopEnvWrapper: env {bad} sits at step_in_episode "
                f"{int(self._raw._step_in_episode.reshape(-1)[bad])} after a decision; it must be a multiple of "
                f"{self.period}, or the next action would land on a held tick.")
        return observations, reward_sum, terminated_any, truncated_any, info


#: Workspace DECISIONS §187 add. 1: hold the first action until the frame stack holds only real, distinct frames.
FIRST_FULL_STACK_ENV = "NETT_FIRST_FULL_STACK"


def first_full_stack_enabled() -> bool:
    """NETT_FIRST_FULL_STACK: unset/empty or "1" = on (the default), "0" = off (the old episode start). Else refuses."""
    import os

    raw = os.environ.get(FIRST_FULL_STACK_ENV, "")
    v = raw.strip()
    if v in ("", "1"):
        return True
    if v == "0":
        return False
    raise ValueError(f"{FIRST_FULL_STACK_ENV}={raw!r}: expected 1 (default) or 0.")


def first_full_stack_ticks(body_env) -> tuple[int, list[str]]:
    """The zero-action ticks after a reset until every slot of the policy input is a real rendered frame of the episode.

    Each temporal wrapper on the body chain needs ``d`` consecutive inputs to fill its output with distinct real
    frames, and its first full output is ``d - 1`` ticks after the reset. Chained, the delays add:
      * FrameStack  (``framestack``)    d = n_stack: the reset fills every slot with the reset frame;
      * Video       (``video``)         d = frames: the same fill;
      * DVSPolarity (``dvs_polarity``)  d = 1 + W, W = NETT_DVS_WINDOW or 1: no previous frame at a reset = a blank
                                        event map, and a window needs W real per-tick maps;
      * gymnasium FrameStackObservation  d = stack_size: the stack ``dvs`` and ``retina`` build under them (zero- or
                                        reset-padded).
    Every other wrapper maps one frame to one frame (a segmenter's previous frame trains the segmenter and never
    reaches the policy input). Returns (ticks, one "name +k" part per temporal wrapper, outermost first).
    """
    import gymnasium as gym

    from ..body.wrappers.dvs_polarity import DVSPolarity
    from ..body.wrappers.framestack import FrameStack
    from ..body.wrappers.video import Video

    ticks, parts = 0, []
    current, seen = body_env, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_step_in_episode") and hasattr(current, "_decision_period"):
            break
        extra = None
        if isinstance(current, FrameStack):
            extra, name = int(current.n_stack) - 1, f"framestack n={current.n_stack}"
        elif isinstance(current, Video):
            extra, name = int(current.frames) - 1, f"video frames={current.frames}"
        elif isinstance(current, DVSPolarity):
            w = 1 if current.window is None else int(current.window)
            extra, name = w, f"dvs_polarity window={w}"
        elif isinstance(current, gym.wrappers.FrameStackObservation):
            extra, name = int(current.stack_size) - 1, f"frame_stack_observation n={current.stack_size}"
        if extra is not None and extra > 0:
            ticks += extra
            parts.append(f"{name} +{extra}")
        current = getattr(current, "_env", None) or getattr(current, "env", None)
    return ticks, parts


def first_full_stack_banner(ticks: int, parts: list[str], enabled: bool) -> str:
    chain = ", ".join(parts) if parts else "single-frame chain"
    if not enabled:
        return (f"[NETT first full stack] OFF ({FIRST_FULL_STACK_ENV}=0): the first action is taken on the reset "
                f"observation (a full stack would need {ticks} hold tick(s): {chain}) (workspace DECISIONS §187 add. 1)")
    if ticks == 0:
        return (f"[NETT first full stack] hold 0 ticks ({chain}): the first action is taken on the reset observation, "
                f"unchanged (workspace DECISIONS §187 add. 1)")
    return (f"[NETT first full stack] hold {ticks} zero-action tick(s) after every reset ({chain}); not queried, not "
            f"stored, not counted as decisions (workspace DECISIONS §187 add. 1)")


class FirstFullStackEnvWrapper:
    """After every reset, step the env ``ticks`` times with a zero action before the agent sees an observation.

    OWNER, workspace DECISIONS §187: "DVS agents should provide their first action after receiving their first
    non-blank observation (the result of two env steps without the agent moving)." §187 add. 1: "Only for models
    that request multiple frames (DVS, 3DCNN, ViViT, etc). The idea is to hold off until the framestack contains all
    unique frames."

    It wraps the skrl adapter (IsaacEnvWrapper) UNDER DecisionLoopEnvWrapper, so train, test and record all get it
    and every held tick is one env tick. ``ticks`` comes from ``first_full_stack_ticks`` (the body chain), never a
    constant. The held ticks are invisible above this wrapper: skrl never queries the policy on them and records no
    transition for them, the trainer's step counts are agent steps only, and the env (repo B ``set_preroll_ticks``,
    nett_isaac/preroll.py) does not count them -- step_in_episode holds at 0, no reward, no test-log row -- so every
    episode keeps its full NETT_STEPS decisions after the hold. ``reset`` returns the observation after the hold; a
    ``step`` that ends the episodes returns that tick's reward, dones and info with the observation after the next
    episode's hold. The record phase never holds (repo B), so this drains nothing there.

    Fail-loud guards: the env must offer ``set_preroll_ticks`` (an older repo B pin refuses); the pending hold must be
    the same on every env (NETT episodes end together); no held tick may end an episode or carry a reward.
    """

    def __init__(self, env, raw, ticks: int) -> None:
        self._env = env
        self._raw = raw
        self.ticks = int(ticks)

    @classmethod
    def wrap(cls, wrapped, body_env):
        """``wrapped`` = the skrl adapter, ``body_env`` = the body-wrapped env under it. 0 ticks or the knob off =
        ``wrapped`` itself."""
        import os

        enabled = first_full_stack_enabled()
        ticks, parts = first_full_stack_ticks(body_env)
        print(first_full_stack_banner(ticks, parts, enabled), flush=True)
        if not enabled or ticks == 0:
            return wrapped
        from ..body.wrappers.acuity import find_acuity

        if find_acuity(body_env):
            raise ValueError(
                f"{FIRST_FULL_STACK_ENV}: an acuity curriculum counts its own step() calls, so its schedule would "
                f"count the {ticks} held tick(s) per episode. Refusing; set {FIRST_FULL_STACK_ENV}=0 for this arm.")
        if os.environ.get("NETT_XSP2_WINDOWS", "").strip().lower() == "hold":
            raise ValueError(
                f"{FIRST_FULL_STACK_ENV} with NETT_XSP2_WINDOWS=hold: the hold-window sink reads the env's "
                f"step_before after each step, and the held ticks overwrite it at every episode start. Refusing.")
        raw = _unwrap_nett(body_env)
        for name in ("set_preroll_ticks", "_preroll_left"):
            if not hasattr(raw, name):
                raise RuntimeError(
                    f"{FIRST_FULL_STACK_ENV}: the unwrapped env {type(raw).__name__} has no {name}; the hold needs "
                    f"repo B's uncounted pre-roll ticks (nett_isaac/preroll.py, DECISIONS §187 add. 1). Use a repo B "
                    f"tree that has it, or set {FIRST_FULL_STACK_ENV}=0.")
        raw.set_preroll_ticks(ticks)
        return cls(wrapped, raw, ticks)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        observations, info = self._env.reset(*args, **kwargs)
        return self._hold(observations), info

    def step(self, actions):
        observations, reward, terminated, truncated, info = self._env.step(actions)
        if bool((torch.as_tensor(terminated).bool() | torch.as_tensor(truncated).bool()).any()):
            observations = self._hold(observations)
        return observations, reward, terminated, truncated, info

    def _hold(self, observations):
        left = self._raw._preroll_left.reshape(-1)
        n = int(left.max())
        if n == 0:
            return observations
        if bool((left != n).any()):
            bad = int((left != n).nonzero()[0])
            raise RuntimeError(
                f"FirstFullStackEnvWrapper: env {bad} has {int(left[bad])} hold tick(s) pending, another has {n}; the "
                f"hold steps every env, so the episodes must start together.")
        zeros = torch.zeros((int(self._env.num_envs), int(self._env.action_space.shape[0])), device=self._env.device)
        for tick in range(n):
            observations, reward, terminated, truncated, _ = self._env.step(zeros)
            if bool((torch.as_tensor(terminated).bool() | torch.as_tensor(truncated).bool()).any()):
                raise RuntimeError(f"FirstFullStackEnvWrapper: held tick {tick + 1} of {n} ended an episode.")
            if bool((torch.as_tensor(reward) != 0).any()):
                raise RuntimeError(f"FirstFullStackEnvWrapper: held tick {tick + 1} of {n} carried a reward.")
        if int(self._raw._preroll_left.max()) != 0:
            raise RuntimeError("FirstFullStackEnvWrapper: hold ticks still pending after the hold.")
        return observations


def _unwrap_nett(env):
    """The NETTEnv under any chain of gym / skrl wrappers (the trainer's _unwrap_env walk)."""
    current, seen = env, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_step_in_episode") and hasattr(current, "_decision_period"):
            return current
        current = (getattr(current, "_env", None) or getattr(current, "env", None)
                   or getattr(current, "_unwrapped", None))
    return env
