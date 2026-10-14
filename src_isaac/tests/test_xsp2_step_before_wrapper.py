"""StepBeforeEnvWrapper: the env's own step_before lands on the memory row skrl writes -- MEASURED.

Workspace DECISIONS §110 (hold windows). The row mapping is checked through skrl's own
SequentialTrainer + PPO + RandomMemory (the classes NETT runs), with a ring that wraps and rollouts
SHORTER than an episode (the case the done-anchored recovery at e9d38af could not handle). The fake
env writes its pre-step counter into the observation, so the recorded value must equal the stored
observation's tag on every row; a one-row shift and a post-step read both fail that, so it has power.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

from nett_skrl.brain.aux.xsp2_aux import XSP2Term, hold_step_before
from nett_skrl.brain.encoders.xsp2 import XSP2Encoder
from nett_skrl.brain.env_wrappers import StepBeforeEnvWrapper, step_before_sink
from nett_skrl.brain.trainer import BrainTrainer, _unwrap_env

P = 3
NUM_ENVS = 4
#: tag channels of the flat observation
SB, ENV, POSE, PREV = 0, 1, 2, 3


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NETT_XSP2_BATCH", "NETT_XSP2_PREDICTOR", "NETT_XSP2_HIDDEN", "NETT_XSP2_REFINE_HIDDEN",
                 "NETT_XSP2_LR_SCALE", "NETT_XSP2_SURPRISE", "NETT_XSP2_WINDOWS",
                 "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_CONV3D_ORTHO_INIT", "NETT_TRAIN_DECISION_PERIOD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_XSP2_BATCH", "16")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        yield


class _DPEnv:
    """NETTEnv's counter semantics: _apply_action sets _last_step_for_outputs = step_in_episode, the
    pose moves only when step_before % P == 0, the counter increments, and an episode of ``ep_len``
    steps truncates and resets in the SAME step. obs = [step_in_episode, env, pose, previous pose].
    ``stride`` = counter increment per env.step (2 = the decimation fault the guard must catch).
    """

    def __init__(self, ep_len, period=P, stride=1):
        self.num_envs = len(ep_len)
        self.num_agents = 1
        self.device = "cpu"
        self.observation_space = gym.spaces.Box(-1e6, 1e6, shape=(4,), dtype=np.float32)
        self.state_space = None
        self.action_space = gym.spaces.Box(-1e6, 1e6, shape=(2,), dtype=np.float32)
        self.screens, self.motor = object(), object()
        self.ep_len = torch.tensor(ep_len)
        self.period, self.stride = period, stride
        self._step_in_episode = torch.zeros(self.num_envs, dtype=torch.long)
        self._last_step_for_outputs = torch.zeros(self.num_envs, dtype=torch.long)
        self.pose = torch.arange(self.num_envs) * 1000
        self.prev = self.pose.clone()

    def _decision_period(self):
        return self.period

    def _obs(self):
        return torch.stack([self._step_in_episode, torch.arange(self.num_envs), self.pose, self.prev],
                           dim=1).float()

    def reset(self):
        self._step_in_episode.zero_()
        return self._obs(), {}

    def step(self, actions):
        step_before = self._step_in_episode.clone()
        self._last_step_for_outputs = step_before
        self.prev = self.pose.clone()
        self.pose = self.pose + (step_before % self.period == 0).long()
        self._step_in_episode = self._step_in_episode + self.stride
        done = self._step_in_episode >= self.ep_len
        self._step_in_episode = torch.where(done, torch.zeros_like(step_before), self._step_in_episode)
        self.prev = torch.where(done, self.pose, self.prev)            # a reset frame has no motion
        zeros = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        return self._obs(), torch.zeros(self.num_envs, 1), zeros, done[:, None], {}


class _EnvWrapper(Wrapper):
    def reset(self):
        return self._env.reset()

    def step(self, actions):
        return self._env.step(actions)

    def state(self):
        return None

    def render(self, *args, **kwargs):
        return None

    def close(self):
        return None


class _Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False)
        self.lin = nn.Linear(4, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

    def compute(self, inputs, role=""):
        mean = torch.tanh(self.lin(inputs["observations"] * 1e-3))
        return mean, {"log_std": self.log_std.expand_as(mean)}


class _Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.lin = nn.Linear(4, 1)

    def compute(self, inputs, role=""):
        return self.lin(inputs["observations"]), {}


def _term(monkeypatch):
    monkeypatch.setenv("NETT_XSP2_WINDOWS", "hold")
    monkeypatch.setenv("NETT_TRAIN_DECISION_PERIOD", str(P))
    space = gym.spaces.Box(low=0, high=255, shape=(6, 80, 128), dtype=np.uint8)
    return XSP2Term(XSP2Encoder(space, features_dim=235, policy_input="fg"))


def _agent(env, n_env, memory_size):
    memory = RandomMemory(memory_size=memory_size, num_envs=n_env, device="cpu")
    models = {"policy": _Policy(env.observation_space, env.action_space, "cpu"),
              "value": _Value(env.observation_space, env.action_space, "cpu")}
    return PPO(models=models, memory=memory, observation_space=env.observation_space,
               action_space=env.action_space, device="cpu",
               cfg={"rollouts": memory_size, "learning_starts": 10**9})


def _train(raw, agents, timesteps, sinks=None):
    """skrl's own trainer over the wrapped env; ``sinks`` override each agent's aux (mutations)."""
    wrapped = _EnvWrapper(raw)
    for agent, sink in zip(agents, sinks or [None] * len(agents)):
        if sink is not None:
            agent._aux = sink
    size = raw.num_envs // len(agents)
    env = StepBeforeEnvWrapper.wrap(wrapped, agents, _unwrap_env(wrapped), [size] * len(agents))
    assert isinstance(env, StepBeforeEnvWrapper)
    trainer = SequentialTrainer(env=env, agents=agents if len(agents) > 1 else agents[0],
                                scopes=[size] * len(agents) if len(agents) > 1 else None,
                                cfg={"timesteps": timesteps, "headless": True, "disable_progressbar": True,
                                     "close_environment_at_exit": False})
    trainer.train()


@pytest.mark.parametrize("memory_size,timesteps,ep_steps", [(10, 47, 20), (15, 50, (5, 6, 7, 8))],
                         ids=["rollout-shorter-than-episode", "dones-in-rollout"])
def test_recorded_step_before_is_each_stored_rows_own_counter(monkeypatch, memory_size, timesteps, ep_steps):
    ep_steps = (ep_steps,) * NUM_ENVS if isinstance(ep_steps, int) else ep_steps
    raw = _DPEnv([s * P for s in ep_steps])
    term = _term(monkeypatch)
    agent = _agent(raw, NUM_ENVS, memory_size)
    term.attach_memory(agent.memory)
    agent._aux = term
    _train(raw, [agent], timesteps)
    mem = agent.memory
    assert mem.filled and int(mem.memory_index) == timesteps % memory_size
    obs = mem.tensors["observations"]
    # THE CLAIM: on every stored row the recorded value is the counter the env stepped from.
    assert torch.equal(term._sb_rows, obs[..., SB].long())
    if ep_steps[0] == 20:
        assert (hold_step_before(mem) < 0).all()               # control: the done flags anchor nothing
    # and the draw built on it is last-triple only, one pose across (o0, o1, o2)
    for seed in range(6):
        torch.manual_seed(seed)
        obs_t, obs_tk, _, _ = term._draw_hold("cpu")
        assert (obs_t[:, SB].long() % P == P - 1).all()
        assert torch.equal(obs_tk[:, SB], obs_t[:, SB] + 1)
        assert torch.equal(obs_t[:, PREV], obs_t[:, POSE]) and torch.equal(obs_tk[:, POSE], obs_t[:, POSE])


class _Shift:
    """MUTATION sink: forwards to the real term, but one row late (memory_index + 1)."""
    needs_step_before = True

    def __init__(self, term, size):
        self.term, self.hold_period, self.size = term, term.hold_period, size

    def record_step_before(self, row, sb):
        self.term.record_step_before((row + 1) % self.size, sb)


class _PostStep(_Shift):
    """MUTATION sink: the counter AFTER the step (what reading _step_in_episode would give)."""

    def record_step_before(self, row, sb):
        self.term.record_step_before(row, sb + 1)


@pytest.mark.parametrize("mutant", [_Shift, _PostStep])
def test_a_mis_mapped_record_disagrees_with_the_stored_rows(monkeypatch, mutant):
    raw = _DPEnv([20 * P] * NUM_ENVS)
    term = _term(monkeypatch)
    agent = _agent(raw, NUM_ENVS, 10)
    term.attach_memory(agent.memory)
    agent._aux = term
    _train(raw, [agent], 47, sinks=[mutant(term, 10)])
    assert not torch.equal(term._sb_rows, agent.memory.tensors["observations"][..., SB].long())
    if mutant is _Shift:
        with pytest.raises(RuntimeError, match="step chain"):   # the term's own check catches a shift
            term._draw_hold("cpu")


def test_two_brains_each_record_their_own_env_slice(monkeypatch):
    raw = _DPEnv([7 * P, 9 * P, 11 * P, 13 * P])
    terms, agents = [], []
    for _ in range(2):
        term, agent = _term(monkeypatch), _agent(raw, 2, 12)
        term.attach_memory(agent.memory)
        agent._aux = term
        terms.append(term)
        agents.append(agent)
    _train(raw, agents, 41)
    for b, (term, agent) in enumerate(zip(terms, agents)):
        obs = agent.memory.tensors["observations"]
        assert torch.equal(obs[..., ENV].long().unique(), torch.tensor([2 * b, 2 * b + 1]))
        assert torch.equal(term._sb_rows, obs[..., SB].long())


def test_wrap_is_a_no_op_without_a_hold_sink_and_refuses_what_it_cannot_check(monkeypatch):
    raw = _DPEnv([20 * P] * NUM_ENVS)
    wrapped = _EnvWrapper(raw)
    plain = _agent(raw, NUM_ENVS, 10)
    assert StepBeforeEnvWrapper.wrap(wrapped, [plain], raw, [NUM_ENVS]) is wrapped
    plain._aux = MagicMock()                                    # a Mock's attribute is truthy, not True
    assert step_before_sink(plain) is None
    term = _term(monkeypatch)
    plain._aux = MagicMock(term=term)                            # composite (with_cltt_ref) -> its .term
    assert step_before_sink(plain) is term
    plain._aux = term
    with pytest.raises(ValueError, match="decision_period=2"):
        StepBeforeEnvWrapper.wrap(wrapped, [plain], _DPEnv([60] * NUM_ENVS, period=2), [NUM_ENVS])
    with pytest.raises(RuntimeError, match="_last_step_for_outputs"):
        StepBeforeEnvWrapper.wrap(wrapped, [plain], object(), [NUM_ENVS])
    with pytest.raises(ValueError, match="scope"):
        StepBeforeEnvWrapper.wrap(wrapped, [plain], raw, [NUM_ENVS // 2])


def test_the_post_step_guard_refuses_two_counts_per_step(monkeypatch):
    term = _term(monkeypatch)
    for stride, ok in ((1, True), (2, False)):
        raw = _DPEnv([20 * P] * NUM_ENVS, stride=stride)
        agent = _agent(raw, NUM_ENVS, 10)
        term.attach_memory(agent.memory)
        agent._aux = term
        env = StepBeforeEnvWrapper.wrap(_EnvWrapper(raw), [agent], raw, [NUM_ENVS])
        env.reset()
        if ok:
            env.step(torch.zeros(NUM_ENVS, 2))                  # control: one count per step passes
        else:
            with pytest.raises(RuntimeError, match="one counted step"):
                env.step(torch.zeros(NUM_ENVS, 2))


def test_the_production_trainer_applies_the_wrapper_only_for_a_hold_sink(monkeypatch):
    seen = []

    class UnitSequentialTrainer:
        def __init__(self, *, env, agents, scopes, cfg):
            seen.append(env)

        def train(self):
            pass

    monkeypatch.setattr("nett_skrl.brain.trainer.SequentialTrainer", UnitSequentialTrainer)
    raw = _DPEnv([20 * P] * NUM_ENVS)
    wrapped = _EnvWrapper(raw)
    agent = _agent(raw, NUM_ENVS, 10)
    BrainTrainer(wrapped, [agent], device="cpu")._run_skrl_train(wrapped, timesteps=1)
    term = _term(monkeypatch)
    term.attach_memory(agent.memory)
    agent._aux = term
    BrainTrainer(wrapped, [agent], device="cpu")._run_skrl_train(wrapped, timesteps=1)
    assert seen[0] is wrapped and isinstance(seen[1], StepBeforeEnvWrapper)
