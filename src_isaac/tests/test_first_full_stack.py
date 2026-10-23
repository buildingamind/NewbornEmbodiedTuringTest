"""NETT_FIRST_FULL_STACK: the first action waits until the policy input holds only real, distinct frames
(workspace DECISIONS §187 add. 1).

OWNER: "Only for models that request multiple frames (DVS, 3DCNN, ViViT, etc). The idea is to hold off until the
framestack contains all unique frames." Held steps take a zero action, are not queried, are not stored, and do not
count toward the episode's decisions; single-frame models are unchanged.

The env below is NETTEnv's counters with repo B's own pre-roll arithmetic (nett_isaac.preroll.preroll_tick, so repo B
must be on PYTHONPATH), under each label's REAL body chain (Body(segmentation_wrappers(spec))), the real skrl adapter,
FirstFullStackEnvWrapper and DecisionLoopEnvWrapper, stacked as brain.py stacks them. Every frame it renders is a new
random field with a known id, so a slot of the policy input can be traced to the tick that rendered it. Every check
has a knob-off control that shows the defect the hold removes.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import campaign_train as ct  # noqa: E402
from nett_isaac.preroll import preroll_tick, validate_preroll_ticks  # noqa: E402  (repo B)
from nett_skrl.body.body import Body  # noqa: E402
from nett_skrl.body.skrl_adapter import IsaacEnvWrapper  # noqa: E402
from nett_skrl.brain.env_wrappers import (  # noqa: E402
    FIRST_FULL_STACK_ENV, DecisionLoopEnvWrapper, FirstFullStackEnvWrapper, decision_loop_period,
    first_full_stack_enabled, first_full_stack_ticks)

H, W = 40, 64
KNOBS = (FIRST_FULL_STACK_ENV, "NETT_DECISION_LOOP", "NETT_DVS_COLOR", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU",
         "NETT_DVS_THRESHOLD", "NETT_DVS_BLUR", "NETT_DVS_BLUR_KERNEL", "NETT_FRAMESTACK_N", "NETT_FRAMESTACK_SHUFFLE",
         "NETT_XSP2_WINDOWS", "NETT_MIRROR_EQUIV", "NETT_AUX_LOSS")

#: (label, extra knobs, expected hold ticks). The coordinator's three families first: 3DCNN (framestack),
#: CNN-UnityRecipe+DVS (DVS, decision loop 2) and ViViT.
CASES = [
    ("3DCNN", {}, 1),
    ("CNN-UnityRecipe+DVS", {"NETT_DECISION_LOOP": "2", "NETT_DVS_COLOR": "1"}, 1),
    ("ViViT-CLTT-Ref", {}, 1),
    ("CNN-UnityRecipe+DVS", {}, 1),
    ("CNN-UnityRecipe+DVS", {"NETT_DECISION_LOOP": "2", "NETT_DVS_WINDOW": "2"}, 2),
    ("CNN2F", {}, 1),
    ("3DCNN", {"NETT_FRAMESTACK_N": "4"}, 3),
    ("3DCNN-DVS6", {"NETT_DVS_COLOR": "1"}, 2),
    ("3DCNN-DVS6", {"NETT_DVS_COLOR": "1", "NETT_DECISION_LOOP": "2", "NETT_DVS_WINDOW": "2"}, 3),
    ("ViT-CLTT-Ref", {}, 1),
    ("ViViT-CLTT-Ref-DVS6", {"NETT_DVS_COLOR": "1"}, 2),
]
IDS = [f"{lbl}{''.join(f'-{k[5:].lower()}{v}' for k, v in kn.items())}" for lbl, kn, _ in CASES]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


class _FakeNETT(gym.Env):
    """NETTEnv's counters and autoreset: decision = step % P == 0 outside the pre-roll (preroll_tick), reward
    1 + step // P on decision ticks only, truncation at step >= episode_steps * P, the reset frame re-rendered in the
    same step. ``frames[i]`` is the i-th render; ``episode_start[e]`` the render id of env e's current reset frame."""

    def __init__(self, n_env=2, episode_steps=4, period=1):
        self.num_envs, self.device, self.period = n_env, torch.device("cpu"), period
        self.cfg = SimpleNamespace(episode_steps=episode_steps)
        self.single_observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (H, W, 3), np.uint8)})
        self.observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (n_env, H, W, 3), np.uint8)})
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = self.single_action_space
        self._step_in_episode = torch.zeros(n_env, dtype=torch.long)
        self._preroll_left = torch.zeros(n_env, dtype=torch.long)
        self._preroll_ticks = 0
        self.frames, self.ticks = [], []                 # ticks: (step_before, preroll_left, decision, action)
        self.episode_start = np.zeros(n_env, dtype=np.int64)
        self._rng = np.random.default_rng(7)

    def _decision_period(self):
        return self.period

    def set_preroll_ticks(self, n):
        self._preroll_ticks = validate_preroll_ticks(n)

    def _render(self):
        f = self._rng.integers(0, 256, (self.num_envs, H, W, 3), dtype=np.uint8)
        self.frames.append(f)
        return {"policy": torch.as_tensor(f)}

    def reset(self, seed=None, options=None):
        self._step_in_episode.zero_()
        self._preroll_left.fill_(self._preroll_ticks)
        obs = self._render()
        self.episode_start[:] = len(self.frames) - 1
        return obs, {}

    def step(self, action):
        a = torch.as_tensor(action, dtype=torch.float32).reshape(self.num_envs, 2).clone()
        before, left = self._step_in_episode.clone(), self._preroll_left.clone()
        if self._preroll_ticks == 0:
            decision, after, left_after = (before % self.period) == 0, before + 1, left
        else:
            decision, after, left_after = preroll_tick(before, left, self.period)
        self.ticks.append((before, left, decision, a))
        reward = (1.0 + (before // self.period).float()) * decision.float()
        self._step_in_episode, self._preroll_left = after, left_after
        trunc = self._step_in_episode >= self.cfg.episode_steps * self.period
        self._step_in_episode[trunc] = 0
        self._preroll_left[trunc] = self._preroll_ticks
        obs = self._render()
        self.episode_start[trunc.numpy()] = len(self.frames) - 1
        return obs, reward, torch.zeros_like(trunc), trunc, {}


def _stack(label, knobs, monkeypatch, **kw):
    """brain.py's stack over the fake env: adapter -> FirstFullStack -> DecisionLoop. Returns (raw, body, env)."""
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)
    period = decision_loop_period()
    raw = _FakeNETT(period=period or 1, **kw)
    body = Body(ct.segmentation_wrappers(ct.MODELS[label])).wrap(raw)
    adapter = IsaacEnvWrapper(body, device="cpu")
    env = DecisionLoopEnvWrapper.wrap(FirstFullStackEnvWrapper.wrap(adapter, body), body, period)
    return raw, body, adapter, env


def _slots(obs, body, knobs):
    """Flat (N, D) policy input -> (N, T, C, H, W): T frames of C channels (RGB 3, DVS luma 2, DVS colour 6)."""
    dvs = any(type(w).__name__ == "DVSPolarity" for w in _chain(body))
    cpf = (6 if knobs.get("NETT_DVS_COLOR") == "1" else 2) if dvs else 3
    c = body.observation_space["policy"].shape[-3]
    return torch.as_tensor(obs).reshape(obs.shape[0], c // cpf, cpf, H, W), dvs


def _chain(env):
    out = []
    while isinstance(env, gym.Wrapper):
        out.append(env)
        env = env.env
    return out


def _frame_ids(slots, raw):
    """(N, T) render id of each RGB slot (-1 = no render matches)."""
    ids = np.full(slots.shape[:2], -1)
    for e in range(slots.shape[0]):
        for t in range(slots.shape[1]):
            s = slots[e, t].to(torch.uint8).permute(1, 2, 0).numpy()
            for i, f in enumerate(raw.frames):
                if np.array_equal(f[e], s):
                    ids[e, t] = i
    return ids


def _full_and_real(obs, body, raw, knobs):
    """True when every slot of every env is a real frame of the current episode, all distinct (RGB: traced to its
    render; DVS: a non-blank event map, no two equal)."""
    slots, dvs = _slots(obs, body, knobs)
    if dvs:
        nonblank = slots.flatten(2).amax(-1) > 0                               # (N, T)
        distinct = all(not torch.equal(slots[e, i], slots[e, j]) for e in range(slots.shape[0])
                       for i in range(slots.shape[1]) for j in range(i + 1, slots.shape[1]))
        return bool(nonblank.all()) and distinct
    ids = _frame_ids(slots, raw)
    return bool(all(len(set(row)) == len(row) and min(row) >= raw.episode_start[e] for e, row in enumerate(ids.tolist())))


def _run_episode(env, raw):
    """Agent steps with a nonzero action until the episodes end; returns (obs at the done step, agent steps)."""
    a = torch.full((raw.num_envs, 2), 0.5)
    for n in range(1, 10_000):
        obs, _, term, trunc, _ = env.step(a)
        if bool((term | trunc).any()):
            return obs, n
    raise AssertionError("no episode end")


# ---------------------------------------------------------------- the hold length, derived from the chain

@pytest.mark.parametrize("label, knobs, ticks", CASES + [("CNN-UnityRecipe", {}, 0)], ids=IDS + ["CNN-UnityRecipe"])
def test_hold_length_is_derived_from_the_body_chain(monkeypatch, capsys, label, knobs, ticks):
    raw, body, adapter, env = _stack(label, knobs, monkeypatch)
    assert first_full_stack_ticks(body)[0] == ticks
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[NETT first full stack]")]
    assert len(line) == 1 and f"hold {ticks} " in line[0] and "§187 add. 1" in line[0]
    assert raw._preroll_ticks == ticks


# ---------------------------------------------------------------- the first policy input, every episode

@pytest.mark.parametrize("label, knobs, ticks", CASES, ids=IDS)
def test_first_policy_input_holds_only_real_distinct_frames(monkeypatch, label, knobs, ticks):
    raw, body, _, env = _stack(label, knobs, monkeypatch)
    obs, _ = env.reset()
    assert len(raw.ticks) == ticks                                             # the reset ran the hold
    assert _full_and_real(obs, body, raw, knobs)
    for _ in range(2):                                                         # and after every episode end
        obs, n = _run_episode(env, raw)
        assert n == raw.cfg.episode_steps
        assert _full_and_real(obs, body, raw, knobs)


@pytest.mark.parametrize("label, knobs, ticks", CASES, ids=IDS)
def test_control_knob_off_first_input_repeats_or_blanks_a_frame(monkeypatch, capsys, label, knobs, ticks):
    monkeypatch.setenv(FIRST_FULL_STACK_ENV, "0")
    raw, body, adapter, env = _stack(label, knobs, monkeypatch)
    assert "OFF" in capsys.readouterr().out and raw._preroll_ticks == 0
    obs, _ = env.reset()
    assert len(raw.ticks) == 0 and not _full_and_real(obs, body, raw, knobs)
    obs, _ = _run_episode(env, raw)
    assert not _full_and_real(obs, body, raw, knobs)


# ---------------------------------------------------------------- held ticks: zero action, uncounted, no reward

@pytest.mark.parametrize("label, knobs, ticks", CASES[:3] + [CASES[4], CASES[7]], ids=IDS[:3] + [IDS[4], IDS[7]])
def test_held_ticks_are_zero_action_and_every_episode_keeps_its_decisions(monkeypatch, label, knobs, ticks):
    steps = 200                                                                # the test's decisions per episode
    raw, body, _, env = _stack(label, knobs, monkeypatch, episode_steps=steps)
    p = raw.period
    env.reset()
    a = torch.tensor([[0.25, -0.5], [0.75, 1.0]])
    rewards, dones = [], []
    for _ in range(2 * steps):
        _, r, term, trunc, _ = env.step(a)
        rewards.append(r.reshape(-1).clone())
        dones.append(bool((term | trunc).all()))
    assert [i + 1 for i, d in enumerate(dones) if d] == [steps, 2 * steps]    # 200 agent steps per episode
    # the agent saw exactly each episode's decision rewards 1..200: nothing from a held tick
    assert torch.equal(torch.stack(rewards)[:, 0], torch.arange(1, steps + 1).float().repeat(2))
    # raw ticks: each episode = hold + 200 decisions x P (the next episode's hold ran inside the done step)
    per = ticks + steps * p
    assert len(raw.ticks) == 2 * per + ticks
    for e in range(3):
        ep = raw.ticks[e * per:(e + 1) * per]
        held, rest = ep[:ticks], ep[ticks:]
        assert all(torch.equal(t[3], torch.zeros(2, 2)) and not bool(t[2].any()) for t in held)
        assert all(int(t[0][0]) == 0 for t in held) and [int(t[1][0]) for t in held] == list(range(ticks, 0, -1))
        if e == 2:
            break
        assert [int(t[0][0]) for t in rest] == list(range(steps * p))           # the step counter never saw the hold
        dec = [t for t in rest if bool(t[2].all())]
        assert len(dec) == steps and all(torch.equal(t[3], a) for t in dec)    # 200 decisions, the agent's action


def test_control_knob_off_has_no_held_ticks(monkeypatch):
    monkeypatch.setenv(FIRST_FULL_STACK_ENV, "0")
    raw, _, _, env = _stack("3DCNN-DVS6", {"NETT_DVS_COLOR": "1"}, monkeypatch, episode_steps=6)
    env.reset()
    for _ in range(12):
        env.step(torch.full((2, 2), 0.5))
    assert len(raw.ticks) == 12 and all(bool(t[2].all()) for t in raw.ticks)


# ---------------------------------------------------------------- the rollout buffer: real PPO through skrl

@pytest.mark.parametrize("hold", [True, False], ids=["on", "off"])
def test_rollout_buffer_holds_only_agent_steps(monkeypatch, hold):
    """Brain -> build_agents -> PPO -> skrl SequentialTrainer over the stack, one rollout of 16 steps x 2 envs
    (rollouts 32 = 16 steps per env, build_agents' scaling)."""
    from skrl.trainers.torch import SequentialTrainer

    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain

    if not hold:
        monkeypatch.setenv(FIRST_FULL_STACK_ENV, "0")
    raw, body, _, env = _stack("CNN2F", {}, monkeypatch, episode_steps=5)
    brain = Brain(encoder="nature_cnn", algorithm="PPO", encoder_cfg={"features_dim": 32},
                  algorithm_cfg={"rollouts": 32, "mini_batches": 2, "learning_epochs": 1, "learning_rate": 1e-3},
                  model={"shared_encoder": True, "hidden_sizes": [16]})
    brain.num_brains = 1                                                       # one agent over both envs
    agent = build_agents(brain, env, torch.device("cpu"))[0]
    assert agent.memory.memory_size == 16 and agent.memory.num_envs == 2
    seen = []
    inner = agent.act

    def act(states, *args, **kwargs):
        seen.append(states.clone())
        return inner(states, *args, **kwargs)

    monkeypatch.setattr(agent, "act", act)
    stored = {}
    inner_update = agent.update

    def update(*args, **kwargs):                                               # read the buffer before PPO consumes it
        stored["states"] = agent.memory.get_tensor_by_name("observations").clone()
        stored["rewards"] = agent.memory.get_tensor_by_name("rewards").clone()
        return inner_update(*args, **kwargs)

    monkeypatch.setattr(agent, "update", update)
    SequentialTrainer(env=env, agents=agent, cfg={"timesteps": 16, "headless": True, "disable_progressbar": True}).train()
    assert len(seen) == 16 and "states" in stored                              # queried once per agent step
    states = stored["states"]                                                  # (16, 2, D)
    assert states.shape[:2] == (16, 2)
    assert torch.equal(states.to(torch.float32), torch.stack(seen).to(torch.float32))   # stored = what the policy saw
    assert torch.equal(stored["rewards"][:, 0, 0], torch.tensor([1., 2., 3., 4., 5.] * 3 + [1.]))
    # raw ticks: 16 agent steps + one hold per episode start (4 starts)
    assert len(raw.ticks) == 16 + (4 if hold else 0)
    firsts = [0, 5, 10, 15]                                                    # the rows that open an episode
    for row in firsts:
        ids = _frame_ids(_slots(states[row], body, {})[0], raw)                # (2 envs, 2 slots)
        assert all((len(set(r)) == len(r)) == hold for r in ids.tolist())      # off: the reset frame twice


# ---------------------------------------------------------------- single frame: untouched

def test_single_frame_is_the_unwrapped_adapter_and_bit_identical(monkeypatch):
    runs = {}
    for knob in ("1", "0"):
        monkeypatch.setenv(FIRST_FULL_STACK_ENV, knob)
        torch.manual_seed(0)
        raw, body, adapter, env = _stack("CNN-UnityRecipe", {}, monkeypatch, episode_steps=3)
        assert env is adapter and raw._preroll_ticks == 0
        obs = [env.reset()[0].clone()]
        for k in range(7):
            obs.append(env.step(torch.full((2, 2), 0.1 * k))[0].clone())
        runs[knob] = (torch.stack(obs), [tuple(x.tolist() for x in t) for t in raw.ticks])
    assert torch.equal(runs["1"][0], runs["0"][0]) and runs["1"][1] == runs["0"][1]
    assert len(runs["1"][1]) == 7


# ---------------------------------------------------------------- refusals

def test_knob_is_strict(monkeypatch):
    assert first_full_stack_enabled() is True
    for v, want in (("1", True), (" 1 ", True), ("", True), ("0", False)):
        monkeypatch.setenv(FIRST_FULL_STACK_ENV, v)
        assert first_full_stack_enabled() is want
    for v in ("on", "true", "2", "off"):
        monkeypatch.setenv(FIRST_FULL_STACK_ENV, v)
        with pytest.raises(ValueError, match=FIRST_FULL_STACK_ENV):
            first_full_stack_enabled()


def test_refuses_an_env_without_the_preroll_and_the_knob_off_is_the_escape(monkeypatch):
    class _OldNETT(_FakeNETT):
        set_preroll_ticks = None

    old = _OldNETT()
    del old._preroll_left
    body = Body(ct.segmentation_wrappers(ct.MODELS["3DCNN"])).wrap(old)
    adapter = IsaacEnvWrapper(body, device="cpu")
    with pytest.raises(RuntimeError, match="set_preroll_ticks|_preroll_left"):
        FirstFullStackEnvWrapper.wrap(adapter, body)
    monkeypatch.setenv(FIRST_FULL_STACK_ENV, "0")
    assert FirstFullStackEnvWrapper.wrap(adapter, body) is adapter               # control


def test_refuses_acuity_and_hold_windows(monkeypatch):
    raw = _FakeNETT()
    body = Body(ct.segmentation_wrappers(ct.MODELS["3DCNN"])).wrap(raw)
    adapter = IsaacEnvWrapper(body, device="cpu")
    monkeypatch.setenv("NETT_XSP2_WINDOWS", "hold")
    with pytest.raises(ValueError, match="hold"):
        FirstFullStackEnvWrapper.wrap(adapter, body)
    monkeypatch.delenv("NETT_XSP2_WINDOWS")
    from nett_skrl.body.wrappers.acuity import AcuityCurriculum

    fake = AcuityCurriculum.__new__(AcuityCurriculum)
    fake.__dict__["_env"] = body
    with pytest.raises(ValueError, match="acuity"):
        FirstFullStackEnvWrapper.wrap(adapter, fake)
    assert isinstance(FirstFullStackEnvWrapper.wrap(adapter, body), FirstFullStackEnvWrapper)   # control


def test_refuses_unequal_pending_holds_a_held_done_and_a_held_reward(monkeypatch):
    raw, body, adapter, env = _stack("3DCNN-DVS6", {"NETT_DVS_COLOR": "1"}, monkeypatch, episode_steps=4)
    env.reset()
    raw._preroll_left[:] = torch.tensor([2, 0])
    with pytest.raises(RuntimeError, match="episodes must start together"):
        env._hold(None)
    raw._preroll_left[:] = 2
    raw.cfg.episode_steps = 0                                                  # every tick truncates
    with pytest.raises(RuntimeError, match="ended an episode"):
        env._hold(None)
    raw.cfg.episode_steps = 4
    raw._preroll_left[:] = 1
    raw.set_preroll_ticks(0)                                                   # the counter path that pays a reward
    raw._step_in_episode.zero_()
    with pytest.raises(RuntimeError, match="carried a reward"):
        env._hold(None)
