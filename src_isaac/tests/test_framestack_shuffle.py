"""NETT_FRAMESTACK_SHUFFLE (workspace DECISIONS §63): replace the t-1 slot of the frame stack.

past -- the same env's frame k steps back, k ~ U{10..60} per env per step, clamped to the
        current episode (k <- min(k, age)); never crosses an episode boundary.
env  -- another env's current frame, a fresh derangement each step.
unset -- byte-identical to the plain stack. Anything else refuses.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from nett_skrl.body.wrappers.framestack import FrameStack

N = 4


class _CountingEnv(gym.Env):
    """Batched (N, 4, 4, 3) frames that encode their origin: ch0 = global step, ch1 = env id,
    ch2 = that env's episode index. `dones[t]` lists the envs whose episode ends at step t."""

    action_space = gym.spaces.Box(-1.0, 1.0, (N, 1), dtype=np.float32)
    observation_space = gym.spaces.Box(0, 65535, (N, 4, 4, 3), dtype=np.uint16)    # step > 255

    def __init__(self, dones=None):
        self.dones = dones or {}
        self.t = 0
        self.ep = np.zeros(N, dtype=np.int64)

    def _obs(self):
        o = np.empty((N, 4, 4, 3), dtype=np.uint16)
        o[..., 0] = self.t
        o[..., 1] = np.arange(N)[:, None, None]
        o[..., 2] = self.ep[:, None, None]
        return o

    def reset(self, *, seed=None, options=None):
        self.t = 0
        self.ep[:] = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        term = np.zeros(N, dtype=bool)
        term[list(self.dones.get(self.t, []))] = True
        self.ep[term] += 1          # auto-reset: the returned obs is the new episode's first frame
        return self._obs(), np.zeros(N), term, np.zeros(N, dtype=bool), {}


def _run(env, steps):
    obs, _ = env.reset()
    out = [obs]
    for _ in range(steps):
        obs, *_ = env.step(np.zeros((N, 1), dtype=np.float32))
        out.append(obs)
    return out


def _prev(obs):
    return obs[..., :3]    # channel_stack_frames: oldest first, so t-1 occupies channels 0..2


def _cur(obs):
    return obs[..., 3:]


def test_unset_is_byte_identical_to_the_plain_stack(monkeypatch):
    dones = {5: [1], 9: [0, 3]}
    monkeypatch.delenv("NETT_FRAMESTACK_SHUFFLE", raising=False)
    plain = _run(FrameStack(_CountingEnv(dones), n_stack=2), 20)
    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", "")
    empty_env = FrameStack(_CountingEnv(dones), n_stack=2)
    assert empty_env.shuffle is None
    empty = _run(empty_env, 20)
    for t, (a, b) in enumerate(zip(plain, empty)):
        assert a.dtype == b.dtype and np.array_equal(a, b)
        # The plain contract itself: t-1 is the previous step's frame, or the scrubbed first frame.
        if t > 0:
            restarted = np.isin(np.arange(N), dones.get(t, []))
            want = np.where(restarted, t, t - 1)
            assert np.array_equal(_prev(a)[:, 0, 0, 0], want)


@pytest.mark.parametrize("bad", ["yes", "1", "pst", "envs", "both"])
def test_invalid_value_refuses(monkeypatch, bad):
    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", bad)
    with pytest.raises(ValueError, match="NETT_FRAMESTACK_SHUFFLE"):
        FrameStack(_CountingEnv(), n_stack=2)


def test_past_draws_k_in_10_to_60_within_the_episode(monkeypatch, capsys):
    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", "past")
    env = FrameStack(_CountingEnv(), n_stack=2)
    assert "[NETT framestack] shuffle=past k~U{10..60}" in capsys.readouterr().out
    out = _run(env, 300)
    ks = []
    for t, obs in enumerate(out):
        prev, cur = _prev(obs), _cur(obs)
        assert np.array_equal(cur[:, 0, 0, 0], np.full(N, t))           # the t slot is untouched
        assert np.array_equal(prev[..., 1], cur[..., 1])                 # same env
        k = t - prev[:, 0, 0, 0].astype(int)
        assert np.all(k >= 0) and np.all(k <= np.minimum(60, t))
        if t >= 60:
            assert np.all(k >= 10)
            ks.extend(k.tolist())
        else:
            assert np.all(k >= min(10, t))                               # clamped only by the age
    assert min(ks) == 10 and max(ks) == 60                               # the whole range is drawn
    assert len(set(ks)) == 51


def test_past_clamps_at_episode_start_and_never_crosses(monkeypatch):
    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", "past")
    dones = {70: [1], 75: [1, 2], 140: [0]}
    out = _run(FrameStack(_CountingEnv(dones), n_stack=2), 200)
    start = np.zeros(N, dtype=int)
    for t, obs in enumerate(out):
        for e in dones.get(t, []):
            start[e] = t
        prev, cur = _prev(obs), _cur(obs)
        assert np.array_equal(prev[..., 2], cur[..., 2])                 # same episode, always
        src = prev[:, 0, 0, 0].astype(int)
        assert np.all(src >= start)                                      # never before this episode
        age = t - start
        k = t - src
        assert np.all(k <= np.minimum(age, 60))
        assert np.all(k >= np.minimum(age, 10))
        # At an episode's first step, t-1 is the first frame itself.
        for e in dones.get(t, []):
            assert src[e] == t


def test_past_reset_clears_the_ring(monkeypatch):
    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", "past")
    env = FrameStack(_CountingEnv(), n_stack=2)
    _run(env, 80)
    obs, _ = env.reset()
    assert np.array_equal(_prev(obs), _cur(obs))
    obs, *_ = env.step(np.zeros((N, 1), dtype=np.float32))
    assert np.array_equal(_prev(obs)[:, 0, 0, 0], np.zeros(N))          # k clamped to age 1


def test_env_takes_another_envs_current_frame(monkeypatch, capsys):
    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", "env")
    env = FrameStack(_CountingEnv({4: [2]}), n_stack=2)
    assert "[NETT framestack] shuffle=env" in capsys.readouterr().out
    perms = set()
    for t, obs in enumerate(_run(env, 200)):
        prev, cur = _prev(obs), _cur(obs)
        src = prev[:, 0, 0, 1].astype(int)
        assert np.array_equal(prev[..., 0], cur[..., 0])                 # the CURRENT step
        assert np.all(src != np.arange(N))                               # never the env itself
        assert sorted(src.tolist()) == list(range(N))                    # a permutation
        perms.add(tuple(src.tolist()))
    assert len(perms) == 9                                               # all derangements of 4


def test_env_refuses_a_single_env(monkeypatch):
    class _One(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros((4, 4, 3), dtype=np.uint8), {}

    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", "env")
    with pytest.raises(ValueError, match=">= 2 envs"):
        FrameStack(_One(), n_stack=2).reset()


def test_shuffle_follows_the_torch_seed(monkeypatch):
    import torch

    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", "past")
    runs = []
    for seed in (3, 3, 4):
        torch.manual_seed(seed)
        runs.append(np.stack(_run(FrameStack(_CountingEnv(), n_stack=2), 120)))
    assert np.array_equal(runs[0], runs[1]) and not np.array_equal(runs[0], runs[2])


@pytest.mark.parametrize("model,refused", [("CNN", True), ("3DCNN-1F", True), ("3DCNN", False)])
def test_campaign_refuses_shuffle_on_an_arm_without_a_stack(monkeypatch, model, refused):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train as campaign

    assert campaign.MODELS[model]["framestack"] is (not refused)
    monkeypatch.setenv("NETT_MODEL", model)
    monkeypatch.setenv("NETT_EXPERIMENT", "parsing")
    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", "past")
    monkeypatch.setenv("NETT_BRAINS", "not-an-int")    # past the guard, main stops here
    with pytest.raises(ValueError) as e:
        campaign.main()
    assert ("has no frame stack" in str(e.value)) is refused
