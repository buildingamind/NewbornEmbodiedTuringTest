"""Action-aligned windows: the alignment is MEASURED through real skrl, not read off a docstring.

⛔ THE CLAIM: for a window drawn by `draw_action_window`, `actions[i, j]` is the action applied
between `obs_t[i]` and `obs_tk[i]`, in order. An off-by-one would still train a forward model --
on the action taken AFTER the frame -- so the fixture is an environment whose observation is an
exact integrator of the applied action, driven by skrl's own SequentialTrainer + PPO +
RandomMemory (the classes NETT runs), with per-env episode resets and a wrapped ring buffer.
`obs_{t+k} - obs_t == sum(a_t .. a_{t+k-1})` then holds only under the right alignment, and the
test also shows the SHIFTED alignment fails, so the check has power.

The vicreg_tt refactor (transit arithmetic moved into action_windows) is pinned against a frozen
copy of vicreg_tt_aux.py at 2141520 over every mask outcome.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

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

from nett_skrl.brain.aux.action_windows import (
    MASK_NO_MOTION, MASK_OFF, draw_action_window, transit_weights, window_mean_abs_turn,
)
from nett_skrl.brain.aux.cltt_ref_aux import episode_window_starts
from nett_skrl.brain.aux.vicreg_tt_aux import VICRegTemporalAuxLoss

_SRC = Path(__file__).resolve().parents[1]
FROZEN_VICREG = _SRC / "tests" / "fixtures" / "frozen" / "vicreg_tt_aux_2141520.py.txt"
#: `git rev-parse 2141520:src_isaac/nett_skrl/brain/aux/vicreg_tt_aux.py`, copied from output.
FROZEN_VICREG_BLOB = "fff88ca3bd7a22a62f4b319a5a42593c9a875665"

NUM_ENVS = 4
EP_LEN = (5, 7, 9, 11)          # per-env episode lengths -> resets at different rows


@pytest.fixture(autouse=True)
def _seeded(monkeypatch):
    for name in ("NETT_AUX_BATCH", "NETT_AUX_VICREG_TT_OFFSETS", "NETT_AUX_TRANSIT_MASK"):
        monkeypatch.delenv(name, raising=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        yield


# ------------------------------------------------------------------ real skrl rollout fixture

class _IntegratorEnv:
    """obs = [position, 1000 * env + episode_step]; position += applied action each step.

    Auto-resets like a vectorized Isaac env: on done, the RETURNED next observation is the new
    episode's first one. Even envs terminate, odd envs truncate, so both signals are exercised.
    """

    def __init__(self):
        self.num_envs = NUM_ENVS
        self.num_agents = 1
        self.device = "cpu"
        self.observation_space = gym.spaces.Box(-1e6, 1e6, shape=(2,), dtype=np.float32)
        self.state_space = None
        self.action_space = gym.spaces.Box(-1e6, 1e6, shape=(2,), dtype=np.float32)
        self.pos = torch.zeros(NUM_ENVS)
        self.step_i = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.applied = []

    def _obs(self):
        tag = torch.arange(NUM_ENVS) * 1000 + self.step_i
        return torch.stack([self.pos, tag.float()], dim=1)

    def reset(self):
        self.pos.zero_()
        self.step_i.zero_()
        return self._obs(), {}

    def step(self, actions):
        a = actions.detach().clone()
        self.applied.append(a)
        self.pos = self.pos + a[:, 0]
        self.step_i = self.step_i + 1
        done = self.step_i >= torch.tensor(EP_LEN)
        even = torch.arange(NUM_ENVS) % 2 == 0
        terminated = (done & even)[:, None]
        truncated = (done & ~even)[:, None]
        self.pos = torch.where(done, torch.zeros_like(self.pos), self.pos)
        self.step_i = torch.where(done, torch.zeros_like(self.step_i), self.step_i)
        return self._obs(), torch.zeros(NUM_ENVS, 1), terminated, truncated, {}


class _EnvWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

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
        Model.__init__(self, observation_space=observation_space, action_space=action_space,
                       device=device)
        GaussianMixin.__init__(self, clip_actions=False)
        self.lin = nn.Linear(2, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

    def compute(self, inputs, role=""):
        # Mean depends on the observation, so the action genuinely is "decided from obs[t]".
        mean = torch.tanh(self.lin(inputs["observations"] * 0.01))
        return mean, {"log_std": self.log_std.expand_as(mean)}


class _Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space,
                       device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.lin = nn.Linear(2, 1)

    def compute(self, inputs, role=""):
        return self.lin(inputs["observations"]), {}


def _skrl_rollout(memory_size=40, timesteps=57):
    """Run skrl's own trainer; return (agent memory, env). timesteps > memory_size WRAPS it."""
    env = _IntegratorEnv()
    wrapped = _EnvWrapper(env)
    memory = RandomMemory(memory_size=memory_size, num_envs=NUM_ENVS, device="cpu")
    models = {"policy": _Policy(env.observation_space, env.action_space, "cpu"),
              "value": _Value(env.observation_space, env.action_space, "cpu")}
    agent = PPO(models=models, memory=memory, observation_space=env.observation_space,
                action_space=env.action_space, device="cpu",
                cfg={"rollouts": memory_size, "learning_starts": 10**9})
    trainer = SequentialTrainer(env=wrapped, agents=agent,
                                cfg={"timesteps": timesteps, "headless": True,
                                     "disable_progressbar": True,
                                     # No atexit hook: it logs to a closed pytest stream.
                                     "close_environment_at_exit": False})
    trainer.train()
    return agent.memory, env


@pytest.mark.parametrize("timesteps", [33, 57], ids=["partial-fill", "wrapped-ring"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_stored_action_at_t_is_the_one_applied_between_obs_t_and_obs_t_plus_1(timesteps, k):
    memory, env = _skrl_rollout(timesteps=timesteps)
    assert bool(memory.filled) == (timesteps > memory.memory_size)
    for _ in range(25):
        win = draw_action_window(memory, k, max_samples=3)
        assert win.actions.shape == (win.batch, k, 2)
        tags = win.obs_t[:, 1]
        # Same env, same episode, consecutive steps: the tag encodes (env, episode step).
        assert torch.equal(win.obs_tk[:, 1], tags + k)
        assert torch.equal((tags // 1000).long(), torch.full_like(tags, win.env).long())
        err = (win.obs_tk[:, 0] - win.obs_t[:, 0] - win.actions[:, :, 0].sum(1)).abs().max()
        assert float(err) < 1e-4, (k, win.env, win.t0, float(err))


def test_the_shifted_alignment_fails_so_the_check_has_power():
    memory, _ = _skrl_rollout(timesteps=33)
    acts = memory.tensors["actions"]
    obs = memory.tensors["observations"]
    starts = episode_window_starts(memory, (3,))
    worst = 0.0
    for t, e in starts[:-1].nonzero().tolist():      # need a[t+1..t+2], i.e. t+2 <= t_max-1
        right = obs[t + 2, e, 0] - obs[t, e, 0] - acts[t:t + 2, e, 0].sum()
        wrong = obs[t + 2, e, 0] - obs[t, e, 0] - acts[t + 1:t + 3, e, 0].sum()
        assert abs(float(right)) < 1e-4
        worst = max(worst, abs(float(wrong)))
    assert worst > 0.1


def test_windows_never_cross_a_reset():
    memory, _ = _skrl_rollout(timesteps=57)
    for k in (1, 4):
        for _ in range(40):
            win = draw_action_window(memory, k, max_samples=4)
            # Episode step strictly increasing by exactly k: a reset would restart it at 0.
            assert torch.equal(win.obs_tk[:, 1] - win.obs_t[:, 1], torch.full((win.batch,), float(k)))


# ------------------------------------------------------------------ refusals and weighting

class _FakeMemory:
    def __init__(self, t_max=30, n_env=3, turn=None, done_at=()):
        ids = torch.arange(t_max)[:, None] * 1000 + torch.arange(n_env)[None, :]
        self.tensors = {
            "observations": ids[..., None].float(),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "actions": torch.zeros(t_max, n_env, 2),
        }
        if turn is not None:
            self.tensors["actions"][..., 0] = turn
        for t, e in done_at:
            self.tensors["terminated"][t, e] = True
        self.memory_size = t_max
        self.filled = True
        self.memory_index = 0


def test_missing_actions_refuses():
    m = _FakeMemory()
    del m.tensors["actions"]
    with pytest.raises(ValueError, match="without the actions between them"):
        draw_action_window(m, 2, 4)


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_k_refuses(k):
    with pytest.raises(ValueError, match="positive integer"):
        draw_action_window(_FakeMemory(), k, 4)


def test_uniform_by_default_reports_mask_off():
    win = draw_action_window(_FakeMemory(), 2, 4)
    assert win.mean_turn == MASK_OFF


def test_transit_weighting_on_a_motionless_rollout_reports_no_motion():
    win = draw_action_window(_FakeMemory(), 2, 4, transit_weighted=True)
    assert win.mean_turn == MASK_NO_MOTION


def test_transit_weighting_selects_the_only_turning_slab_and_counts_every_returned_action():
    turn = torch.zeros(30, 3)
    # Motion ONLY at row 20 of env 1. With batch 4 and k 3 a slab covers rows t0..t0+5, so the
    # rows that can see it are t0 in 15..20 -- including t0 = 18..20, where row 20 is reached
    # only through the k-1 trailing actions. Weighting over anchors alone would miss those.
    turn[20, 1] = 1.0
    m = _FakeMemory(turn=turn)
    seen = set()
    for _ in range(200):
        win = draw_action_window(m, 3, 4, transit_weighted=True)
        assert win.env == 1 and 15 <= win.t0 <= 20, (win.env, win.t0)
        assert win.mean_turn == pytest.approx(1.0 / 6.0)
        assert float(win.actions[..., 0].abs().sum()) > 0
        seen.add(win.t0)
    assert seen >= {18, 19, 20}


def test_transit_weighting_respects_episode_boundaries():
    turn = torch.ones(30, 3)
    m = _FakeMemory(turn=turn, done_at=[(t, e) for t in range(30) for e in (0, 2)])
    for _ in range(50):
        win = draw_action_window(m, 2, 4, transit_weighted=True)
        assert win.env == 1


# ------------------------------------------------------------------ vicreg_tt stays identical

def _load_frozen_vicreg():
    name = "nett_skrl.brain.aux._frozen_vicreg_tt_aux_2141520"
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(FROZEN_VICREG))
    spec = importlib.util.spec_from_file_location(name, str(FROZEN_VICREG), loader=loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Enc(nn.Module):
    features_dim = 4

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1))


def test_frozen_vicreg_fixture_is_the_pre_refactor_file():
    data = FROZEN_VICREG.read_bytes()
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == FROZEN_VICREG_BLOB
    assert b"window_mean_abs_turn" not in data


@pytest.mark.parametrize("scenario", ["mask-off", "moving", "motionless", "no-actions",
                                      "resets"])
def test_vicreg_tt_select_window_is_identical_to_the_pre_refactor_code(monkeypatch, scenario):
    if scenario != "mask-off":
        monkeypatch.setenv("NETT_AUX_TRANSIT_MASK", "1")
    torch.manual_seed(0)
    old_cls = _load_frozen_vicreg().VICRegTemporalAuxLoss
    new = VICRegTemporalAuxLoss(_Enc())
    old = old_cls(_Enc())
    turn = torch.rand(48, 5) * 2 - 1
    if scenario == "motionless":
        turn = torch.zeros(48, 5)
    m = _FakeMemory(t_max=48, n_env=5, turn=turn,
                    done_at=[(9, 1), (30, 3), (17, 0)] if scenario == "resets" else ())
    if scenario == "no-actions":
        del m.tensors["actions"]
    n_draws, states = 0, set()
    for batch in (2, 5, 11):
        for avail in (40, 32):
            for seed in range(8):
                torch.manual_seed(seed)
                got_new = new._select_window(m, 5, avail, batch)
                state_new = torch.get_rng_state()
                torch.manual_seed(seed)
                got_old = old._select_window(m, 5, avail, batch)
                assert got_new == got_old, (scenario, batch, avail, seed)
                assert new.last_window_turn == old.last_window_turn
                states.add(min(new.last_window_turn, 0.0))
                # Same RNG consumption, so every later draw in the loss is unchanged too.
                assert torch.equal(state_new, torch.get_rng_state())
                n_draws += 1
    assert n_draws == 48
    # ⛔ Each scenario must reach the branch it is named for, or the identity is vacuous there.
    expected = {"mask-off": {-1.0}, "moving": {0.0}, "motionless": {-3.0},
                "no-actions": {-2.0}, "resets": {0.0}}[scenario]
    assert states == expected, (scenario, states)


def test_order_statistics_are_computed_on_the_cpu_so_the_strict_update_cannot_kill_them():
    """⛔ THE GPU-ONLY DEATH. The PPO update runs under use_deterministic_algorithms(True,
    warn_only=False) and only the AUX BACKWARD is relaxed, so a CUDA `median(dim=...)` (indices
    output) or `quantile` RAISES at the first optimizer step -- the same failure compact_vit.py
    records for adaptive_avg_pool2d, and one a CPU test host can never reproduce. The reachable
    proxy is that these statistics come back on the CPU, having been computed there."""
    from nett_skrl.brain.aux.token_term import column_shift, parked_transit
    match = torch.randint(0, 40, (6, 40))
    assert column_shift(match, 8).device.type == "cpu"
    turn = torch.rand(6)
    parked, transit = parked_transit(turn)
    assert bool((parked | transit).all()) and int(transit.sum()) > 0


def test_shared_arithmetic_matches_the_inline_original_bitwise():
    actions = torch.randn(40, 4, 2)
    starts = torch.rand(33, 4) > 0.3
    avail, n_env, batch = 40, 4, 8
    turn = actions[:avail, :n_env, 0].abs().float().cpu()
    csum = torch.cat([torch.zeros(1, turn.shape[1]), turn.cumsum(0)], dim=0)
    win = (csum[batch:] - csum[:avail - batch + 1]) / float(batch)
    weights = win.masked_fill(~starts, 0.0).flatten().clamp_min(0.0)
    got = window_mean_abs_turn(actions, avail, n_env, batch)
    assert torch.equal(got, win)
    assert torch.equal(transit_weights(got, starts), weights)
