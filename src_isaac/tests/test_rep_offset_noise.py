"""NETT_TEST_REP_OFFSET, repo A half: Trainer.eval re-seeds the test action noise once from env.test_noise_seed()
(repo B rep_offset.py), so a continued test does not replay the noise the agent-build seed gave the earlier test."""

from __future__ import annotations

import torch

import nett_skrl.brain.trainer as trainer_mod
from nett_skrl.brain.trainer import BrainTrainer


class _Env:
    """One-env stand-in; test_noise_seed() is what NETTEnv returns (None = no offset)."""

    def __init__(self, seed):
        self.num_envs = 1
        self.num_agents = 1
        self.device = torch.device("cpu")
        self._seed = seed

    def test_noise_seed(self):
        return self._seed

    def eval_noise_key(self):
        return None

    def reset(self):
        return torch.zeros(1, 2), {}

    def step(self, actions):
        z = torch.zeros(1, 1)
        return torch.zeros(1, 2), z, z.bool(), z.bool(), {}

    def state(self):
        return None


class _NoisyAgent:
    """act() draws from the GLOBAL torch RNG, as skrl's Normal.rsample does in a sampled test."""

    def __init__(self):
        self.draws = []

    def enable_training_mode(self, enabled, *, apply_to_models=False):
        pass

    def act(self, observations, states, *, timestep, timesteps):
        a = torch.randn(observations.shape[0], 2)
        self.draws.append(a.clone())
        return a, {}


def _run(monkeypatch, seed, pre_seed=123, steps=5):
    monkeypatch.setattr(trainer_mod, "_EVAL_STOCHASTIC", True)
    torch.manual_seed(pre_seed)   # the agent-build seed (agent_factory: base + global brain id)
    agent = _NoisyAgent()
    BrainTrainer(_Env(seed), [agent], device="cpu").eval(total_timesteps=steps, show_progress=False)
    return torch.cat(agent.draws)


def _stream(seed, steps=5):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([torch.randn(1, 2, generator=g) for _ in range(steps)])


def test_no_offset_keeps_the_agent_build_stream(monkeypatch):
    assert torch.equal(_run(monkeypatch, None), _stream(123))


def test_offset_seed_replaces_the_stream(monkeypatch, capsys):
    got = _run(monkeypatch, 987654321)
    assert torch.equal(got, _stream(987654321))
    assert not torch.equal(got, _stream(123))
    assert ("[NETT eval] NETT_TEST_REP_OFFSET: test noise re-seeded 987654321 before the first action"
            in capsys.readouterr().out)


def test_no_offset_prints_no_reseed_line(monkeypatch, capsys):
    _run(monkeypatch, None)
    assert "re-seeded" not in capsys.readouterr().out


def test_offset_seed_is_independent_of_the_prior_state(monkeypatch):
    assert torch.equal(_run(monkeypatch, 42, pre_seed=1), _run(monkeypatch, 42, pre_seed=2))


def test_eval_restores_the_rng_after_an_offset_test(monkeypatch):
    torch.manual_seed(5)
    want = torch.get_rng_state()
    monkeypatch.setattr(trainer_mod, "_EVAL_STOCHASTIC", True)
    BrainTrainer(_Env(42), [_NoisyAgent()], device="cpu").eval(total_timesteps=3, show_progress=False)
    assert torch.equal(torch.get_rng_state(), want)


def test_deterministic_eval_never_reseeds(monkeypatch):
    # NETT_EVAL_STOCHASTIC=0: raw_env is None, so test_noise_seed is never consulted.
    monkeypatch.setattr(trainer_mod, "_EVAL_STOCHASTIC", False)

    class _Boom(_Env):
        def test_noise_seed(self):
            raise AssertionError("consulted under deterministic eval")

    BrainTrainer(_Boom(1), [_NoisyAgent()], device="cpu").eval(total_timesteps=2, show_progress=False)
