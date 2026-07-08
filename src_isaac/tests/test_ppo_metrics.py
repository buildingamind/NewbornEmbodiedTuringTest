"""Tests for the SB3/Unity-parity PPO health metrics (nett_skrl.brain.ppo_metrics).

skrl's stock PPO omits ``approx_kl``, ``clip_fraction``, ``explained_variance``,
``entropy``, ``learning_rate`` and ``clip_range`` that the original SB3/Unity
tensorboard logged. ``track_ppo_health_metrics`` recovers them. These tests use
a lightweight mock agent (no Isaac, no GPU) that mimics the skrl agent surface
the function touches, so they exercise the real metric math and the failure
modes that bit during development:

  * ``MetricsPPO`` must override ``update`` (skrl's method) — overriding
    ``_update`` silently no-ops.
  * the rollout buffer may live on CPU (hybrid memory) while the policy is on
    another device — sampled tensors must be moved before the forward pass.
  * ``states`` is ``None`` when the env has no separate state space.
"""
from __future__ import annotations

import types

import pytest
import torch

from skrl.agents.torch.ppo import PPO
from nett_skrl.brain.ppo_metrics import MetricsPPO, track_ppo_health_metrics


def test_metrics_ppo_overrides_skrl_update_method():
    # The bug that produced zero new metrics: overriding _update (which skrl's
    # PPO does not define) instead of update.
    assert issubclass(MetricsPPO, PPO)
    assert "update" in vars(MetricsPPO), "MetricsPPO must override `update`"
    assert MetricsPPO.update is not PPO.update


class _MockMemory:
    """Returns CPU tensors (as the hybrid CPU memory does) for a 2-minibatch roll."""

    def __init__(self, n=16, obs=4, act=2):
        self.n, self.obs, self.act = n, obs, act
        g = torch.Generator().manual_seed(0)
        self._values = torch.rand(n, 1, generator=g)
        self._returns = torch.rand(n, 1, generator=g)
        self._obs = torch.rand(n, obs, generator=g)
        self._actions = torch.rand(n, act, generator=g)
        self._log_prob = -torch.rand(n, 1, generator=g)

    def get_tensor_by_name(self, name):
        return {"values": self._values, "returns": self._returns}[name]

    def sample_all(self, names, mini_batches=1):
        # names == ["observations", "states", "actions", "log_prob"]; states None
        half = self.n // mini_batches
        out = []
        for i in range(mini_batches):
            s = slice(i * half, (i + 1) * half)
            out.append([self._obs[s], None, self._actions[s], self._log_prob[s]])
        return out


class _MockPolicy:
    """Gaussian-ish policy: act() returns a slightly shifted log_prob so KL>0."""

    def act(self, inputs, role):
        n = inputs["taken_actions"].shape[0]
        return None, {"log_prob": torch.full((n, 1), -0.5)}

    def distribution(self, role):
        return torch.distributions.Normal(torch.zeros(2), torch.ones(2))


class _MockAgent:
    def __init__(self, mini_batches=2, device="cpu"):
        self.device = device
        self.cfg = types.SimpleNamespace(ratio_clip=0.2, mini_batches=mini_batches)
        self.memory = _MockMemory()
        self.policy = _MockPolicy()
        self.optimizer = types.SimpleNamespace(param_groups=[{"lr": 3e-4}])
        self.scheduler = None
        self._observation_preprocessor = lambda x: x
        self._state_preprocessor = lambda x: x
        self.tracking_data = {}

    def track_data(self, tag, value):
        self.tracking_data.setdefault(tag, []).append(value)


def test_all_six_metrics_are_tracked_and_finite():
    agent = _MockAgent()
    track_ppo_health_metrics(agent)
    td = agent.tracking_data
    for tag in [
        "Loss / KL divergence",
        "Loss / Clip fraction",
        "Loss / Explained variance",
        "Policy / Entropy",
        "Policy / Clip range",
        "Learning / Learning rate",
    ]:
        assert tag in td and td[tag], f"missing metric: {tag}"
        val = td[tag][-1]
        assert val == val and abs(val) != float("inf"), f"{tag} not finite: {val}"


def test_metric_values_are_correct():
    agent = _MockAgent()
    track_ppo_health_metrics(agent)
    td = agent.tracking_data
    # clip range == cfg.ratio_clip; lr == optimizer lr
    assert td["Policy / Clip range"][-1] == pytest.approx(0.2)
    assert td["Learning / Learning rate"][-1] == pytest.approx(3e-4)
    # KL = E[(r-1) - log r] with r = exp(new - old); both >= 0
    assert td["Loss / KL divergence"][-1] >= 0.0
    assert 0.0 <= td["Loss / Clip fraction"][-1] <= 1.0
    # explained variance = 1 - Var(returns - values)/Var(returns)
    v = agent.memory._values.flatten()
    r = agent.memory._returns.flatten()
    expected_ev = float(1.0 - (r - v).var(unbiased=False) / r.var(unbiased=False))
    assert td["Loss / Explained variance"][-1] == pytest.approx(expected_ev, rel=1e-5)


def test_handles_cpu_buffer_with_offdevice_policy_request():
    # Sampled tensors are CPU; requesting device='cpu' must not raise (the .to
    # move is exercised); None states must be tolerated.
    agent = _MockAgent(device="cpu")
    track_ppo_health_metrics(agent)  # must not raise
    assert "Loss / KL divergence" in agent.tracking_data


def test_best_effort_never_raises_on_broken_policy():
    agent = _MockAgent()

    def boom(*a, **k):
        raise RuntimeError("policy exploded")

    agent.policy.act = boom
    # KL/clip/entropy silently skipped, but EV/LR/clip-range still logged.
    track_ppo_health_metrics(agent)
    assert "Loss / Explained variance" in agent.tracking_data
    assert "Learning / Learning rate" in agent.tracking_data
    assert "Loss / KL divergence" not in agent.tracking_data
