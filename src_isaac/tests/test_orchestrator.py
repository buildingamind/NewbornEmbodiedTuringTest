"""Unit tests for MultiBrainTrainer's per-env-slice dispatch.

Mocks both the env and the skrl agents so this runs without Isaac Sim and
without skrl's optimizer step. The point is to lock the slicing contract:
N agents each see only their env-slice for act/record_transition.
"""

from __future__ import annotations

import json
import csv

import torch

from nett_skrl.brain.trainer import MultiBrainTrainer, TrainCfg
from nett_skrl.runtime.task_runner import (
    _is_tolerated_isaac_teardown_exit,
    _training_boundaries,
    _write_eval_metrics,
)
from nett_skrl.runtime.task import TaskConfig


class _FakeEnv:
    """Minimal skrl-wrapper-compatible env yielding random tensors."""

    def __init__(self, num_envs: int, obs_dim: int = 4, act_dim: int = 2,
                 device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self._t = 0

    def _obs(self) -> torch.Tensor:
        return torch.randn(self.num_envs, self.obs_dim, device=self.device)

    def reset(self):
        return self._obs(), {}

    def step(self, actions):
        # Verify action stack shape matches (num_envs, act_dim).
        assert actions.shape == (self.num_envs, self.act_dim), actions.shape
        self._t += 1
        return (
            self._obs(),
            torch.randn(self.num_envs, 1, device=self.device),
            torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device),
            {},
        )


class _FakeAgent:
    """Records every call so the test can assert per-slice indexing."""

    def __init__(self, act_dim: int = 2) -> None:
        self.act_dim = act_dim
        self.act_calls: list = []
        self.record_calls: list = []
        self.post_calls = 0
        self.pre_calls = 0
        self.init_calls = 0
        self.mode = None
        self.saved: list[str] = []

    def init(self, trainer_cfg=None):
        self.init_calls += 1

    def enable_training_mode(self, enabled: bool, *, apply_to_models: bool = False) -> None:
        self.mode = "train" if enabled else "eval"

    def pre_interaction(self, *, timestep, timesteps):
        self.pre_calls += 1

    def act(self, observations, states, *, timestep, timesteps):
        self.act_calls.append(observations.clone())
        # Stub action: zeros, same batch dim as obs.
        return torch.zeros(observations.shape[0], self.act_dim), {}

    def record_transition(self, **kwargs):
        self.record_calls.append({k: v for k, v in kwargs.items() if k != "infos"})

    def post_interaction(self, *, timestep, timesteps):
        self.post_calls += 1

    def track_data(self, *args, **kwargs):
        pass

    def save(self, path: str):
        self.saved.append(path)


def test_trainer_dispatches_each_slice_to_its_agent():
    n = 3
    env = _FakeEnv(num_envs=n)
    agents = [_FakeAgent() for _ in range(n)]
    trainer = MultiBrainTrainer(env, agents, device="cpu")
    trainer.train(TrainCfg(total_timesteps=4))

    # Each agent saw the same number of act + record_transition calls as steps.
    for i, agent in enumerate(agents):
        assert len(agent.act_calls) == 4, f"agent {i}: {len(agent.act_calls)}"
        assert len(agent.record_calls) == 4
        # Each call's observation batch should be (1, obs_dim) — single slice.
        for obs in agent.act_calls:
            assert obs.shape == (1, env.obs_dim)
        assert agent.post_calls == 4


def test_trainer_eval_switches_mode_to_eval():
    env = _FakeEnv(num_envs=2)
    agents = [_FakeAgent() for _ in range(2)]
    trainer = MultiBrainTrainer(env, agents, device="cpu")
    means = trainer.eval(total_timesteps=3)
    assert set(means.keys()) == {0, 1}
    assert all(a.mode == "eval" for a in agents)


def test_trainer_eval_zero_steps_returns_zero_without_reset():
    env = _FakeEnv(num_envs=2)
    agents = [_FakeAgent() for _ in range(2)]
    trainer = MultiBrainTrainer(env, agents, device="cpu")
    assert trainer.eval(total_timesteps=0) == {0: 0.0, 1: 0.0}


def test_trainer_raises_on_env_agent_mismatch():
    env = _FakeEnv(num_envs=3)
    agents = [_FakeAgent() for _ in range(2)]
    try:
        MultiBrainTrainer(env, agents, device="cpu")
    except ValueError as e:
        assert "num_envs" in str(e)
    else:
        raise AssertionError("expected ValueError on size mismatch")


def test_trainer_saves_final_checkpoint_per_agent_experiment_dir(tmp_path):
    """``MultiBrainTrainer.train`` must write ``final_agent.pt`` into each
    agent's own ``experiment_dir`` (the same dir skrl writes ``agent_{N}.pt``
    + ``best_agent.pt`` to). NETT's old ``brain_{i}/models/`` layout is gone;
    checkpoints are owned end-to-end by skrl now."""
    env = _FakeEnv(num_envs=2)
    agents = [_FakeAgent() for _ in range(2)]
    # The trainer reads ``agent.experiment_dir`` to locate the checkpoint dir.
    # In real runs this is populated by ``Agent.init()`` from
    # ``cfg.experiment.directory + experiment_name``; here we set it directly.
    agents[0].experiment_dir = str(tmp_path / "wandb_runs" / "brain_1")
    agents[1].experiment_dir = str(tmp_path / "wandb_runs" / "brain_2")
    trainer = MultiBrainTrainer(env, agents, device="cpu")
    trainer.train(TrainCfg(total_timesteps=2, hparams_dir=tmp_path))
    assert agents[0].saved[-1].endswith("wandb_runs/brain_1/checkpoints/final_agent.pt")
    assert agents[1].saved[-1].endswith("wandb_runs/brain_2/checkpoints/final_agent.pt")
    # Hparams JSON backup still written (skrl auto-captures hparams to wandb,
    # but the local backup keeps offline tooling working).
    assert (tmp_path / "logs" / "hparams.json").exists()
    timing = json.loads((tmp_path / "logs" / "train_timing.json").read_text())
    assert timing["env_timesteps"] == 2
    assert timing["train_steps"] == 4
    assert timing["train_steps_per_second"] >= timing["env_steps_per_second"]


class _Intrinsic:
    def __init__(self, value: float):
        self.value = value
        self.watch_calls = 0
        self.update_calls = 0

    def watch(self, *args):
        self.watch_calls += 1

    def compute(self, **kwargs):
        return torch.full_like(kwargs["rewards"], self.value)

    def update(self):
        self.update_calls += 1


def test_intrinsic_reward_is_added_before_record_transition():
    class _ConstantRewardEnv(_FakeEnv):
        def step(self, actions):
            obs, reward, terminated, truncated, infos = super().step(actions)
            return obs, torch.ones_like(reward), terminated, truncated, infos

    env = _ConstantRewardEnv(num_envs=1)
    agent = _FakeAgent()
    intrinsic = _Intrinsic(value=0.5)
    trainer = MultiBrainTrainer(env, [agent], device="cpu")
    trainer.train(
        TrainCfg(total_timesteps=1),
        intrinsic_reward_adapters=[intrinsic],
    )

    recorded_reward = agent.record_calls[0]["rewards"]
    assert intrinsic.watch_calls == 1
    assert intrinsic.update_calls == 1
    assert recorded_reward.item() == 1.5


def test_eval_checkpoint_boundaries_use_eval_and_checkpoint_milestones():
    assert _training_boundaries(
        200_000,
        eval_freq=50_000,
        checkpoint_freq=60_000,
    ) == [50_000, 60_000, 100_000, 120_000, 150_000, 180_000, 200_000]


def test_only_known_isaac_teardown_exit_codes_are_tolerated():
    assert _is_tolerated_isaac_teardown_exit(-11)
    assert _is_tolerated_isaac_teardown_exit(139)
    assert not _is_tolerated_isaac_teardown_exit(1)


def test_write_eval_metrics_appends_csv_and_jsonl(tmp_path):
    config = TaskConfig(
        "Object1",
        tmp_path / "run",
        ["train", "test"],
        current_mode="test",
        eval_step=50_000,
    )

    class _Brain:
        steps_per_episode = 200
        test_iterations = {"Object1": 2}

    _write_eval_metrics(config, {0: 0.25, 1: 0.5}, _Brain())
    _write_eval_metrics(config, {0: 0.75}, _Brain())

    csv_path = tmp_path / "run" / "Object1" / "logs" / "eval_metrics.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["eval_step"] for r in rows] == ["50000", "50000", "50000"]
    assert [r["brain_id"] for r in rows] == ["1", "2", "1"]
    assert rows[0]["timesteps"] == "400"

    jsonl = (tmp_path / "run" / "Object1" / "logs" / "eval_metrics.jsonl").read_text().splitlines()
    assert len(jsonl) == 3
    assert json.loads(jsonl[0])["condition"] == "Object1"
