"""Tests for the Brain model factories' handling of NETTEnv obs quirks.

NETTEnv emits ``(B, H, W, 3) uint8`` natively, but skrl's torch memory expects
flat samples. ``_features_forward`` must paper over flat→4D reshape,
HWC→CHW permute, uint8→float scale, and cross-device moves.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.agent_factory import build_agents as _build_agents
from nett_skrl.brain.brain import Brain, IntrinsicRewardAdapter
from nett_skrl.brain.models import features_forward as _features_forward
from nett_skrl.observation import to_chw_space as _hwc_to_chw_space
from nett_skrl.brain.models import ValueCritic, model_cfg_from
from nett_skrl.brain.trainer import BrainTrainer, TrainCfg
from nett_skrl.brain.encoders import NETTFeatureExtractor, Resnet10CNN, Resnet18CNN, SmallCNN
from nett_skrl.brain.rewards import E3B, ICM, PseudoCounts, RIDE
from nett_skrl.brain.env_adapter import NettIsaacLabWrapper
from nett_skrl.brain.registry import algorithm_spec, register_reward


def test_hwc_to_chw_space_image():
    s = gym.spaces.Box(low=0, high=255, shape=(64, 128, 3), dtype=np.uint8)
    t = _hwc_to_chw_space(s)
    assert t.shape == (3, 64, 128)
    assert t.dtype == np.uint8


def test_hwc_to_chw_space_non_image_passthrough():
    s = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
    assert _hwc_to_chw_space(s) is s


@pytest.mark.parametrize("encoder_cls", [SmallCNN, Resnet10CNN, Resnet18CNN])
def test_native_encoders_accept_hwc_and_flat_obs(encoder_cls):
    space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    encoder = encoder_cls(space, features_dim=16)
    hwc = torch.zeros(2, 64, 64, 3, dtype=torch.uint8)
    flat = hwc.view(2, -1)
    assert encoder(hwc).shape == (2, 16)
    assert encoder(flat).shape == (2, 16)


@pytest.mark.parametrize("encoder_cls", [SmallCNN, Resnet10CNN, Resnet18CNN])
@pytest.mark.parametrize("resolution", [16, 32, 64])
def test_native_encoders_accept_schema_valid_resolutions(encoder_cls, resolution):
    space = gym.spaces.Box(low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8)
    encoder = encoder_cls(space, features_dim=16)
    obs = torch.zeros(2, resolution, resolution, 3, dtype=torch.uint8)
    assert encoder(obs).shape == (2, 16)


def test_native_encoders_accept_channel_stacked_video_obs():
    space = gym.spaces.Box(low=0, high=255, shape=(16, 16, 9), dtype=np.uint8)
    encoder = SmallCNN(space, features_dim=16)
    obs = torch.zeros(2, 16, 16, 9, dtype=torch.uint8)
    assert encoder(obs).shape == (2, 16)


def test_unregistered_heavy_encoder_name_fails_at_brain_construction():
    with pytest.raises(KeyError, match="register_encoder"):
        Brain(encoder="DinoV1")


def test_unsupported_intrinsic_reward_fails_at_brain_construction():
    with pytest.raises(ImportError, match="RND"):
        Brain(reward="RND")


class _Stub:
    """Minimal stand-in for a skrl Model — only what `_features_forward` reads."""

    def __init__(self, obs_space, encoder, trunk):
        self.observation_space = obs_space
        self.encoder = encoder
        self.trunk = trunk


def _make_stub_for(obs_shape: tuple[int, int, int]):
    encoder = nn.Conv2d(obs_shape[2], 4, kernel_size=1)  # accepts CHW
    trunk = nn.Identity()
    obs_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    return _Stub(obs_space, encoder, trunk)


def test_features_forward_reshapes_flat_obs():
    stub = _make_stub_for((4, 8, 3))
    flat = torch.zeros(2, 4 * 8 * 3, dtype=torch.uint8)
    out = _features_forward(stub, {"observations": flat})
    assert out.shape == (2, 4, 4, 8)  # (B, out_channels, H, W)


def test_features_forward_uint8_to_float():
    stub = _make_stub_for((4, 8, 3))
    img = torch.full((1, 4, 8, 3), 255, dtype=torch.uint8)
    out = _features_forward(stub, {"observations": img})
    assert out.dtype == torch.float32


def test_features_forward_states_key_fallback():
    stub = _make_stub_for((4, 8, 3))
    img = torch.zeros(1, 4, 8, 3, dtype=torch.uint8)
    # No "observations" key — falls back to "states" (skrl 2.x convention).
    out = _features_forward(stub, {"states": img})
    assert out.shape == (1, 4, 4, 8)


def test_brain_env_reward_types_from_legacy_reward_string():
    assert Brain(reward="closeness").env_reward_types() == ("closeness",)
    assert Brain(reward="closeness,completeness").env_reward_types() == (
        "closeness",
        "completeness",
    )
    assert Brain(reward="unsupervised").env_reward_types() == ()


def test_registered_intrinsic_reward_builds_adapters():
    class DummyIntrinsic:
        def __init__(self, env=None, device=None, **kwargs):
            self.env = env
            self.device = device

    register_reward("DummyIntrinsicForBrainTest", DummyIntrinsic)
    brain = Brain(reward="DummyIntrinsicForBrainTest")

    class _Env:
        num_envs = 2

    adapters = brain._build_intrinsic_adapters(_Env(), torch.device("cpu"))
    assert len(adapters) == 2
    assert all(isinstance(adapter.reward, DummyIntrinsic) for adapter in adapters)


@pytest.mark.parametrize("reward_cls", [ICM, E3B, RIDE, PseudoCounts])
def test_builtin_intrinsic_rewards_compute_reward_shape(reward_cls):
    reward = reward_cls(device="cpu", beta=0.5, latent_dim=8)
    observations = torch.zeros(2, 12)
    next_observations = torch.ones(2, 12)
    actions = torch.zeros(2, 2)
    extrinsic = torch.zeros(2, 1)
    value = reward.compute(
        observations=observations,
        actions=actions,
        rewards=extrinsic,
        terminated=torch.zeros(2, 1, dtype=torch.bool),
        truncated=torch.zeros(2, 1, dtype=torch.bool),
        next_observations=next_observations,
    )
    assert value.shape == extrinsic.shape
    assert torch.isfinite(value).all()


def test_intrinsic_reward_adapter_passes_samples_to_update():
    class SampleAwareReward:
        def __init__(self, *args, **kwargs):
            self.updated_with = None

        def compute(self, **kwargs):
            return torch.ones_like(kwargs["rewards"])

        def update(self, samples):
            self.updated_with = samples

    adapter = IntrinsicRewardAdapter(
        SampleAwareReward,
        env=None,
        device=torch.device("cpu"),
        weight=1.0,
        update_enabled=True,
        kwargs={},
    )
    observations = torch.zeros(1, 4)
    next_observations = torch.ones(1, 4)
    actions = torch.zeros(1, 2)
    rewards = torch.zeros(1, 1)
    terminated = torch.zeros(1, 1, dtype=torch.bool)
    truncated = torch.zeros(1, 1, dtype=torch.bool)

    adapter.watch(observations, actions, rewards, terminated, truncated, next_observations)
    adapter.compute(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        next_observations=next_observations,
    )
    adapter.update()

    assert adapter.reward.updated_with["observations"] is observations
    assert adapter.reward.updated_with["next_observations"] is next_observations


def test_brain_test_uses_full_episode_steps(monkeypatch):
    brain = Brain()
    brain.steps_per_episode = 50
    brain.test_iterations = {"Object1": 3}
    seen = {}

    class _Trainer:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self, total_timesteps):
            seen["steps"] = total_timesteps
            return {}

    monkeypatch.setattr("nett_skrl.brain.brain.BrainTrainer", _Trainer)
    monkeypatch.setattr(
        "nett_skrl.brain.brain._build_skrl_agents",
        lambda brain, wrapped, device, **_: [object()],
    )
    monkeypatch.setattr(
        "nett_skrl.brain.brain._load_latest_checkpoints_fn",
        lambda agents, config: None,
    )

    class _Env:
        num_envs = 1
        observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (4, 8, 3), np.uint8)})
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    class _Config:
        device = 0
        condition = "Object1"
        path = "."

    brain.test(_Env(), _Config())
    assert seen["steps"] == 150


def test_calc_iterations_derives_parallel_envs_per_brain_from_rollout_minibatch_size():
    brain = Brain(algorithm_cfg={"rollouts": 1024, "mini_batches": 2})
    brain.calc_iterations(
        num_brains=1,
        iterations_per_episode={"Object1": 1},
        episodes={"train": 1},
        steps_per_episode=200,
    )
    assert brain.envs_per_brain == 2


def test_brain_train_uses_chunk_timesteps_and_loads_resume_checkpoint(monkeypatch, tmp_path):
    brain = Brain(wandb={"mode": "disabled"})
    brain.train_iterations = 100
    seen = {}

    class _Trainer:
        def __init__(self, wrapped, agents, device):
            seen["agents"] = agents

        def train(self, cfg, **kwargs):
            seen["steps"] = cfg.total_timesteps

    monkeypatch.setattr("nett_skrl.brain.brain.BrainTrainer", _Trainer)
    monkeypatch.setattr(
        "nett_skrl.brain.brain._build_skrl_agents",
        lambda brain, wrapped, device, **_: ["agent"],
    )

    def _load(agents, config, *, context="test"):
        seen["load"] = (agents, context)

    monkeypatch.setattr("nett_skrl.brain.brain._load_latest_checkpoints_fn", _load)

    class _Env:
        num_envs = 1
        observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (4, 8, 3), np.uint8)})
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    class _Config:
        device = 0
        condition = "Object1"
        current_mode = "train"
        dry_run = False
        train_timesteps = 25
        train_global_step = 50
        path = tmp_path

    brain.train(_Env(), _Config())
    assert seen["steps"] == 25
    assert seen["load"] == (["agent"], "train")


def test_nett_wrapper_uses_actual_policy_observation_space():
    class _WrappedLikeEnv:
        num_envs = 2
        observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(0, 255, (3, 16, 16), dtype=np.uint8)}
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        def reset(self, seed=None):
            return {"policy": np.zeros((2, 3, 16, 16), dtype=np.uint8)}, {}

    wrapped = NettIsaacLabWrapper(_WrappedLikeEnv(), device="cpu")
    assert wrapped.observation_space.shape == (3, 16, 16)
    obs, _ = wrapped.reset()
    assert obs.shape == (2, 3 * 16 * 16)
    assert obs.device.type == "cpu"


def test_nett_wrapper_exposes_skrl_optional_env_api():
    class _MinimalEnv:
        num_envs = 1
        observation_space = gym.spaces.Box(0, 255, (4,), dtype=np.uint8)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    wrapped = NettIsaacLabWrapper(_MinimalEnv(), device="cpu")
    assert wrapped.num_agents == 1
    assert wrapped.state() is None
    assert wrapped.render() is None
    assert wrapped.close() is None


class _FakeSkrlEnv:
    num_envs = 1
    num_agents = 1
    observation_space = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def reset(self):
        return torch.zeros(1, 64 * 64 * 3), {}

    def step(self, actions):
        return (
            torch.zeros(1, 64 * 64 * 3),
            torch.zeros(1, 1),
            torch.zeros(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
            {},
        )

    def state(self):
        return None

    def render(self, *args, **kwargs):
        return None

    def close(self):
        pass


def _tiny_algorithm_cfg(algorithm="PPO", *, rollouts=8, mini_batch_size=2):
    if algorithm in {"SAC", "TD3", "DDPG"}:
        return {"memory_size": rollouts, "batch_size": mini_batch_size}
    return {"rollouts": rollouts, "mini_batches": max(1, rollouts // mini_batch_size)}


@pytest.mark.parametrize(
    "algorithm",
    ["PPO", "A2C", "TRPO", "RPO", "CEM", "SAC", "TD3", "DDPG"],
)
def test_supported_algorithms_run_one_fake_step(algorithm):
    env = _FakeSkrlEnv()
    brain = Brain(algorithm=algorithm, algorithm_cfg=_tiny_algorithm_cfg(algorithm))
    agent = _build_agents(brain, env, torch.device("cpu"))[0]
    BrainTrainer(env, [agent], device="cpu").train(TrainCfg(total_timesteps=1))


@pytest.mark.parametrize(
    "algorithm",
    ["PPO", "A2C", "TRPO", "RPO", "CEM", "SAC", "TD3", "DDPG"],
)
def test_supported_algorithms_build_expected_model_keys(algorithm):
    env = _FakeSkrlEnv()
    brain = Brain(algorithm=algorithm, algorithm_cfg=_tiny_algorithm_cfg(algorithm))
    agent = _build_agents(brain, env, torch.device("cpu"))[0]
    assert set(agent.models) == set(algorithm_spec(brain.algorithm).model_keys)


def test_ppo_uses_conservative_nett_stability_defaults():
    env = _FakeSkrlEnv()
    brain = Brain(algorithm="PPO", algorithm_cfg=_tiny_algorithm_cfg(rollouts=64, mini_batch_size=4))
    agent = _build_agents(brain, env, torch.device("cpu"))[0]
    assert agent.cfg.learning_rate == pytest.approx((1e-5, 1e-5))
    assert agent.cfg.rollouts == 64
    assert agent.cfg.mini_batches == 16
    assert agent.cfg.value_loss_scale == pytest.approx(0.25)
    assert agent.cfg.grad_norm_clip == pytest.approx(0.25)
    assert agent.memory.memory_size == 64


def test_algorithm_cfg_can_override_ppo_stability_defaults():
    env = _FakeSkrlEnv()
    brain = Brain(
        algorithm="PPO",
        algorithm_cfg={
            "learning_rate": 2e-5,
            "rollouts": 16,
            "mini_batches": 4,
            "value_loss_scale": 0.75,
            "grad_norm_clip": 0.5,
        },
    )
    agent = _build_agents(brain, env, torch.device("cpu"))[0]
    assert agent.cfg.learning_rate == pytest.approx((2e-5, 2e-5))
    assert agent.cfg.rollouts == 16
    assert agent.cfg.mini_batches == 4
    assert agent.cfg.value_loss_scale == pytest.approx(0.75)
    assert agent.cfg.grad_norm_clip == pytest.approx(0.5)
    assert agent.memory.memory_size == 16


def test_off_policy_agents_keep_replay_buffer_size():
    env = _FakeSkrlEnv()
    brain = Brain(algorithm="SAC", algorithm_cfg={"memory_size": 123, "batch_size": 2})
    agent = _build_agents(brain, env, torch.device("cpu"))[0]
    assert agent.memory.memory_size == 123


def test_value_critic_output_is_bounded_and_finite():
    space = gym.spaces.Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    critic = ValueCritic(
        encoder_cls=SmallCNN,
        encoder_kwargs={"features_dim": 8},
        observation_space=space,
        action_space=action_space,
        device=torch.device("cpu"),
        cfg=model_cfg_from({"hidden_sizes": [8], "value_bound": 3.0}),
    )
    with torch.no_grad():
        critic.value_head.weight.fill_(1000.0)
        critic.value_head.bias.fill_(1000.0)
    value, _ = critic.compute({"observations": torch.full((4, 16, 16, 3), 255, dtype=torch.uint8)})
    assert torch.isfinite(value).all()
    assert value.abs().max() <= 3.0


def test_ppo_first_update_keeps_weights_finite():
    env = _FakeSkrlEnv()
    brain = Brain(
        algorithm="PPO",
        encoder_cfg={"features_dim": 16},
        algorithm_cfg={"rollouts": 128, "mini_batches": 4},
    )
    agent = _build_agents(brain, env, torch.device("cpu"))[0]
    BrainTrainer(env, [agent], device="cpu").train(TrainCfg(total_timesteps=65))
    for model in agent.models.values():
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), name


def test_native_encoders_do_not_inherit_sb3_base_class():
    assert issubclass(SmallCNN, NETTFeatureExtractor)
    assert issubclass(Resnet10CNN, NETTFeatureExtractor)
    assert issubclass(Resnet18CNN, NETTFeatureExtractor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_features_forward_moves_cpu_obs_to_cuda_model():
    stub = _make_stub_for((4, 8, 3))
    stub.encoder = stub.encoder.cuda()
    img = torch.zeros(1, 4, 8, 3, dtype=torch.uint8)  # CPU
    out = _features_forward(stub, {"observations": img})
    assert out.device.type == "cuda"


# ---------- wandb wiring ----------

class _FakeTaskConfig:
    """Minimal duck-typed TaskConfig for `_build_agents(config=...)` tests."""

    def __init__(self, *, condition: str, current_mode: str, run_name: str, tmp):
        self.condition = condition
        self.current_mode = current_mode
        # Orchestrator lays out {output}/{run_name}/{condition}/; the wandb
        # helper reads `config.path.parent.name` as the run name.
        self.path = tmp / run_name / condition
        self.path.mkdir(parents=True, exist_ok=True)


def _normalize_kwargs(agent_cfg):
    """Return the wandb_kwargs dict whether the cfg dataclass stores it directly or not."""
    return dict(agent_cfg.experiment.wandb_kwargs)


def test_wandb_disabled_keeps_directory_but_skips_wandb(tmp_path):
    """Even with wandb disabled, skrl's `experiment.directory` and
    `experiment_name` must be populated so checkpoints land in the
    canonical `{output}/wandb_runs/brain_{i}/checkpoints/` path."""
    env = _FakeSkrlEnv()
    brain = Brain(algorithm="PPO", algorithm_cfg=_tiny_algorithm_cfg(),
                  wandb={"mode": "disabled"})
    config = _FakeTaskConfig(condition="cond", current_mode="train",
                             run_name="run", tmp=tmp_path)
    agent = _build_agents(brain, env, torch.device("cpu"), config=config)[0]
    exp = agent.cfg.experiment
    assert exp.wandb is False
    assert exp.wandb_kwargs == {}
    # Directory + experiment_name still set so checkpoints work without wandb
    assert exp.directory == str(config.path / "wandb_runs")
    assert exp.experiment_name == "brain_1"


def test_wandb_offline_populates_skrl_experiment(tmp_path):
    env = _FakeSkrlEnv()
    brain = Brain(algorithm="PPO", algorithm_cfg=_tiny_algorithm_cfg(),
                  wandb={"project": "p", "entity": "e", "mode": "offline",
                         "tags": ["custom"], "notes": "n"})
    config = _FakeTaskConfig(condition="condA", current_mode="train",
                             run_name="run42", tmp=tmp_path)
    agent = _build_agents(brain, env, torch.device("cpu"), config=config)[0]

    exp = agent.cfg.experiment
    assert exp.wandb is True
    assert exp.experiment_name == "brain_1"
    assert exp.directory.endswith("wandb_runs")

    kw = _normalize_kwargs(agent.cfg)
    assert kw["project"] == "p"
    assert kw["entity"] == "e"
    assert kw["mode"] == "offline"
    assert kw["group"] == "condA"
    assert kw["name"] == "run42/condA/brain_1"
    assert kw["job_type"] == "train"
    assert kw["reinit"] == "create_new"
    assert kw["resume"] == "allow"
    assert kw["id"].startswith("nett-")
    assert kw["sync_tensorboard"] is True
    # auto-tags + user tags
    assert kw["tags"] == ["run42", "condA", "brain_1", "train", "custom"]
    # local wandb cache co-located with NETT output
    assert kw["dir"] == str(config.path)
    assert kw["notes"] == "n"


def test_wandb_entity_and_notes_omitted_when_none(tmp_path):
    env = _FakeSkrlEnv()
    brain = Brain(algorithm="PPO", algorithm_cfg=_tiny_algorithm_cfg(),
                  wandb={"mode": "online"})
    config = _FakeTaskConfig(condition="c", current_mode="test",
                             run_name="r", tmp=tmp_path)
    agent = _build_agents(brain, env, torch.device("cpu"), config=config)[0]

    kw = _normalize_kwargs(agent.cfg)
    assert "entity" not in kw  # None → skip (let wandb use default)
    assert "notes" not in kw
    assert kw["mode"] == "online"
    assert kw["job_type"] == "test"
    assert kw["resume"] == "allow"


def test_wandb_scalar_mirror_routes_tracking_data_to_wandb():
    """The per-agent scalar mirror should call ``wandb.run.log`` with the
    same aggregated values skrl is about to flush to its writer.

    The mirror's init wrap looks up the Run via ``_wandb_runs_by_id`` keyed
    by ``agent.cfg.experiment.wandb_kwargs["id"]`` — NOT via the global
    ``wandb.run`` (which skrl's ``reinit="create_new"`` mode leaves unset).
    """
    from nett_skrl.brain.experiment import attach_wandb_scalar_mirror
    from nett_skrl.brain import experiment as exp_mod

    class _FakeRun:
        def __init__(self) -> None:
            self.logged: list[tuple[dict, int]] = []

        def log(self, data, step=None) -> None:
            self.logged.append((dict(data), step))

    class _FakeExperimentCfg:
        wandb_kwargs = {"id": "test-run-id-xyz"}

    class _FakeCfg:
        experiment = _FakeExperimentCfg()

    class _FakeAgent:
        def __init__(self) -> None:
            self.tracking_data = {
                "Loss / Policy loss": [1.0, 3.0],
                "Loss / Value loss": [0.25, 0.75],
                "Loss / Entropy loss": [-0.02, -0.04],
                "Policy / Standard deviation": [0.6, 0.8],
                "Reward/Total (max)": [0.2, 0.9],
                "Reward/Total (min)": [-0.5, 0.1],
            }
            self.cfg = _FakeCfg()
            self.init_calls = 0
            self.write_calls = 0

        def init(self, *, trainer_cfg=None):
            self.init_calls += 1

        def write_tracking_data(self, *, timestep, timesteps):
            self.write_calls += 1
            # mimic skrl: clear after flush
            self.tracking_data = {k: [] for k in self.tracking_data}

    agent = _FakeAgent()
    attach_wandb_scalar_mirror(agent)
    # Idempotent
    attach_wandb_scalar_mirror(agent)

    # Pre-populate the captured-runs map as ``_install_wandb_init_capture``
    # would after the real ``wandb.init`` call returns the Run.
    fake_run = _FakeRun()
    exp_mod._wandb_runs_by_id["test-run-id-xyz"] = fake_run
    try:
        agent.init(trainer_cfg={})
        assert agent._nett_wandb_run is fake_run, \
            "init wrap should resolve the Run by id, not via wandb.run"
        agent.write_tracking_data(timestep=42, timesteps=1000)
    finally:
        exp_mod._wandb_runs_by_id.pop("test-run-id-xyz", None)

    assert agent.write_calls == 1, "original write_tracking_data should still run"
    assert len(fake_run.logged) == 1
    payload, step = fake_run.logged[0]
    assert step is None
    assert payload["Stats/nett_timestep"] == 42
    assert payload["Loss / Policy loss"] == pytest.approx(2.0)     # mean
    assert payload["Reward/Total (max)"] == pytest.approx(0.9)      # max
    assert payload["Reward/Total (min)"] == pytest.approx(-0.5)     # min
    assert payload["train/policy_gradient_loss"] == pytest.approx(2.0)
    assert payload["train/value_loss"] == pytest.approx(0.5)
    assert payload["train/entropy_loss"] == pytest.approx(-0.03)
    assert payload["train/std"] == pytest.approx(0.7)
    assert payload["train/loss"] == pytest.approx(2.47)
    assert payload["train/n_updates"] == 1


def test_finish_agent_wandb_runs_calls_finish_and_clears():
    """``finish_agent_wandb_runs`` calls ``.finish()`` on each captured Run
    and clears ``_nett_wandb_run`` so subsequent calls are no-ops.
    """
    from nett_skrl.brain.experiment import finish_agent_wandb_runs

    class _FakeRun:
        def __init__(self) -> None:
            self.finished = 0

        def finish(self) -> None:
            self.finished += 1

    class _A: pass

    run_a, run_b = _FakeRun(), _FakeRun()
    a, b, c = _A(), _A(), _A()
    a._nett_wandb_run = run_a
    b._nett_wandb_run = run_b
    # c has no _nett_wandb_run — must not error
    finish_agent_wandb_runs([a, b, c])
    assert run_a.finished == 1
    assert run_b.finished == 1
    assert a._nett_wandb_run is None
    assert b._nett_wandb_run is None
    # Idempotent: second call doesn't double-finish
    finish_agent_wandb_runs([a, b, c])
    assert run_a.finished == 1
    assert run_b.finished == 1


def test_wandb_scalar_mirror_no_op_without_run():
    """When ``_nett_wandb_run`` is None the mirror falls through cleanly."""
    from nett_skrl.brain.experiment import attach_wandb_scalar_mirror

    class _FakeAgent:
        def __init__(self) -> None:
            self.tracking_data = {"k": [1.0]}
            self.write_calls = 0

        def init(self, *, trainer_cfg=None):
            pass

        def write_tracking_data(self, *, timestep, timesteps):
            self.write_calls += 1

    agent = _FakeAgent()
    attach_wandb_scalar_mirror(agent)
    agent._nett_wandb_run = None
    agent.write_tracking_data(timestep=0, timesteps=1)
    assert agent.write_calls == 1  # original ran, no wandb call attempted


def test_wandb_scalar_mirror_logs_episode_total_rewards():
    from nett_skrl.brain.experiment import attach_wandb_scalar_mirror

    class _FakeRun:
        def __init__(self) -> None:
            self.logged: list[dict] = []

        def log(self, data, step=None) -> None:
            self.logged.append(dict(data))

    class _FakeAgent:
        def __init__(self) -> None:
            self.tracking_data = {}
            self.record_calls = 0

        def init(self, *, trainer_cfg=None):
            pass

        def record_transition(self, **kwargs):
            self.record_calls += 1

        def write_tracking_data(self, *, timestep, timesteps):
            pass

    agent = _FakeAgent()
    attach_wandb_scalar_mirror(agent)
    agent._nett_wandb_run = _FakeRun()

    agent.record_transition(
        rewards=torch.tensor([[1.0], [2.0]]),
        terminated=torch.tensor([[False], [False]]),
        truncated=torch.tensor([[False], [False]]),
        timestep=1,
    )
    agent.record_transition(
        rewards=torch.tensor([[3.0], [4.0]]),
        terminated=torch.tensor([[True], [False]]),
        truncated=torch.tensor([[False], [False]]),
        timestep=2,
    )
    agent.record_transition(
        rewards=torch.tensor([[5.0], [6.0]]),
        terminated=torch.tensor([[False], [False]]),
        truncated=torch.tensor([[False], [True]]),
        timestep=3,
    )

    assert agent.record_calls == 3
    logged = agent._nett_wandb_run.logged
    assert [row["rollout/ep_rew_total"] for row in logged] == [4.0, 12.0]
    assert [row["rollout/env_index"] for row in logged] == [0, 1]
    assert [row["rollout/episode"] for row in logged] == [1, 2]
    assert [row["Stats/nett_timestep"] for row in logged] == [2, 3]


def test_wandb_invalid_mode_rejected_at_brain_construction():
    with pytest.raises(ValueError, match="brain.wandb.mode"):
        Brain(algorithm="PPO", algorithm_cfg=_tiny_algorithm_cfg(),
              wandb={"mode": "bogus"})


def test_build_agents_without_config_skips_wandb_wiring():
    env = _FakeSkrlEnv()
    brain = Brain(algorithm="PPO", algorithm_cfg=_tiny_algorithm_cfg(),
                  wandb={"project": "p", "mode": "offline"})
    # No config kwarg → wandb wiring is skipped so the existing test fixtures
    # (which build agents without TaskConfig) keep working.
    agent = _build_agents(brain, env, torch.device("cpu"))[0]
    assert agent.cfg.experiment.wandb is False


def test_wandb_each_brain_gets_unique_run_name(tmp_path):
    class _TwoBrainEnv(_FakeSkrlEnv):
        num_envs = 2

    env = _TwoBrainEnv()
    brain = Brain(algorithm="PPO", algorithm_cfg=_tiny_algorithm_cfg(),
                  wandb={"mode": "offline"})
    config = _FakeTaskConfig(condition="c", current_mode="train",
                             run_name="r", tmp=tmp_path)
    agents = _build_agents(brain, env, torch.device("cpu"), config=config)

    names = [_normalize_kwargs(a.cfg)["name"] for a in agents]
    # Per-brain cfgs MUST hold distinct names so skrl's wandb.init creates
    # distinct runs rather than reinit'ing one run twice with the same name.
    assert names == ["r/c/brain_1", "r/c/brain_2"]
    # And the experiment_dir (used as the TB transport target) is per-brain.
    assert agents[0].cfg.experiment.experiment_name == "brain_1"
    assert agents[1].cfg.experiment.experiment_name == "brain_2"
    # The two cfg.experiment objects must be independent — otherwise the
    # second agent's overrides would clobber the first.
    assert agents[0].cfg.experiment is not agents[1].cfg.experiment


# ---------- checkpoint wiring ----------

from nett_skrl.brain.experiment import pick_checkpoint as _pick_checkpoint  # noqa: E402


def test_checkpoint_freq_drives_skrl_checkpoint_interval(tmp_path):
    env = _FakeSkrlEnv()
    brain = Brain(algorithm="PPO", algorithm_cfg=_tiny_algorithm_cfg(),
                  checkpoint_freq=500, wandb={"mode": "disabled"})
    config = _FakeTaskConfig(condition="c", current_mode="train",
                             run_name="r", tmp=tmp_path)
    agent = _build_agents(brain, env, torch.device("cpu"), config=config)[0]
    # Drives skrl's native auto-checkpoint cadence and the milestone runner's
    # global-step aliases.
    assert agent.cfg.experiment.checkpoint_interval == 500
    # Whole-agent .pt file format (matches Agent.load's expectations).
    assert agent.cfg.experiment.store_separately is False


def test_checkpoint_freq_none_disables_skrl_auto_checkpoint(tmp_path):
    env = _FakeSkrlEnv()
    brain = Brain(algorithm="PPO", algorithm_cfg=_tiny_algorithm_cfg(),
                  checkpoint_freq=None, wandb={"mode": "disabled"})
    config = _FakeTaskConfig(condition="c", current_mode="train",
                             run_name="r", tmp=tmp_path)
    agent = _build_agents(brain, env, torch.device("cpu"), config=config)[0]
    # 0 means "no periodic save"; BrainTrainer still writes a single
    # final_agent.pt at the end of training.
    assert agent.cfg.experiment.checkpoint_interval == 0


def test_pick_checkpoint_prefers_final_then_latest_then_best(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    # Empty dir → None
    assert _pick_checkpoint(d) is None

    # best only → best
    (d / "best_agent.pt").touch()
    assert _pick_checkpoint(d) == d / "best_agent.pt"

    # numbered checkpoints outrank best
    (d / "agent_10.pt").touch()
    (d / "agent_100.pt").touch()
    (d / "agent_50.pt").touch()
    assert _pick_checkpoint(d) == d / "agent_100.pt"

    # final_agent.pt outranks everything
    (d / "final_agent.pt").touch()
    assert _pick_checkpoint(d) == d / "final_agent.pt"

    # Non-numeric agent_*.pt files are ignored (e.g. agent_abc.pt)
    d2 = tmp_path / "checkpoints2"
    d2.mkdir()
    (d2 / "agent_abc.pt").touch()
    (d2 / "best_agent.pt").touch()
    assert _pick_checkpoint(d2) == d2 / "best_agent.pt"


def test_pick_checkpoint_missing_dir_returns_none(tmp_path):
    assert _pick_checkpoint(tmp_path / "does_not_exist") is None


def test_load_latest_checkpoints_reads_skrl_path(tmp_path, monkeypatch):
    """`load_latest_checkpoints` must look under `wandb_runs/brain_i/checkpoints/`,
    matching the path skrl writes to via cfg.experiment.directory + experiment_name."""
    from nett_skrl.brain.experiment import load_latest_checkpoints

    ckpt_dir = tmp_path / "wandb_runs" / "brain_1" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    fake_ckpt = ckpt_dir / "agent_42.pt"
    fake_ckpt.touch()

    loaded = []

    class _FakeAgent:
        def load(self, path):
            loaded.append(path)

    class _Cfg:
        path = tmp_path

    load_latest_checkpoints([_FakeAgent()], _Cfg())
    assert loaded == [str(fake_ckpt)]


def test_load_latest_checkpoints_falls_back_silently_when_missing(tmp_path):
    from nett_skrl.brain.experiment import load_latest_checkpoints

    class _FakeAgent:
        def load(self, path):
            raise AssertionError("load must not be called when no checkpoint exists")

    class _Cfg:
        path = tmp_path  # no wandb_runs/brain_1/checkpoints/ present

    # Must not raise; logs a warning and moves on.
    load_latest_checkpoints([_FakeAgent()], _Cfg())
