"""NETT_FIXED_ACTION_STD (workspace DECISIONS §187 add. 9; owner 2026-10-23: "make sure that action noise is fixed at 1").

Knob on: log_std is a persistent zero BUFFER (not a parameter) -- untrained, outside the optimizer, kept under the
checkpoint key ``log_std``, and a checkpoint with a non-zero log_std refuses to load. Knob off: the original trainable
nn.Parameter, bit for bit. Every "on" claim is paired with an "off" control that shows the property is not automatic.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.models import GaussianActor, ModelCfg, ValueCritic
from nett_skrl.brain.models.multivariate_gaussian_actor import MultivariateGaussianActor
from nett_skrl.brain.models.utils import action_std as A
from nett_skrl.brain.models.utils.features import MIRROR_ENV

ENV = A.FIXED_STD_ENV


class _TinyEncoder(nn.Module):
    def __init__(self, observation_space, **kw):
        super().__init__()
        self.features_dim = 8
        self.net = nn.Linear(int(np.prod(observation_space.shape)), 8)

    def forward(self, x):
        return self.net(x.float().flatten(1))


class _TinyImageEncoder(nn.Module):
    """For the mirror path: a (1, 4, 6) CHW image."""

    def __init__(self, observation_space, **kw):
        super().__init__()
        self.features_dim = 8
        self.net = nn.Linear(int(np.prod(observation_space.shape)), 8)

    def forward(self, x):
        return self.net(x.float().flatten(1))


def _spaces(img=False):
    obs = gym.spaces.Box(low=0, high=1, shape=(1, 4, 6) if img else (4,), dtype=np.float32)
    act = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return obs, act


def _actor(cls=GaussianActor, img=False, **cfgkw):
    obs, act = _spaces(img)
    torch.manual_seed(0)
    return cls(encoder_cls=_TinyImageEncoder if img else _TinyEncoder, encoder_kwargs={}, observation_space=obs,
               action_space=act, device="cpu", cfg=ModelCfg(hidden_sizes=[8], **cfgkw))


# ── the knob is strict ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("val,want", [(None, False), ("", False), ("0", False), (" 1 ", True), ("1", True)])
def test_knob_values(monkeypatch, val, want):
    if val is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, val)
    assert A.fixed_action_std_enabled() is want


@pytest.mark.parametrize("val", ["true", "yes", "2", "0.0", "on", "-1"])
def test_knob_refuses_anything_else(monkeypatch, val):
    monkeypatch.setenv(ENV, val)
    with pytest.raises(ValueError, match=ENV):
        A.fixed_action_std_enabled()
    with pytest.raises(ValueError, match=ENV):
        _actor()


def test_knob_refuses_a_nonzero_initial_log_std(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(ValueError, match="initial_log_std"):
        _actor(initial_log_std=-0.5)


# ── structure: buffer vs parameter ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [GaussianActor, MultivariateGaussianActor])
def test_on_log_std_is_a_zero_buffer_not_a_parameter(monkeypatch, capsys, cls):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(A, "_banner_printed", False)
    m = _actor(cls)
    names = dict(m.named_parameters())
    assert "log_std" not in names and all(p is not m.log_std for p in m.parameters())
    assert "log_std" in dict(m.named_buffers()) and torch.equal(m.log_std, torch.zeros(2))
    assert "log_std" in m.state_dict()                               # checkpoint key kept (policy/log_std)
    if cls is MultivariateGaussianActor:
        assert "tril_offdiag" not in names and torch.equal(m.tril_offdiag, torch.zeros(1))
    assert A.BANNER in capsys.readouterr().out


@pytest.mark.parametrize("cls", [GaussianActor, MultivariateGaussianActor])
def test_off_control_log_std_is_a_trainable_parameter(monkeypatch, cls):
    monkeypatch.delenv(ENV, raising=False)
    m = _actor(cls)
    names = dict(m.named_parameters())
    assert "log_std" in names and names["log_std"].requires_grad
    if cls is MultivariateGaussianActor:
        assert "tril_offdiag" in names


def test_off_is_bit_identical_to_the_original_construction(monkeypatch):
    """Off: same parameter list (names, order, values) and same outputs as the pre-knob code path, which built
    ``nn.Parameter(torch.full((n,), initial_log_std))`` at the same point of __init__."""
    monkeypatch.delenv(ENV, raising=False)
    a, b = _actor(), _actor()
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    for (_, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q)
    x = torch.rand(5, 4)
    ma, oa = a.compute({"observations": x, "states": x}, role="policy")
    mb, ob = b.compute({"observations": x, "states": x}, role="policy")
    assert torch.equal(ma, mb) and torch.equal(oa["log_std"], ob["log_std"])


def test_on_and_off_share_every_other_weight(monkeypatch):
    """Knob on changes only log_std's kind: the seeded encoder/trunk/mean weights are identical."""
    monkeypatch.delenv(ENV, raising=False)
    off = _actor()
    monkeypatch.setenv(ENV, "1")
    on = _actor()
    pon = dict(on.named_parameters())
    for n, p in off.named_parameters():
        if n == "log_std":
            continue
        assert torch.equal(p, pon[n]), n


# ── the sampled std is 1 ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [GaussianActor, MultivariateGaussianActor])
def test_on_sampled_action_std_is_one(monkeypatch, cls):
    monkeypatch.setenv(ENV, "1")
    m = _actor(cls, clip_actions=False)
    x = torch.rand(1, 4).expand(20000, 4)
    with torch.no_grad():
        acts, out = m.act({"observations": x, "states": x}, role="policy")
    std = (acts - out["mean_actions"]).std(0)
    assert torch.allclose(std, torch.ones(2), atol=0.02)
    assert float(m.distribution(role="policy").stddev.mean()) == pytest.approx(1.0) if cls is GaussianActor else True


# ── real skrl PPO: the optimizer never sees it; updates leave it at exactly 0 ─────────────────────

def _ppo(monkeypatch, tmp_path, on, train=True):
    from skrl.agents.torch.ppo import PPO
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer

    monkeypatch.chdir(tmp_path)                                       # skrl writes runs/ under the cwd
    if on:
        monkeypatch.setenv(ENV, "1")
    else:
        monkeypatch.delenv(ENV, raising=False)

    class Bandit(gym.Env):
        """Reward -(a - 0.3)^2: a policy-gradient signal on the std (narrower = better) on every step."""
        observation_space, action_space = _spaces()

        def reset(self, seed=None, options=None):
            return np.full(4, 0.5, np.float32), {}

        def step(self, a):
            r = -float(np.sum((np.asarray(a) - 0.3) ** 2))
            return np.full(4, 0.5, np.float32), r, False, True, {}

    env = wrap_env(gym.vector.SyncVectorEnv([Bandit, Bandit]))
    torch.manual_seed(0)
    pol = _actor(clip_actions=False, max_log_std=0.0)
    obs, act = _spaces()
    val = ValueCritic(encoder_cls=_TinyEncoder, encoder_kwargs={}, observation_space=obs, action_space=act, device="cpu",
                      cfg=ModelCfg(hidden_sizes=[8]))
    cfg = dict(rollouts=16, mini_batches=2, learning_epochs=4, learning_rate=1e-2, entropy_loss_scale=0.0,
               experiment={"directory": str(tmp_path), "write_interval": 0, "checkpoint_interval": 0})
    agent = PPO(models={"policy": pol, "value": val}, memory=RandomMemory(memory_size=16, num_envs=2, device="cpu"),
                cfg=cfg, observation_space=env.observation_space, action_space=env.action_space, device="cpu")
    if train:
        SequentialTrainer(env=env, agents=agent, cfg={"timesteps": 160, "headless": True, "disable_progressbar": True}).train()
    return agent, pol


def test_on_real_ppo_updates_leave_log_std_exactly_zero(monkeypatch, tmp_path):
    agent, pol = _ppo(monkeypatch, tmp_path, on=True)
    in_opt = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
    assert id(pol.log_std) not in in_opt
    assert torch.equal(pol.log_std, torch.zeros(2))                   # 10 updates later: still exactly 0
    assert pol.mean_layer.weight.grad is not None                     # the rest of the policy did train


def test_off_control_real_ppo_moves_log_std(monkeypatch, tmp_path):
    """The same run with the knob off: log_std is optimised and leaves 0 (downward under this reward; the clamp at
    max 0 lets it go below). This is the drift the knob exists to stop."""
    agent, pol = _ppo(monkeypatch, tmp_path, on=False)
    in_opt = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
    assert id(pol.log_std) in in_opt
    assert float(pol.log_std.detach().min()) < 0.0


@pytest.mark.parametrize("on", [True, False], ids=["on", "off"])
def test_production_path_brain_build_agents_ppo(monkeypatch, tmp_path, on):
    """The launcher's path: Brain -> build_agents (NETT optimizer patches) -> skrl PPO -> SequentialTrainer over brain.py's
    stack (test_first_full_stack's fake env), one rollout + update. On: log_std absent from the optimizer and exactly 0
    after; off (control): in the optimizer and moved."""
    import test_first_full_stack as tffs
    from skrl.trainers.torch import SequentialTrainer

    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain

    monkeypatch.chdir(tmp_path)                                       # skrl writes runs/ under the cwd
    for k in tffs.KNOBS:
        monkeypatch.delenv(k, raising=False)
    if on:
        monkeypatch.setenv(ENV, "1")
    else:
        monkeypatch.delenv(ENV, raising=False)
    torch.manual_seed(0)
    _, _, _, env = tffs._stack("CNN2F", {}, monkeypatch, episode_steps=5)
    brain = Brain(encoder="nature_cnn", algorithm="PPO", encoder_cfg={"features_dim": 32},
                  algorithm_cfg={"rollouts": 32, "mini_batches": 2, "learning_epochs": 2, "learning_rate": 1e-2},
                  model={"shared_encoder": True, "hidden_sizes": [16]})
    brain.num_brains = 1
    agent = build_agents(brain, env, torch.device("cpu"))[0]
    pol = agent.models["policy"]
    in_opt = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
    before = {n: p.detach().clone() for n, p in pol.named_parameters()}
    SequentialTrainer(env=env, agents=agent, cfg={"timesteps": 16, "headless": True, "disable_progressbar": True}).train()
    assert not torch.equal(pol.mean_layer.weight.detach(), before["mean_layer.weight"])   # an update happened
    if on:
        assert "log_std" not in before and id(pol.log_std) not in in_opt
        assert torch.equal(pol.log_std, torch.zeros(2)) and torch.equal(pol.state_dict()["log_std"], torch.zeros(2))
    else:
        assert id(pol.log_std) in in_opt and not torch.equal(pol.log_std.detach(), before["log_std"])


# ── checkpoints ────────────────────────────────────────────────────────────────────────────────────

def test_on_checkpoint_roundtrip_and_old_tools_key(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "1")
    a = _actor()
    p = tmp_path / "agent.pt"
    torch.save({"policy": a.state_dict()}, p)
    d = torch.load(p, map_location="cpu", weights_only=False)
    assert torch.equal(d["policy"]["log_std"], torch.zeros(2))       # sigma_check / probe_policy_readout read this
    b = _actor()
    b.load_state_dict(d["policy"])
    assert torch.equal(b.log_std, torch.zeros(2))
    monkeypatch.delenv(ENV, raising=False)
    c = _actor()                                                      # knob-on checkpoint -> knob-off code loads
    c.load_state_dict(d["policy"])
    assert torch.equal(c.log_std.detach(), torch.zeros(2))


@pytest.mark.parametrize("bad", [[0.0, -0.0004], [0.0041, 0.0042]])
def test_on_refuses_a_checkpoint_with_nonzero_log_std(monkeypatch, bad):
    """Both measured shapes refuse: the h1 smoke's [0, -.0004] (std < 1) and B1's [+.004, +.004] (std 1 only through the
    max_log_std clamp; a std-1-by-construction policy must not silently adopt it)."""
    monkeypatch.delenv(ENV, raising=False)
    sd = _actor().state_dict()
    sd["log_std"] = torch.tensor(bad)
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(RuntimeError, match="not exactly 0"):
        _actor().load_state_dict(sd)


def _seed_checkpoint(agent, run, log_std):
    """skrl's own Agent.save, then policy/log_std overwritten: the file a CONTINUE_FROM row seeds into brain_1."""
    from nett_skrl.recording.checkpoints import brain_checkpoint_dir

    d = brain_checkpoint_dir(run, 1)
    d.mkdir(parents=True)
    agent.save(str(d / "final_agent.pt"))
    blob = torch.load(d / "final_agent.pt", map_location="cpu", weights_only=False)
    blob["policy"]["log_std"] = torch.tensor(log_std)
    torch.save(blob, d / "final_agent.pt")


@pytest.mark.parametrize("context", ["train", "test"])
def test_on_refusal_survives_the_agent_load_path(monkeypatch, tmp_path, context):
    """skrl Agent.load via load_latest_checkpoints, the path of a CONTINUE_FROM resume (context train) and of every test.
    Its train path logs a failed load and carries on from fresh weights; the sigma refusal must raise there too."""
    from types import SimpleNamespace

    from nett_skrl.brain.models.utils.action_std import FixedActionStdCheckpointError
    from nett_skrl.recording.checkpoints import load_latest_checkpoints

    src, _ = _ppo(monkeypatch, tmp_path, on=False, train=False)
    run = tmp_path / "run"
    _seed_checkpoint(src, run, [0.0, -0.0004])
    agent, pol = _ppo(monkeypatch, tmp_path, on=True, train=False)
    with pytest.raises(FixedActionStdCheckpointError, match="not exactly 0"):
        load_latest_checkpoints([agent], SimpleNamespace(path=run, dry_run=False), context=context)
    assert torch.equal(pol.log_std, torch.zeros(2))


@pytest.mark.parametrize("context", ["train", "test"])
def test_agent_load_controls(monkeypatch, tmp_path, context):
    """Controls for the test above, same knob on both sides: on + a zero log_std loads whole; off + the h1 log_std loads
    it. (The checkpoint's optimizer state must match, so the source agent is built under the same knob.)"""
    from types import SimpleNamespace

    from nett_skrl.recording.checkpoints import load_latest_checkpoints

    for i, (on, ls) in enumerate([(True, [0.0, 0.0]), (False, [0.0, -0.0004])]):
        src, _ = _ppo(monkeypatch, tmp_path, on=on, train=False)
        run = tmp_path / f"run{i}"
        _seed_checkpoint(src, run, ls)
        agent, pol = _ppo(monkeypatch, tmp_path, on=on, train=False)
        load_latest_checkpoints([agent], SimpleNamespace(path=run, dry_run=False), context=context)
        assert torch.equal(pol.log_std.detach(), torch.tensor(ls))


def test_test_of_a_knob_on_run_without_the_knob_refuses(monkeypatch, tmp_path):
    """A knob-on run retested with the knob forgotten: the policy fits (zeros into the Parameter) but the saved Adam state
    has one tensor fewer, so skrl's load raises and TEST refuses (MissingCheckpointError). Loud, not a silent std drift."""
    from types import SimpleNamespace

    from nett_skrl.recording.checkpoints import MissingCheckpointError, load_latest_checkpoints

    src, _ = _ppo(monkeypatch, tmp_path, on=True, train=False)
    _seed_checkpoint(src, tmp_path / "run", [0.0, 0.0])
    agent, _ = _ppo(monkeypatch, tmp_path, on=False, train=False)
    with pytest.raises(MissingCheckpointError, match="parameter group"):
        load_latest_checkpoints([agent], SimpleNamespace(path=tmp_path / "run", dry_run=False), context="test")


def test_other_train_load_failures_keep_their_old_handling(monkeypatch, tmp_path):
    """The re-raise is specific to the sigma refusal: a different load failure in a train resume is still logged and
    skipped, as before (gate_a_resume.py's preflight owns that case)."""
    from types import SimpleNamespace

    from nett_skrl.recording.checkpoints import brain_checkpoint_dir, load_latest_checkpoints

    agent, _ = _ppo(monkeypatch, tmp_path, on=True, train=False)
    d = brain_checkpoint_dir(tmp_path / "run", 1)
    d.mkdir(parents=True)
    torch.save({"policy": {"bogus": torch.zeros(1)}}, d / "final_agent.pt")      # strict load -> RuntimeError
    load_latest_checkpoints([agent], SimpleNamespace(path=tmp_path / "run", dry_run=False), context="train")


def test_on_refuses_multivariate_nonzero_offdiag(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    sd = _actor(MultivariateGaussianActor).state_dict()
    sd["tril_offdiag"] = torch.tensor([0.1])
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(RuntimeError, match="tril_offdiag"):
        _actor(MultivariateGaussianActor).load_state_dict(sd)


def test_on_buffer_follows_the_module_device_and_dtype(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    m = _actor().double()
    assert m.log_std.dtype == torch.float64 and torch.equal(m.log_std, torch.zeros(2, dtype=torch.float64))


# ── the mirror-equivariant path ────────────────────────────────────────────────────────────────────

def test_on_with_mirror_equiv(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(MIRROR_ENV, "1")
    m = _actor(img=True, clip_actions=False)
    x = torch.rand(6, 24)
    mean, out = m.compute({"observations": x, "states": x}, role="policy")
    assert torch.equal(out["log_std"], torch.zeros_like(mean))
    xf = x.view(6, 1, 4, 6).flip(-1).reshape(6, 24)                   # mirror equivariance is intact
    mean_f, _ = m.compute({"observations": xf, "states": xf}, role="policy")
    assert torch.allclose(mean_f[:, 0], -mean[:, 0], atol=1e-6) and torch.allclose(mean_f[:, 1], mean[:, 1], atol=1e-6)
    with torch.no_grad():
        xs = torch.rand(1, 24).expand(20000, 24)
        acts, o = m.act({"observations": xs, "states": xs}, role="policy")
    assert torch.allclose((acts - o["mean_actions"]).std(0), torch.ones(2), atol=0.02)
