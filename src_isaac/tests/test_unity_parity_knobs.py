"""Unity/SB3-parity knobs for the rA10 replication arms (FINDINGS 4dh.44a, replica audit).

Each knob is unset = unchanged; these tests pin both the default and the parity setting.
"""

from __future__ import annotations

import types

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.models import GaussianActor, ModelCfg
from nett_skrl.brain.ppo_metrics import (
    MetricsPPO,
    _peb_compute_gae,
    apply_diag_schedules,
    minibatch_advantage_norm,
)


class _TinyEncoder(nn.Module):
    def __init__(self, observation_space, **kw):
        super().__init__()
        self.features_dim = 8
        self.net = nn.Linear(int(np.prod(observation_space.shape)), 8)

    def forward(self, x):
        return self.net(x.float().flatten(1))


def _actor(cfg):
    obs = gym.spaces.Box(low=0, high=255, shape=(4,), dtype=np.float32)
    act = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return GaussianActor(encoder_cls=_TinyEncoder, encoder_kwargs={}, observation_space=obs,
                         action_space=act, device="cpu", cfg=cfg)


def _std_at(cfg, log_std=5.0):
    m = _actor(cfg)
    with torch.no_grad():
        m.log_std.fill_(log_std)
    m.act({"observations": torch.rand(3, 4), "states": torch.rand(3, 4)}, role="policy")
    return float(m.distribution(role="policy").stddev.mean())


# ── log-std clamp ────────────────────────────────────────────────────────────
def test_default_log_std_is_clamped_at_skrl_two():
    assert _std_at(ModelCfg()) == pytest.approx(np.exp(2.0), rel=1e-5)


def test_clip_log_std_false_removes_the_clamp():
    assert _std_at(ModelCfg(clip_log_std=False)) == pytest.approx(np.exp(5.0), rel=1e-5)


def test_max_log_std_float_still_caps():
    assert _std_at(ModelCfg(max_log_std=1.0)) == pytest.approx(np.exp(1.0), rel=1e-5)


# ── time-limit handling (NETT_DIAG_PEB=T) ────────────────────────────────────
def _gae_inputs():
    # 4 steps x 1 env, truncation at step 1 (episode end), values all 10.
    r = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    z = torch.zeros(4, 1, dtype=torch.bool)
    tr = z.clone(); tr[1] = True
    v = torch.full((4, 1), 10.0)
    return dict(rewards=r, terminated=z, truncated=tr, values=v, last_values=torch.full((1,), 10.0),
                discount_factor=0.9, lambda_coefficient=1.0, time_limit_bootstrap=False)


def test_peb_t_return_at_the_limit_is_the_reward_only(monkeypatch):
    monkeypatch.setenv("NETT_DIAG_PEB", "T")
    returns, _ = _peb_compute_gae(**_gae_inputs())
    assert float(returns[1]) == pytest.approx(2.0)                 # r_T, V = 0 after the limit
    assert float(returns[0]) == pytest.approx(1.0 + 0.9 * 2.0)     # no leak across the limit


def test_default_bootstraps_across_the_limit(monkeypatch):
    monkeypatch.delenv("NETT_DIAG_PEB", raising=False)
    returns, _ = _peb_compute_gae(**_gae_inputs())
    assert float(returns[1]) != pytest.approx(2.0)                 # stock skrl: r_T + gamma*...


def test_peb_t_refuses_beside_skrl_time_limit_bootstrap(monkeypatch):
    monkeypatch.setenv("NETT_DIAG_PEB", "T")
    kw = _gae_inputs(); kw["time_limit_bootstrap"] = True
    with pytest.raises(ValueError, match="gamma\\*V"):
        _peb_compute_gae(**kw)


def test_peb_unknown_mode_refuses(monkeypatch):
    monkeypatch.setenv("NETT_DIAG_PEB", "terminal")
    with pytest.raises(ValueError, match="expected off"):
        _peb_compute_gae(**_gae_inputs())


# ── Adam eps ─────────────────────────────────────────────────────────────────
def _sched_agent():
    return types.SimpleNamespace(cfg=types.SimpleNamespace(entropy_loss_scale=0.0),
                                 optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3, "eps": 1e-8}]),
                                 scheduler=None)


def test_adam_eps_default_unchanged(monkeypatch):
    for k in ("NETT_ADAM_EPS", "NETT_DIAG_ENT_START", "NETT_DIAG_LR_ANNEAL"):
        monkeypatch.delenv(k, raising=False)
    a = _sched_agent(); apply_diag_schedules(a, timestep=0, timesteps=10)
    assert a.optimizer.param_groups[0]["eps"] == 1e-8


def test_adam_eps_set(monkeypatch):
    monkeypatch.setenv("NETT_ADAM_EPS", "1e-5")
    a = _sched_agent(); apply_diag_schedules(a, timestep=0, timesteps=10)
    assert a.optimizer.param_groups[0]["eps"] == 1e-5


def test_adam_reads_eps_from_the_group():
    # the mechanism the knob relies on: a changed group eps changes the step
    p1 = nn.Parameter(torch.ones(3)); p2 = nn.Parameter(torch.ones(3))
    o1 = torch.optim.Adam([p1], lr=0.1); o2 = torch.optim.Adam([p2], lr=0.1)
    o2.param_groups[0]["eps"] = 10.0
    for p, o in ((p1, o1), (p2, o2)):
        p.grad = torch.full((3,), 1e-3); o.step()
    assert not torch.allclose(p1, p2)


# ── per-minibatch advantage standardisation ──────────────────────────────────
class _Mem:
    def __init__(self, adv):
        self.adv = adv

    def sample(self, names, batch_size, mini_batches=1, sequence_length=1):
        n = len(self.adv) // mini_batches
        return [[torch.zeros(n, 1), self.adv[i * n:(i + 1) * n].clone()] for i in range(mini_batches)]


def _adv_agent():
    adv = torch.tensor([[0.0], [1.0], [2.0], [3.0], [10.0], [20.0], [30.0], [40.0]])
    return types.SimpleNamespace(_tensors_names=["observations", "advantages"], memory=_Mem(adv))


def test_adv_norm_default_is_no_override(monkeypatch):
    monkeypatch.delenv("NETT_ADV_NORM", raising=False)
    a = _adv_agent()
    assert minibatch_advantage_norm(a) is None and "sample" not in vars(a.memory)


def test_adv_norm_minibatch_standardises_each_batch(monkeypatch):
    monkeypatch.setenv("NETT_ADV_NORM", "minibatch")
    a = _adv_agent()
    assert minibatch_advantage_norm(a) is True
    for _, adv in a.memory.sample(names=None, batch_size=8, mini_batches=2):
        assert float(adv.mean()) == pytest.approx(0.0, abs=1e-6)
        assert float(adv.std()) == pytest.approx(1.0, abs=1e-5)
    del a.memory.sample
    raw = a.memory.sample(names=None, batch_size=8, mini_batches=2)
    assert float(raw[1][1].mean()) == pytest.approx(25.0)             # restored


def test_adv_norm_unknown_refuses(monkeypatch):
    monkeypatch.setenv("NETT_ADV_NORM", "batch")
    with pytest.raises(ValueError, match="expected 'rollout'"):
        minibatch_advantage_norm(_adv_agent())


def test_metrics_ppo_update_installs_and_removes_the_override():
    import inspect
    src = inspect.getsource(MetricsPPO.update)
    assert src.index("minibatch_advantage_norm(self)") < src.index("super().update")
    assert "del self.memory.sample" in src and "finally:" in src


# ── env parsing: log-std model cfg, value standardisation ────────────────────
def _campaign(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train
    return campaign_train


@pytest.mark.parametrize("val,want", [(None, {}), ("none", {"clip_log_std": False}), ("1.5", {"max_log_std": 1.5})])
def test_log_std_model_cfg_from_env(monkeypatch, val, want):
    ct = _campaign(monkeypatch)
    if val is None:
        monkeypatch.delenv("NETT_MAX_LOG_STD", raising=False)
    else:
        monkeypatch.setenv("NETT_MAX_LOG_STD", val)
    got = ct.log_std_model_cfg_from_env()
    assert got == want
    ModelCfg(**got)                                                    # keys are real ModelCfg fields
    # ⛔ AND legal in schema.json: c5e60d9 passed the dataclass check and every NETT_MAX_LOG_STD=none
    # launch then died at config validation ('clip_log_std' was unexpected, chicken 2026-09-24).
    import jsonschema
    from nett_skrl.nett import _load_schema
    model_schema = _load_schema()["properties"]["brain"]["properties"]["model"]
    jsonschema.validate({**got, "clip_actions": False}, model_schema)


@pytest.mark.parametrize("val,want", [(None, True), ("on", True), ("off", False)])
def test_value_std_enabled(monkeypatch, val, want):
    from nett_skrl.brain.agent_factory import value_std_enabled
    if val is None:
        monkeypatch.delenv("NETT_VALUE_STD", raising=False)
    else:
        monkeypatch.setenv("NETT_VALUE_STD", val)
    assert value_std_enabled() is want


def test_value_std_unknown_refuses(monkeypatch):
    from nett_skrl.brain.agent_factory import value_std_enabled
    monkeypatch.setenv("NETT_VALUE_STD", "0")
    with pytest.raises(ValueError, match="expected 'on'"):
        value_std_enabled()
