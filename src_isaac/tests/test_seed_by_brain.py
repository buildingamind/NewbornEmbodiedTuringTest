"""NETT_SEED_BY_BRAIN=1 (workspace DECISIONS §187 add. 10, owner: "Seeding should always be a result of brain number or
test episode"). CPU, single process, no Kit.

Flag off: TaskConfig.seed is the condition hash, stream_seed == config.seed, weight_init_base == torch.initial_seed()
(the pre-patch behaviour). Flag on: the base seed is the constant 2014423175 for every condition, the single-brain
process stream is constant + brain offset, and the weight init that build_agents runs (manual_seed(weight_init_base +
global brain id), then build_models_for_algorithm) depends on the brain number only -- the same for two conditions at one
offset, different across offsets, and unaffected by how much global RNG the env load consumed before it.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import nett_skrl.runtime.task as T
from nett_skrl.brain.encoders import NatureCNN
from nett_skrl.brain.models import ModelCfg, build_models_for_algorithm

CONST = 2014423175  # (int(sha256(b"nett")[:8], "big") * 7919) % (2**31 - 1)
_SRC = Path(__file__).resolve().parents[1]


def _cfg(cond, off=0, nb=1):
    return T.TaskConfig(cond, "/tmp/nett_seed_by_brain_unused", ["train"], brain_id_offset=off, num_brains=nb, num_envs=nb)


def _weights(cfg, local_id=0, burn=0):
    """build_agents' weight-init sequence (agent_factory.py) after an env load that consumed `burn` draws."""
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)  # NETTEnv.seed(cfg.seed)
    torch.rand(burn + 1); np.random.rand(burn + 1)
    if T.seed_by_brain():
        T.reseed_streams(T.stream_seed(cfg))  # environment.py, flag on
    base = T.weight_init_base(cfg)
    torch.manual_seed(base + int(cfg.brain_id_offset or 0) + local_id)
    m = build_models_for_algorithm(
        SimpleNamespace(actor_type="gaussian", critic_type="value", model_keys=("policy", "value")),
        encoder_cls=NatureCNN, encoder_kwargs=dict(features_dim=128, conv_dim=64, spatial_pool=False),
        observation_space=gym.spaces.Box(0, 255, (3, 80, 128), np.uint8),
        action_space=gym.spaces.Box(-1.0, 1.0, (2,), np.float32), device="cpu",
        cfg=ModelCfg(hidden_sizes=[], shared_encoder=True, clip_actions=False, value_bound=None, max_log_std=0.0))
    h = hashlib.sha256()
    for key in ("policy", "value"):
        for name, t in sorted(m[key].state_dict().items()):
            h.update(name.encode()); h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture
def flag(monkeypatch):
    def set_(v):
        if v is None:
            monkeypatch.delenv("NETT_SEED_BY_BRAIN", raising=False)
        else:
            monkeypatch.setenv("NETT_SEED_BY_BRAIN", v)
        monkeypatch.setattr(T, "_MULTI_BRAIN_NOTE_DONE", False)
    return set_


@pytest.mark.parametrize("v", [None, "0", ""])
def test_flag_off_keeps_the_condition_seed(flag, v):
    flag(v)
    a, b = _cfg("fork-2", 4), _cfg("ship-2", 4)
    want = (int.from_bytes(hashlib.sha256(b"fork-2").digest()[:8], "big") * 7919) % (2 ** 31 - 1)
    assert a.seed == want and a.seed != b.seed
    assert T.stream_seed(a) == a.seed
    torch.manual_seed(1234)
    assert T.weight_init_base(a) == torch.initial_seed() == 1234
    assert _weights(a) != _weights(b)  # positive control: the condition reaches the init when the flag is off


def test_flag_on_seed_is_condition_free_and_stream_is_brain_keyed(flag, capsys):
    flag("1")
    for cond in ("fork-2", "ship-2", "fork-1", "binding-Object1"):
        c = _cfg(cond, 5)
        assert c.seed == CONST and T.stream_seed(c) == CONST + 5 and T.weight_init_base(c) == CONST
    assert "[NETT seed]" not in capsys.readouterr().out
    m = _cfg("fork-2", 4, nb=2)
    assert T.stream_seed(m) == CONST and T.stream_seed(m) == CONST  # second read: note printed once only
    assert capsys.readouterr().out.count("[NETT seed] NETT_SEED_BY_BRAIN=1 with num_brains=2") == 1


def test_flag_on_init_is_a_function_of_brain_number_only(flag):
    flag("1")
    w = {(cond, off): _weights(_cfg(cond, off), burn=burn)
         for cond, burn in (("fork-2", 0), ("ship-2", 4096), ("fork-1", 17)) for off in (4, 5)}
    assert w[("fork-2", 4)] == w[("ship-2", 4)] == w[("fork-1", 4)]
    assert w[("fork-2", 5)] == w[("ship-2", 5)] == w[("fork-1", 5)]
    assert w[("fork-2", 4)] != w[("fork-2", 5)]
    # multi-brain brain b == single-brain offset b (global brain id)
    assert _weights(_cfg("ship-2", 4, nb=2), local_id=1) == w[("fork-2", 5)]


def test_build_agents_uses_weight_init_base_then_global_brain_seed():
    src = (_SRC / "nett_skrl" / "brain" / "agent_factory.py").read_text()
    i_base = src.index("base_seed = weight_init_base(config)")
    i_seed = src.index("torch.manual_seed(base_seed + global_brain_id)")
    i_build = src.index("models = build_models_for_algorithm(", i_seed)
    assert i_base < i_seed < i_build
    env_src = (_SRC / "nett_skrl" / "environment" / "environment.py").read_text()
    assert "reseed_streams(stream_seed(config))" in env_src
