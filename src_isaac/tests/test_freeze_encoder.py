"""NETT_FREEZE_ENCODER=1 (BL2 frozen-encoder arms, owner 2026-11-03): each model's freshly initialised encoder is
frozen (random features) and only the heads train. Checked on the four BASELINE backbones at 128x80: after an Adam
step on a loss through encoder and head, every encoder tensor is byte-identical and the head moved."""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.brain.agent_factory import freeze_encoders_at_init  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402

H, W = 80, 128


class _M(nn.Module):
    def __init__(self, label, channels):
        super().__init__()
        spec = ct.MODELS[label]
        cfg = {k: v for k, v in spec["cfg"].items() if k != "trainable"}
        self.encoder = encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, H, W), np.uint8), **cfg)
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        return self.head(self.encoder(x))


@pytest.mark.parametrize("label,channels", [("CNN-D512P", 3), ("ViT-P8-666K", 3), ("3DCNN", 6), ("ViViT-P8-T2", 6)])
def test_freeze_at_init_keeps_the_encoder_and_trains_the_head(label, channels):
    torch.manual_seed(0)
    models = {"policy": _M(label, channels), "value": _M(label, channels)}
    enc0 = {r: {k: v.clone() for k, v in m.encoder.state_dict().items()} for r, m in models.items()}
    head0 = {r: m.head.weight.detach().clone() for r, m in models.items()}
    got = freeze_encoders_at_init(models)
    assert set(got) == {"policy", "value"} and all(n > 0 for n in got.values())
    params = [p for m in models.values() for p in m.parameters()]   # skrl's optimizer sees every parameter
    opt = torch.optim.Adam(params, lr=1e-2)
    x = torch.rand(2, channels, H, W) * 255
    loss = sum(m(x).pow(2).sum() for m in models.values())
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(params, 0.5)
    opt.step()
    for r, m in models.items():
        assert not any(p.requires_grad for p in m.encoder.parameters())
        assert all(p.grad is None for p in m.encoder.parameters())
        for k, v in m.encoder.state_dict().items():
            assert torch.equal(v, enc0[r][k]), f"{label} {r} encoder.{k} changed"
        assert all(p.requires_grad for p in m.head.parameters())
        assert not torch.equal(m.head.weight, head0[r])


def test_freeze_at_init_does_not_change_the_weights():
    torch.manual_seed(3); a = _M("CNN-D512P", 3)
    torch.manual_seed(3); b = _M("CNN-D512P", 3)
    freeze_encoders_at_init({"policy": b})
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])


def test_freeze_at_init_skips_models_without_an_encoder():
    assert freeze_encoders_at_init({"policy": nn.Linear(2, 2)}) == {}


def test_factory_reads_the_knob():
    src = (_SRC / "nett_skrl/brain/agent_factory.py").read_text()
    assert 'os.environ.get("NETT_FREEZE_ENCODER", "")' in src
    assert "freeze_encoders_at_init(models)" in src
    assert "are exclusive" in src   # never combined with NETT_FROZEN_ENC_FROM
