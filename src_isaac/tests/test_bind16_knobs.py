"""The three bind16 knobs ported from the presenter's scratch trees (workspace PROGRAM_BIND16, owner 2026-10-27).

1. MODELS["ViT-P8CONV-682K"]: CompactViT, patch 8, embed 140, conv stem, no aux; 681,612 encoder params at 128x80.
2. NETT_SHARED_ENCODER_DUP: unset / 0 = the deduplicated default (unchanged); 1 = skrl's raw chain (repeats kept).
3. NETT_FROZEN_ENC_FROM (+ NETT_FROZEN_ENC_SHA256): encoder.* loaded strictly per role from a checkpoint, frozen.
"""
import hashlib
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
from nett_skrl.brain.agent_factory import load_frozen_encoders  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402
from nett_skrl.brain.skrl_patches import shared_encoder_dup, unique_parameters  # noqa: E402


# ------------------------------------------------------------------ 1. the label
def test_vit_p8conv_label_is_the_bind16_encoder():
    spec = ct.MODELS["ViT-P8CONV-682K"]
    assert spec["encoder"] == "compact_vit" and spec["framestack"] is False and "aux" not in spec
    assert spec["cfg"] == {**ct.VIT_CFG, "patch_size": 8, "embed_dim": 140, "stem": "conv"}
    cfg = {k: v for k, v in spec["cfg"].items() if k != "trainable"}
    enc = encoder_mapping["compact_vit"](gym.spaces.Box(0, 255, (3, 80, 128), np.uint8), **cfg)
    assert sum(p.numel() for p in enc.parameters()) == 681_612


# ------------------------------------------------------------------ 2. DUP
@pytest.mark.parametrize("v,want", [(None, False), ("", False), ("0", False), ("off", False), ("1", True), ("true", True)])
def test_shared_encoder_dup_parse(monkeypatch, v, want):
    if v is None:
        monkeypatch.delenv("NETT_SHARED_ENCODER_DUP", raising=False)
    else:
        monkeypatch.setenv("NETT_SHARED_ENCODER_DUP", v)
    assert shared_encoder_dup() is want


def test_shared_encoder_dup_refuses_garbage(monkeypatch):
    monkeypatch.setenv("NETT_SHARED_ENCODER_DUP", "maybe")
    with pytest.raises(ValueError):
        shared_encoder_dup()


def test_unique_parameters_default_dedupes_and_dup_keeps_repeats(monkeypatch):
    shared, a, b = nn.Linear(2, 2), nn.Linear(2, 1), nn.Linear(2, 1)
    chain = list(shared.parameters()) + list(a.parameters()) + list(shared.parameters()) + list(b.parameters())
    monkeypatch.delenv("NETT_SHARED_ENCODER_DUP", raising=False)
    assert len(unique_parameters(iter(chain))) == 6
    monkeypatch.setenv("NETT_SHARED_ENCODER_DUP", "1")
    assert len(unique_parameters(iter(chain))) == 8


# ------------------------------------------------------------------ 3. FROZEN-ENC
class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        self.head = nn.Linear(2, 1)


def _ckpt(tmp_path, src):
    p = tmp_path / "enc.pt"
    sd = {f"encoder.{k}": v.clone() for k, v in src.encoder.state_dict().items()}
    torch.save({"policy": sd, "value": dict(sd)}, p)
    return p


def test_frozen_encoders_load_strictly_and_freeze_only_the_encoder(tmp_path):
    torch.manual_seed(0); src = _M()
    torch.manual_seed(1); models = {"policy": _M(), "value": _M()}
    p = _ckpt(tmp_path, src)
    got = load_frozen_encoders(models, str(p), hashlib.sha256(p.read_bytes()).hexdigest())
    assert got == {"policy": 4, "value": 4}
    for m in models.values():
        for k, v in m.encoder.state_dict().items():
            assert torch.equal(v, src.encoder.state_dict()[k])
        assert not any(q.requires_grad for q in m.encoder.parameters())
        assert all(q.requires_grad for q in m.head.parameters())


def test_frozen_encoders_refuse_a_wrong_sha_and_a_mismatched_encoder(tmp_path):
    p = _ckpt(tmp_path, _M())
    with pytest.raises(RuntimeError, match="sha256"):
        load_frozen_encoders({"policy": _M()}, str(p), "0" * 64)
    other = _M(); other.encoder = nn.Linear(4, 2)
    with pytest.raises(RuntimeError):
        load_frozen_encoders({"policy": other}, str(p))
