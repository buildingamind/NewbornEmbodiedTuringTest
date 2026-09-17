"""The ONE shared EMA teacher: frozen, stepped once per compute, and out of every param set.

⛔ THE THREE FAILURES THIS PINS ARE ALL SILENT.
1. A teacher registered as a submodule puts a second copy of the trunk into `head.parameters()`
   -- which is what AuxLossPPO hands the optimizer -- so the "frozen" target would be trained.
2. A lazily-built teacher raises INSIDE update 1 on a GPU arm, because the encoder then carries
   the shared feature cache and its tensor is non-leaf (deepcopy refuses). CPU tests never see
   it, so the cache is populated here on purpose.
3. A teacher stepped once per TERM instead of once per compute decays twice per minibatch.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.aux.ema_teacher import EMATeacher
from nett_skrl.brain.aux.with_cltt_ref import WithCLTTRef
from nett_skrl.brain.encoders.compact_vit import CompactViT
from nett_skrl.brain.models.utils.features import (
    _CACHE_ATTR, features_forward, shared_feature_cache,
)

H, W, C = 80, 128, 6
SMALL = dict(features_dim=32, patch_size=16, embed_dim=32, depth=1, num_heads=2)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("NETT_AUX_EMA_DECAY", raising=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        yield


def _encoder():
    return CompactViT(gym.spaces.Box(0, 255, shape=(H, W, C), dtype=np.uint8), **SMALL)


def _prepared(enc, b=2):
    with torch.no_grad():
        return enc._prepare_image(torch.randint(0, 256, (b, H, W, C), dtype=torch.uint8))


def test_teacher_is_frozen_and_unregistered():
    enc = _encoder()
    holder = EMATeacher(enc)
    assert not isinstance(holder, nn.Module)
    assert all(not p.requires_grad for p in holder.module.parameters())
    assert holder.module is not enc
    # Tokens from the teacher carry no graph, whatever the ambient grad mode.
    tokens, grid = holder.tokens(_prepared(enc))
    assert not tokens.requires_grad and grid == (5, 8)


def test_the_teacher_never_reaches_the_optimizer_through_head():
    enc = _encoder()

    class _Term(nn.Module):
        needs_teacher = True
        needs_memory = False

        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 4)

        def attach_teacher(self, t):
            self.teacher_ref = t          # a plain object: nn.Module does not register it

        def compute(self, encoder, observations):
            return torch.zeros(())

    comp = WithCLTTRef(enc, _Term(), "t")
    head_ids = {id(p) for p in comp.head.parameters()}
    teacher_ids = {id(p) for p in comp._teacher.module.parameters()}
    assert head_ids and teacher_ids and not (head_ids & teacher_ids)
    assert not ({id(p) for p in comp.parameters()} & teacher_ids)


def test_ema_step_moves_the_target_toward_the_trunk_by_exactly_one_minus_decay(monkeypatch):
    monkeypatch.setenv("NETT_AUX_EMA_DECAY", "0.5")
    enc = _encoder()
    holder = EMATeacher(enc)
    before = {n: p.clone() for n, p in holder.module.named_parameters()}
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(1.0)
    holder.step(enc)
    for n, p in holder.module.named_parameters():
        assert torch.allclose(p, 0.5 * before[n] + 0.5 * (before[n] + 1.0))
    assert holder.updates == 1


def test_construction_survives_a_populated_feature_cache():
    """⛔ The GPU-arm failure: at the first update the encoder holds a NON-LEAF cached tensor and
    `copy.deepcopy` refuses it. Reproduced here by populating the cache for real."""
    enc = _encoder()

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = enc
            self.trunk = nn.Linear(enc.features_dim, 4)

    with shared_feature_cache(enc):
        x = torch.randint(0, 256, (2, H, W, C), dtype=torch.uint8)
        features_forward(_Head(), {"observations": x})
        cached = getattr(enc, _CACHE_ATTR)
        assert cached is not None and cached[1].requires_grad and cached[1].grad_fn is not None
        holder = EMATeacher(enc)                       # must not raise
        # The live cache is untouched, and the copy does not carry one.
        assert getattr(enc, _CACHE_ATTR) is cached
        assert getattr(holder.module, _CACHE_ATTR, None) is None


@pytest.mark.parametrize("raw", ["-0.1", "1.5", "none", ""])
def test_decay_knob_refuses_values_outside_the_unit_interval(monkeypatch, raw):
    monkeypatch.setenv("NETT_AUX_EMA_DECAY", raw)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EMATeacher(_encoder())
