"""The keep-mask rule must change with the slot count, or a test object gets deleted.

The parsing test shows TWO objects on two monitors plus the chamber. With K>2 a
rule that keeps ONE slot can suppress one of the two alternatives outright, and the
arm would score at chance for a reason unrelated to the hypothesis. These tests pin
the rule that prevents it. They fail against the pre-change code, which had no
``_keep_mask`` at all and multiplied by a single chosen slot unconditionally.
"""

import os

import gymnasium as gym
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("NETT_SEG_"):
            monkeypatch.delenv(name, raising=False)


class FakeEnv(gym.Env):
    def __init__(self, channels=6):
        self.observation_space = gym.spaces.Box(0, 255, (80, 128, channels), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, np.uint8), 0.0, False, False, {}


def _build(queries, monkeypatch, rule=None):
    monkeypatch.setenv("NETT_SEG_QUERIES", str(queries))
    if rule is not None:
        monkeypatch.setenv("NETT_SEG_MASK_RULE", rule)
    from nett_skrl.body.wrappers.gwm_seg import GwmSeg
    w = GwmSeg(FakeEnv())
    w._ensure(3)
    return w


def _masks(w, batch=4):
    x = torch.rand(batch, 3, 80, 128, device=w.device)
    with torch.no_grad():
        return w._model.get_masks(x)


@pytest.mark.parametrize("queries,expect_not_bg", [(2, False), (3, True), (5, True)])
def test_auto_rule_follows_slot_count(queries, expect_not_bg, monkeypatch):
    w = _build(queries, monkeypatch)
    w._keep_mask(_masks(w))
    assert bool(w.last_stats["seg/mask_rule_not_background"]) is expect_not_bg


@pytest.mark.parametrize("queries", [3, 5])
def test_multi_slot_keeps_every_non_background_slot(queries, monkeypatch):
    """⛔ THE FATAL CASE. Keeping one slot at K>2 can delete a test alternative."""
    w = _build(queries, monkeypatch)
    m = _masks(w)
    keep = w._keep_mask(m)
    bg = int(w.last_stats["seg/selected_slot"])
    others = sum(m[:, j] for j in range(queries) if j != bg)
    torch.testing.assert_close(keep[:, 0], others, rtol=1e-4, atol=1e-5)
    # and it must retain strictly more than the largest single non-background slot
    biggest_single = max(
        float(m[:, j].mean()) for j in range(queries) if j != bg
    )
    assert float(keep.mean()) > biggest_single


@pytest.mark.parametrize("queries", [3, 5])
def test_background_is_the_largest_slot(queries, monkeypatch):
    w = _build(queries, monkeypatch)
    m = _masks(w)
    w._keep_mask(m)
    bg = int(w.last_stats["seg/selected_slot"])
    assert bg == int(torch.argmax(m.mean(dim=(0, 2, 3))).item())


def test_two_slots_still_keeps_the_foreground_not_the_complement(monkeypatch):
    """K=2 must be byte-for-byte the reference behaviour, or the baseline moves."""
    w = _build(2, monkeypatch)
    m = _masks(w)
    keep = w._keep_mask(m)
    slot = int(w.last_stats["seg/selected_slot"])
    torch.testing.assert_close(keep, m[:, slot:slot + 1], rtol=0, atol=0)
    assert float(m[:, slot].mean()) <= float(m[:, 1 - slot].mean())


def test_explicit_rule_overrides_and_garbage_raises(monkeypatch):
    w = _build(5, monkeypatch, rule="foreground")
    w._keep_mask(_masks(w))
    assert not bool(w.last_stats["seg/mask_rule_not_background"])
    monkeypatch.setenv("NETT_SEG_MASK_RULE", "whatever")
    from nett_skrl.body.wrappers.gwm_seg import GwmSeg
    with pytest.raises(ValueError, match="NETT_SEG_MASK_RULE"):
        GwmSeg(FakeEnv())


@pytest.mark.parametrize("queries", [2, 3, 5])
def test_dilution_diagnostics_are_recorded(queries, monkeypatch):
    """Dilution is invisible in the loss; it must be visible in the logs."""
    w = _build(queries, monkeypatch)
    stats = w.slot_diagnostics(_masks(w))
    assert set(stats) == {
        "seg/confident_pixels", "seg/slot_pair_cosine_max",
        "seg/slot_occ_min", "seg/slot_occ_max",
    }
    assert 0.0 <= stats["seg/confident_pixels"] <= 1.0
    assert -1.0 <= stats["seg/slot_pair_cosine_max"] <= 1.0 + 1e-6
    assert stats["seg/slot_occ_min"] <= stats["seg/slot_occ_max"]
