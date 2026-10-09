"""ViT-CLTT-Ref size ladder (workspace DECISIONS §82): ViT-CLTT-Ref-250K / -500K / -1M / -2M.

Pinned here, built on CPU at the campaign eye (128x80, 2 frames x 3 channels):
  (a) each rung's ENCODER parameter count is exactly the one its MODELS comment declares, and the
      base "ViT-CLTT-Ref" is 804,320 (the positive control: the same builder reproduces the base);
  (b) the cltt_ref aux head adds the same 329,216 parameters at every rung (it is built on
      features_dim, not embed_dim), so a rung's trainable total is encoder + 329,216 + 1,541;
  (c) the rungs order strictly by size, and each differs from the base in embed_dim only (the module
      asserts this at import; re-checked here so a weakened import assert still fails a test).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128
LADDER = {"ViT-CLTT-Ref-250K": 256_056, "ViT-CLTT-Ref": 804_320, "ViT-CLTT-Ref-500K": 510_056,
          "ViT-CLTT-Ref-1M": 994_680, "ViT-CLTT-Ref-2M": 2_117_632}
AUX_HEAD = 329_216


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train

    return campaign_train


def _encoder(campaign, label):
    spec = campaign.MODELS[label]
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    ch = 3 * (campaign._FRAMESTACK_N if spec["framestack"] else 1)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (ch, H, W), np.uint8), **kw)


def _n(m):
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("label", sorted(LADDER))
def test_encoder_count_matches_declared(campaign, label):
    assert _n(_encoder(campaign, label)) == LADDER[label]


@pytest.mark.parametrize("label", sorted(LADDER))
def test_aux_head_is_constant(campaign, label):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    enc = _encoder(campaign, label)
    aux = CLTTReferenceAuxLoss(enc)
    shared = {id(p) for p in enc.parameters()}
    assert sum(p.numel() for p in aux.parameters() if id(p) not in shared) == AUX_HEAD


def test_rungs_are_one_knob_off_base(campaign):
    base = campaign.MODELS["ViT-CLTT-Ref"]
    sizes = []
    for label in ("ViT-CLTT-Ref-250K", "ViT-CLTT-Ref-500K", "ViT-CLTT-Ref-1M", "ViT-CLTT-Ref-2M"):
        r = campaign.MODELS[label]
        assert {k: v for k, v in r.items() if k != "cfg"} == {k: v for k, v in base.items() if k != "cfg"}
        diff = {k for k in set(r["cfg"]) | set(base["cfg"]) if r["cfg"].get(k) != base["cfg"].get(k)}
        assert diff == {"embed_dim"}, (label, diff)
        sizes.append(LADDER[label])
    assert sizes == sorted(sizes) and len(set(sizes)) == 4
