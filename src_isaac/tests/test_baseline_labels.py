"""BASELINE series labels (owner 2026-10-28; workspace notes/chicken/presenter_i800_i805/baseline/PLAN_BASELINE.md).

Two backbones at ~600-700K encoder parameters with 512 features in the bind16 env (128x80): CNN-D512P (NatureCNN,
pooled readout) and ViT-P8-666K (CompactViT patch 8, linear stem); their +DVS / +CLTT-Ref / +MAE variations; and the
ViViT-P8-T2 video twin (early fusion, CLS). Counts are pinned at 128x80 so an unintended change fails here.
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402

H, W = 80, 128


def _count(label, channels):
    spec = ct.MODELS[label]
    cfg = {k: v for k, v in spec["cfg"].items() if k != "trainable"}
    enc = encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, H, W), np.uint8), **cfg)
    return sum(p.numel() for p in enc.parameters())


@pytest.mark.parametrize("label,channels,want", [
    ("CNN-D512P", 3, 666_272),
    ("ViT-P8-666K", 3, 659_704),
    ("CNN-D512P+DVS", 6, 672_416),        # DVS6: NETT_DVS_COLOR=1 -> 6 event channels
    ("ViT-P8-666K+DVS", 6, 688_120),
    ("CNN-D512P+CLTT-Ref", 3, 666_272),
    ("ViT-P8-666K+CLTT-Ref", 3, 659_704),
    ("ViT-P8-666K+MAE", 3, 659_704),
    ("ViViT-P8-T2", 6, 688_120),          # 2-frame stack, early fusion
    ("3DCNN", 6, 695_981),                # the existing video twin of D512P
])
def test_baseline_encoder_counts_at_128x80(label, channels, want):
    assert _count(label, channels) == want
    assert 600_000 <= want <= 700_000


def test_baseline_label_specs():
    m = ct.MODELS
    assert m["CNN-D512P"] == dict(encoder="nature_cnn", framestack=False,
                                  cfg={"trainable": True, "features_dim": 512, "conv_dim": 64, "spatial_pool": True})
    assert m["ViT-P8-666K"]["cfg"] == {**ct.VIT_CFG, "patch_size": 8, "embed_dim": 148}
    assert m["ViT-P8-666K"]["cfg"]["stem"] == "linear" and m["ViT-P8-666K"]["cfg"]["features_dim"] == 512
    for base in ("CNN-D512P", "ViT-P8-666K"):
        assert m[f"{base}+DVS"] == {**m[base], "pre": ("dvs_polarity",)}
        assert m[f"{base}+CLTT-Ref"] == {**m[base], "aux": "cltt_ref", "aux_weight": 1.0}
    assert m["ViT-P8-666K+MAE"] == {**m["ViT-P8-666K"], "aux": "mae", "aux_weight": 1.0}
    v = m["ViViT-P8-T2"]
    assert v["encoder"] == "compact_vivit" and v["framestack"] is True
    assert v["cfg"]["patch_size"] == 8 and v["cfg"]["temporal_mode"] == "early" and v["cfg"]["pool"] == "cls"
    assert v["cfg"]["features_dim"] == 512
