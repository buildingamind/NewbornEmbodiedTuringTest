"""Owner, workspace DECISIONS §162: "ViViT-CLTT-Ref-T2" -- ViViT-CLTT-Ref with temporal_mode "early".

"early" IS a 2-frame-deep patch: each token is ONE Linear over both frames' 16x16 patches at a location,
concatenated T-major -- proven below to equal a Conv3d with a (2, 16, 16) kernel and (2, 16, 16) stride
built from the same weights. Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.compact_vivit import _patchify
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the default eye
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _build(spec, channels=6, seed=0):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, H, W), np.uint8), **kwargs)


def _n(module):
    return sum(p.numel() for p in module.parameters())


def test_t2_label_is_vivit_cltt_ref_plus_early_only(campaign):
    ref, t2 = campaign.MODELS["ViViT-CLTT-Ref"], campaign.MODELS["ViViT-CLTT-Ref-T2"]
    assert ref["cfg"]["temporal_mode"] == "joint"
    assert {**t2["cfg"]} == {**ref["cfg"], "temporal_mode": "early"}
    assert {k: v for k, v in t2.items() if k != "cfg"} == {k: v for k, v in ref.items() if k != "cfg"}


def test_t2_parameter_counts_at_128x80(campaign):
    M = campaign.MODELS
    ref, t2 = _build(M["ViViT-CLTT-Ref"]), _build(M["ViViT-CLTT-Ref-T2"])
    assert (_n(ref), _n(t2)) == (694_016, 804_320)
    assert (_n(ref.patch_embed), _n(t2.patch_embed)) == (110_736, 221_328)
    assert ref.temporal_pos_embed.numel() == 288 and t2.temporal_pos_embed is None
    assert tuple(t2.spatial_pos_embed.shape) == (1, 40, 144)
    assert _n(AUX_LOSSES["cltt_ref"](t2).head) == _n(AUX_LOSSES["cltt_ref"](ref).head) == 329_216
    x = torch.randint(0, 256, (2, 6, H, W), dtype=torch.uint8).float()
    assert t2(x).shape == ref(x).shape == (2, 512)


def test_early_tokens_are_a_two_frame_deep_tubelet(campaign):
    enc = _build(campaign.MODELS["ViViT-CLTT-Ref-T2"])
    T, C, P, D = enc.num_frames, enc.base_channels, enc.patch_size, enc.embed_dim
    assert (T, C, P) == (2, 3, 16)
    x = torch.rand(2, T * C, H, W, generator=torch.Generator().manual_seed(1))
    frames = enc._frames(x)
    tok = enc.patch_embed(torch.cat([_patchify(frames[:, t], P) for t in range(T)], dim=-1))
    # the same weights as one Conv3d over (B, C, T, H, W): kernel (T, P, P), stride (T, P, P)
    w3 = enc.patch_embed.weight.view(D, T, C, P, P).permute(0, 2, 1, 3, 4)
    vol = x.view(2, T, C, H, W).permute(0, 2, 1, 3, 4)
    conv = F.conv3d(vol, w3, enc.patch_embed.bias, stride=(T, P, P))      # (B, D, 1, H/P, W/P)
    assert conv.shape[2] == 1
    torch.testing.assert_close(tok, conv.flatten(2).transpose(1, 2), rtol=1e-5, atol=1e-5)
