"""Owner request 2026-10-05 (researcher2): "3DCNN-DVS" and NETT_RANDOM_FIRST_FRAME.

3DCNN-DVS  -- "3DCNN" fed dvs_polarity events: 2 channels per frame, stated in the cfg and
              checked EXACTLY, so an RGB stack cannot reach it and a DVS stack cannot reach "3DCNN".
RFF        -- opt-in: unset, campaign_train passes no random_first_frame key (Environment's
              False governs); set, it passes True and tags the run from the same flag.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.compact_3dcnn import Compact3DCNN
from nett_skrl.brain.encoders.utils.temporal import validate_framestack_depth
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the default eye
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _space(channels):
    return gym.spaces.Box(low=0, high=255, shape=(channels, H, W), dtype=np.uint8)


def _build(spec, channels):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(0)
    return encoder_mapping[spec["encoder"]](_space(channels), **kwargs)


def _params(enc):
    return sum(p.numel() for p in enc.parameters())


# ---------------------------------------------------------------- validator

def test_validator_default_path_is_unchanged():
    assert validate_framestack_depth(6, 2, "x") == 2
    assert validate_framestack_depth(3, 1, "x") == 1
    with pytest.raises(ValueError, match="not a plausible image depth"):
        validate_framestack_depth(4, 2, "x")          # 2/frame still refused without the field


def test_validator_with_channels_per_frame_is_exact():
    assert validate_framestack_depth(4, 2, "x", channels_per_frame=2) == 2
    for total in (6, 2, 8):
        with pytest.raises(ValueError, match="channels_per_frame=2"):
            validate_framestack_depth(total, 2, "x", channels_per_frame=2)
    with pytest.raises(ValueError):
        validate_framestack_depth(0, 2, "x", channels_per_frame=0)


# ---------------------------------------------------------------- 3DCNN-DVS

def test_registry_entry_is_3dcnn_plus_dvs_only(campaign):
    base, dvs = campaign.MODELS["3DCNN"], campaign.MODELS["3DCNN-DVS"]
    assert dvs["encoder"] == base["encoder"] == "compact_3dcnn"
    assert {**dvs["cfg"]} == {**base["cfg"], "channels_per_frame": 2}
    assert dvs["framestack"] is True and dvs["pre"] == ["dvs_polarity"]
    assert not dvs.get("aux")
    # dvs_polarity INNERMOST, then framestack -- the order the wrapper docstring requires.
    assert campaign.segmentation_wrappers(dvs) == ["dvs_polarity", "framestack"]


def test_3dcnn_dvs_builds_at_four_channels_with_matched_capacity(campaign):
    base = _build(campaign.MODELS["3DCNN"], 6)
    dvs = _build(campaign.MODELS["3DCNN-DVS"], 4)
    assert dvs.base_channels == 2 and dvs.num_frames == 2
    assert _params(base) == 695_981 and _params(dvs) == 695_405
    assert abs(_params(dvs) / _params(base) - 1) < 0.001
    x = torch.randint(0, 256, (3, 4, H, W), dtype=torch.uint8).float()
    assert dvs(x).shape == (3, 512)


def test_3dcnn_dvs_refuses_an_rgb_stack_and_3dcnn_refuses_events(campaign):
    with pytest.raises(ValueError, match="channels_per_frame=2"):
        _build(campaign.MODELS["3DCNN-DVS"], 6)
    with pytest.raises(ValueError, match="not a plausible image depth"):
        _build(campaign.MODELS["3DCNN"], 4)


def test_3dcnn_default_constructor_is_byte_identical():
    """channels_per_frame=None must not move a weight: same seed, same tensors."""
    torch.manual_seed(1)
    a = Compact3DCNN(_space(6), features_dim=512, conv_dim=77)
    torch.manual_seed(1)
    b = Compact3DCNN(_space(6), features_dim=512, conv_dim=77, channels_per_frame=None)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_3dcnn_dvs_keeps_time_major_channel_layout(campaign):
    """Frame t-1's (ON, OFF) are channels 0-1 and frame t's are 2-3 (framestack is T-major).
    Events only in frame t must reach ONLY the stem's t tap: zero the t-1 kernel slice and the
    output must not change."""
    enc = _build(campaign.MODELS["3DCNN-DVS"], 4).eval()
    x = torch.zeros(1, 4, H, W)
    x[:, 2:] = 255.0
    with torch.no_grad():
        before = enc(x)
        enc.conv3d.weight[:, :, 0].zero_()           # kernel tap 0 = frame t-1
        after = enc(x)
    assert torch.allclose(before, after)


# ---------------------------------------------------------------- RFF knob

def test_random_first_frame_is_opt_in_and_derived(campaign, monkeypatch):
    src = _TRAIN.read_text()
    assert '**({"random_first_frame": True}' in src
    assert 'if _env_flag("NETT_RANDOM_FIRST_FRAME") else {})' in src
    assert '["random-first-frame"] if _env_flag("NETT_RANDOM_FIRST_FRAME")' in src
    import inspect
    from nett_skrl.environment.environment import Environment, _ENV_CFG_FIELDS

    assert inspect.signature(Environment.__init__).parameters["random_first_frame"].default is False
    assert ("screens.random_first_frame", "random_first_frame", bool) in _ENV_CFG_FIELDS
    monkeypatch.delenv("NETT_RANDOM_FIRST_FRAME", raising=False)
    assert campaign._env_flag("NETT_RANDOM_FIRST_FRAME") is False
    monkeypatch.setenv("NETT_RANDOM_FIRST_FRAME", "1")
    assert campaign._env_flag("NETT_RANDOM_FIRST_FRAME") is True
    monkeypatch.setenv("NETT_RANDOM_FIRST_FRAME", "flase")
    with pytest.raises(Exception):
        campaign._env_flag("NETT_RANDOM_FIRST_FRAME")


def test_environment_prints_the_resolved_start_frame_per_phase():
    src = (Path(__file__).resolve().parents[1] / "nett_skrl" / "environment" / "environment.py").read_text()
    assert 'print(f"[NETT screens] random_first_frame={bool(_screens.random_first_frame)} "' in src
    assert 'f"phase={config.current_mode}", flush=True)' in src
