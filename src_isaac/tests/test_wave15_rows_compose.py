"""Wave 15's rows must compose END TO END, and each must be read against the INCUMBENT.

⛔ WHY THIS FILE EXISTS SEPARATELY FROM THE UNIT SUITES. `test_dvs_polarity.py` and
`test_unity_vit.py` stub the environment, so neither can see what happens when the arm's real
wrapper list, the encoder built from the resulting space, and the aux's view construction are
composed in the order `Body.wrap` builds them. That gap is not hypothetical: running row 08 this
way surfaced that `framestack` hands the encoder a NumPy array, which no unit test here touches.

⚠ AND IT PAIRS EVERY ASSERTION WITH ViT-CLTT-Ref. A new row hitting a snag proves nothing about
the row until the incumbent is shown not to hit it -- the NumPy hand-off above is identical on
both, i.e. pre-existing, and a suite without the control would have reported it as a defect in
the new wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

# ⚠ The repo's own convention (tests/test_cltt_ref_aux.py:15-16): `examples` is not a package on
# the default path, so a bare `from campaign_train import ...` passes only when the caller happened
# to set PYTHONPATH. Put it on the path here rather than relying on the invocation.
_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "examples"))

from campaign_train import MODELS, segmentation_wrappers  # noqa: E402
from nett_skrl.body.body import Body
from nett_skrl.body.wrappers.registry import validate_wrappers
from nett_skrl.brain.aux.cltt_views import current_frame_stack
from nett_skrl.brain.registry import encoder_mapping

H, W, N = 80, 128, 4

#: Every wave-15 row, with the per-frame channel count its `env:` block declares.
WAVE15 = {
    "ViT-CLTT-Ref": 3,          # the incumbent -- the control, not a row
    "ViT-CLTT-Ref-H8": 3,
    "ViT-CLTT-Ref-P8": 3,
    "ViT-CLTT-Ref-Sp": 3,
    "ViT-CLTT-Ref-Conv": 3,
    "ViT-CLTT-Ref-D6": 3,
    "ViT-CLTT-Ref-MLP4": 3,
    "UnityViT-CLTT-Ref": 3,
    "ViT-CLTT-Ref-DVS": 2,
}
ROWS = [m for m in WAVE15 if m != "ViT-CLTT-Ref"]


class FakeIsaac(gym.Env):
    """Batched Isaac-shaped env serving a bright block that MOVES every step."""

    def __init__(self, moving=True):
        self.observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(0, 255, (N, H, W, 3), dtype=np.uint8)})
        self.action_space = gym.spaces.Box(-1, 1, (N, 2))
        self.t, self.moving = 0, moving

    def _obs(self):
        self.t += 1
        f = np.full((N, H, W, 3), 40, dtype=np.uint8)
        c = (self.t * 6) % 90 if self.moving else 10
        f[:, 20:40, c:c + 20, :] = 220
        return {"policy": torch.as_tensor(f)}

    def reset(self, **kw):
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        z = np.zeros(N, dtype=bool)
        return self._obs(), np.zeros(N), z, z.copy(), {}


def drive(model, monkeypatch, steps=2, moving=True):
    """Build the row exactly as the driver does and take `steps` steps."""
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", str(WAVE15[model]))
    spec = MODELS[model]
    names = segmentation_wrappers(spec)
    env = Body(wrappers=validate_wrappers(names)).wrap(FakeIsaac(moving=moving))
    env.reset()
    obs = None
    for _ in range(steps):
        obs, *_ = env.step(torch.zeros(N, 2))
    x = obs["policy"]
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    batched = env.observation_space["policy"]
    per_env = gym.spaces.Box(0, 255, tuple(batched.shape[1:]), dtype=np.uint8)
    cfg = dict(spec["cfg"])
    cfg.pop("trainable", None)
    enc = encoder_mapping[spec["encoder"]](per_env, **cfg)
    return names, batched, x, enc


@pytest.mark.parametrize("model", ROWS)
def test_row_runs_from_wrappers_through_encoder_to_backward(model, monkeypatch):
    """The whole chain, including the aux's view construction and a real backward pass."""
    _, _, x, enc = drive(model, monkeypatch)
    assert enc(x).shape == (N, 512)
    prepared = enc._prepare_image(x)
    view = current_frame_stack(prepared)
    assert view.shape == prepared.shape
    enc.encode_prepared(view).sum().backward()
    grad = sum(p.grad.abs().sum().item() for p in enc.parameters() if p.grad is not None)
    assert grad > 0, "no gradient reached the encoder through the auxiliary path"


@pytest.mark.parametrize("model", ROWS)
def test_row_matches_the_incumbents_plumbing(model, monkeypatch):
    """⚠ THE CONTROL. Output type and rank must match ViT-CLTT-Ref's -- a difference here is a
    difference in the BODY, which no row in this wave is supposed to change except row 08's
    channel count."""
    _, base_space, base_x, _ = drive("ViT-CLTT-Ref", monkeypatch)
    _, space, x, _ = drive(model, monkeypatch)
    assert type(x) is type(base_x)
    assert x.dtype == base_x.dtype
    assert len(space.shape) == len(base_space.shape)
    assert space.shape[0] == base_space.shape[0] == N
    assert space.shape[2:] == base_space.shape[2:] == (H, W)


def test_only_the_dvs_row_changes_the_channel_count():
    """Six variants and the Unity row must see the incumbent's 6 channels; only row 08 differs."""
    import count_params as cp
    w, h, _ = cp.eye_wh()
    base = cp.wrapped_space(MODELS["ViT-CLTT-Ref"], w, h)[0].shape[-1]
    assert base == 6
    for model in ROWS:
        ch = cp.wrapped_space(MODELS[model], w, h)[0].shape[-1]
        assert ch == (4 if model == "ViT-CLTT-Ref-DVS" else 6), f"{model} sees {ch} channels"


def test_the_dvs_row_orders_its_wrapper_innermost():
    """⛔ dvs_polarity is temporal and must consume RAW frames. Reversed, it would difference two
    already-stacked tensors and the channel axis would mean neither thing."""
    names = segmentation_wrappers(MODELS["ViT-CLTT-Ref-DVS"])
    assert names == ["dvs_polarity", "framestack"]
    assert names.index("dvs_polarity") < names.index("framestack")


def test_a_moving_stimulus_produces_events_and_a_still_one_produces_none(monkeypatch):
    """⛔ THE TWO-SIDED CHECK. "Events fire" alone is satisfied by a wrapper that outputs noise;
    "no events when still" alone is satisfied by one that outputs zeros. Only the pair binds.

    ⚠ IT ALSO PINS THAT EVENTS ARE SPARSE. A stimulus sweeping the whole frame every step lights
    up a couple of percent of pixels, because events are EDGE signals. A falsifier gating on
    absolute occupancy must be calibrated against this, not against intuition -- wave 15's row 08
    originally required occupancy "above ~0.3", which no run could ever have met.
    """
    _, _, moving, _ = drive("ViT-CLTT-Ref-DVS", monkeypatch, moving=True)
    _, _, still, _ = drive("ViT-CLTT-Ref-DVS", monkeypatch, moving=False)
    occ_moving = (moving > 0).float().mean().item()
    assert still.max().item() == 0, "a stationary stimulus emitted events"
    assert 0.0 < occ_moving < 0.20, f"expected sparse edge events, got occupancy {occ_moving}"


def test_the_dvs_row_would_die_without_its_channels_per_frame_knob(monkeypatch):
    """⛔ THE ENV BLOCK IS LOAD-BEARING, AND THE FAILURE IS LOUD -- which is the only reason a
    default of 3 is tolerable. 4 channels under the RGB default is 4 % 3 != 0."""
    monkeypatch.delenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", raising=False)
    _, _, x, enc = drive("ViT-CLTT-Ref-DVS", monkeypatch)
    monkeypatch.delenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", raising=False)
    with pytest.raises(ValueError, match="CHANNELS_PER_FRAME"):
        current_frame_stack(enc._prepare_image(x))
