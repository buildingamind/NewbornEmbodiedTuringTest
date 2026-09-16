"""Behaviour of the two-channel (ON, OFF) event wrapper.

⚠ EVERY TEST HERE DRIVES `DVSPolarity` THROUGH `reset()`/`step()`. Exercising `_luma` or
`_events` directly would make the helper the subject and leave the wrapper -- the thing an arm
actually runs -- untested; a previous suite on this fleet went fully green against a broken
caller for exactly that reason.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
from nett_skrl.body.wrappers.registry import _load_wrapper

H, W = 8, 12


class FakeEnv(gym.Env):
    """Serves a scripted list of frames. Batched iff the frames carry a leading env axis."""

    def __init__(self, frames, *, hwc=True, as_tensor=True, dict_obs=True, n_env=None,
                 device="cpu"):
        self.frames = list(frames)
        self.i = 0
        self.as_tensor = as_tensor
        self.dict_obs = dict_obs
        self.device = device
        self.n_env = n_env
        c = 3
        shape = (H, W, c) if hwc else (c, H, W)
        if n_env is not None:
            shape = (n_env,) + shape
        box = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({"policy": box}) if dict_obs else box
        self.action_space = gym.spaces.Discrete(2)

    def _emit(self):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        out = torch.as_tensor(f, device=self.device) if self.as_tensor else np.asarray(f)
        return {"policy": out, "extra": "keep-me"} if self.dict_obs else out

    def reset(self, **kw):
        self.i = 0
        return self._emit(), {}

    def step(self, action):
        n = self.n_env or 1
        z = np.zeros(n, dtype=bool)
        return self._emit(), np.zeros(n), z, z.copy(), {}


def _pol(obs):
    """Pull the policy image out and return it as a numpy array."""
    x = obs["policy"] if isinstance(obs, dict) else obs
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def const(v, *, hwc=True, n_env=None, c=3):
    """A uniform frame at value ``v`` (scalar, or a per-channel triple)."""
    px = np.array(v if np.ndim(v) else [v] * c, dtype=np.float64)
    f = np.broadcast_to(px, (H, W, c)).astype(np.uint8)
    if not hwc:
        f = np.moveaxis(f, -1, 0)
    if n_env is not None:
        f = np.broadcast_to(f, (n_env,) + f.shape).copy()
    return f


def drive(frames, **kw):
    """reset() then step() through every remaining frame; return the list of event maps."""
    env = DVSPolarity(FakeEnv(frames, **kw))
    out = [_pol(env.reset()[0])]
    for _ in range(len(frames) - 1):
        out.append(_pol(env.step(0)[0]))
    return out


# ---------------------------------------------------------------- space & construction

def test_output_space_is_two_channel_uint8_hwc():
    env = DVSPolarity(FakeEnv([const(0)]))
    sp = env.observation_space["policy"]
    assert sp.shape == (H, W, 2)
    assert sp.dtype == np.uint8
    assert (sp.low.min(), sp.high.max()) == (0, 255)


def test_output_space_is_two_channel_chw():
    env = DVSPolarity(FakeEnv([const(0, hwc=False)], hwc=False))
    assert env.observation_space["policy"].shape == (2, H, W)


def test_output_space_keeps_the_batch_axis():
    env = DVSPolarity(FakeEnv([const(0, n_env=4)], n_env=4))
    assert env.observation_space["policy"].shape == (4, H, W, 2)


def test_non_image_space_rides_through_untouched():
    inner = FakeEnv([const(0)])
    inner.observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(-1, 1, shape=(5,))})
    assert DVSPolarity(inner).observation_space["policy"].shape == (5,)


def test_zero_threshold_is_rejected():
    # ⛔ At 0 both `d >= 0` and `d <= 0` hold wherever d == 0, so EVERY unchanged pixel would
    # fire BOTH polarities -- a degenerate input returning a full-field success.
    with pytest.raises(ValueError, match="both polarities|BOTH polarities"):
        DVSPolarity(FakeEnv([const(0)]), threshold=0.0)


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError):
        DVSPolarity(FakeEnv([const(0)]), threshold=-1.0)


def test_registry_resolves_the_name():
    assert _load_wrapper("dvs_polarity") is DVSPolarity


def test_registry_entry_is_distinct_from_dvs():
    assert _load_wrapper("dvs") is not DVSPolarity


# ---------------------------------------------------------------- the event transform

def test_first_frame_emits_no_events():
    # ⛔ Differencing frame 0 against a zero buffer would report the whole image as one ON
    # event at every reset -- a full-field flash perfectly correlated with episode start.
    assert _pol(DVSPolarity(FakeEnv([const(200)])).reset()[0]).sum() == 0


def test_brightening_fires_on_only():
    ev = drive([const(10), const(200)])[1]
    assert (ev[..., 0] == 255).all()
    assert (ev[..., 1] == 0).all()


def test_darkening_fires_off_only():
    ev = drive([const(200), const(10)])[1]
    assert (ev[..., 0] == 0).all()
    assert (ev[..., 1] == 255).all()


def test_no_change_fires_nothing():
    assert drive([const(120), const(120), const(120)])[2].sum() == 0


def test_polarities_are_mutually_exclusive():
    for a, b in ((10, 200), (200, 10), (120, 120), (120, 121)):
        ev = drive([const(a), const(b)])[1]
        assert not ((ev[..., 0] > 0) & (ev[..., 1] > 0)).any()


@pytest.mark.parametrize("delta,fires", [(29, False), (30, True), (31, True)])
def test_threshold_boundary_is_inclusive(delta, fires):
    ev = drive([const(100), const(100 + delta)])[1]
    assert bool(ev[..., 0].any()) is fires


def test_threshold_knob_is_honoured():
    env = DVSPolarity(FakeEnv([const(100), const(105)]), threshold=4.0)
    env.reset()
    assert _pol(env.step(0)[0])[..., 0].any()


def test_events_are_relative_to_the_previous_frame_not_the_first():
    # A ramp that never moves more than the threshold in one step must stay silent even
    # though its total excursion is far above it.
    frames = [const(v) for v in (100, 120, 140, 160, 180)]
    assert sum(e.sum() for e in drive(frames)[1:]) == 0


def test_colour_change_at_matched_luminance_is_invisible():
    # ⛔ THE LOAD-BEARING CLAIM. Two frames with very different RGB but the SAME BT.601
    # luminance must produce no events -- that is what "colour information is lost" means,
    # and it is the reason this arm cannot take the background-brightness shortcut.
    a = np.array([255.0, 0.0, 0.0])                       # luma 76.2
    b = np.array([0.0, 255.0 * 0.299 / 0.587, 0.0])       # luma 76.2
    assert abs(np.dot(a, (0.299, 0.587, 0.114)) - np.dot(b, (0.299, 0.587, 0.114))) < 1e-6
    assert drive([const(a), const(b)])[1].sum() == 0


def test_luma_weights_are_bt601():
    # Pure blue at 255 -> luma 29.07, below the default 30 threshold; pure green -> 149.7.
    assert drive([const(0), const([0.0, 0.0, 255.0])])[1].sum() == 0
    assert drive([const(0), const([0.0, 255.0, 0.0])])[1][..., 0].all()


def test_spatially_local_change_stays_local():
    a = const(100)
    b = a.copy()
    b[2:4, 3:6, :] = 200
    ev = drive([a, b])[1]
    on = ev[..., 0] > 0
    assert on[2:4, 3:6].all()
    assert not on[6:, 9:].any()          # far from the patch, outside any blur spill


def test_chw_input_puts_channels_at_axis_minus_three():
    ev = drive([const(10, hwc=False), const(200, hwc=False)], hwc=False)[1]
    assert ev.shape == (2, H, W)
    assert (ev[0] == 255).all() and (ev[1] == 0).all()


# ---------------------------------------------------------------- statefulness

def test_reset_clears_the_previous_frame():
    env = DVSPolarity(FakeEnv([const(10), const(200)]))
    env.reset()
    env.step(0)                                  # buffer now holds the bright frame
    env.frames = [const(200)]
    assert _pol(env.reset()[0]).sum() == 0


def test_done_zeroes_only_the_finished_env():
    n = 3
    frames = [const(10, n_env=n), const(200, n_env=n)]
    env = DVSPolarity(FakeEnv(frames, n_env=n))
    inner = env.env
    env.reset()
    done = np.array([True, False, False])
    orig_step = inner.step
    inner.step = lambda a: (orig_step(a)[0], np.zeros(n), done, np.zeros(n, bool), {})
    ev = _pol(env.step(0)[0])
    # ⚠ Across an episode cut the difference is a teleport, not motion.
    assert ev[0].sum() == 0
    assert ev[1][..., 0].all() and ev[2][..., 0].all()


def test_envs_are_independent():
    n = 3
    a = const(100, n_env=n)
    b = a.copy()
    b[1] = 200                                   # only env 1 moves
    ev = drive([a, b], n_env=n)[1]
    assert ev[0].sum() == 0 and ev[2].sum() == 0
    assert ev[1][..., 0].all()


# ---------------------------------------------------------------- type / device / plumbing

def test_tensor_in_tensor_out_same_device():
    env = DVSPolarity(FakeEnv([const(10), const(200)]))
    env.reset()
    out = env.step(0)[0]["policy"]
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.uint8
    assert out.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_in_cuda_out():
    env = DVSPolarity(FakeEnv([const(10), const(200)], device="cuda"))
    env.reset()
    assert env.step(0)[0]["policy"].is_cuda


def test_numpy_in_numpy_out():
    env = DVSPolarity(FakeEnv([const(10), const(200)], as_tensor=False))
    env.reset()
    out = env.step(0)[0]["policy"]
    assert isinstance(out, np.ndarray) and out.dtype == np.uint8


def test_other_dict_keys_survive():
    env = DVSPolarity(FakeEnv([const(10), const(200)]))
    env.reset()
    assert env.step(0)[0]["extra"] == "keep-me"


def test_bare_observation_is_supported():
    env = DVSPolarity(FakeEnv([const(10), const(200)], dict_obs=False))
    env.reset()
    assert _pol(env.step(0)[0]).shape == (H, W, 2)


def test_reward_and_flags_pass_through():
    env = DVSPolarity(FakeEnv([const(10), const(200)]))
    env.reset()
    _, rew, term, trunc, info = env.step(0)
    assert np.asarray(rew).shape == (1,) and not np.asarray(term).any() and isinstance(info, dict)


# ---------------------------------------------------------------- knobs

def test_blur_off_is_a_pure_difference(monkeypatch):
    monkeypatch.setenv("NETT_DVS_BLUR", "0")
    a = const(100)
    b = a.copy()
    b[3, 5, :] = 200                             # ONE pixel
    env = DVSPolarity(FakeEnv([a, b]))
    assert env.blur is False
    env.reset()
    ev = _pol(env.step(0)[0])
    assert ev[..., 0].sum() == 255               # exactly that pixel, no spill


def test_blur_on_by_default_spreads_a_single_pixel():
    a = const(100)
    b = a.copy()
    b[3, 5, :] = 255
    env = DVSPolarity(FakeEnv([a, b]), threshold=1.0)
    env.reset()
    ev = _pol(env.step(0)[0])
    assert ev[..., 0].sum() > 255                # the blur put events on neighbours too


def test_threshold_env_var_is_read(monkeypatch):
    monkeypatch.setenv("NETT_DVS_THRESHOLD", "7.5")
    assert DVSPolarity(FakeEnv([const(0)])).threshold == 7.5


def test_more_than_three_channels_is_refused():
    # ⛔ The ordering rule, enforced. >3 channels means a stacking wrapper ran first and this
    # would difference across a frame boundary with a channel axis that no longer means colour.
    six = np.zeros((H, W, 6), dtype=np.uint8)
    env = DVSPolarity(FakeEnv([const(0)]))
    env.env.frames = [six]
    with pytest.raises(ValueError, match="BEFORE"):
        env.reset()
