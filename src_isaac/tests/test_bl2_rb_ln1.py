"""BL2 Random Brightness / first-layer LayerNorm arms (owner 2026-11-04: "For binding, I'd like to train 6 brains of
object 1 on Random Brightness and 6 brains on 1st layer LayerNorm for 3DCNN + CLTT (attached)").

  3DCNN-CLTT-Ref+RB  = 3DCNN-CLTT-Ref + pre ("random_brightness",): one factor ~ U[0.7, 1.3] per env per TRAINING
                       episode, every pixel, clamped; test unaugmented.
  3DCNN-CLTT-Ref+LN1 = 3DCNN-CLTT-Ref + cfg stem_layernorm=True: LayerNorm over the Conv3d stem map's C x H' x W',
                       elementwise affine, before cnn2d's first ReLU, at train and test.

Pinned: each label is the twin with that ONE change; the wrapper's factor is constant within an episode and across
both stacked frames, redraws exactly on done, lies in the range, is logged, is identity outside training and refuses
unbound; the LayerNorm is off by default (no module, no parameter), sits where the definition says, normalises,
receives gradient, is used by encode_spatial too, and refuses the duplicate-frame path. Fixtures use the input the
real env supplies -- {"policy": uint8 NHWC tensor} with tensor done flags -- on CUDA when present.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.body.wrappers import brightness as BR  # noqa: E402
from nett_skrl.body.wrappers.framestack import FrameStack  # noqa: E402
from nett_skrl.body.wrappers.registry import validate_wrappers  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.encoders.compact_3dcnn import Compact3DCNN  # noqa: E402
from nett_skrl.brain.models.utils.init import orthogonal_init  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
H, W = 80, 128      # BL2's eye (NETT_RES=128)
TWIN = "3DCNN-CLTT-Ref"


# ── labels ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_rb_label_is_the_twin_plus_the_wrapper_only():
    rb = ct.MODELS["3DCNN-CLTT-Ref+RB"]
    assert {k: v for k, v in rb.items() if k != "pre"} == ct.MODELS[TWIN]
    assert tuple(rb["pre"]) == ("random_brightness",)
    assert ct.segmentation_wrappers(rb) == ["random_brightness", "framestack"]   # innermost, before framestack
    assert validate_wrappers(["random_brightness"]) == [BR.RandomBrightness]


def test_ln1_label_is_the_twin_plus_the_cfg_flag_only():
    ln = ct.MODELS["3DCNN-CLTT-Ref+LN1"]
    assert {k: v for k, v in ln.items() if k != "cfg"} == {k: v for k, v in ct.MODELS[TWIN].items() if k != "cfg"}
    assert ln["cfg"] == {**ct.MODELS[TWIN]["cfg"], "stem_layernorm": True}
    assert "stem_layernorm" not in ct.MODELS[TWIN]["cfg"] and "pre" not in ct.MODELS[TWIN]


# ── LayerNorm ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _encoder(label, seed=0):
    spec = ct.MODELS[label]
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    space = gym.spaces.Box(low=0, high=255, shape=(3 * ct._FRAMESTACK_N, H, W), dtype=np.uint8)
    torch.manual_seed(seed)
    enc = encoder_mapping[spec["encoder"]](space, **kwargs)
    enc.apply(lambda m: orthogonal_init(m, 2 ** 0.5))
    return enc


def _obs(seed=1, n=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3 * ct._FRAMESTACK_N, H, W), generator=g, dtype=torch.uint8).float()


def test_ln_off_registers_nothing_and_keeps_the_twin_bitwise():
    twin, ln = _encoder(TWIN), _encoder("3DCNN-CLTT-Ref+LN1")
    assert not hasattr(twin, "stem_ln")
    extra = set(ln.state_dict()) - set(twin.state_dict())
    assert extra == {"stem_ln.weight", "stem_ln.bias"}
    # The pre-existing tensors are drawn identically (LayerNorm consumes no RNG and is registered last).
    for k, v in twin.state_dict().items():
        assert torch.equal(v, ln.state_dict()[k]), k
    n_twin = sum(p.numel() for p in twin.parameters())
    n_ln = sum(p.numel() for p in ln.parameters())
    assert n_ln - n_twin == 2 * 32 * ((H - 1) // 2 + 1) * ((W - 1) // 2 + 1) == 163840


def test_ln_sits_after_the_stem_over_c_h_w_and_normalises():
    ln = _encoder("3DCNN-CLTT-Ref+LN1")
    assert tuple(ln.stem_ln.normalized_shape) == (32, 40, 64) and ln.stem_ln.elementwise_affine
    assert torch.all(ln.stem_ln.weight == 1) and torch.all(ln.stem_ln.bias == 0)
    x = ln._prepare_image(_obs())
    with torch.no_grad():
        raw = ln._fold_time(ln._stem_conv(x.view(x.shape[0], 2, 3, H, W).permute(0, 2, 1, 3, 4).contiguous()))
        y = ln._temporal_stem(x)
    assert torch.allclose(y, torch.nn.functional.layer_norm(raw, raw.shape[1:], eps=ln.stem_ln.eps))
    flat = y.flatten(1)
    assert torch.allclose(flat.mean(1), torch.zeros(3), atol=1e-4)
    assert torch.allclose(flat.std(1, unbiased=False), torch.ones(3), atol=1e-3)
    # forward = linear(cnn2d(LN(stem))): the norm precedes cnn2d's first ReLU.
    assert isinstance(ln.cnn2d[0], torch.nn.ReLU)
    with torch.no_grad():
        assert torch.equal(ln(_obs()), ln.linear(ln.cnn2d(ln._temporal_stem(ln._prepare_image(_obs())))))


def test_ln_gets_gradient_and_encode_spatial_sees_it():
    ln = _encoder("3DCNN-CLTT-Ref+LN1")
    ln(_obs()).sum().backward()
    assert ln.stem_ln.weight.grad is not None and ln.stem_ln.weight.grad.abs().sum() > 0
    assert ln.stem_ln.bias.grad.abs().sum() > 0
    with torch.no_grad():
        ln.stem_ln.bias.fill_(5.0)       # a changed norm must change the spatial map
        a = ln.encode_spatial(_obs())
        ln.stem_ln.bias.zero_()
        b = ln.encode_spatial(_obs())
    assert not torch.equal(a, b)


def test_ln_refuses_duplicate_frame():
    space = gym.spaces.Box(low=0, high=255, shape=(3, H, W), dtype=np.uint8)
    with pytest.raises(ValueError, match="stem_layernorm"):
        Compact3DCNN(space, duplicate_frame=True, stem_layernorm=True)


# ── Random brightness ───────────────────────────────────────────────────────────────────────────────────────────────
class _StubEnv(gym.Env):
    action_space = None
    metadata: dict = {}
    render_mode = None
    spec = None

    def __init__(self, n=4, h=12, w=20, device="cpu", done_at=None):
        self.n, self.h, self.w, self.device = n, h, w, device
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(n, h, w, 3), dtype=np.uint8)
        self.done_at = done_at or {}   # step index -> env ids that finish on that step
        self.t = 0
        g = torch.Generator().manual_seed(0)
        self.base = torch.randint(0, 256, (n, h, w, 3), generator=g, dtype=torch.uint8)

    def _frame(self):
        # A different (valid uint8) frame every step, so the factor is checked on changing content.
        return ((self.base.int() + 7 * self.t) % 256).to(torch.uint8).to(self.device)

    def reset(self, **kwargs):
        self.t = 0
        return {"policy": self._frame()}, {}

    def step(self, action):
        self.t += 1
        done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for i in self.done_at.get(self.t, []):
            done[i] = True
        return ({"policy": self._frame()}, torch.zeros(self.n, device=self.device), done,
                torch.zeros(self.n, dtype=torch.bool, device=self.device), {})

    def close(self):
        pass


def _bound(tmp_path, phase="train", seed=(7, 3), **kw):
    env = BR.RandomBrightness(_StubEnv(**kw))
    env.bind_phase(phase, log_dir=tmp_path, offset=3, seed=list(seed))
    return env


def _expected(raw, f):
    return (raw.float() * torch.as_tensor(f, dtype=torch.float32).view(-1, 1, 1, 1)).round().clamp(0, 255).to(torch.uint8)


def test_unbound_refuses(tmp_path):
    env = BR.RandomBrightness(_StubEnv())
    with pytest.raises(RuntimeError, match="never bound"):
        env.reset()


@pytest.mark.parametrize("device", DEVICES)
def test_factor_is_per_episode_and_redraws_exactly_on_done(tmp_path, device):
    env = _bound(tmp_path, device=device, done_at={3: [1], 5: [0, 2]})
    obs, _ = env.reset()
    f0 = env._factor.copy()
    assert np.all((f0 >= 0.7) & (f0 <= 1.3)) and len(set(f0)) == 4
    assert torch.equal(obs["policy"].cpu(), _expected(env.env._frame().cpu(), f0))
    history = [f0]
    for t in range(1, 8):
        obs, *_ = env.step(None)
        history.append(env._factor.copy())
        assert obs["policy"].dtype == torch.uint8 and obs["policy"].device.type == device
        assert torch.equal(obs["policy"].cpu(), _expected(env.env._frame().cpu(), env._factor))
    changed = [set(np.where(history[t] != history[t - 1])[0]) for t in range(1, 8)]
    assert changed == [set(), set(), {1}, set(), {0, 2}, set(), set()]
    rows = list(csv.reader(open(tmp_path / "brightness_train_off3.csv")))
    assert rows[0] == ["step", "env_id", "factor", "phase"]
    assert [(int(r[0]), int(r[1])) for r in rows[1:]] == [(0, 0), (0, 1), (0, 2), (0, 3), (3, 1), (5, 0), (5, 2)]


def test_the_frame_returned_with_done_carries_the_new_episodes_factor(tmp_path):
    env = _bound(tmp_path, done_at={2: [0]})
    env.reset()
    env.step(None)
    old = env._factor[0]
    obs, *_ = env.step(None)           # env 0 auto-resets: this frame is the new episode's first
    assert env._factor[0] != old
    assert torch.equal(obs["policy"][0], _expected(env.env._frame(), env._factor)[0])


def test_both_stacked_frames_share_the_episode_factor(tmp_path):
    """Through the real FrameStack: a stack never mixes two factors (the buffer refills on done)."""
    inner = _bound(tmp_path, done_at={2: [0, 3]})
    stack = FrameStack(inner, n_stack=2)
    stack.reset()
    for _ in range(4):
        obs, *_ = stack.step(None)
        p = np.asarray(obs["policy"])                     # (n, h, w, 6), T-major channels
        for i in range(4):
            prev, cur = p[i, ..., :3], p[i, ..., 3:]
            raw_cur = inner.env._frame()[i].numpy()
            assert np.array_equal(cur, _expected(torch.as_tensor(raw_cur)[None], [inner._factor[i]])[0].numpy())
            if inner.env.t == 2 and i in (0, 3):
                assert np.array_equal(prev, cur)          # refilled with the new episode's (augmented) first frame
            else:
                raw_prev = ((inner.env.base[i].int() + 7 * (inner.env.t - 1)) % 256).to(torch.uint8)
                assert np.array_equal(prev, _expected(raw_prev[None], [inner._factor[i]])[0].numpy())


@pytest.mark.parametrize("phase", ["test", "record", "validate"])
def test_identity_outside_training(tmp_path, phase):
    env = _bound(tmp_path, phase=phase, done_at={1: [0, 1, 2, 3]})
    obs, _ = env.reset()
    assert torch.equal(obs["policy"], env.env._frame()) and np.all(env._factor == 1.0)
    obs, *_ = env.step(None)
    assert torch.equal(obs["policy"], env.env._frame())
    assert len(list(csv.reader(open(tmp_path / f"brightness_{phase}_off3.csv")))) == 1   # header only


def test_same_seed_same_draws_and_numpy_frames(tmp_path):
    a, b = _bound(tmp_path / "a"), _bound(tmp_path / "b")
    a.reset(), b.reset()
    assert np.array_equal(a._factor, b._factor)
    c = _bound(tmp_path / "c", seed=(7, 4))
    c.reset()
    assert not np.array_equal(a._factor, c._factor)
    x = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3) * 9
    a._factor = np.array([1.3, 0.7])
    out = a._apply(x)
    assert isinstance(out, np.ndarray) and out.dtype == np.uint8
    assert np.array_equal(out, np.clip(np.round(x.astype(np.float32) * np.array([1.3, 0.7], np.float32)[:, None, None, None]), 0, 255))


def test_refuses_stacked_or_float_frames(tmp_path):
    env = _bound(tmp_path)
    with pytest.raises(ValueError, match="channels"):
        env._geometry(torch.zeros(2, 4, 4, 6, dtype=torch.uint8))
    env._factor = np.ones(2)
    with pytest.raises(TypeError, match="uint8"):
        env._apply(torch.zeros(2, 4, 4, 3))


def test_runner_binds_train_and_seeds_by_brain(tmp_path, monkeypatch):
    from nett_skrl.runtime import task_runner as tr

    env = BR.RandomBrightness(_StubEnv())
    logs = []
    rc = SimpleNamespace(brain_id_offset=5, train_start_step=0, seed=11, path=tmp_path,
                         logger=SimpleNamespace(info=lambda *a: logs.append(a)))
    tr._bind_brightness(env, "train", rc)
    assert env._bound and env._train and env._log_path == tmp_path / "logs" / "brightness_train_off5.csv"
    env.reset()
    ref = np.random.default_rng([11, 5, 0]).uniform(0.7, 1.3, size=4)
    assert np.array_equal(env._factor, ref)
    tr._bind_brightness(_StubEnv(), "train", rc)    # no wrapper on the chain: a no-op
