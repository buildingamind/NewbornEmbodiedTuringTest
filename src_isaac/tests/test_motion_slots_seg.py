"""U35 slots-from-motion segmenters: MoTokFlowSeg, SASeg, SAFlowSeg (owner request 2026-10-05).

Covers: the three labels and their registry seats; forward at the real 448x280 eye; loss finite and
falling on a synthetic moving-square clip; the mask no-grad guard; previous-frame handling at
episode resets; persistence; and that MoTokSeg / GwmSeg are untouched.
"""

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers import motion_seg, sa_seg
from nett_skrl.body.wrappers.gwm_seg import GwmSeg
from nett_skrl.body.wrappers.motion_seg import FLOW_HW, MoTokFlowSeg, expert_flow_px
from nett_skrl.body.wrappers.motok_seg import MoTokSeg
from nett_skrl.body.wrappers.sa_seg import SAFlowSeg, SASeg
from nett_skrl.body.wrappers.segmentation import SegmentationObservationWrapper, state_filename

NEW = [MoTokFlowSeg, SASeg, SAFlowSeg]


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("NETT_SEG_", "NETT_EXPERT_FLOW")) or name == "NETT_ROLLOUTS":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NETT_SEG_DEVICE", "cpu")
    threads = torch.get_num_threads()
    torch.set_num_threads(min(4, threads))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield
    torch.set_num_threads(threads)


def _campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train as campaign

    return campaign


# ---------------------------------------------------------------------------------------------
# synthetic two-region clip: the square's texture is drawn from the SAME distribution as the
# background's, so only its motion against the static background separates the two regions
# ---------------------------------------------------------------------------------------------
def _texture(rng, h, w, block=4):
    t = rng.integers(60, 200, (h // block + 1, w // block + 1, 3)).astype(np.uint8)
    return np.repeat(np.repeat(t, block, 0), block, 1)[:h, :w]


class MovingSquare(gym.Env):
    """``n`` envs, each a square moving at ``speed`` px/step, bouncing; auto-reset at ``ep_len``
    (the returned obs is then the NEW episode's first frame, as in Isaac)."""

    def __init__(self, n=4, h=56, w=96, side=16, speed=3, ep_len=1000, seed=0, done_at=None):
        self.n, self.h, self.w, self.side, self.speed, self.ep_len = n, h, w, side, speed, ep_len
        self.rng = np.random.default_rng(seed)
        self.bg, self.tex = _texture(self.rng, h, w), _texture(self.rng, side, side)
        self.observation_space = gym.spaces.Box(0, 255, (n, h, w, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.pos, self.vel, self.t = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n, int)
        self.done_at = done_at or {}          # {step: [env ids]} forced terminations
        self.calls = 0

    def _init(self, i):
        self.pos[i] = [self.rng.uniform(0, self.h - self.side), self.rng.uniform(0, self.w - self.side)]
        a = self.rng.uniform(0, 2 * np.pi)
        self.vel[i] = self.speed * np.array([np.sin(a), np.cos(a)])
        self.t[i] = 0

    def frames(self):
        out = np.repeat(self.bg[None], self.n, 0).copy()
        for i in range(self.n):
            y, x = (int(round(v)) for v in self.pos[i])
            out[i, y : y + self.side, x : x + self.side] = self.tex
        return out

    def reset(self, **kwargs):
        for i in range(self.n):
            self._init(i)
        return self.frames(), {}

    def step(self, action):
        self.calls += 1
        self.pos += self.vel
        self.t += 1
        lim = np.array([self.h - self.side, self.w - self.side])
        for d in range(2):
            out = (self.pos[:, d] < 0) | (self.pos[:, d] > lim[d])
            self.vel[out, d] *= -1
            self.pos[:, d] = np.clip(self.pos[:, d], 0, lim[d])
        done = self.t >= self.ep_len
        done[self.done_at.get(self.calls, [])] = True
        for i in np.where(done)[0]:
            self._init(i)
        return self.frames(), np.zeros(self.n), done.copy(), np.zeros(self.n, bool), {}


class RealEye(gym.Env):
    num_envs = 16
    shape = (16, 280, 448, 3)

    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 255, self.shape, np.uint8)
        self.action_space = gym.spaces.Discrete(2)


# ---------------------------------------------------------------------------------------------
# 1. labels and registry
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("label,seg,cls", [
    ("MoTokFlow-Seg-UnityRecipe", "motokflow_seg", MoTokFlowSeg),
    ("SA-Seg-UnityRecipe", "sa_seg", SASeg),
    ("SAFlow-Seg-UnityRecipe", "saflow_seg", SAFlowSeg),
])
def test_labels_are_the_motok_recipe_but_for_the_segmenter(monkeypatch, label, seg, cls):
    from nett_skrl.body import Body
    from nett_skrl.body.wrappers.registry import _load_wrapper
    from nett_skrl.body.wrappers.segmentation import body_has_segmenter

    campaign = _campaign(monkeypatch)
    spec, ref = campaign.MODELS[label], campaign.MODELS["MoTok-Seg-UnityRecipe"]
    assert {k: v for k, v in spec.items() if k != "seg"} == {k: v for k, v in ref.items() if k != "seg"}
    assert spec["seg"] == seg and spec["framestack"] is False and "seg_after" not in spec
    assert campaign.segmentation_wrappers(spec) == [seg]
    assert _load_wrapper(seg) is cls
    assert body_has_segmenter(Body(wrappers=campaign.segmentation_wrappers(spec)).wrappers)


@pytest.mark.parametrize("cls", NEW)
def test_constructs_with_no_env_set(cls):
    w = cls(RealEye())
    assert w.kind == {MoTokFlowSeg: "motok_flow", SASeg: "sa", SAFlowSeg: "sa_flow"}[cls]
    assert w.motion_weight == (0.0 if cls is SASeg else 1.0)
    assert w.UNITY_TRAINING_KNOBS


# ---------------------------------------------------------------------------------------------
# 2. forward at the real eye
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls", NEW)
def test_forward_real_eye_hwc_and_nhwc(cls):
    w = cls(RealEye())
    w.train_every = 0
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 256, RealEye.shape, dtype=np.uint8)
    out = w.observation(batch)
    assert out.shape == RealEye.shape and out.dtype == np.uint8
    single = w.observation(batch[0])
    assert single.shape == RealEye.shape[1:]
    assert torch.is_tensor(w.observation(torch.from_numpy(batch)))
    x = torch.from_numpy(batch[:2]).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        masks = w._model.get_masks(x)
    assert masks.shape[0] == 2 and masks.shape[2:] == (280, 448)
    torch.testing.assert_close(masks.sum(1), torch.ones(2, 280, 448), atol=1e-5, rtol=0)
    # the policy's frame is the raw frame times a [0,1] mask: never brighter
    assert (out.astype(int) <= batch.astype(int)).all()


@pytest.mark.parametrize("cls", NEW)
def test_refuses_a_stacked_input(cls):
    w = cls(RealEye())
    with pytest.raises(ValueError, match="BEFORE framestack"):
        w.observation(np.zeros((2, 24, 32, 6), np.uint8))


# ---------------------------------------------------------------------------------------------
# 3. loss finite and falling on the synthetic clip
# ---------------------------------------------------------------------------------------------
def _clip_batches(n_batches, seed=0, batch=4):
    """(prev, cur, target=cur, valid) 10-channel samples from consecutive MovingSquare frames."""
    env = MovingSquare(n=batch, seed=seed)
    prev, _ = env.reset()
    out = []
    for _ in range(n_batches):
        cur, *_ = env.step(0)
        p = torch.from_numpy(prev).permute(0, 3, 1, 2).float() / 255
        c = torch.from_numpy(cur).permute(0, 3, 1, 2).float() / 255
        out.append(torch.cat([p, c, c, torch.ones(batch, 1, *c.shape[2:])], 1))
        prev = cur
    return out


@pytest.mark.parametrize("cls", NEW)
def test_loss_finite_and_decreasing_on_moving_square(cls, monkeypatch):
    monkeypatch.setattr(sa_seg, "SA_WARMUP_STEPS", 0)      # the paper's 10k warm-up would hold lr ~0
    w = cls(MovingSquare())
    w._ensure(3)
    batch = _clip_batches(3)[-1]          # ONE pair batch: different pairs carry different flow
    losses = [w._opt_step(batch) for _ in range(20)]
    assert all(loss is not None and np.isfinite(loss) for loss in losses), losses
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), losses
    assert w.last_stats["seg/train_steps"] == 20
    assert np.isfinite(w.last_stats["seg/recon_loss"])
    if cls is SASeg:
        assert np.isnan(w.last_stats["seg/motion_loss"])
    else:
        assert np.isfinite(w.last_stats["seg/motion_loss"]) and w.last_stats["seg/motion_loss"] > 0
        assert w.last_stats["seg/flow_absmax"] > 0         # the square moved; flow saw it


def test_motion_weight_zero_is_exactly_motok_loss(monkeypatch):
    """MoTokFlow at weight 0 computes MoTokSeg's loss on the same weights, to the bit."""
    monkeypatch.setenv("NETT_SEG_MOTION_WEIGHT", "0")
    flow = MoTokFlowSeg(MovingSquare())
    flow._ensure(3)
    plain = MoTokSeg(MovingSquare())
    plain._ensure(3)
    plain._model.load_state_dict(flow._model.state_dict())
    batch = _clip_batches(1)[0]
    torch.testing.assert_close(flow._loss(batch), plain._loss(batch[:, 6:9]), atol=0, rtol=0)


def test_one_pass_recon_and_masks_equal_motoknet_methods():
    w = MoTokFlowSeg(MovingSquare())
    w._ensure(3)
    x = torch.rand(3, 3, 56, 96)
    (recon, commit), masks = w._recon_and_masks(x)
    r2, c2 = w._model.reconstruct(x)
    torch.testing.assert_close(recon, r2, atol=0, rtol=0)
    torch.testing.assert_close(commit, c2, atol=0, rtol=0)
    torch.testing.assert_close(masks, w._model.get_masks(x), atol=0, rtol=0)


def test_flow_is_in_gwm_pixels_on_every_grid():
    """A +4 px x-translation at the 80x128 grid reads +4 -- also after resampling to SA's 64x96."""
    from nett_skrl.brain.aux.expert_flow import ExpertBlockFlow

    g = torch.Generator().manual_seed(0)
    base = torch.rand(1, 3, 80, 128, generator=g)
    shifted = torch.roll(base, shifts=4, dims=3)
    flow, px = expert_flow_px(ExpertBlockFlow(), base, shifted, FLOW_HW)
    inner = (slice(None), 0, slice(16, -16), slice(16, -16))
    assert torch.equal(flow, px)
    assert torch.allclose(px[inner], torch.full_like(px[inner], 4.0))
    small, _ = expert_flow_px(ExpertBlockFlow(), base, shifted, (64, 96))
    assert small.shape[-2:] == (64, 96)
    assert torch.allclose(small[inner], torch.full_like(small[inner], 4.0))


# ---------------------------------------------------------------------------------------------
# 4. the isolation guard
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls", NEW)
def test_guard_raises_on_leaked_mask(cls):
    w = cls(MovingSquare())
    w._ensure(3)

    def leaking(frame):
        with torch.enable_grad():
            return torch.full((frame.shape[0], 2, *frame.shape[2:]), 0.5, requires_grad=True)

    w._model.get_masks = leaking
    with pytest.raises(RuntimeError, match="mask carries requires_grad=True"):
        w.reset()


@pytest.mark.parametrize("cls", NEW)
def test_policy_gradient_never_reaches_the_segmenter(cls):
    w = cls(MovingSquare())
    obs, _ = w.reset()
    image = torch.from_numpy(np.asarray(obs)).float().requires_grad_(False)
    policy = torch.nn.Linear(3, 1)
    policy(image).mean().backward()
    assert policy.weight.grad is not None
    assert all(p.grad is None for p in w._model.parameters())


# ---------------------------------------------------------------------------------------------
# 5. previous frame at episode resets
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls", NEW)
def test_reset_and_done_envs_never_form_a_pair(cls, monkeypatch):
    monkeypatch.setenv("NETT_SEG_CADENCE", "rollout")
    monkeypatch.setenv("NETT_SEG_ROLLOUT_FRAMES", "1000000")      # store, never train
    env = MovingSquare(n=4, done_at={2: [1], 3: [0, 3]})
    w = cls(env)
    w.reset()                                                     # no previous frame anywhere
    for _ in range(4):
        w.step(0)
    valid = torch.stack(w._roll_valid).tolist()
    assert valid == [
        [False, False, False, False],    # reset frame: nothing before it
        [True, True, True, True],        # step 1: within episode
        [True, False, True, True],       # step 2: env 1 ended -> obs is a NEW episode's frame
        [False, True, True, False],      # step 3: envs 0 and 3 ended; env 1 pairs inside its new episode
        [True, True, True, True],        # step 4
    ]
    w.reset()
    w.step(0)
    assert torch.stack(w._roll_valid)[-2].tolist() == [False] * 4   # reset clears the cache


@pytest.mark.parametrize("order", ["env", "time"])
def test_rollout_rebuilds_prev_from_the_sequence(monkeypatch, order):
    """Every trained sample's prev is that env's frame one step earlier (the rollout's first step
    pairs with the cache from the previous rollout), and its valid flag is the stored one."""
    monkeypatch.setenv("NETT_SEG_CADENCE", "rollout")
    monkeypatch.setenv("NETT_SEG_ROLLOUT_FRAMES", "12")
    monkeypatch.setenv("NETT_SEG_ROLLOUT_ORDER", order)
    monkeypatch.setenv("NETT_SEG_BATCH", "4")
    env = MovingSquare(n=4, done_at={5: [2]})
    w = MoTokFlowSeg(env)
    seen = []
    w._opt_step = lambda b: seen.append((b * 255).round().to(torch.uint8)) or 0.0
    raw = [env.reset()[0]]
    w.observation(raw[0])               # direct call, empty cache: no pair yet
    for _ in range(5):
        raw.append(env.step(0)[0])
        w._pending_done = np.isin(np.arange(4), env.done_at.get(env.calls, []))
        w.observation(raw[-1])
    # 6 calls x 4 envs = 24 frames -> two rollouts of 12 (3 calls each), 3 batches of 4 each
    assert len(seen) == 6
    frames = torch.from_numpy(np.stack(raw)).permute(0, 1, 4, 2, 3)   # (T, E, 3, H, W)
    samples = torch.cat(seen)
    for k, s in enumerate(samples):
        roll, j = divmod(k, 12)
        ti, ei = (j // 4, j % 4) if order == "time" else (j % 3, j // 3)
        ti += 3 * roll
        assert torch.equal(s[3:6], frames[ti, ei])
        assert torch.equal(s[6:9], frames[ti, ei])                     # TRAIN_ON=raw target
        if ti > 0:
            assert torch.equal(s[0:3], frames[ti - 1, ei])
        expect_valid = ti > 0 and not (ti == 5 and ei == 2)
        assert bool(s[9, 0, 0] == 255) == expect_valid, (k, ti, ei)


def test_masked_target_is_the_policy_frame(monkeypatch):
    monkeypatch.setenv("NETT_SEG_CADENCE", "rollout")
    monkeypatch.setenv("NETT_SEG_TRAIN_ON", "masked")
    monkeypatch.setenv("NETT_SEG_ROLLOUT_FRAMES", "1000000")
    env = MovingSquare(n=2)
    w = MoTokFlowSeg(env)
    obs, _ = w.reset()
    out = w.step(0)[0]
    assert torch.equal(w._roll_tgt[-1], torch.from_numpy(out).permute(0, 3, 1, 2))
    assert torch.equal(w._roll[-1], torch.from_numpy(env.frames()).permute(0, 3, 1, 2))


# ---------------------------------------------------------------------------------------------
# 6. SA specifics
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [SASeg, SAFlowSeg])
@pytest.mark.parametrize("name,value,match", [
    ("NETT_SEG_FG_SLOT", "1", "no identity"),
    ("NETT_SEG_LR", "1e-4", "paper's optimiser"),
    ("NETT_SEG_WD", "0", "paper's optimiser"),
    ("NETT_SEG_SA_SLOTS", "0", "NETT_SEG_SA_SLOTS"),
])
def test_sa_refuses(monkeypatch, cls, name, value, match):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=match):
        cls(RealEye())


def test_motion_weight_on_the_recon_only_control_is_refused(monkeypatch):
    monkeypatch.setenv("NETT_SEG_MOTION_WEIGHT", "1")
    with pytest.raises(ValueError, match="reconstruction-only control"):
        SASeg(RealEye())
    monkeypatch.setenv("NETT_SEG_MOTION_WEIGHT", "lots")
    for cls in (SAFlowSeg, MoTokFlowSeg):
        with pytest.raises(ValueError, match="NETT_SEG_MOTION_WEIGHT"):
            cls(RealEye())


def test_sa_slot_count_knob(monkeypatch):
    monkeypatch.setenv("NETT_SEG_SA_SLOTS", "6")
    w = SASeg(RealEye())
    w._ensure(3)
    assert w._model.get_masks(torch.rand(1, 3, 280, 448)).shape[1] == 6


def test_sa_paper_shapes_and_params():
    from nett_skrl.brain.aux.slot_attention_ae import SA_RES, SlotAttentionAutoEncoder

    net = SlotAttentionAutoEncoder(num_slots=4)
    assert sum(p.numel() for p in net.parameters()) == 890_308
    sa = net.slot_attention
    assert (sa.dim, sa.iters, sa.eps, sa.mlp[0].out_features) == (64, 3, 1e-8, 128)
    convs = [m for m in net.encoder_cnn if isinstance(m, torch.nn.Conv2d)]
    assert [(c.kernel_size, c.stride, c.out_channels) for c in convs] == [((5, 5), (1, 1), 64)] * 4
    recon, alphas = net.decode(net.prepare(torch.rand(2, 3, 280, 448)))
    assert recon.shape == (2, 3, *SA_RES) and alphas.shape == (2, 4, *SA_RES)


def test_sa_eval_is_deterministic_and_train_samples_noise():
    w = SASeg(RealEye())
    w._ensure(3)
    x = torch.rand(2, 3, 64, 96)
    with torch.no_grad():
        a, b = w._model.get_masks(x), w._model.get_masks(x)
        assert torch.equal(a, b)
        w._model.train()
        c, d = w._model.get_masks(x), w._model.get_masks(x)
        w._model.eval()
    assert not torch.equal(c, d)
    assert "slot_attention.eval_noise" in w._model.state_dict()


def test_sa_selection_is_per_frame():
    """Frame 0's background is slot 0, frame 1's is slot 2: each frame drops ITS largest slot."""
    w = SASeg(RealEye())
    m = torch.zeros(2, 4, 4, 4)
    for frame, (bg, obj) in enumerate(((0, 1), (2, 3))):     # 12-px background, 4-px object
        m[frame, bg] = 1.0
        m[frame, bg, 0] = 0.0
        m[frame, obj, 0] = 1.0
    keep = w._keep_mask(m)
    torch.testing.assert_close(keep[0, 0], m[0, 1])
    torch.testing.assert_close(keep[1, 0], m[1, 3])
    assert w.last_stats["seg/selected_slot_agree"] == 0.5
    for key in ("seg/fg_slot", "seg/fg_area", "seg/mask_rule_not_background", "seg/selected_slot",
                "seg/kept_area"):
        assert key in w.last_stats


def test_sa_lr_schedule_is_table_11():
    w = SASeg(RealEye())
    assert w._lr_at(0) == 0.0
    assert w._lr_at(5_000) == pytest.approx(2e-4 * 0.5 ** 0.05)
    assert w._lr_at(10_000) == pytest.approx(4e-4 * 0.5 ** 0.1)
    assert w._lr_at(110_000) == pytest.approx(4e-4 * 0.5 ** 1.1)


# ---------------------------------------------------------------------------------------------
# 7. persistence through the train -> test boundary
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls", NEW)
def test_test_phase_masks_with_the_trained_segmenter(cls, monkeypatch, tmp_path):
    monkeypatch.setattr(sa_seg, "SA_WARMUP_STEPS", 0)
    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "1")
    monkeypatch.setenv("NETT_SEG_BATCH", "2")
    trained = cls(MovingSquare(n=2))
    trained.reset()
    for _ in range(3):
        trained.step(0)
    assert trained.last_stats["seg/train_steps"] > 0
    path = trained.save_state(tmp_path / state_filename(0))
    trained.train_every = 0
    x = MovingSquare(n=2, seed=5).reset()[0]
    expected = trained.observation(x)
    tested = cls(MovingSquare(n=2))
    tested.bind_phase("test", path)
    np.testing.assert_array_equal(tested.observation(x), expected)
    assert tested.last_stats["seg/frozen"] == 1.0 and tested.last_stats["seg/loaded"] == 1.0
    fresh = cls(MovingSquare(n=2))
    fresh.train_every = 0
    assert not np.array_equal(fresh.observation(x), expected)
    with pytest.raises(ValueError, match="holds a"):
        (MoTokSeg if cls is MoTokFlowSeg else (SAFlowSeg if cls is SASeg else SASeg))(
            MovingSquare(n=2)).bind_phase("test", path)


# ---------------------------------------------------------------------------------------------
# 8. MoTokSeg and GwmSeg are untouched
# ---------------------------------------------------------------------------------------------
def test_existing_segmenters_are_untouched(monkeypatch):
    """Nothing in the new modules patches the existing classes, their labels or their seats.
    (Source: `git diff 71a5fec -- segmentation.py motok_seg.py gwm_seg.py` is empty.)"""
    from nett_skrl.body.wrappers.registry import _WRAPPER_SPECS

    for cls, module in ((SegmentationObservationWrapper, "nett_skrl.body.wrappers.segmentation"),
                        (MoTokSeg, "nett_skrl.body.wrappers.motok_seg"),
                        (GwmSeg, "nett_skrl.body.wrappers.gwm_seg")):
        for name, attr in vars(cls).items():
            fn = getattr(attr, "__func__", attr)
            if callable(fn) and hasattr(fn, "__module__"):
                assert fn.__module__ == module, (cls.__name__, name, fn.__module__)
    assert _WRAPPER_SPECS["motok_seg"] == ("nett_skrl.body.wrappers.motok_seg", "MoTokSeg")
    assert _WRAPPER_SPECS["gwm_seg"] == ("nett_skrl.body.wrappers.gwm_seg", "GwmSeg")
    recipe = {"encoder": "nature_cnn", "cfg": {"trainable": True, "features_dim": 128, "conv_dim": 64,
                                               "spatial_pool": False}, "framestack": False}
    old = {"encoder": "nature_cnn", "cfg": {"trainable": True, "features_dim": 512, "conv_dim": 75}}
    campaign = _campaign(monkeypatch)
    assert campaign.MODELS["MoTok-Seg-UnityRecipe"] == {**recipe, "seg": "motok_seg"}
    assert campaign.MODELS["MoTok-Seg"] == {**old, "framestack": False, "seg": "motok_seg"}
    for q, name in ((2, "CNN2F+GWM-Seg"), (3, "CNN2F+GWM-Seg-Q3"), (5, "CNN2F+GWM-Seg-Q5")):
        assert campaign.MODELS[name] == {**old, "framestack": True, "seg": "gwm_seg",
                                         "seg_after": True, "seg_queries": q}


def test_motokflow_builds_motoknet_unchanged():
    """Same seed -> the same MoTokNet weights and masks as MoTokSeg: the architecture is MoTok's."""
    torch.manual_seed(3)
    a = MoTokSeg(RealEye())
    a._ensure(3)
    torch.manual_seed(3)
    b = MoTokFlowSeg(RealEye())
    b._ensure(3)
    sa, sb = a._model.state_dict(), b._model.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert type(a._optim) is type(b._optim) and a._optim.defaults == b._optim.defaults
    x = torch.rand(2, 3, 280, 448)
    with torch.no_grad():
        assert torch.equal(a._model.get_masks(x), b._model.get_masks(x))
    assert motion_seg.MoTokFlowSeg._ensure is MoTokSeg._ensure
