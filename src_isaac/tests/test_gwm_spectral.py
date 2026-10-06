"""U35: GWM-Seg with K>2 regions, a spectral figure/ground merge, and RAFT-S trained from scratch."""

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.body import Body
from nett_skrl.body.wrappers.framestack import FrameStack
from nett_skrl.body.wrappers.gwm_seg import GwmSeg
from nett_skrl.body.wrappers.gwm_spectral import GwmSpectralSeg, spectral_bipartition
from nett_skrl.brain.aux.raft_small import RAFTSmall, unsupervised_flow_loss

LABELS = {"CNN2F+GWM-Seg-K4-Spectral": "expert", "CNN2F+GWM-Seg-K4-Spectral-RAFTS": "raft_scratch",
          "CNN2F+GWM-Seg-K4-Spectral-RAFTPT": "raft_pretrained"}


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("NETT_SEG_", "NETT_GWM_")) or name == "NETT_EXPERT_FLOW":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NETT_SEG_DEVICE", "cpu")
    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "0")
    monkeypatch.setenv("NETT_SEG_BATCH", "2")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(threads)
    # export_seg_flow writes os.environ directly; never leak it into the rest of the session
    os.environ.pop("NETT_GWM_FLOW", None)


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train
    return campaign_train


class Frames(gym.Env):
    """A different frame every step (so a trained segmenter genuinely moves)."""

    num_envs = 2
    device = "cpu"

    def __init__(self, channels=6):
        self.shape = (self.num_envs, 24, 32, channels)
        self.observation_space = gym.spaces.Box(0, 255, self.shape, np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.rng = np.random.default_rng(3)

    def _obs(self):
        return self.rng.integers(0, 256, self.shape, dtype=np.uint8)

    def reset(self, **kwargs):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), {}


# ───────────────────────────── labels and registration ─────────────────────────────

@pytest.mark.parametrize("label,flow", LABELS.items())
def test_both_labels_construct_and_carry_their_flow(label, flow, campaign, monkeypatch):
    spec = campaign.MODELS[label]
    control = campaign.MODELS["CNN2F+GWM-Seg"]
    differing = {k for k in set(spec) | set(control) if spec.get(k) != control.get(k)}
    assert differing == {"seg", "seg_queries", "seg_flow"}, differing
    assert spec["seg_queries"] == 4 and spec["seg_flow"] == flow
    assert campaign.segmentation_wrappers(spec) == ["framestack", "gwm_seg_spectral"]
    monkeypatch.setenv("NETT_SEG_QUERIES", str(spec["seg_queries"]))
    campaign.export_seg_flow(label, spec)
    assert os.environ["NETT_GWM_FLOW"] == flow
    wrapped = Body(wrappers=campaign.segmentation_wrappers(spec)).wrap(Frames(3))
    seg = wrapped.env
    assert isinstance(seg, GwmSpectralSeg) and isinstance(seg.env, FrameStack)
    assert seg.flow_mode == flow and seg.num_queries == 4
    obs, _ = wrapped.reset()
    assert obs.shape == (2, 6, 24, 32)
    n_dorsal = sum(p.numel() for p in seg._model.dorsal.parameters())
    n_ventral = sum(p.numel() for p in seg._model.ventral.parameters())
    print(f"\n{label}: ventral {n_ventral} params, dorsal({type(seg._model.dorsal).__name__}) {n_dorsal}")
    assert (n_dorsal > 0) == (flow == "raft_scratch")
    assert [g["name"] for g in seg._optim.param_groups] == (["ventral", "raft"] if n_dorsal else ["ventral"])


def test_existing_gwm_labels_are_unchanged(campaign, monkeypatch):
    base = dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 512, "conv_dim": 75},
                framestack=True, seg="gwm_seg", seg_after=True)
    for label, q in (("CNN2F+GWM-Seg", 2), ("CNN2F+GWM-Seg-Q3", 3), ("CNN2F+GWM-Seg-Q5", 5)):
        assert campaign.MODELS[label] == dict(base, seg_queries=q), label
        assert campaign.segmentation_wrappers(campaign.MODELS[label]) == ["framestack", "gwm_seg"]
        campaign.export_seg_flow(label, campaign.MODELS[label])
        assert "NETT_GWM_FLOW" not in os.environ
        monkeypatch.setenv("NETT_GWM_FLOW", "expert")
        with pytest.raises(ValueError, match="has no seg_flow"):
            campaign.export_seg_flow(label, campaign.MODELS[label])
        monkeypatch.delenv("NETT_GWM_FLOW")
    from nett_skrl.body.wrappers.registry import _WRAPPER_SPECS
    assert _WRAPPER_SPECS["gwm_seg"] == ("nett_skrl.body.wrappers.gwm_seg", "GwmSeg")
    assert type(GwmSeg(Frames())).__name__ == "GwmSeg"


def test_a_row_cannot_contradict_the_labels_flow(campaign, monkeypatch):
    monkeypatch.setenv("NETT_GWM_FLOW", "expert")
    with pytest.raises(ValueError, match="contradicts"):
        campaign.export_seg_flow("CNN2F+GWM-Seg-K4-Spectral-RAFTS", campaign.MODELS["CNN2F+GWM-Seg-K4-Spectral-RAFTS"])


# ───────────────────────────── refusals ─────────────────────────────

@pytest.mark.parametrize("env,match", [
    ({"NETT_GWM_FLOW": "raft"}, "expected one of"),
    ({"NETT_SEG_QUERIES": "2"}, ">= 3"),
    ({"NETT_SEG_BACKBONE_LR": "1e-5"}, "does not apply"),
    ({"NETT_SEG_MASK_RULE": "not_background"}, "must be 'auto'"),
    ({"NETT_SEG_FG_SLOT": "1"}, "must be 'auto'"),
    ({"NETT_GWM_FLOW": "raft_scratch", "NETT_EXPERT_FLOW": "1"}, "contradicts"),
    ({"NETT_GWM_FLOW": "expert", "NETT_EXPERT_FLOW": "0"}, "contradicts"),
    ({"NETT_GWM_SPECTRAL_TAU": "0"}, "must be > 0"),
    ({"NETT_GWM_FLOW": "raft_pretrained", "NETT_GWM_RAFT_LR": "1e-4"}, "do not apply to the frozen"),
    ({"NETT_GWM_FLOW": "raft_pretrained", "NETT_GWM_RAFT_ITERS": "8"}, "do not apply to the frozen"),
    ({"NETT_GWM_FLOW": "raft_scratch", "NETT_GWM_RAFT_PT_ITERS": "12"}, "raft_pretrained only"),
    ({"NETT_GWM_RAFT_PT_SCALE": "2"}, "raft_pretrained only"),
    ({"NETT_GWM_FLOW": "raft_pretrained", "NETT_GWM_RAFT_PT_ITERS": "0"}, "must be >= 1"),
    ({"NETT_GWM_FLOW": "raft_pretrained", "NETT_GWM_RAFT_PT_SCALE": "0.5"}, "UPsampling"),
    ({"NETT_GWM_FLOW": "raft_pretrained", "NETT_EXPERT_FLOW": "1"}, "contradicts"),
])
def test_knobs_that_would_run_something_else_are_refused(env, match, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match=match):
        GwmSpectralSeg(Frames())


@pytest.mark.parametrize("env,flow", [
    ({"NETT_EXPERT_FLOW": "1"}, "expert"),                    # the GWM-Seg row template, copied
    ({"NETT_EXPERT_FLOW": "true", "NETT_GWM_FLOW": "expert"}, "expert"),
    ({"NETT_EXPERT_FLOW": "0", "NETT_GWM_FLOW": "raft_scratch"}, "raft_scratch"),
])
def test_an_agreeing_expert_flow_knob_still_constructs(env, flow, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert GwmSpectralSeg(Frames()).flow_mode == flow


# ───────────────────────────── the merge ─────────────────────────────

def test_get_masks_is_the_ventral_forward():
    env = GwmSpectralSeg(Frames())
    env.reset()
    x = torch.rand(3, 3, 24, 32)
    with torch.no_grad():
        got = env._model.get_masks(x)
        ref = env._model.ventral(x).softmax(1)
        feats = env._model.ventral.enc3(env._model.ventral.enc2(env._model.ventral.enc1(x)))
    assert torch.equal(got, ref)
    assert torch.equal(env._model.last_feats, feats)


def _synthetic(groups, areas_px, C=8, seed=0, noise=0.05):
    """Hard K-slot masks on a 1x(sum areas) strip tiled to 16x? grid; slot k has the feature of its group."""
    g = torch.Generator().manual_seed(seed)
    K = len(groups)
    H, W = 16, 32
    lab = torch.zeros(H * W, dtype=torch.long)
    start = 0
    for k, a in enumerate(areas_px):
        lab[start:start + a] = k
        start += a
    assert start == H * W
    lab = lab[torch.randperm(H * W, generator=g)].view(H, W)
    masks = F.one_hot(lab, K).permute(2, 0, 1).float().unsqueeze(0)          # (1,K,H,W)
    protos = torch.randn(max(groups) + 1, C, generator=g)
    f = protos[torch.tensor(groups)][lab]                                    # (H,W,C)
    f = f + noise * torch.randn(f.shape, generator=g)
    return masks, f.permute(2, 0, 1).unsqueeze(0)                            # feats at full res


@pytest.mark.parametrize("groups,areas,expect_fg", [
    ([0, 0, 1, 1], [40, 60, 200, 212], [True, True, False, False]),      # small pair = figure
    ([0, 1, 0, 1], [40, 200, 60, 212], [True, False, True, False]),      # not an order artifact
    ([1, 0, 0, 0], [300, 70, 70, 72], [False, True, True, True]),        # 3 vs 1: the 3 are smaller
])
def test_spectral_merge_groups_the_right_segments(groups, areas, expect_fg):
    masks, feats = _synthetic(groups, areas)
    fg, ncut, eig2, gap = spectral_bipartition(masks, feats, tau=0.1)
    assert fg[0].tolist() == expect_fg
    assert float(gap[0]) == pytest.approx(0.0, abs=1e-9)     # the sweep found the exact min Ncut
    assert float(ncut[0]) < 0.05


def test_spectral_merge_is_deterministic_and_slot_equivariant():
    masks, feats = _synthetic([0, 1, 0, 1, 1], [50, 100, 60, 150, 152], seed=4)
    runs = [spectral_bipartition(masks, feats, 0.1) for _ in range(3)]
    for r in runs[1:]:
        for a, b in zip(runs[0], r):
            assert torch.equal(a, b)
    perm = torch.tensor([3, 0, 4, 1, 2])
    fg_p = spectral_bipartition(masks[:, perm], feats, 0.1)[0]
    assert torch.equal(fg_p[0], runs[0][0][0][perm])
    # both sides non-empty, always (random masks and features, many images)
    m = torch.rand(64, 4, 12, 16).softmax(1)
    fg = spectral_bipartition(m, torch.randn(64, 32, 3, 4), 0.1)[0]
    assert fg.any(1).all() and (~fg).any(1).all()


def test_kept_mask_is_the_figure_slots_and_every_stat_is_emitted():
    env = GwmSpectralSeg(Frames())
    raw, _ = env.env.reset()
    obs = env.observation(raw)
    st = env.last_stats
    for k in ("seg/fg_slot", "seg/fg_area", "seg/mask_rule_not_background", "seg/selected_slot",
              "seg/kept_area", "seg/spectral_fg_rule_area", "seg/spectral_fg_slots",
              "seg/spectral_ncut", "seg/spectral_ncut_exact_gap", "seg/spectral_eig2",
              *(f"seg/spectral_fg_frac_slot{k}" for k in range(4))):
        assert k in st and np.isfinite(st[k]), k
    assert st["seg/spectral_fg_rule_area"] == 1.0 and st["seg/selected_slot"] == -1.0
    x = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        masks = env._model.get_masks(x[:, -3:])
    fg = spectral_bipartition(masks, env._model.last_feats, env.tau)[0]
    keep = (masks * fg[:, :, None, None].float()).sum(1, keepdim=True)
    expect = ((x[:, -3:] * keep).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    np.testing.assert_array_equal(obs[..., -3:], expect)


# ───────────────────────────── RAFT-S ─────────────────────────────

def test_raft_small_forward_backward_is_finite_and_sized_like_the_reference():
    m = RAFTSmall(iters=3)
    n = sum(p.numel() for p in m.parameters())
    print(f"\nRAFT-S params {n}")
    assert n == 990_162          # reference RAFT-S is ~1.0M; pins the architecture
    a, b = torch.rand(2, 3, 40, 60), torch.rand(2, 3, 40, 60)     # not a multiple of 8: pad path
    f12, f21 = m(a, b), m(b, a)
    assert len(f12) == 3 and f12[-1].shape == (2, 2, 40, 60)
    loss, sc = unsupervised_flow_loss(a, b, f12, f21)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(np.isfinite(v) for v in sc.values())


def test_photometric_loss_on_a_translating_texture_reports_the_recovered_flow():
    """~30 AdamW steps on one texture translated by (dx, dy) = (+3, +1). Asserts only that the
    unsupervised loss went down; the recovered forward AND backward flow are printed, because a
    falling loss here can be an input-independent bias (see the report), not matching."""
    torch.manual_seed(0)
    m = RAFTSmall(iters=4)
    opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=1e-4)     # the wrapper's default
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(30):
        a = F.interpolate(torch.rand(2, 3, 20, 32, generator=g), size=(80, 128), mode="bilinear",
                          align_corners=False)
        b = torch.roll(a, shifts=(1, 3), dims=(2, 3))
        f12, f21 = m(a, b), m(b, a)
        loss, sc = unsupervised_flow_loss(a, b, f12, f21, use_occlusion=False)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    fw = f12[-1].detach()[..., 10:-10, 10:-10].mean(dim=(0, 2, 3)).tolist()
    bw = f21[-1].detach()[..., 10:-10, 10:-10].mean(dim=(0, 2, 3)).tolist()
    print(f"\nloss first5 {np.mean(losses[:5]):.3f} last5 {np.mean(losses[-5:]):.3f}; "
          f"recovered fwd (dx,dy)=({fw[0]:+.2f},{fw[1]:+.2f}) [true +3,+1], "
          f"bwd ({bw[0]:+.2f},{bw[1]:+.2f}) [true -3,-1]")
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def _spy_steps(env):
    """Record, per optimiser step, which groups had gradients."""
    calls = []
    orig = env._optim.step

    def step(*a, **k):
        calls.append({g["name"]: any(p.grad is not None for p in g["params"]) for g in env._optim.param_groups})
        return orig(*a, **k)

    env._optim.step = step
    return calls


def test_raft_trains_only_on_its_own_loss_and_the_ventral_only_on_detached_flow(monkeypatch):
    monkeypatch.setenv("NETT_GWM_FLOW", "raft_scratch")
    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "1")
    monkeypatch.setenv("NETT_GWM_RAFT_ITERS", "2")
    monkeypatch.setenv("NETT_GWM_RAFT_BATCH", "3")
    env = GwmSpectralSeg(Frames())
    env.reset()
    calls = _spy_steps(env)
    raft0 = [p.detach().clone() for p in env._model.dorsal.parameters()]
    ven0 = [p.detach().clone() for p in env._model.ventral.parameters()]
    for _ in range(3):
        env.step(0)
    st = env.last_stats
    assert st["seg/train_steps"] >= 1 and st["seg/raft_steps"] == st["seg/train_steps"]
    assert calls and len(calls) == 2 * st["seg/train_steps"]
    assert calls[0::2] == [{"ventral": False, "raft": True}] * len(calls[0::2])     # RAFT step
    assert calls[1::2] == [{"ventral": True, "raft": False}] * len(calls[1::2])     # segmenter step
    assert any(not torch.equal(a, b) for a, b in zip(raft0, env._model.dorsal.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(ven0, env._model.ventral.parameters()))
    for k in ("seg/raft_loss", "seg/raft_census", "seg/raft_flow_absmax", "seg/flow_absmax", "seg/recon_loss"):
        assert np.isfinite(st[k]), k
    assert not env._model.training


def test_expert_arm_trains_the_ventral_with_no_dorsal_group(monkeypatch):
    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "1")
    env = GwmSpectralSeg(Frames())
    env.reset()
    calls = _spy_steps(env)
    for _ in range(3):
        env.step(0)
    assert calls and all(c == {"ventral": True} for c in calls)
    assert np.isfinite(env.last_stats["seg/recon_loss"])


def test_raft_state_round_trips_and_test_masks_need_no_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("NETT_GWM_FLOW", "raft_scratch")
    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "1")
    monkeypatch.setenv("NETT_GWM_RAFT_ITERS", "2")
    env = GwmSpectralSeg(Frames())
    env.reset()
    for _ in range(3):
        env.step(0)
    path = env.save_state(tmp_path / "s.pt")
    state = torch.load(path, weights_only=False)
    assert any(k.startswith("dorsal.") for k in state["model"])
    assert len(state["optim"]["param_groups"]) == 2
    probe = np.random.default_rng(9).integers(0, 256, (2, 24, 32, 6), dtype=np.uint8)
    want = env.observation(probe)

    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "0")
    fresh = GwmSpectralSeg(Frames())
    fresh._pending_state = state
    calls = []
    fresh_forward = RAFTSmall.forward
    monkeypatch.setattr(RAFTSmall, "forward", lambda *a, **k: calls.append(1) or fresh_forward(*a, **k))
    np.testing.assert_array_equal(fresh.observation(probe), want)
    assert calls == []          # masking at test runs the ventral only
