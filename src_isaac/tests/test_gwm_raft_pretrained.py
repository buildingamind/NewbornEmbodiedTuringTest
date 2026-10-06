"""U35 / DECISIONS §73: frozen pretrained RAFT (torchvision raft_large C_T_V2) as GWM-Seg's flow target.

Checks the exception's scope: frozen three ways, in no optimizer, no gradient into it, not
checkpointed, not needed at test; flow in the expert path's units (gate (i) of FINDINGS
§4dh.44h.118: known-translation warps); the weight-file sha check and its construction print.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.body.wrappers.gwm_spectral import GwmSpectralSeg
from nett_skrl.brain.aux import raft_pretrained as rp
from nett_skrl.brain.aux.expert_flow import ExpertBlockFlow

from test_gwm_spectral import Frames, _spy_steps, defaults  # noqa: F401  (autouse fixture)


def _cached() -> bool:
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights
    except ImportError:
        return False
    return os.path.exists(rp._weights_file(Raft_Large_Weights.C_T_V2))


pytestmark = pytest.mark.skipif(
    not _cached(), reason="pretrained RAFT weights not in the torch hub cache; fetch once with "
    "python -c 'from torchvision.models.optical_flow import raft_large as r, Raft_Large_Weights as W; "
    "r(weights=W.C_T_V2)'")


@pytest.fixture(scope="module")
def raft():
    with torch.random.fork_rng(devices=[]):
        return rp.FrozenRAFTLarge(iters=12, scale=2.0, chunk=256)


def _textured(H=80, W=128, B=2, pad=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, 3, (H + 2 * pad) // 4 + 2, (W + 2 * pad) // 4 + 2, generator=g)
    return F.interpolate(x, size=(H + 2 * pad, W + 2 * pad), mode="bicubic", align_corners=False).clamp(0, 1)


def _shifted_pair(dx, dy, H=80, W=128, pad=16):
    """frame_t1 is frame_t with its content moved by (+dx right, +dy down)."""
    big = _textured(H, W, pad=pad)
    return big[:, :, pad:pad + H, pad:pad + W], big[:, :, pad - dy:pad - dy + H, pad - dx:pad - dx + W]


# ───────────────────────────── frozen, and out of every optimizer ─────────────────────────────

def test_construction_prints_the_sha_and_the_net_is_frozen(capsys):
    r = rp.FrozenRAFTLarge(iters=7, scale=2.0)
    out = capsys.readouterr().out
    assert f"[NETT pretrained] RAFT raft_large C_T_V2 frozen sha256={rp.EXPECTED_SHA256[:12]} iters=7" in out
    assert r.num_params == 5_257_536
    assert not r.net.training
    assert all(not p.requires_grad for p in r.net.parameters())


def test_a_different_weight_file_is_refused(monkeypatch):
    monkeypatch.setattr(rp, "EXPECTED_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="refusing to run a different file"):
        rp.FrozenRAFTLarge()


def test_no_gradient_reaches_or_leaves_the_frozen_net(raft):
    a, b = _shifted_pair(3, -2)
    a.requires_grad_(True)
    with torch.enable_grad():
        f = raft(a, b)
    assert not f.requires_grad and f.grad_fn is None
    assert all(p.grad is None for p in raft.net.parameters())


def test_shim_has_no_parameters_and_no_state(raft):
    shim = rp.PretrainedFlow(raft)
    assert list(shim.parameters()) == [] and shim.state_dict() == {}
    assert shim.forward_single(*_shifted_pair(2, 2)).shape == (2, 2, 80, 128)


def test_wrapper_trains_the_ventral_only_and_never_moves_raft(monkeypatch):
    monkeypatch.setenv("NETT_GWM_FLOW", "raft_pretrained")
    monkeypatch.setenv("NETT_GWM_RAFT_PT_ITERS", "2")
    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "1")
    env = GwmSpectralSeg(Frames())
    env.reset()
    net = env._model.dorsal.raft.net
    raft_ids = {id(p) for p in net.parameters()}
    assert [g["name"] for g in env._optim.param_groups] == ["ventral"]
    assert not raft_ids & {id(p) for g in env._optim.param_groups for p in g["params"]}
    before = [p.detach().clone() for p in net.parameters()]
    calls = _spy_steps(env)
    for _ in range(3):
        env.step(0)
    assert calls and all(c == {"ventral": True} for c in calls)
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))
    assert all(p.grad is None for p in net.parameters()) and not net.training
    st = env.last_stats
    assert st["seg/flow_mode_raft_pretrained"] == 1.0 and st["seg/flow_mode_raft"] == 0.0
    assert np.isfinite(st["seg/recon_loss"]) and np.isfinite(st["seg/flow_absmax"])


def test_raft_is_not_checkpointed_and_test_masks_need_no_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("NETT_GWM_FLOW", "raft_pretrained")
    monkeypatch.setenv("NETT_GWM_RAFT_PT_ITERS", "2")
    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "1")
    env = GwmSpectralSeg(Frames())
    env.reset()
    for _ in range(2):
        env.step(0)
    state = torch.load(env.save_state(tmp_path / "s.pt"), weights_only=False)
    assert not any(k.startswith("dorsal.") for k in state["model"])
    assert len(state["optim"]["param_groups"]) == 1
    probe = np.random.default_rng(9).integers(0, 256, (2, 24, 32, 6), dtype=np.uint8)
    want = env.observation(probe)

    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "0")
    fresh = GwmSpectralSeg(Frames())
    fresh._pending_state = state
    calls = []
    orig = rp.FrozenRAFTLarge.__call__
    monkeypatch.setattr(rp.FrozenRAFTLarge, "__call__", lambda *a, **k: calls.append(1) or orig(*a, **k))
    np.testing.assert_array_equal(fresh.observation(probe), want)
    assert calls == []


# ───────────────────────────── resolution and units ─────────────────────────────

@pytest.mark.parametrize("hw,scale,want", [
    ((80, 128), 2.0, (160, 256)),        # the live eye at the default
    ((80, 128), 1.0, (128, 208)),        # short side floor 128 binds: s = 1.6
    ((24, 32), 2.0, (128, 176)),         # the unit-test frames
    ((160, 256), 2.0, (320, 512)),
])
def test_raft_input_size(hw, scale, want):
    assert rp.raft_input_size(*hw, scale) == want
    assert all(v % 8 == 0 and v >= 128 for v in want)


@pytest.mark.parametrize("dx,dy", [(3, -2), (6, 4), (-10, 5)])
def test_gate_i_known_translation_is_recovered_in_segmenter_pixels(raft, dx, dy):
    f = raft(*_shifted_pair(dx, dy))[:, :, 12:-12, 12:-12]          # interior: no border band
    epe = ((f[:, 0] - dx) ** 2 + (f[:, 1] - dy) ** 2).sqrt().mean().item()
    assert epe < 0.15, epe
    assert abs(f[:, 0].mean().item() - dx) < 0.15 and abs(f[:, 1].mean().item() - dy) < 0.15


def test_pretrained_and_expert_flow_agree_in_sign_and_magnitude(raft):
    a, b = _shifted_pair(6, -4)            # both components on the expert's stride-2 lattice
    pt = raft(a, b)[:, :, 16:-16, 16:-16].flatten(2).median(-1).values.mean(0)
    ex = ExpertBlockFlow(radius=12, stride=2, patch=8).forward_single(a, b)[:, :, 16:-16, 16:-16]
    ex = ex.flatten(2).median(-1).values.mean(0)
    assert torch.equal(torch.sign(pt.round()), torch.sign(ex)), (pt, ex)
    assert (pt - ex).abs().max().item() < 1.0, (pt, ex)
