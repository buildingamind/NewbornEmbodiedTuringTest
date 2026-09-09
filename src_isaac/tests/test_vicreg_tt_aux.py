"""CPU acceptance tests pinning temporal VICReg to its one-factor contrast."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

# Resolve THIS worktree, never the fleet's pinned runtime clone.
_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "examples"))

from campaign_train import MODELS, VIT_CFG, VIVIT_CFG
from nett_skrl.brain.aux import vicreg_aux, vicreg_tt_aux
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES, AuxLossPPO, PPO
from nett_skrl.brain.aux.simclr_aux import _augment
from nett_skrl.brain.aux.vicreg_aux import VICRegAuxLoss, VICRegExpander, vicreg_loss
from nett_skrl.brain.aux.vicreg_tt_aux import VICRegTemporalAuxLoss, vicreg_terms
from nett_skrl.brain.registry import encoder_mapping


@pytest.fixture(autouse=True)
def cpu_defaults(monkeypatch):
    for name in (
        "NETT_AUX_BATCH", "NETT_AUX_VICREG_TT_OFFSETS",
        "NETT_VICREG_INV", "NETT_VICREG_VAR", "NETT_VICREG_COV",
    ):
        monkeypatch.delenv(name, raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            yield
    finally:
        torch.set_num_threads(threads)


class FakeMemory:
    def __init__(self, raw, *, filled=True, memory_index=0):
        self.tensors = {"observations": raw}
        self.memory_size = raw.shape[0]
        self.filled = filled
        self.memory_index = memory_index

    def get_tensor_by_name(self, name):
        raise AssertionError("Never transfer the whole observation buffer!")


class IdentityEncoder(nn.Module):
    """Record identifiable (t, env) entries and retain a trainable backbone."""

    def __init__(self, channels=6):
        super().__init__()
        self.features_dim = channels
        self.scale = nn.Parameter(torch.ones(channels))
        self.views = []

    def _prepare_image(self, observations):
        assert not torch.is_grad_enabled()
        return observations

    def encode_prepared(self, view):
        assert torch.is_grad_enabled()
        self.views.append(view.detach().clone())
        return torch.sin(view / 1000 * torch.arange(1, self.features_dim + 1)) * self.scale


def identity_memory(t_max=16, num_envs=3, channels=6, **kwargs):
    ids = torch.arange(t_max)[:, None] * 1000 + torch.arange(num_envs)[None, :]
    return FakeMemory(ids[..., None].expand(-1, -1, channels).float().clone(), **kwargs)


def fixed_draws(monkeypatch, draws):
    pending = iter(draws)
    calls = []

    def randint(high, size):
        value = next(pending)
        assert size == () and 0 <= value < high
        calls.append(high)
        return torch.tensor(value)

    monkeypatch.setattr(torch, "randint", randint)
    return calls


@pytest.fixture
def identity_augmentation(monkeypatch):
    def augment(view, *, scale_min, jitter):
        assert torch.is_grad_enabled()
        return view

    monkeypatch.setattr(vicreg_tt_aux, "_augment", augment)


def test_incumbent_head_geometry_and_reused_functions():
    aux = VICRegTemporalAuxLoss(IdentityEncoder(channels=512))
    incumbent_head = VICRegExpander(512, 512, 512)
    assert type(aux.head) is VICRegExpander
    assert sum(p.numel() for p in aux.head.parameters()) == 790_016
    assert aux.head.state_dict().keys() == incumbent_head.state_dict().keys()
    assert next(aux.head.parameters()).device.type == "cpu"
    assert vicreg_tt_aux.vicreg_loss is vicreg_loss
    assert vicreg_tt_aux._off_diagonal is vicreg_aux._off_diagonal
    assert vicreg_tt_aux._augment is _augment


def test_defaults_and_coefficient_overrides_match_incumbent(monkeypatch):
    encoder = IdentityEncoder()
    aux = VICRegTemporalAuxLoss(encoder)
    incumbent = VICRegAuxLoss(encoder)
    assert (aux.inv, aux.var, aux.cov) == (incumbent.inv, incumbent.var, incumbent.cov) == (3, 30, 10)
    assert aux.max_samples == incumbent.max_samples == 48
    assert aux.crop_scale_min == incumbent.crop_scale_min == 0.5
    assert aux.jitter == incumbent.jitter == 0.2
    assert aux.offsets == (8,)
    assert aux.last_terms is None
    for name, value in (("INV", "1.5"), ("VAR", "7"), ("COV", "0.25")):
        monkeypatch.setenv(f"NETT_VICREG_{name}", value)
    monkeypatch.setenv("NETT_AUX_BATCH", "8")
    aux = VICRegTemporalAuxLoss(encoder, crop_scale_min=0.7, jitter=0.1, max_samples=10)
    incumbent = VICRegAuxLoss(encoder)
    assert (aux.inv, aux.var, aux.cov) == (incumbent.inv, incumbent.var, incumbent.cov) == (1.5, 7, 0.25)
    assert aux.max_samples == incumbent.max_samples == 8
    assert (aux.crop_scale_min, aux.jitter) == (0.7, 0.1)
    monkeypatch.delenv("NETT_AUX_BATCH")
    assert VICRegTemporalAuxLoss(encoder, max_samples=10).max_samples == 10


@pytest.mark.parametrize(
    "filled,memory_index,batch,start,expected_batch,expected_highs",
    [(True, 1, 4, 8, 4, [3, 11]), (False, 9, 4, 1, 4, [3, 4]),
     (False, 7, 48, 0, 5, [3]), (False, 6, 4, 0, 4, [3, 1])],
)
def test_temporal_contiguous_one_stream_without_whole_buffer_transfer(
    monkeypatch, identity_augmentation,
    filled, memory_index, batch, start, expected_batch, expected_highs,
):
    # Offset pinned to 2 so the tiny fake buffers below can hold a window; the
    # PRODUCTION default is 8 and is pinned by test_default_offset_is_calibrated.
    monkeypatch.setenv("NETT_AUX_VICREG_TT_OFFSETS", "2")
    monkeypatch.setenv("NETT_AUX_BATCH", str(batch))
    calls = fixed_draws(monkeypatch, [2, start] if len(expected_highs) == 2 else [2])
    encoder = IdentityEncoder()
    aux = VICRegTemporalAuxLoss(encoder)
    aux.attach_memory(identity_memory(filled=filled, memory_index=memory_index))
    loss = aux.compute(encoder, torch.full((1,), float("nan")))
    assert torch.isfinite(loss)
    assert calls == expected_highs
    assert aux.num_frames == 2
    assert len(encoder.views) == 2
    anchor_t = torch.arange(start, start + expected_batch)
    for view, offset in zip(encoder.views, (0, 2)):
        ids = view[:, 0].long()
        assert torch.equal(ids % 1000, torch.full((expected_batch,), 2))
        assert torch.equal(ids // 1000, anchor_t + offset)
        assert torch.equal(view, view[:, :1].expand_as(view))
    assert torch.equal(encoder.views[1] - encoder.views[0], torch.full_like(encoder.views[0], 2000))
    assert isinstance(aux.last_terms, tuple) and all(isinstance(x, float) for x in aux.last_terms)
    torch.testing.assert_close(loss.detach(), torch.tensor(sum(
        term * coefficient for term, coefficient in zip(aux.last_terms, (aux.inv, aux.var, aux.cov))
    )))


def test_views_augmented_independently_outside_no_grad(monkeypatch):
    encoder = IdentityEncoder()
    aux = VICRegTemporalAuxLoss(encoder, crop_scale_min=0.7, jitter=0.1)
    aux.attach_memory(identity_memory())
    calls = []

    def augment(view, *, scale_min, jitter):
        assert torch.is_grad_enabled()
        assert (scale_min, jitter) == (0.7, 0.1)
        draw = torch.rand(())
        calls.append((view.clone(), draw))
        return view + draw

    monkeypatch.setattr(vicreg_tt_aux, "_augment", augment)
    aux.compute(encoder, torch.empty(0))
    assert len(calls) == 2
    assert calls[0][1] != calls[1][1]
    for encoded, (prepared, draw) in zip(encoder.views, calls):
        torch.testing.assert_close(encoded, prepared + draw)


def test_backbone_receives_nonzero_gradients_with_real_augmentation(monkeypatch):
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    space = gym.spaces.Box(0, 255, shape=(8, 12, 6), dtype=np.uint8)
    encoder = encoder_mapping["simclr_cltt"](space, features_dim=32, conv_dim=8).cpu()
    # This encoder's legacy reward projector is unused by encode_prepared.
    encoder.projector.requires_grad_(False)
    aux = VICRegTemporalAuxLoss(encoder)
    # 20 timesteps, not 12: this test runs at the PRODUCTION offset (8) rather than
    # pinning a small one, so the buffer must hold an anchor window plus the offset.
    raw = torch.randint(256, (20, 2, int(np.prod(space.shape))), dtype=torch.uint8)
    aux.attach_memory(FakeMemory(raw))
    fixed_draws(monkeypatch, [1, 2])
    aux.compute(encoder, torch.empty(0)).backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in encoder.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0 for p in aux.head.parameters())


@pytest.mark.parametrize("t_max", [0, 2, 3])
def test_too_few_anchors_raises(monkeypatch, t_max):
    monkeypatch.setenv("NETT_AUX_VICREG_TT_OFFSETS", "2")
    encoder = IdentityEncoder()
    aux = VICRegTemporalAuxLoss(encoder)
    aux.attach_memory(identity_memory(t_max))
    with pytest.raises(ValueError, match="variance/covariance terms are degenerate at B=1"):
        aux.compute(encoder, torch.empty(0))


def test_batch_cap_of_one_raises(monkeypatch):
    monkeypatch.setenv("NETT_AUX_VICREG_TT_OFFSETS", "2")
    monkeypatch.setenv("NETT_AUX_BATCH", "1")
    encoder = IdentityEncoder()
    aux = VICRegTemporalAuxLoss(encoder)
    aux.attach_memory(identity_memory())
    with pytest.raises(ValueError, match="B_eff >= 2, got 1"):
        aux.compute(encoder, torch.empty(0))


def test_memory_required():
    encoder = IdentityEncoder()
    with pytest.raises(RuntimeError, match="attach_memory"):
        VICRegTemporalAuxLoss(encoder).compute(encoder, torch.empty(0))


def test_unaligned_offsets_refused_and_logged_only_once(monkeypatch):
    monkeypatch.setenv("NETT_AUX_VICREG_TT_OFFSETS", "2")
    encoder = IdentityEncoder(channels=9)
    aux = VICRegTemporalAuxLoss(encoder)
    aux.attach_memory(identity_memory(channels=9))
    messages = []
    monkeypatch.setattr(vicreg_tt_aux.logger, "info", lambda *args: messages.append(args))
    for _ in range(2):
        with pytest.raises(ValueError, match="multiples of T=3"):
            aux.compute(encoder, torch.empty(0))
        assert not encoder.views
        assert len(messages) == 1
        assert messages[0][0].startswith("VICRegTemporalAuxLoss:")
        assert messages[0][1:] == ((2,), 3)


@pytest.mark.parametrize("channels,offsets", [(9, "3, 6"), (3, "1,2"), (6, "2,4")])
def test_extra_offsets_sum_imported_loss_and_diagnostics(
    monkeypatch, identity_augmentation, channels, offsets,
):
    monkeypatch.setenv("NETT_AUX_VICREG_TT_OFFSETS", offsets)
    encoder = IdentityEncoder(channels=channels)
    aux = VICRegTemporalAuxLoss(encoder)
    aux.attach_memory(identity_memory(channels=channels))
    pairs = []

    def record_loss(z1, z2, i, v, c):
        pairs.append((z1, z2))
        return vicreg_loss(z1, z2, i, v, c)

    monkeypatch.setattr(vicreg_tt_aux, "vicreg_loss", record_loss)
    loss = aux.compute(encoder, torch.empty(0))
    assert aux.num_frames == channels // 3
    assert len(encoder.views) == 3 and len(pairs) == 2
    assert pairs[0][0] is pairs[1][0]  # Encode and expand the anchor only once.
    for view, offset in zip(encoder.views[1:], aux.offsets):
        torch.testing.assert_close(view - encoder.views[0], torch.full_like(view, offset * 1000))
    expected = sum(vicreg_loss(z1, z2, aux.inv, aux.var, aux.cov) for z1, z2 in pairs)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    expected_terms = torch.stack([torch.stack(vicreg_terms(*pair)) for pair in pairs]).sum(0)
    torch.testing.assert_close(torch.tensor(aux.last_terms), expected_terms)


@pytest.mark.parametrize("offsets", ["", "0", "-2", "two", "2.5", "2,", ",2", "2,,4"])
def test_invalid_offsets_raise(monkeypatch, offsets):
    monkeypatch.setenv("NETT_AUX_VICREG_TT_OFFSETS", offsets)
    with pytest.raises(ValueError, match="NETT_AUX_VICREG_TT_OFFSETS"):
        VICRegTemporalAuxLoss(IdentityEncoder())


@pytest.mark.parametrize("batch,dim,eps", [(2, 7, 1e-4), (17, 32, 1e-4), (8, 12, 1e-3)])
def test_term_decomposition_matches_incumbent(batch, dim, eps):
    z1, z2 = torch.randn(batch, dim), torch.randn(batch, dim)
    inv, var, cov = vicreg_terms(z1, z2, eps)
    i, v, c = 3.0, 30.0, 10.0
    assert torch.allclose(inv * i + var * v + cov * c, vicreg_loss(z1, z2, i, v, c, eps), rtol=1e-7, atol=0)


def test_registry_and_campaign_match_incumbents():
    assert "vicreg_tt" in AUX_LOSSES
    aux = AUX_LOSSES["vicreg_tt"](IdentityEncoder())
    assert isinstance(aux, VICRegTemporalAuxLoss) and aux.needs_memory is True
    for prefix in ("ViT", "ViViT"):
        incumbent = MODELS[f"{prefix}+VICReg"]
        candidate = MODELS[f"{prefix}-VICReg-TT"]
        assert candidate["aux"] == "vicreg_tt" and incumbent["aux"] == "vicreg"
        for field in ("encoder", "cfg", "framestack"):
            assert candidate[field] == incumbent[field]
        assert candidate["framestack"] is (prefix == "ViViT")
        assert candidate["aux_weight"] == incumbent.get("aux_weight", 1.0) == 1.0
        assert "reward" not in candidate


@pytest.mark.parametrize("has_memory", [True, False])
def test_ppo_memory_hook_unchanged(monkeypatch, has_memory):
    encoder = IdentityEncoder()
    memory = identity_memory() if has_memory else None
    registrations = []

    def init_ppo(self, *args, **kwargs):
        self.memory = kwargs["memory"]
        self.policy = SimpleNamespace(encoder=encoder)

        def add_param_group(group):
            assert self._aux._memory is memory
            registrations.append(group)

        self.optimizer = SimpleNamespace(add_param_group=add_param_group)

    monkeypatch.setattr(PPO, "__init__", init_ppo)
    if has_memory:
        agent = AuxLossPPO(aux_loss="vicreg_tt", aux_weight=1.0, memory=memory)
        assert agent._aux._memory is memory
        assert registrations[0]["params"] == list(agent._aux.head.parameters())
    else:
        with pytest.raises(ValueError, match="draws its own temporal windows"):
            AuxLossPPO(aux_loss="vicreg_tt", aux_weight=1.0, memory=None)
        assert not registrations


def test_incumbent_still_augments_same_prepared_tensor_twice(monkeypatch):
    encoder = IdentityEncoder()
    incumbent = VICRegAuxLoss(encoder)
    prepared = identity_memory().tensors["observations"][:, 0]
    calls = []

    def augment(view, **kwargs):
        assert torch.is_grad_enabled()
        calls.append(view)
        return view

    monkeypatch.setattr(vicreg_aux, "_augment", augment)
    assert torch.isfinite(incumbent.compute(encoder, prepared))
    assert len(calls) == 2 and calls[0] is calls[1] is prepared


def test_incumbent_file_byte_identical_to_base():
    """vicreg_tt is a ONE-FACTOR change: the incumbent it is measured against must not move.

    Two scored arms (ViT+VICReg, ViViT+VICReg) carry the `vicreg` label. If this file
    drifts, every published number under that label silently changes definition and the
    vicreg-tt+ contrast stops identifying the view construction.

    SKIPS rather than errors when origin/feat/isaac is not fetched (a shallow or
    remote-less clone). A skip is honest about not having checked; a hard error here would
    be noise, and quietly passing would be the one outcome that must never happen.
    """
    relative = "src_isaac/nett_skrl/brain/aux/vicreg_aux.py"
    baseline = subprocess.run(
        ["git", "show", f"origin/feat/isaac:{relative}"],
        cwd=_SRC.parent, capture_output=True,
    )
    if baseline.returncode != 0:
        pytest.skip(
            "origin/feat/isaac not available in this clone, so the incumbent could not be "
            f"compared against its base: {baseline.stderr.decode(errors='replace').strip()}"
        )
    assert (_SRC.parent / relative).read_bytes() == baseline.stdout
    result = subprocess.run(
        ["git", "diff", "--stat", "origin/feat/isaac", "--", relative],
        cwd=_SRC.parent, check=True, capture_output=True,
    )
    assert result.stdout == b""


@pytest.mark.parametrize(
    "name,cfg,expected",
    [("compact_vivit", VIVIT_CFG, 694_016), ("compact_vit", VIT_CFG, 804_320)],
)
def test_measured_encoder_parameters_at_live_eye(name, cfg, expected):
    # eye_resolution=(128, 80) is W x H; the live HWC space has this shape.
    space = gym.spaces.Box(low=0, high=255, shape=(80, 128, 6), dtype=np.uint8)
    encoder = encoder_mapping[name](space, **cfg).cpu()
    assert sum(p.numel() for p in encoder.parameters()) == expected


def test_default_offset_is_calibrated_to_the_rollout():
    """The production default is 8, not cltt_ref's 2, and the reason is measured.

    At offset 2 the median viewpoint change between the paired frames is 4.0 deg
    -- under two pixels at the live eye's 2.34 deg/px field average, and far
    inside _augment's own crop (41% linear zoom at scale_min=0.5). The temporal
    signal would be invisible under the augmentation and a null would carry no
    information about temporal pairing. See the module docstring and
    notes/researcher/vicreg-tt-plus.md section 6a.

    This is a REGISTERED EXPERIMENTAL PARAMETER, not a style choice: an arm
    launched at a different offset is not the arm the falsifier was written
    against. Change it only with a fresh measurement and a note update.
    """
    assert VICRegTemporalAuxLoss(IdentityEncoder()).offsets == (8,)


def test_default_offset_is_aligned_to_the_live_stack_depth():
    """8 must survive the shared-frame guard at the live framestack depth.

    ViViT-VICReg-TT runs framestack=True with num_frames=2, so T=2 and the guard
    requires offsets that are multiples of 2. This is the arithmetic the queue
    row depends on; if T ever changes, this test fails rather than the arm
    silently raising mid-run.
    """
    for live_T in (1, 2):
        assert 8 % live_T == 0
