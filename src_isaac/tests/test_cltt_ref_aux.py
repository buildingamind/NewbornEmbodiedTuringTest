"""CPU acceptance tests for reference CLTT: temporal identity and gradient controls."""

from pathlib import Path
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

from campaign_train import MODELS, VIT_CFG
from nett_skrl.brain.aux.cltt_aux import CLTTAuxLoss
from nett_skrl.brain.aux.cltt_ref_aux import (
    CLTTReferenceAuxLoss,
    CLTTReferenceProjectionHead,
)
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES, AuxLossPPO, PPO
from nett_skrl.brain.aux.simclr_aux import SimCLRProjectionHead, nt_xent
from nett_skrl.brain.registry import encoder_mapping


@pytest.fixture(autouse=True)
def cpu_defaults(monkeypatch):
    for name in ("NETT_AUX_BATCH", "NETT_AUX_CLTT_REF_OFFSETS", "NETT_AUX_CLTT_REF_TEMP"):
        monkeypatch.delenv(name, raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
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
    """Preparation is identity; record exactly which (t, env) entries are encoded."""

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
        # Vary the features nonlinearly so different temporal offsets have
        # different geometry, even though raw entries each repeat one identity.
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


def test_reference_head_geometry():
    head = CLTTReferenceProjectionHead(512)
    assert [type(layer) for layer in head.net] == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Linear]
    assert (head.net[0].in_features, head.net[0].out_features) == (512, 512)
    assert head.net[1].num_features == 512
    assert (head.net[3].in_features, head.net[3].out_features) == (512, 128)
    assert head.net[3].bias is None
    assert sum(p.numel() for p in head.parameters()) == 329_216
    torch.testing.assert_close(head(torch.randn(4, 512)).norm(dim=-1), torch.ones(4))


@pytest.mark.parametrize(
    "filled,memory_index,batch,start,expected_batch,expected_highs",
    [(True, 1, 4, 8, 4, [3, 9]), (False, 9, 4, 1, 4, [3, 2]),
     (False, 7, 96, 0, 3, [3]), (False, 8, 4, 0, 4, [3, 1])],
)
def test_one_stream_contiguous_windows_without_whole_buffer_transfer(
    monkeypatch, filled, memory_index, batch, start, expected_batch, expected_highs,
):
    monkeypatch.setenv("NETT_AUX_BATCH", str(batch))
    draws = [2, start] if len(expected_highs) == 2 else [2]
    calls = fixed_draws(monkeypatch, draws)
    encoder = IdentityEncoder()
    aux = CLTTReferenceAuxLoss(encoder)
    aux.attach_memory(identity_memory(filled=filled, memory_index=memory_index))
    # Poisoned PPO observations prove the loss draws from its attached memory.
    loss = aux.compute(encoder, torch.full((1,), float("nan")))
    assert torch.isfinite(loss)
    assert calls == expected_highs
    assert aux.num_frames == 2
    assert aux.offsets == (2, 4)
    assert len(encoder.views) == 3
    anchor_t = torch.arange(start, start + expected_batch)
    for view, offset in zip(encoder.views, (0, 2, 4)):
        ids = view[:, 0].long()
        assert torch.equal(ids % 1000, torch.full((expected_batch,), 2))
        assert torch.equal(ids // 1000, anchor_t + offset)
        assert torch.equal(view, view[:, :1].expand_as(view))


def test_three_frame_default_offsets_refused_after_logging(monkeypatch):
    encoder = IdentityEncoder(channels=9)
    aux = CLTTReferenceAuxLoss(encoder)
    aux.attach_memory(identity_memory(channels=9))
    messages = []
    monkeypatch.setattr("nett_skrl.brain.aux.cltt_ref_aux.logger.info", lambda *args: messages.append(args))
    # A caught refusal must not let a later call bypass the bound; log only once.
    for _ in range(2):
        with pytest.raises(ValueError) as error:
            aux.compute(encoder, torch.empty(0))
        message = str(error.value)
        assert "T=3" in message and "offsets=(2, 4)" in message
        assert "share a literally identical frame" in message
        assert "multiples of T=3" in message
        assert len(messages) == 1
        # ⛔ The batch must be in this line. Every level claim about NT-Xent depends on
        # B (chance is 2*ln(2B-1)), and the line used to carry offsets and stack depth
        # and not B -- which is how a read protocol got published against an assumed
        # B=96 when the realised B was ~45.
        assert messages[0][1:3] == ((2, 4), 3)
        assert len(messages[0][1:]) > 2, "offsets and T alone leave the level unbacked"
        fmt = messages[0][0] % messages[0][1:]
        assert "batch B=" in fmt and "chance" in fmt
        assert not encoder.views  # Refuse before any backbone encoding.


@pytest.mark.parametrize("channels,offsets", [(9, "3,6"), (3, "1,2")])
def test_offsets_aligned_to_stack_depth_are_accepted(monkeypatch, channels, offsets):
    monkeypatch.setenv("NETT_AUX_CLTT_REF_OFFSETS", offsets)
    encoder = IdentityEncoder(channels=channels)
    aux = CLTTReferenceAuxLoss(encoder)
    aux.attach_memory(identity_memory(channels=channels))
    assert torch.isfinite(aux.compute(encoder, torch.empty(0)))
    assert aux.num_frames == channels // 3
    assert len(encoder.views) == 3


def test_backbone_receives_nonzero_gradients(monkeypatch):
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    space = gym.spaces.Box(0, 255, shape=(8, 12, 6), dtype=np.uint8)
    encoder = encoder_mapping["simclr_cltt"](space, features_dim=32, conv_dim=8).cpu()
    # The legacy CLTTReward projector is not in forward()/encode_prepared().
    # Freeze ONLY that unused branch in this fixture; all backbone/fc params
    # remain trainable. The new loss owns the reference projector separately.
    encoder.projector.requires_grad_(False)
    aux = CLTTReferenceAuxLoss(encoder)
    raw = torch.randint(256, (12, 2, int(np.prod(space.shape))), dtype=torch.uint8)
    aux.attach_memory(FakeMemory(raw))
    fixed_draws(monkeypatch, [1, 2])
    loss = aux.compute(encoder, torch.empty(0))
    loss.backward()
    for name, parameter in encoder.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert parameter.grad.norm() > 0, name
    for name, parameter in aux.head.named_parameters():
        assert parameter.grad is not None and parameter.grad.norm() > 0, name


def test_both_offsets_are_summed(monkeypatch):
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    encoder = IdentityEncoder()
    aux = CLTTReferenceAuxLoss(encoder)
    memory = identity_memory()
    aux.attach_memory(memory)
    monkeypatch.setenv("NETT_AUX_CLTT_REF_OFFSETS", "2")
    single = CLTTReferenceAuxLoss(encoder)
    single.head.load_state_dict(aux.head.state_dict())
    single.attach_memory(memory)
    fixed_draws(monkeypatch, [1, 3, 1, 3])
    one_loss = single.compute(encoder, torch.empty(0))
    two_loss = aux.compute(encoder, torch.empty(0))
    raw = memory.tensors["observations"]
    z = [aux.head(encoder.encode_prepared(raw[3 + k:7 + k, 1])) for k in (0, 2, 4)]
    terms = [nt_xent(z[0], positive, 0.5) for positive in z[1:]]
    torch.testing.assert_close(one_loss, terms[0])
    torch.testing.assert_close(two_loss, terms[0] + terms[1])
    assert not torch.isclose(one_loss, two_loss)


@pytest.mark.parametrize("t_max", [0, 4, 5])
def test_too_few_anchors_raises(t_max):
    encoder = IdentityEncoder()
    aux = CLTTReferenceAuxLoss(encoder)
    aux.attach_memory(identity_memory(t_max))
    with pytest.raises(ValueError, match="B_eff >= 2"):
        aux.compute(encoder, torch.empty(0))


def test_memory_required():
    encoder = IdentityEncoder()
    aux = CLTTReferenceAuxLoss(encoder)
    with pytest.raises(RuntimeError, match="attach_memory"):
        aux.compute(encoder, torch.empty(0))


@pytest.mark.parametrize("offsets", ["", "0", "-2", "two", "2.5", "2,", ",2", "2,,4"])
def test_invalid_offsets_raise(monkeypatch, offsets):
    monkeypatch.setenv("NETT_AUX_CLTT_REF_OFFSETS", offsets)
    with pytest.raises(ValueError, match="NETT_AUX_CLTT_REF_OFFSETS"):
        CLTTReferenceAuxLoss(IdentityEncoder())


def test_defaults_overrides_and_one_time_logging(monkeypatch):
    encoder = IdentityEncoder(channels=9)
    aux = CLTTReferenceAuxLoss(encoder)
    assert aux.max_samples == 96 and aux.temperature == 0.5
    monkeypatch.setenv("NETT_AUX_CLTT_REF_OFFSETS", "3, 6")
    monkeypatch.setenv("NETT_AUX_CLTT_REF_TEMP", "0.7")
    aux = CLTTReferenceAuxLoss(encoder)
    assert aux.offsets == (3, 6) and aux.temperature == 0.7
    aux.attach_memory(identity_memory(channels=9))
    messages = []
    monkeypatch.setattr("nett_skrl.brain.aux.cltt_ref_aux.logger.info", lambda *args: messages.append(args))
    aux.compute(encoder, torch.empty(0))
    aux.compute(encoder, torch.empty(0))
    assert len(messages) == 1
    assert messages[0][1:3] == ((3, 6), 3)
    assert "batch B=" in (messages[0][0] % messages[0][1:])


def test_incumbent_and_campaign_registry():
    assert AUX_LOSSES["cltt"] is not AUX_LOSSES["cltt_ref"]
    encoder = IdentityEncoder()
    incumbent = AUX_LOSSES["cltt"](encoder)
    assert isinstance(incumbent, CLTTAuxLoss)
    assert isinstance(incumbent.head, SimCLRProjectionHead)
    assert isinstance(AUX_LOSSES["cltt_ref"](encoder), CLTTReferenceAuxLoss)
    for prefix in ("SimCLR", "ViT"):
        assert MODELS[f"{prefix}-CLTT-Ref"]["aux"] == "cltt_ref"
        assert MODELS[f"{prefix}-CLTT"]["aux"] == "cltt"
        assert MODELS[f"{prefix}-CLTT-Ref"]["framestack"] is True
        assert MODELS[f"{prefix}-CLTT-Ref"]["aux_weight"] == 1.0


@pytest.mark.parametrize("has_memory", [True, False])
def test_ppo_attaches_memory_before_registering_head(monkeypatch, has_memory):
    encoder = IdentityEncoder()
    memory = identity_memory() if has_memory else None
    registrations = []

    def init_ppo(self, *args, **kwargs):
        self.memory = memory
        self.policy = SimpleNamespace(encoder=encoder)

        def add_param_group(group):
            assert self._aux._memory is memory
            registrations.append(group)

        self.optimizer = SimpleNamespace(add_param_group=add_param_group)

    monkeypatch.setattr(PPO, "__init__", init_ppo)
    if has_memory:
        agent = AuxLossPPO(aux_loss="cltt_ref", aux_weight=1.0)
        assert agent._aux._memory is memory
        assert registrations[0]["params"] == list(agent._aux.head.parameters())
    else:
        with pytest.raises(ValueError, match="draws its own temporal windows"):
            AuxLossPPO(aux_loss="cltt_ref", aux_weight=1.0)
        assert not registrations


@pytest.mark.parametrize(
    "name,cfg,expected",
    [("simclr_cltt", {"trainable": True, "features_dim": 512, "conv_dim": 77}, 701_831),
     ("compact_vit", VIT_CFG, 804_320)],
)
def test_measured_encoder_parameters_at_live_eye(name, cfg, expected):
    # eye_resolution=(128, 80) is HWC (80, 128, 6), NOT the archived square eye.
    space = gym.spaces.Box(low=0, high=255, shape=(80, 128, 6), dtype=np.uint8)
    encoder = encoder_mapping[name](space, **cfg).cpu()
    assert sum(p.numel() for p in encoder.parameters()) == expected


# ---------------------------------------------------------------------------
# Gate A control for cltt_ref. The vicreg_tt control does NOT transfer: it scores the
# temporal positive against another AUGMENTATION of the anchor, and cltt_ref augments
# nothing. So this asks the two questions actually open here, both of which a falling
# loss is consistent with.
# ---------------------------------------------------------------------------

from nett_skrl.brain.aux.cltt_ref_aux import nt_xent_diagnostics  # noqa: E402


def _norm(x):
    return torch.nn.functional.normalize(x, dim=-1)


def test_a_trivially_solvable_task_reads_as_solved():
    """TOO EASY: adjacent stacks off one env stream are near-identical images. If the
    softmax is solved from update 1, the gradient carries no pressure toward the object --
    and the loss is near zero either way."""
    z = _norm(torch.randn(16, 8))
    d = nt_xent_diagnostics(z, z + 1e-4 * torch.randn(16, 8), 0.2)
    assert d["pos_acc"] > 0.95
    assert d["pos_sim"] > d["neg_sim"]


def test_an_uninformative_pairing_reads_at_chance():
    """TOO HARD: if the temporal offset carries nothing, the positive is not findable."""
    d = nt_xent_diagnostics(_norm(torch.randn(32, 8)), _norm(torch.randn(32, 8)), 0.2)
    assert d["pos_acc"] < 0.2, d
    assert d["chance"] == pytest.approx(1 / 63)


def test_the_shuffled_null_is_the_comparator_not_zero():
    """pos_acc at or below shuffled_acc means the pairing carried no information -- the
    reading the vicreg-tt+ kill was really about, expressed for a contrastive objective."""
    z = _norm(torch.randn(24, 8))
    good = nt_xent_diagnostics(z, z + 1e-4 * torch.randn(24, 8), 0.2)
    null = nt_xent_diagnostics(_norm(torch.randn(24, 8)), _norm(torch.randn(24, 8)), 0.2)
    assert good["pos_acc"] > good["shuffled_acc"]
    assert null["pos_acc"] <= null["shuffled_acc"] + 0.1


def test_positive_and_negative_similarity_are_reported_untempered():
    """Divided by temperature they are not comparable across a temperature change, and the
    fleet has already been bitten by a statistic whose definition moved under its name."""
    z1, z2 = _norm(torch.randn(8, 4)), _norm(torch.randn(8, 4))
    a = nt_xent_diagnostics(z1, z2, 0.2)
    b = nt_xent_diagnostics(z1, z2, 0.5)
    assert a["pos_sim"] == pytest.approx(b["pos_sim"], abs=1e-5)
    assert -1.001 <= a["pos_sim"] <= 1.001


def test_the_batch_size_is_reported_beside_the_accuracy():
    """chance is 1/(2B-1), so an accuracy without its B is not interpretable."""
    d = nt_xent_diagnostics(_norm(torch.randn(10, 4)), _norm(torch.randn(10, 4)), 0.2)
    assert d["batch"] == 10 and d["chance"] == pytest.approx(1 / 19)


def test_the_diag_is_off_by_default_and_goes_through_env_flag(monkeypatch):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss
    monkeypatch.delenv("NETT_AUX_CLTT_REF_DIAG", raising=False)
    assert CLTTReferenceAuxLoss(IdentityEncoder()).diag is False
    for spelling in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("NETT_AUX_CLTT_REF_DIAG", spelling)
        assert CLTTReferenceAuxLoss(IdentityEncoder()).diag is True, spelling
