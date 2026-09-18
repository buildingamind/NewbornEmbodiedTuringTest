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
from nett_skrl.brain.aux.knobs import NOT_MEASURED
from nett_skrl.brain.aux.cltt_ref_aux import (
    CLTTReferenceAuxLoss,
    CLTTReferenceProjectionHead,
    nt_xent_same_frame_masked,
    positive_alias_mask,
    same_frame_mask,
)
from nett_skrl.brain.aux.knobs import NOT_MEASURED
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
        self.tensors = {"observations": raw,
                        "terminated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool),
                        "truncated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool)}
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
    [(True, 0, 4, 8, 4, [3, 9]), (False, 9, 4, 1, 4, [3, 2]),
     (False, 7, 96, 0, 3, [3]), (False, 8, 4, 0, 4, [3, 1])],
)
def test_one_stream_contiguous_windows_without_whole_buffer_transfer(
    monkeypatch, filled, memory_index, batch, start, expected_batch, expected_highs,
):
    # Pin this sampler regression's original geometry independently of loss defaults.
    monkeypatch.setenv("NETT_AUX_CLTT_REF_OFFSETS", "2,4")
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


def test_three_frame_default_offsets_accepted_with_one_time_logging(monkeypatch):
    encoder = IdentityEncoder(channels=9)
    aux = CLTTReferenceAuxLoss(encoder)
    aux.attach_memory(identity_memory(channels=9))
    messages = []
    monkeypatch.setattr("nett_skrl.brain.aux.cltt_ref_aux.logger.info", lambda *args: messages.append(args))
    for _ in range(2):
        assert torch.isfinite(aux.compute(encoder, torch.empty(0)))
        assert len(messages) == 1
        # ⛔ The batch must be in this line. Every level claim about NT-Xent depends on
        # B (chance is 2*ln(2B-1)), and the line used to carry offsets and stack depth
        # and not B -- which is how a read protocol got published against an assumed
        # B=96 when the realised B was ~45.
        assert messages[0][1:3] == ((1, 2), 3)
        assert len(messages[0][1:]) > 2, "offsets and T alone leave the level unbacked"
        fmt = messages[0][0] % messages[0][1:]
        assert "batch B=" in fmt and "chance" in fmt
    assert len(encoder.views) == 6


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
    monkeypatch.setenv("NETT_AUX_CLTT_REF_OFFSETS", "2,4")
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
    # ⛔ THE REFERENCE IS THE MASKED FORM NOW. The objective changed on 2026-09-17 (the anchor's
    # own frame left its own negatives), and this line is where that change is visible: against
    # plain `nt_xent` the module reads 1.8461 where the unmasked objective gives 2.1058 at B=4.
    terms = [nt_xent_same_frame_masked(z[0], positive, 0.5, k)[0]
             for k, positive in zip((2, 4), z[1:])]
    torch.testing.assert_close(one_loss, terms[0])
    torch.testing.assert_close(two_loss, terms[0] + terms[1])
    assert not torch.isclose(one_loss, two_loss)


@pytest.mark.parametrize("t_max", [0, 2, 3])
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
    assert aux.max_samples == 512 and aux.temperature == 0.5
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
    # ⛔ THIS LOOP'S CONTRACT IS "PREFIXES THAT HAVE BOTH VARIANTS" -- it asserts `-CLTT` as well
    # as `-CLTT-Ref`. Adding "3DCNN" here failed on KeyError: '3DCNN-CLTT', correctly: there is no
    # 3DCNN incumbent and there must not be one, because the incumbent `cltt` family is frozen so
    # already-scored arms keep their definition. A Ref-only member gets its own assertion.
    for prefix in ("SimCLR", "ViT"):
        assert MODELS[f"{prefix}-CLTT-Ref"]["aux"] == "cltt_ref"
        assert MODELS[f"{prefix}-CLTT"]["aux"] == "cltt"
        assert MODELS[f"{prefix}-CLTT-Ref"]["framestack"] is True
        assert MODELS[f"{prefix}-CLTT-Ref"]["aux_weight"] == 1.0

    # Ref-only members: the aux is attached to an encoder that has no `cltt` incumbent twin.
    for name in ("3DCNN-CLTT-Ref",):
        assert MODELS[name]["aux"] == "cltt_ref"
        assert MODELS[name]["framestack"] is True
        assert MODELS[name]["aux_weight"] == 1.0
        # ⛔ THE ONE-FACTOR PROPERTY IS THE POINT OF THE ARM AND IS ASSERTED, NOT ASSUMED.
        # If cfg ever drifts from its plain twin the contrast stops being "adds cltt_ref" and
        # becomes "adds cltt_ref AND something else", which no result would reveal.
        base = name[: -len("-CLTT-Ref")]
        assert MODELS[name]["cfg"] == MODELS[base]["cfg"]
        assert MODELS[name]["encoder"] == MODELS[base]["encoder"]
        assert MODELS[base].get("aux") is None


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
    assert d["pos_chance"] == pytest.approx(1 / 63)


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
    """pos_chance is 1/|candidates|, so an accuracy without its B is not interpretable.

    ⚠ `chance` (the LOSS floor) and `pos_chance` (the argmax probability) are different numbers
    under different keys since 2026-09-17: the diagnostic's key used to be `chance` too and
    silently overwrote the loss floor in `last_scalars` whenever the diagnostic was on."""
    d = nt_xent_diagnostics(_norm(torch.randn(10, 4)), _norm(torch.randn(10, 4)), 0.2)
    assert d["batch"] == 10 and d["pos_chance"] == pytest.approx(1 / 19)
    assert d["duplicates"] == NOT_MEASURED and d["dup_sim"] == NOT_MEASURED


def test_the_diag_is_off_by_default_and_goes_through_env_flag(monkeypatch):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss
    monkeypatch.delenv("NETT_AUX_CLTT_REF_DIAG", raising=False)
    assert CLTTReferenceAuxLoss(IdentityEncoder()).diag is False
    for spelling in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("NETT_AUX_CLTT_REF_DIAG", spelling)
        assert CLTTReferenceAuxLoss(IdentityEncoder()).diag is True, spelling


@pytest.mark.parametrize("kind", ["cltt_ref", "vicreg_tt"])
def test_sampler_refuses_window_spanning_reset(monkeypatch, kind):
    """Every candidate crosses a reset, so neither sampler may encode a pair."""
    from nett_skrl.brain.aux.vicreg_tt_aux import VICRegTemporalAuxLoss

    monkeypatch.setenv("NETT_AUX_CLTT_REF_OFFSETS", "4")
    monkeypatch.setenv("NETT_AUX_VICREG_TT_OFFSETS", "4")
    monkeypatch.setattr("nett_skrl.brain.aux.vicreg_tt_aux._augment", lambda view, **kw: view)
    encoder = IdentityEncoder()
    memory = identity_memory(t_max=6, num_envs=1)
    memory.tensors["terminated"] = torch.zeros(6, 1, 1, dtype=torch.bool)
    memory.tensors["truncated"] = torch.zeros(6, 1, 1, dtype=torch.bool)
    memory.tensors["terminated"][2] = True
    cls = CLTTReferenceAuxLoss if kind == "cltt_ref" else VICRegTemporalAuxLoss
    aux = cls(encoder)
    aux.attach_memory(memory)
    with pytest.raises(ValueError, match="episode"):
        aux.compute(encoder, torch.empty(0))
    assert encoder.views == []


@pytest.mark.parametrize("signal", ["terminated", "truncated", "keys"])
def test_episode_window_helper_checks_interior_and_allows_done_endpoint(signal):
    from nett_skrl.brain.aux.cltt_ref_aux import episode_window_starts

    memory = identity_memory(t_max=10, num_envs=2)
    if signal == "keys":
        keys = torch.zeros(10, 2, 2, dtype=torch.long)
        keys[:, :, 0] = torch.arange(2)
        keys[4:7, 0, 1] = 1  # endpoints can match while the interior is another episode
        memory.tensors = {"observations": memory.tensors["observations"], "keys": keys}
    else:
        memory.tensors[signal][3, 0] = True
        memory.tensors[signal][6, 0] = True
    valid = episode_window_starts(memory, (2, 4))
    assert not valid[:, 0].any()
    assert valid[:, 1].all()
    valid = episode_window_starts(memory, (2,))
    assert valid[1, 0]  # ends AT the last observation of the episode
    assert not valid[2, 0]  # crosses the outgoing transition at t=3
    assert valid[4, 0]  # starts immediately after the reset


@pytest.mark.parametrize("missing", ["terminated", "truncated"])
def test_episode_window_helper_requires_both_done_signals(missing):
    from nett_skrl.brain.aux.cltt_ref_aux import episode_window_starts

    memory = identity_memory()
    del memory.tensors[missing]
    memory.get_tensor_by_name = lambda name: memory.tensors[name]
    with pytest.raises(ValueError, match=f"missing '{missing}'"):
        episode_window_starts(memory, (2, 4))


def test_episode_window_helper_supports_getter_and_rollout_ring_seam():
    from nett_skrl.brain.aux.cltt_ref_aux import episode_window_starts

    memory = identity_memory(t_max=10, num_envs=1, memory_index=5)
    tensors = memory.tensors
    memory.tensors = {"observations": tensors["observations"]}
    memory.get_tensor_by_name = lambda name: tensors[name]
    valid = episode_window_starts(memory, (2,))[:, 0]
    assert valid.tolist() == [True, True, True, False, False, True, True, True]


def test_cltt_ref_does_not_support_transit_masking_and_that_is_asserted_not_assumed():
    """⛔ `NETT_AUX_TRANSIT_MASK` IS READ BY `vicreg_tt_aux` AND BY NOTHING ELSE.

    This test exists because the parametrisation below USED to sweep `mask=[False, True]`
    across `kind=[cltt_ref, vicreg_tt]`, and for `cltt_ref` that factor VARIED NOTHING: the
    module never reads the variable, so the two cases were byte-identical runs. Both passed,
    and a test whose name and parameters both say "mask" is exactly what makes a reader
    conclude the feature is covered. (Found by the lion seat, reproduced by commander, and it
    gated a registered precondition on a procedure with no implementation.)

    ⇒ The absence is now an ASSERTION. If someone adds transit masking to cltt_ref, this fails
    and points at the registration that depends on the answer -- which is the behaviour a
    silent zero-variance parameter could never have.
    [[a-repeat-that-varies-nothing]] [[a-knob-nothing-reads-runs-the-control]]
    """
    import re
    from nett_skrl.brain.aux import cltt_ref_aux
    src = Path(cltt_ref_aux.__file__).read_text()
    # The one "transit" in this module is a comment about done-flag TRANSITIONS.
    reads = re.findall(r"NETT_AUX_TRANSIT_MASK", src)
    assert reads == [], (
        f"cltt_ref_aux now reads NETT_AUX_TRANSIT_MASK ({len(reads)} sites). That is a real "
        f"capability change: the 14-condition wave's rows 04/05 have a registered precondition "
        f"written against this absence, and notes/researcher/fourteen-condition-wave.md section "
        f"7 must be updated in the same change.")


@pytest.mark.parametrize("kind", ["cltt_ref", "vicreg_tt"])
@pytest.mark.parametrize("mask", [False, True])
def test_sampler_excludes_reset_and_shrinks_batch(monkeypatch, kind, mask):
    """⚠ `mask` is a live factor for vicreg_tt ONLY; cltt_ref ignores it by construction and
    the case above asserts that. Kept across both so the RESET-exclusion behaviour -- which is
    what this test is actually about -- is checked for each module under both settings."""
    from nett_skrl.brain.aux import vicreg_tt_aux

    monkeypatch.setenv("NETT_AUX_CLTT_REF_OFFSETS", "2")
    monkeypatch.setenv("NETT_AUX_VICREG_TT_OFFSETS", "2")
    monkeypatch.setenv("NETT_AUX_TRANSIT_MASK", str(int(mask)))
    monkeypatch.setattr(vicreg_tt_aux, "_augment", lambda view, **kw: view)
    encoder = IdentityEncoder()
    memory = identity_memory(t_max=12, num_envs=1)
    memory.tensors["truncated"][5] = True
    # Weight is concentrated at the reset: mask weighting must still exclude it.
    memory.tensors["actions"] = torch.zeros(12, 1, 2)
    memory.tensors["actions"][4:6, 0, 0] = 100
    cls = CLTTReferenceAuxLoss if kind == "cltt_ref" else vicreg_tt_aux.VICRegTemporalAuxLoss
    aux = cls(encoder)
    aux.attach_memory(memory)
    for _ in range(8):
        encoder.views.clear()
        assert torch.isfinite(aux.compute(encoder, torch.empty(0)))
        anchor, positive = [view[:, 0].long() // 1000 for view in encoder.views]
        assert len(anchor) == 4
        assert torch.equal(positive - anchor, torch.full_like(anchor, 2))
        assert torch.equal(anchor // 6, positive // 6)


@pytest.mark.parametrize("kind", ["cltt_ref", "vicreg_tt"])
def test_offline_temporal_losses_keep_internal_window_batches(monkeypatch, kind):
    import replay_harness as rh
    from nett_skrl.brain.aux import vicreg_tt_aux

    monkeypatch.setenv("NETT_AUX_CLTT_REF_OFFSETS", "2")
    monkeypatch.setenv("NETT_AUX_VICREG_TT_OFFSETS", "2")
    monkeypatch.setattr(vicreg_tt_aux, "_augment", lambda view, **kw: view)
    encoder = IdentityEncoder()
    obs = identity_memory(t_max=12, num_envs=2).tensors["observations"]
    keys = torch.stack((torch.arange(2).expand(12, -1),
                        (torch.arange(12) // 6)[:, None].expand(-1, 2),
                        (torch.arange(12) % 6)[:, None].expand(-1, 2)), dim=-1)
    aux, losses = rh.train_offline(encoder, kind, obs, None, 2, 3e-4, 17, batch=2, keys=keys)
    assert len(losses) == 2 and all(np.isfinite(losses))
    assert all(len(view) == 4 for view in encoder.views)  # internal safe slab, not CLI batch=2
    assert torch.equal(aux._memory.tensors["keys"], keys)


# ── Strided-capture gaps ──────────────────────────────────────────────────────
# REGRESSION: the step-advance term of the boundary mask had ZERO coverage until
# 2026-09-13. Disabling it left the whole suite green, and the only thing that
# caught the real defect was an end-to-end run on an actual capture.
#
# ⛔ THE FIXTURE MUST BE LONG. cltt_ref/vicreg_tt still carry a count guard
# (`avail = t_max - max(offsets); if batch < 2: raise`) ABOVE the contiguity
# check. A short gapped capture is refused by the COUNT guard, so the contiguity
# path never runs and the test passes while proving nothing. These captures are
# long enough that the count guard cannot fire: the only thing that can refuse
# them is adjacency.

def strided_keys(t_max, num_envs, *, window=8, every=32):
    """(env, episode, step) for a capture that records `window` contiguous frames
    every `every` steps -- the real shape of chicken's captures."""
    step = torch.empty(t_max, dtype=torch.long)
    for i in range(t_max):
        burst, within = divmod(i, window)
        step[i] = burst * every + within
    keys = torch.zeros(t_max, num_envs, 3, dtype=torch.long)
    keys[:, :, 0] = torch.arange(num_envs)[None, :]
    keys[:, :, 2] = step[:, None]
    return keys


def test_capture_gap_refused_even_when_env_and_episode_are_constant():
    """A 25-step gap with env and episode UNCHANGED must still break the window.

    An identity-only boundary mask sees one env and one episode here and calls
    the whole capture contiguous. Only the step-advance term can refuse it.
    """
    from nett_skrl.brain.aux.cltt_ref_aux import (
        episode_window_batch, episode_window_starts)
    memory = identity_memory(t_max=128, num_envs=2)
    memory.tensors = {"observations": memory.tensors["observations"],
                      "keys": strided_keys(128, 2)}
    # Count guard cannot fire: avail = 128 - 8 = 120, far above 2.
    assert 128 - 8 >= 2
    valid = episode_window_starts(memory, (8,))
    assert not valid.any(), (
        "offset 8 needs 9 adjacent frames; an 8-frame burst cannot supply them"
    )
    with pytest.raises(ValueError, match="episode-contiguous"):
        episode_window_batch(memory, (8,), 64)


def test_capture_gap_allows_windows_that_fit_inside_one_burst():
    """The refusal must be geometric, not blanket: offsets that fit still work."""
    from nett_skrl.brain.aux.cltt_ref_aux import (
        episode_window_batch, episode_window_starts)
    memory = identity_memory(t_max=128, num_envs=2)
    memory.tensors = {"observations": memory.tensors["observations"],
                      "keys": strided_keys(128, 2)}
    valid = episode_window_starts(memory, (4,))
    assert valid.any(), "a 5-frame span fits inside an 8-frame burst"
    batch, _starts = episode_window_batch(memory, (4,), 64)
    assert batch == 4, f"8-frame burst with offset 4 admits batch 4, got {batch}"


# ---------------------------------------------------------------------------
# ⭐⭐⭐ THE FRAMESTACK QUESTION FOR 3DCNN-CLTT-Ref, ANSWERED BY EXECUTION.
# The open design question was "how do you do temporal contrastive learning on an encoder
# whose input is already a framestack?". The registry assertions above cannot answer it --
# they only check a dict. These run the real Compact3DCNN through the real aux.
# ⛔⛔⛔ AND THE ANSWER IS NOT THE ONE THE OLD DOCSTRING GIVES. `088a785` replaced whole
# temporally-offset stacks (offsets 2,4, guarded to multiples of T) with the reference-faithful
# SINGLE CURRENT FRAME, REPEATED across the encoder's T slots (offsets now 1,2, guard removed
# because views can no longer overlap). cltt_views.current_frame_stack says so itself: "it is a
# static view, so a motion encoder sees no within-view motion."
# ⇒ For 3DCNN, whose first Conv3d exists ONLY to read motion across the stack, every auxiliary
# view is a STILL IMAGE. The aux trains the spatial pathway and hands the temporal kernel a
# constant. That is a real property of this arm and it is asserted here so it cannot change
# silently. [[source-version-is-a-per-phase-fact]] [[a-knob-nothing-reads-runs-the-control]]
def _compact_3dcnn(num_frames=2, height=16, width=16):
    import gymnasium as gym
    from nett_skrl.brain.encoders.compact_3dcnn import Compact3DCNN
    space = gym.spaces.Box(low=0, high=255,
                           shape=(height, width, 3 * num_frames), dtype="uint8")
    return Compact3DCNN(space, features_dim=32, num_frames=num_frames, conv_dim=8)


def _raw_frames(t_max=24, num_envs=2, num_frames=2, height=16, width=16):
    torch.manual_seed(0)
    return torch.rand(t_max, num_envs, height * width * 3 * num_frames) * 255.0


def test_cltt_ref_runs_on_a_framestack_native_encoder():
    """The interface holds: Compact3DCNN supplies features_dim, _prepare_image and
    encode_prepared (the last two inherited), which is all the aux touches."""
    enc = _compact_3dcnn(num_frames=2)
    loss = AUX_LOSSES["cltt_ref"](enc)
    loss.attach_memory(FakeMemory(_raw_frames()))
    value = loss.compute(enc, torch.empty(0))
    assert loss.num_frames == 2, loss.num_frames          # T discovered, not passed
    assert torch.is_tensor(value) and torch.isfinite(value), value
    assert value.requires_grad, "backbone gradients must flow, as for ViT-CLTT-Ref"


def test_a_cltt_ref_view_is_temporally_CONSTANT_so_3dcnn_sees_no_motion():
    """⛔ THE COST OF THE REFERENCE-FAITHFUL ADAPTER, MEASURED RATHER THAN ASSUMED."""
    from nett_skrl.brain.aux.cltt_views import current_frame_stack
    prepared = torch.arange(2 * 6 * 4 * 4, dtype=torch.float32).reshape(2, 6, 4, 4)
    view = current_frame_stack(prepared)
    assert view.shape == prepared.shape, "input geometry must be preserved"
    B, CT, H, W = view.shape
    slots = view.view(B, CT // 3, 3, H, W)
    assert torch.equal(slots[:, 0], slots[:, 1]), (
        "the T slots must be identical -- this is what makes the view static")
    # and the frame kept is the CURRENT one (T-major: last three channels)
    assert torch.equal(slots[:, -1], prepared[:, -3:])


# =============================================================================================
# The anchor's own frame leaves its own negatives (owner decision 2026-09-17), and the
# diagnostic that describes it. ⛔ ONE derivation of "which (row, column) pairs are the same
# frame" -- `same_frame_mask` -- consumed by both the loss and the diagnostic.

import itertools  # noqa: E402
import math  # noqa: E402

import torch.nn.functional as _F  # noqa: E402

from nett_skrl.brain.aux.simclr_aux import nt_xent as _plain_nt_xent  # noqa: E402


def _stream(steps, d=32, drift=0.15, seed=0):
    """A PURE FUNCTION OF THE FRAME: one embedding per frame, smoothly drifting, no periodicity.

    This is the regime the tie argument is about -- an encoder that cannot tell which view a
    frame arrived in, so the two encodings of one frame are identical to the last bit.
    """
    g = torch.Generator().manual_seed(seed)
    traj = torch.cumsum(torch.randn(steps, d, generator=g) * drift, dim=0)
    return _F.normalize(traj + torch.randn(1, d, generator=g) * 3.0, dim=-1)


def _sliced(batch, offset, **kw):
    """z1, z2 built by the MODULE'S OWN SLICING, so the duplicates are really there."""
    frames = _stream(batch + offset, **kw)
    return frames[:batch], frames[offset:offset + batch]


@pytest.mark.parametrize("batch,offset", list(itertools.product((3, 5, 8, 12, 498), (1, 2, 4, 8))))
def test_the_mask_count_is_the_number_of_valid_duplicate_pairs(batch, offset):
    """⛔ A COUNT, NOT A SPOT CHECK, and over the edge cases: the k rows at the start of each
    half have NO duplicate (i - k < 0), and at k >= B nothing does."""
    mask, count = same_frame_mask(batch, offset)
    assert count == max(0, 2 * (batch - offset)), (batch, offset)
    assert int(mask.sum()) == count
    assert bool((mask == mask.t()).all()), "the relation is symmetric by construction"
    assert not bool(mask.diagonal().any()), "the diagonal is the OTHER mask and stays separate"
    for i in range(min(offset, batch)):                     # the edge rows, explicitly
        assert not bool(mask[i].any()), f"row {i} < k has no duplicate"


def test_an_offset_of_zero_is_refused_because_the_duplicate_would_be_the_positive():
    with pytest.raises(ValueError, match="not a frame offset"):
        same_frame_mask(8, 0)
    with pytest.raises(ValueError, match="no similarity matrix"):
        same_frame_mask(0, 1)


def test_the_masked_pairs_are_exactly_the_entries_whose_two_frames_are_identical():
    """The mask is derived from indices; this checks the indices against the PIXELS -- that the
    entries it removes are the ones whose row and column really are one frame."""
    batch, offset = 10, 3
    z1, z2 = _sliced(batch, offset)
    z = torch.cat([z1, z2])
    same = torch.zeros(2 * batch, 2 * batch, dtype=torch.bool)
    for a in range(2 * batch):
        for b in range(2 * batch):
            fa = a if a < batch else offset + (a - batch)    # frame index of row a
            fb = b if b < batch else offset + (b - batch)
            same[a, b] = (fa == fb) and (a != b)
    mask, count = same_frame_mask(batch, offset)
    assert torch.equal(mask, same), "the index derivation disagrees with the frame identities"
    cos = torch.mm(z, z.t())
    assert torch.allclose(cos[mask], torch.ones(count), atol=1e-6), (
        "a masked entry must be an exact duplicate under a pure-function encoder")


def test_the_masked_entries_are_minus_inf_and_hold_no_softmax_mass():
    batch, offset, temp = 10, 3, 0.1
    z1, z2 = _sliced(batch, offset)
    z = torch.cat([z1, z2])
    sim = torch.mm(z, z.t()) / temp
    sim.fill_diagonal_(float("-inf"))
    mask, count = same_frame_mask(batch, offset)
    masked = sim.masked_fill(mask, float("-inf"))
    assert bool(torch.isinf(masked[mask]).all()) and int(torch.isinf(masked).sum()) == count + 2 * batch
    assert float(torch.softmax(masked, dim=1)[mask].sum()) == 0.0


def test_removing_the_duplicate_moves_the_loss_and_the_positive_gains_the_mass():
    """⚠ THE OBJECTIVE CHANGED. Not bitwise identity -- the opposite: this is the size of it."""
    batch, offset, temp = 12, 2, 0.1
    z1, z2 = _sliced(batch, offset)
    unmasked = float(_plain_nt_xent(z1, z2, temp))
    masked, count = nt_xent_same_frame_masked(z1, z2, temp, offset)
    assert float(masked) < unmasked
    assert count == 2 * (batch - offset)
    # the fraction of the candidate set removed, per row that has a duplicate: 1 of (2B - 1)
    assert count / float((2 * batch) ** 2) < 0.05


def test_the_ln2_floor_is_NOT_removed_by_this_mask_and_here_is_why():
    """⛔ I DISAGREE WITH ONE LINE OF THE SPEC, AND THIS IS THE DERIVATION.

    The claim was that with the anchor's own frame excluded, the n_offsets*ln2 tie floor no
    longer applies. It still does. Row i's positive is z2[i] = frame t0+k+i -- and THAT FRAME
    ALSO SITS IN THE FIRST HALF, at column i+k, whenever i+k < B. It is not the anchor's frame,
    so this mask does not touch it, and its similarity to the anchor is EXACTLY the positive's
    (same two frames). A tie at the maximum caps p(positive) at 1/2, so the loss cannot go below
    ln2 for those rows either way.

    ⇒ Going below n_offsets*ln2 still means the tie was broken, i.e. one frame embedded two
    ways. Removing that floor as well would mean masking (i, i+k) and (B+j, B+j-k) -- the
    POSITIVE's frame rather than the anchor's -- which removes real negatives and is a second
    owner decision, not this one.
    """
    batch, offset, temp = 12, 2, 0.1
    z1, z2 = _sliced(batch, offset)
    z = torch.cat([z1, z2])
    for i in range(batch - offset):
        assert torch.equal(z1[i + offset], z2[i]), "the positive's frame is in the first half"
        assert float(torch.dot(z1[i], z1[i + offset])) == pytest.approx(
            float(torch.dot(z1[i], z2[i])), abs=1e-6), "and it ties the positive exactly"
    sim = torch.mm(z, z.t()) / temp
    sim.fill_diagonal_(float("-inf"))
    mask, _ = same_frame_mask(batch, offset)
    p = torch.softmax(sim.masked_fill(mask, float("-inf")), dim=1)
    labels = (torch.arange(2 * batch) + batch) % (2 * batch)
    p_pos = p.gather(1, labels[:, None]).squeeze(1)
    assert float(p_pos[:batch - offset].max()) <= 0.5 + 1e-6
    assert float(nt_xent_same_frame_masked(z1, z2, temp, offset)[0]) >= math.log(2)


def test_an_offset_at_or_above_the_batch_masks_nothing_and_reports_that_it_did_not():
    """⛔ "Masked" must never be claimed where nothing was masked -- the count is what says so."""
    z1, z2 = _sliced(4, 4)
    loss, count = nt_xent_same_frame_masked(z1, z2, 0.1, 4)
    assert count == 0
    assert torch.equal(loss, _plain_nt_xent(z1, z2, 0.1)), (
        "with no duplicate in range the objective is the unmasked one, exactly")


def test_the_masked_loss_still_carries_gradient_to_the_encoder():
    """masked_fill is out-of-place and the graph survives it; a fill_ on a leaf would not."""
    encoder = IdentityEncoder()
    aux = CLTTReferenceAuxLoss(encoder)
    aux.attach_memory(identity_memory())
    loss = aux.compute(encoder, torch.empty(0))
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in encoder.parameters()), "no gradient reached the encoder"
    for name, p in aux.head.named_parameters():
        assert p.grad is not None and float(p.grad.norm()) > 0, name


def test_the_regime_and_the_mask_count_go_out_on_every_call():
    """From tfevents alone a reader must be able to tell an old arm from a new one."""
    encoder = IdentityEncoder()
    aux = CLTTReferenceAuxLoss(encoder)
    aux.attach_memory(identity_memory())
    aux.compute(encoder, torch.empty(0))
    s = aux.last_scalars
    assert s["same_frame_masked"] == 1.0
    assert s["masked_pairs"] == float(sum(max(0, 2 * (int(s["B"]) - k)) for k in aux.offsets))
    assert 0.0 < s["masked_frac"] < 1.0
    assert s["chance"] == pytest.approx(len(aux.offsets) * math.log(2 * s["B"] - 1))


def _lookup_encoder(batch, offset, dim=48, seed=3):
    """A lookup-table encoder whose similarity PEAKS at lag `offset` -- the best case the
    diagnostic is meant to recognise -- sliced exactly as the module slices."""
    g = torch.Generator().manual_seed(seed)
    e = _F.normalize(torch.randn(batch + 3 * offset, dim, generator=g), dim=-1)
    f = _F.normalize(torch.stack([e[j] + 0.9 * e[j + offset] + 0.9 * e[max(j - offset, 0)]
                                  for j in range(batch + offset)]), dim=-1)
    return f[:batch], f[offset:offset + batch]


@pytest.mark.parametrize("batch,offset", [(12, 2), (20, 3), (30, 5), (16, 1)])
def test_a_perfect_encoder_reads_at_the_ATTAINABLE_ceiling_which_is_not_one(batch, offset):
    """⛔ THE REQUESTED CONTROL WAS "pos_acc ~ 1.0 FOR A PERFECT ENCODER". IT CANNOT BE, AND THE
    CEILING HAS A CLOSED FORM.

    The slab is contiguous, so the frame at t - k is a candidate as well. An encoder that has
    learned temporal invariance at lag k is TIME-HOMOGENEOUS -- its similarity depends on |Δt| --
    so sim(t, t-k) EQUALS sim(t, t+k) exactly, the argmax breaks the tie by column index, and
    the rows that have a backward twin miss by construction. Only the k rows at the start of
    each half have none:

        ceiling = 0.5 + k / 2B

    ⇒ `pos_ceiling` is emitted beside `pos_acc` so the comparison is against what is attainable.
    A reading materially ABOVE it is not a better encoder: it is a direction-sensitive one.
    """
    z1, z2 = _lookup_encoder(batch, offset)
    d = nt_xent_diagnostics(z1, z2, 0.1, duplicate_offset=offset)
    assert d["pos_ceiling"] == pytest.approx(0.5 + offset / (2.0 * batch))
    assert d["pos_acc"] == pytest.approx(d["pos_ceiling"], abs=1e-6), d
    assert d["shuffled_acc"] < d["pos_acc"]


def test_a_random_encoder_reads_at_chance_on_the_signal_and_on_the_null():
    """Both statistics must collapse together when there is nothing to find."""
    batch, offset = 64, 2
    g = torch.Generator().manual_seed(11)
    f = _F.normalize(torch.randn(batch + offset, 48, generator=g), dim=-1)
    d = nt_xent_diagnostics(f[:batch], f[offset:offset + batch], 0.1, duplicate_offset=offset)
    rows = 2 * batch
    tol = 3.0 * math.sqrt(d["pos_chance"] * (1 - d["pos_chance"]) / rows)
    assert abs(d["pos_acc"] - d["pos_chance"]) <= tol + 1.0 / rows, d
    assert abs(d["shuffled_acc"] - d["pos_chance"]) <= tol + 1.0 / rows, d


def test_dup_sim_is_what_separates_invariance_from_a_broken_tie():
    """⛔ `pos_acc` CANNOT SEPARATE THEM and this is the statistic that can.

    Two encoders reach a low loss by opposite routes: one has learned that t and t+k are the
    same thing; the other embeds ONE frame differently depending on which view it arrived in
    (BatchNorm in train() sees two different slabs -- the live route). A pure function of the
    image gives cos(the two encodings of one frame) = 1 EXACTLY. Anything below is the second
    route, and the loss it reaches is not evidence of invariance.
    """
    batch, offset = 12, 2
    z1, z2 = _lookup_encoder(batch, offset)
    invariant = nt_xent_diagnostics(z1, z2, 0.1, duplicate_offset=offset)
    g = torch.Generator().manual_seed(5)
    view_bias = 0.25 * _F.normalize(torch.randn(1, z2.shape[-1], generator=g), dim=-1)
    broken = nt_xent_diagnostics(z1, _F.normalize(z2 + view_bias, dim=-1), 0.1,
                                 duplicate_offset=offset)
    assert invariant["dup_sim"] == pytest.approx(1.0, abs=1e-6)
    assert broken["dup_sim"] < 0.999
    # ... and pos_acc alone does NOT separate them, which is why dup_sim is emitted.
    assert abs(broken["pos_acc"] - invariant["pos_acc"]) < 0.2


def test_the_null_cannot_land_on_a_duplicate_or_on_a_copy_of_the_positive():
    """⛔ THE OLD NULL WAS `labels + 1`, WHICH ALIASED THE DUPLICATE at offset 1 and never at
    offset 2 -- so it read (B-1)/4B = 0.2495 on every arm and was measuring the same pinned
    argmax the signal was."""
    batch, offset = 12, 2
    z1, z2 = _lookup_encoder(batch, offset)
    dup, _ = same_frame_mask(batch, offset)
    alias, _ = positive_alias_mask(batch, offset)
    labels = (torch.arange(2 * batch) + batch) % (2 * batch)
    for _ in range(20):
        d = nt_xent_diagnostics(z1, z2, 0.1, duplicate_offset=offset)
        assert d["shuffled_acc"] <= 0.5
    # The draw itself: reconstruct the valid set and confirm it excludes all three.
    valid = torch.ones(2 * batch, 2 * batch, dtype=torch.bool)
    valid.fill_diagonal_(False)
    valid.scatter_(1, labels[:, None], False)
    valid &= ~dup & ~alias
    assert not bool((valid & dup).any()) and not bool((valid & alias).any())
    assert not bool(valid.diagonal().any())
    assert not bool(valid.gather(1, labels[:, None]).any())


def test_the_diagnostic_cannot_move_training_bitwise(monkeypatch):
    """⛔ THE DIAGNOSTIC FIX IS DIAGNOSTIC-ONLY, AND THIS IS THE EVIDENCE. Same seed, same
    window, diag off vs on: the loss and EVERY parameter gradient must be bitwise identical.
    The new null draws random numbers, so it uses a private generator -- with the global one,
    turning the diagnostic on would change every window drawn after it."""
    losses, grads = [], []
    for diag in ("0", "1"):
        monkeypatch.setenv("NETT_AUX_CLTT_REF_DIAG", diag)
        torch.manual_seed(20260917)
        encoder = IdentityEncoder()
        aux = CLTTReferenceAuxLoss(encoder)
        aux.attach_memory(identity_memory())
        fixed_draws(monkeypatch, [1, 3, 1, 3])
        loss = aux.compute(encoder, torch.empty(0))
        loss.backward()
        losses.append(loss.detach().clone())
        grads.append({n: p.grad.detach().clone()
                      for n, p in itertools.chain(encoder.named_parameters(),
                                                  aux.head.named_parameters())
                      if p.grad is not None})
        # the RNG stream must also be where it would have been without the diagnostic
        grads[-1]["__rng__"] = torch.random.get_rng_state()
    assert torch.equal(losses[0], losses[1]), (float(losses[0]), float(losses[1]))
    assert set(grads[0]) == set(grads[1]) and len(grads[0]) > 1
    for name in grads[0]:
        assert torch.equal(grads[0][name], grads[1][name]), name


def _diag_scalars(monkeypatch, diag: str) -> dict:
    monkeypatch.setenv("NETT_AUX_CLTT_REF_DIAG", diag)
    encoder = IdentityEncoder()
    aux = CLTTReferenceAuxLoss(encoder)
    aux.attach_memory(identity_memory())
    aux.compute(encoder, torch.empty(0))
    return aux.last_scalars


def test_the_diagnostic_keys_are_the_same_set_on_both_paths(monkeypatch):
    """⛔ ONE SOURCE OF TRUTH FOR THE KEY SET, CHECKED ON BOTH PATHS. With the flag off, these
    nine keys used to be ABSENT from the tag list -- the exact failure the sentinel exists to
    prevent: "did not run" and "ran and found nothing" become the same absence, and the
    sentinel-aware aggregation cannot publish a fire rate for a key it never saw.

    ⚠ Pinned against `DIAG_KEYS` rather than a literal list, so a key added to the diagnostic
    tomorrow cannot be sentinel-less on one of the two paths. cltt_patch is held to the same
    list by its own test; this is the sibling that was left behind when that one was fixed.
    """
    from nett_skrl.brain.aux.cltt_ref_aux import DIAG_KEYS

    off = _diag_scalars(monkeypatch, "0")
    on = _diag_scalars(monkeypatch, "1")
    assert set(DIAG_KEYS) <= set(off), sorted(set(DIAG_KEYS) - set(off))
    assert set(DIAG_KEYS) <= set(on), sorted(set(DIAG_KEYS) - set(on))
    assert set(off) == set(on), "the two paths must publish the SAME tags, not overlapping ones"
    for key in DIAG_KEYS:
        if key == "batch":
            assert off[key] == on[key] > 0, "the batch is known whether or not the diag ran"
        else:
            assert off[key] == NOT_MEASURED, key
    assert on["pos_acc"] != NOT_MEASURED and 0.0 <= on["pos_acc"] <= 1.0
    # ⚠ The LOSS floor is not a diagnostic key and must survive on both paths -- it was the
    # collision that started this (`chance` overwritten by the diagnostic's own probability).
    assert off["chance"] == on["chance"] > 0 and "pos_chance" in off
