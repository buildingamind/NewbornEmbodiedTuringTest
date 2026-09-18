"""Numeric and positive-identity regressions for the two distinct CLTT references."""

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_cltt_ref_aux import FakeMemory


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    import os

    for name in os.environ:
        if name.startswith(("NETT_AUX_", "NETT_VICREG_")):
            monkeypatch.delenv(name)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        yield
    torch.set_num_threads(threads)


class Encoder(nn.Module):
    def __init__(self, frames=2):
        super().__init__()
        self.features_dim = 12
        self.fc = nn.Linear(3 * frames, 12)
        self.views = []

    def _prepare_image(self, x):
        return x

    def encode_prepared(self, x):
        self.views.append(x.detach().clone())
        return self.fc(x.mean((2, 3)))


def memory(length=16, frames=2):
    # Distinct RGB channels and history slots detect C-major or whole-stack views.
    t = torch.arange(length).float()[:, None, None, None, None]
    c = torch.arange(3 * frames).float()[None, None, :, None, None]
    return FakeMemory((t + c / 100).expand(-1, 1, -1, 2, 2).clone())


@pytest.mark.parametrize("diagnostics", [False, True])
def test_vicreg_variance_matches_hand_calculation(diagnostics):
    from nett_skrl.brain.aux.vicreg_aux import vicreg_loss
    from nett_skrl.brain.aux.vicreg_tt_aux import vicreg_terms

    # Unbiased variances: view 1 = [0, 1/4], view 2 = [1, 4].
    a = torch.tensor([[0., -.5], [0., 0.], [0., .5]], dtype=torch.float64)
    b = torch.tensor([[-1., -2.], [0., 0.], [1., 2.]], dtype=torch.float64)
    eps = 1e-4
    expected = ((1 - math.sqrt(eps)) + (1 - math.sqrt(.25 + eps))) / 4
    actual = vicreg_terms(a, b)[1] if diagnostics else vicreg_loss(a, b, 0, 1, 0)
    assert actual.item() == pytest.approx(expected, abs=1e-12)
    # Covariance is summed across views, WITHOUT the variance's factor of 1/2.
    assert vicreg_terms(a, b)[2].item() == pytest.approx(4.)
    assert vicreg_loss(a, b, 0, 0, 1).item() == pytest.approx(4.)


def test_buildingamind_default_batch_and_effective_batch():
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    encoder = Encoder(frames=1)
    aux = CLTTReferenceAuxLoss(encoder)
    assert aux.max_samples == 512
    assert aux.temperature == .5
    aux.attach_memory(memory(length=514, frames=1))
    assert torch.isfinite(aux.compute(encoder, torch.empty(0)))
    assert aux.last_scalars["B"] == 512


@pytest.mark.parametrize("frames", [1, 2, 3])
def test_buildingamind_three_frame_positives_and_summed_loss(monkeypatch, frames):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss
    from nett_skrl.brain.aux.simclr_aux import nt_xent

    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    encoder = Encoder(frames)
    aux = CLTTReferenceAuxLoss(encoder)
    assert aux.offsets == (1, 2)
    aux.attach_memory(memory(frames=frames))
    loss = aux.compute(encoder, torch.empty(0))
    views = encoder.views.copy()
    assert len(views) == 3
    for offset, view in enumerate(views):
        torch.testing.assert_close(view - views[0], torch.full_like(view, offset))
        for slot in range(frames):
            torch.testing.assert_close(view[:, 3 * slot:3 * slot + 3], view[:, -3:])
    # The newest RGB frame, with no augmentation, supplies each view.
    assert (views[0][0, 0, 0, 0] % 1).item() == pytest.approx((frames - 1) * .03, abs=1e-6)
    zs = [aux.head(encoder.encode_prepared(v)) for v in views]
    # ⛔ A DELIBERATE, OWNER-APPROVED DEVIATION FROM THE REFERENCE (2026-09-17, workspace
    # DECISIONS), RECORDED HERE RATHER THAN RELAXED AWAY. The reference
    # (ChicksAndDNNs_ViewInvariance) draws contiguous sliding windows from one unshuffled
    # stream, so each anchor's OWN FRAME is among its negatives there too -- the port was
    # faithful. The owner decided the CLTT objective should exclude it. This test therefore
    # pins the masked form, and the line below records how far that puts us from the
    # reference, so the distance is a number somebody can read rather than a lost property.
    from nett_skrl.brain.aux.cltt_ref_aux import nt_xent_same_frame_masked

    expected = sum(nt_xent_same_frame_masked(zs[0], z, .5, k, mask_positive_twin=aux.mask_twin)[0]
                   for k, z in zip(aux.offsets, zs[1:]))
    torch.testing.assert_close(loss, expected)
    reference = sum(nt_xent(zs[0], z, .5) for z in zs[1:])
    assert float(loss) < float(reference), (
        "removing a maximal-similarity candidate can only lower the cross-entropy")
    assert float(reference) - float(loss) > 1e-3, (
        "if the two agree, the mask stopped doing anything and the deviation is silent")
    # Keep exact self masking: identical normalized vectors have loss ln(2B-1).
    identical = F.normalize(torch.ones(4, 8), dim=1)
    assert nt_xent(identical, identical, .5).item() == pytest.approx(math.log(7))
    loss.backward()
    assert encoder.fc.weight.grad.norm() > 0


def schneider(encoder):
    from nett_skrl.brain.aux.cltt_schneider_aux import CLTTSchneiderAuxLoss

    return CLTTSchneiderAuxLoss(encoder)


def test_schneider_projector_defaults_and_raw_output():
    aux = schneider(Encoder())
    assert aux.max_samples == 256 and aux.temperature == 1.
    assert (aux.tau_minus, aux.tau_plus) == (0, 1)
    layers = list(aux.head.net)
    assert [type(x) for x in layers] == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Linear]
    assert (layers[0].in_features, layers[0].out_features) == (12, 256)
    assert layers[1].num_features == 256 and layers[2].inplace
    assert (layers[3].in_features, layers[3].out_features) == (256, 128)
    assert layers[3].bias is not None
    with torch.no_grad():
        layers[3].weight.zero_()
        layers[3].bias.fill_(2)
    torch.testing.assert_close(aux.head(torch.randn(4, 12)), torch.full((4, 128), 2.))


def test_schneider_next_frame_joint_batch_cosine_loss_and_gradients(monkeypatch):
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    encoder = Encoder()
    aux = schneider(encoder)
    aux.attach_memory(memory())
    projected = []
    handle = aux.head.register_forward_hook(lambda m, args, out: projected.append(out))
    loss = aux.compute(encoder, torch.full((1,), float("nan")))
    handle.remove()
    assert len(encoder.views) == len(projected) == 1  # Reference BN sees both views jointly.
    anchor, positive = encoder.views[0].chunk(2)
    torch.testing.assert_close(positive - anchor, torch.ones_like(anchor))
    torch.testing.assert_close(anchor[:, :3], anchor[:, 3:])
    z = projected[0]
    # Independent reference-style logits: cosine, positive first, no self term.
    similarities = F.cosine_similarity(z[:, None], z[None, :], dim=-1)
    terms = []
    for i in range(8):
        positive_idx = (i + 4) % 8
        terms.append(-similarities[i, positive_idx] + torch.logsumexp(
            similarities[i, torch.arange(8) != i], dim=0))
    torch.testing.assert_close(loss, torch.stack(terms).mean())
    loss.backward()
    assert encoder.fc.weight.grad.norm() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in aux.head.parameters())


@pytest.mark.parametrize("past,future", [(0, 1), (2, 0), (2, 3)])
@pytest.mark.parametrize("boundary", ["terminated", "truncated", "keys", "ring"])
def test_schneider_temporal_support_excludes_self_and_resets(monkeypatch, past, future, boundary):
    monkeypatch.setenv("NETT_AUX_CLTT_SCHNEIDER_TAU_MINUS", str(past))
    monkeypatch.setenv("NETT_AUX_CLTT_SCHNEIDER_TAU_PLUS", str(future))
    encoder = Encoder(frames=1)
    aux = schneider(encoder)
    mem = memory(length=24, frames=1)
    if boundary == "keys":
        keys = torch.zeros(24, 1, 3, dtype=torch.long)
        keys[:, 0, 1] = torch.arange(24) // 12
        keys[:, 0, 2] = torch.arange(24) % 12
        mem.tensors["keys"] = keys
    elif boundary == "ring":
        mem.memory_index = 12
    else:
        mem.tensors[boundary][11] = True
    aux.attach_memory(mem)
    seen = set()
    for _ in range(12):
        aux.compute(encoder, torch.empty(0))
        a, b = encoder.views[-1][:, 0, 0, 0].long().chunk(2)
        delta = b - a
        assert (delta != 0).all() and (delta >= -past).all() and (delta <= future).all()
        assert torch.equal(a // 12, b // 12)
        assert len(a) == 12 - past - future
        seen.update(delta.tolist())
    assert seen == set(range(-past, 0)) | set(range(1, future + 1))


@pytest.mark.parametrize("name,value", [
    ("TAU_MINUS", "-1"), ("TAU_PLUS", "-1"), ("TAU_PLUS", "0"),
    ("TAU_PLUS", "1.5"), ("TEMP", "0"), ("TEMP", "nan"),
])
def test_schneider_invalid_configuration(monkeypatch, name, value):
    # Import before the expected exception so pre-fix absence cannot masquerade as validation.
    from nett_skrl.brain.aux.cltt_schneider_aux import CLTTSchneiderAuxLoss

    monkeypatch.setenv("NETT_AUX_CLTT_SCHNEIDER_" + name, value)
    with pytest.raises(ValueError, match="NETT_AUX_CLTT_SCHNEIDER"):
        CLTTSchneiderAuxLoss(Encoder())


def test_schneider_registration_and_memory_contract(monkeypatch):
    from nett_skrl.brain.aux import CLTTSchneiderAuxLoss
    from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES, AuxLossPPO, PPO
    from campaign_train import MODELS

    encoder = Encoder()
    aux = AUX_LOSSES["cltt_schneider"](encoder)
    assert isinstance(aux, CLTTSchneiderAuxLoss) and aux.needs_memory
    with pytest.raises(RuntimeError, match="attach_memory"):
        aux.compute(encoder, torch.empty(0))
    mem = memory(length=3)
    mem.tensors["terminated"][0] = True
    aux.attach_memory(mem)
    with pytest.raises(ValueError, match="episode"):
        aux.compute(encoder, torch.empty(0))
    assert not encoder.views
    from types import SimpleNamespace
    groups = []

    def init_ppo(self, *args, **kwargs):
        self.memory = mem
        self.policy = SimpleNamespace(encoder=encoder)
        self.optimizer = SimpleNamespace(add_param_group=groups.append)

    monkeypatch.setattr(PPO, "__init__", init_ppo)
    agent = AuxLossPPO(aux_loss="cltt_schneider", aux_weight=1.)
    assert agent._aux._memory is mem
    assert groups[0]["params"] == list(agent._aux.head.parameters())
    for prefix in ("SimCLR", "ViT"):
        assert MODELS[prefix + "-CLTT-Schneider"]["aux"] == "cltt_schneider"


# --- SimCLR colour distortion: the measured shortcut, and what must stay absent -----


def _colour_only_top1(x, seed, **kw):
    """Identify the positive pair using per-channel mean+std ALONE -- no spatial content.

    This is the audit's protocol. If two views of the same frame are findable from 12
    colour numbers, NT-Xent can be minimised without reference to content.
    """
    from nett_skrl.brain.aux.simclr_aux import _augment

    torch.manual_seed(seed)
    v1, v2 = _augment(x, **kw), _augment(x, **kw)
    desc = lambda v: torch.cat([v.mean(dim=(2, 3)), v.std(dim=(2, 3))], dim=1)
    d1, d2 = desc(v1), desc(v2)
    d1 = (d1 - d1.mean(0)) / (d1.std(0) + 1e-8)
    d2 = (d2 - d2.mean(0)) / (d2.std(0) + 1e-8)
    return float((torch.cdist(d1, d2).argmin(1) == torch.arange(len(x))).float().mean())


def test_colour_distortion_closes_the_colour_shortcut():
    """⛔ THE POINT OF THE FIX, PINNED AS A MEASUREMENT AND NOT AS A CALL.

    SimCLR Sec 3 Fig 5: crop ALONE is solvable from colour histograms. Asserting that
    _augment 'calls a gain' would pass on an implementation that changed nothing; this
    asserts the leak is SMALLER, which is the property the fix exists for.
    """
    torch.manual_seed(0)
    # Each sample gets its own colour signature, which is exactly what the shortcut reads.
    base = torch.rand(48, 3, 1, 1) * torch.ones(48, 3, 16, 24)
    x = torch.cat([base, base], dim=1).clamp(0, 1)          # 2-frame stack

    pre = sum(_colour_only_top1(x, s, colour_gain=0.0, grayscale_p=0.0) for s in range(3)) / 3
    post = sum(_colour_only_top1(x, s) for s in range(3)) / 3
    assert post < pre, f"colour distortion must reduce the colour-only leak: {pre:.3f} -> {post:.3f}"


def test_colour_ops_keep_rgb_together_across_a_frame_stack():
    """A 6-channel stack is TWO RGB frames, not six independent planes.

    Distorting planes independently would desynchronise the two frames of one stack and
    hand the temporal objective a difference that is not motion.
    """
    from nett_skrl.brain.aux.simclr_aux import _augment

    torch.manual_seed(3)
    frame = torch.rand(4, 3, 12, 16)
    x = torch.cat([frame, frame], dim=1)                     # both frames identical
    out = _augment(x, scale_min=1.0, jitter=0.0)             # no crop, no brightness/contrast
    assert torch.allclose(out[:, :3], out[:, 3:], atol=1e-6), (
        "identical frames in one stack must receive the identical colour transform"
    )


def test_colour_ops_are_skipped_when_channels_are_not_whole_rgb_frames():
    """A channel count that is not a multiple of 3 has no defined R/G/B grouping.

    Guessing one would distort a non-colour channel, so the colour ops must no-op rather
    than reinterpret the stack.
    """
    from nett_skrl.brain.aux.simclr_aux import _augment

    torch.manual_seed(4)
    x = torch.rand(2, 4, 8, 8)
    before = x.clone()
    out = _augment(x, scale_min=1.0, jitter=0.0)
    assert torch.allclose(out, before, atol=1e-6), "4-channel input must pass colour ops untouched"


def test_horizontal_flip_is_still_absent():
    """⛔ DELIBERATE DEVIATION FROM THE PAPER. The imprinting test is a left/right
    two-alternative choice, so a flip destroys the label. 'Match the reference recipe'
    would break the task. This pins the absence so nobody restores it for fidelity."""
    from nett_skrl.brain.aux import simclr_aux

    # Look for the OPERATION, not the word -- this function's own docstring explains at
    # length why the flip is absent, so a substring check on "flip" fails on the
    # documentation that exists to protect it.
    import ast, inspect

    tree = ast.parse(inspect.getsource(simclr_aux._augment).lstrip())
    calls = {
        ast.unparse(n.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
    }
    assert not any("flip" in c.lower() for c in calls), (
        f"horizontal flip must not be added to _augment; found {sorted(calls)}"
    )


def test_colour_distortion_can_be_disabled_to_reproduce_a_pre_fix_arm(monkeypatch):
    from nett_skrl.brain.aux.simclr_aux import _augment

    monkeypatch.setenv("NETT_SIMCLR_NO_COLOUR_JITTER", "1")
    torch.manual_seed(5)
    frame = torch.rand(2, 3, 8, 8)
    x = torch.cat([frame, frame], dim=1)
    out = _augment(x, scale_min=1.0, jitter=0.0)
    assert torch.allclose(out, x, atol=1e-6), "the opt-out must restore exactly the pre-fix pipeline"
