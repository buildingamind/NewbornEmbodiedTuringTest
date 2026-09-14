"""Architecture, supervision, and PPO integration contracts for dual streams."""

import importlib
import os

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name in os.environ:
        if name.startswith(("NETT_AUX_", "NETT_EOO_", "NETT_GWM_")):
            monkeypatch.delenv(name)
    threads = torch.get_num_threads()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(enabled, warn_only=warn)


from nett_skrl.brain.encoders.hwc_feature_extractor import HWCFeatureExtractor


class Host(HWCFeatureExtractor):
    def __init__(self, channels=6):
        super().__init__(gym.spaces.Box(0, 1, (24, 32, channels), dtype=np.float32), 12)
        self.conv = nn.Conv2d(3, 12, 3, stride=2, padding=1)

    def encode_spatial(self, obs):
        return self.conv(self._prepare_image(obs)[:, -3:]).relu()

    def forward(self, obs):
        return self.encode_spatial(obs).mean((2, 3))


def build(kind, **kwargs):
    from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
    host = Host(**kwargs)
    return host, AUX_LOSSES[kind](host)


@pytest.mark.parametrize("kind", ["eoo_dual", "gwm_dual"])
@pytest.mark.parametrize("size", [(24, 32), (25, 33), (80, 128)])
def test_shapes_and_independent_output_gradients(kind, size):
    host, aux = build(kind)
    a, b = torch.rand(2, 3, *size), torch.rand(2, 3, *size)
    masks = aux.masks(host, a)
    fwd, bwd = aux.head.dorsal(a, b)
    assert masks.shape == (2, 2, *size)
    assert fwd.shape == bwd.shape == (2, 2, *size)
    torch.testing.assert_close(masks.sum(1), torch.ones(2, *size))
    vp, dp = [*host.parameters(), *aux.head.ventral.parameters()], list(aux.head.dorsal.parameters())
    assert not ({id(p) for p in vp} & {id(p) for p in dp})
    grads = torch.autograd.grad(masks[:, 1].square().mean(), vp + dp, allow_unused=True)
    assert all(g is not None and g.abs().sum() > 0 for g in grads[:len(vp)])
    assert all(g is None for g in grads[len(vp):])
    grads = torch.autograd.grad(fwd.square().mean() + bwd.square().mean(), vp + dp, allow_unused=True)
    assert all(g is None for g in grads[:len(vp)])
    assert all(g is not None and g.abs().sum() > 0 for g in grads[len(vp):])
    # Swapping time swaps the two outputs; Conv3d actually consumes both frames.
    swapped = aux.head.dorsal(b, a)
    torch.testing.assert_close(fwd, swapped[1])
    assert not torch.allclose(fwd, bwd)
    first = aux.head.dorsal.enc3d[0]
    assert isinstance(first, nn.Conv3d) and first.kernel_size == (2, 3, 3)


@pytest.mark.parametrize("kind", ["eoo_dual", "gwm_dual"])
def test_ventral_preserves_spatial_information(kind):
    host, aux = build(kind)
    assert not any(isinstance(m, nn.Linear) for m in aux.head.modules())
    observed = []
    hook = host.conv.register_forward_hook(lambda m, args, out: observed.append(out.shape))
    # Same RGB mean, different object location: flat global pooling loses this.
    left = torch.zeros(1, 3, 32, 48)
    left[:, :, 8:24, 4:20] = 1
    right = left.flip(-1)
    ml, mr = aux.masks(host, left), aux.masks(host, right)
    hook.remove()
    assert observed == [torch.Size([1, 12, 16, 24])] * 2
    assert ml[:, 1].std() > 1e-3
    assert (ml - mr).abs().mean() > 1e-3


@pytest.mark.parametrize("kind", ["eoo_dual", "gwm_dual"])
def test_compute_trains_both_streams_and_splits_t_major(kind):
    host, aux = build(kind)
    obs = torch.rand(2, 24, 32, 6)
    seen, ventral_seen = [], []
    hook = aux.head.dorsal.enc3d[0].register_forward_pre_hook(lambda m, args: seen.append(args[0].detach()))
    host_hook = host.conv.register_forward_pre_hook(lambda m, args: ventral_seen.append(args[0].detach()))
    opt = torch.optim.Adam([*host.parameters(), *aux.head.parameters()], lr=1e-3)
    before = host.conv.weight.detach().clone()
    for step in range(3):
        opt.zero_grad()
        loss = aux.compute(host, obs.flatten(1))
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        for stream in (host, aux.head.ventral, aux.head.dorsal):
            grads = [p.grad for p in stream.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            if step >= 1:
                assert sum(g.abs().sum() for g in grads) > 0
        opt.step()
    hook.remove()
    host_hook.remove()
    torch.testing.assert_close(seen[0][:, :, 0], obs[..., :3].permute(0, 3, 1, 2))
    torch.testing.assert_close(seen[0][:, :, 1], obs[..., 3:].permute(0, 3, 1, 2))
    assert len(ventral_seen) == 3
    for frame in ventral_seen:
        torch.testing.assert_close(frame, obs[..., -3:].permute(0, 3, 1, 2))
    assert not torch.equal(before, host.conv.weight)
    assert host(obs).shape == (2, host.features_dim)
    assert all(np.isfinite(v) for v in aux.last_scalars.values())


@pytest.mark.parametrize("kind", ["eoo_dual", "gwm_dual"])
def test_budget_and_framestack_guard(kind):
    from nett_skrl.brain.encoders.nature_cnn import NatureCNN
    from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
    from test_cltt_ref_aux import MODELS
    label = "EoO" if kind == "eoo_dual" else "GWM"
    host = NatureCNN(gym.spaces.Box(0, 255, (80, 128, 6), dtype=np.uint8),
                     **MODELS[f"CNN+{label}-Dual"]["cfg"])
    aux = AUX_LOSSES[kind](host)
    assert aux.parameter_counts == {
        "host": 773995, "ventral_decoder": 66498, "dorsal": 75906, "total": 916399,
    }
    assert not any(name.startswith("enc") for name, _ in aux.head.ventral.named_children())
    with pytest.raises(ValueError, match="framestack"):
        build(kind, channels=3)


def test_eoo_objectness_prefers_explainable_region_over_all_foreground():
    from nett_skrl.brain.aux.eoo_dual_aux import objectness_loss
    a = torch.zeros(1, 3, 24, 32)
    b = a.clone()
    b[:, :, :, 16:] = 1
    flow = torch.zeros(1, 2, 24, 32)
    all_fg = torch.full((1, 1, 24, 32), .99, requires_grad=True)
    selective = torch.full_like(all_fg, .01)
    selective[:, :, :, :10] = .99
    full = objectness_loss(a, b, all_fg, flow, flow)
    assert objectness_loss(a, b, selective, flow, flow) < full
    grad, = torch.autograd.grad(full, all_fg)
    assert grad[:, :, :, :8].mean() < 0  # Admit explained pixels.
    assert grad[:, :, :, 24:].mean() > 0  # Reject unexplained pixels.


@pytest.mark.parametrize("kind", ["eoo_dual", "gwm_dual"])
def test_joint_objective_has_gradients_to_masks_and_flow(kind):
    module = importlib.import_module(f"nett_skrl.brain.aux.{kind}_aux")
    masks = torch.rand(2, 2, 24, 32).softmax(1).requires_grad_()
    flow = (torch.rand(2, 2, 24, 32) * .1).requires_grad_()
    if kind == "eoo_dual":
        loss = module.objectness_loss(torch.rand(2, 3, 24, 32), torch.rand(2, 3, 24, 32), masks[:, 1:2], flow, -flow)
    else:
        loss = module.flow_reconstruction_loss(masks, flow)
    grads = torch.autograd.grad(loss, (masks, flow))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("kind", ["eoo_dual", "gwm_dual"])
def test_split_backward_optimizer_step(kind):
    from nett_skrl.brain.skrl_patches import strict_determinism, relaxed_determinism
    host, aux = build(kind)
    # CUDA exercises the actual nondeterministic grid_sample kernel when available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    host.to(device)
    aux.to(device)
    opt = torch.optim.Adam([*host.parameters(), *aux.head.parameters()], lr=1e-3)
    before = {k: p.detach().clone() for k, p in aux.head.named_parameters()}
    with strict_determinism():
        for step in range(3):
            opt.zero_grad()
            obs = torch.rand(2, 24, 32, 6, device=device)
            loss = aux.compute(host, obs)
            # A real policy readout and a separate auxiliary encoder invocation.
            host(obs).square().mean().backward(retain_graph=True)
            policy_grad = host.conv.weight.grad.clone()
            with relaxed_determinism():
                loss.backward()
            if step >= 1:
                aux_grad = host.conv.weight.grad - policy_grad
                assert torch.isfinite(aux_grad).all() and aux_grad.abs().sum() > 0
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            opt.step()
    for prefix in ("ventral", "dorsal"):
        assert any(not torch.equal(before[k], p) for k, p in aux.head.named_parameters() if k.startswith(prefix))


def test_registration_keeps_legacy_arms():
    from nett_skrl.brain.aux import EoODualAuxLoss, GWMDualAuxLoss
    from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
    from test_cltt_ref_aux import MODELS
    for label, kind, cls in [("EoO", "eoo", EoODualAuxLoss), ("GWM", "gwm", GWMDualAuxLoss)]:
        assert MODELS[f"CNN2F+{label}"]["aux"] == kind
        assert f"CNN2F+{label}-Dual" not in MODELS
        arm = MODELS[f"CNN+{label}-Dual"]
        assert arm["cfg"]["input_frames"] == 1
        assert arm["aux"] == kind + "_dual" and arm["framestack"]
        assert isinstance(AUX_LOSSES[arm["aux"]](Host()), cls)


@pytest.mark.parametrize("kind", ["eoo_dual", "gwm_dual"])
def test_ppo_registers_and_checkpoints_every_stream_parameter(monkeypatch, kind):
    from types import SimpleNamespace
    from nett_skrl.brain.aux.ppo_aux import AuxLossPPO, PPO
    host = Host()

    def init(self, *args, **kwargs):
        self.policy = SimpleNamespace(encoder=host)
        self.optimizer = torch.optim.Adam(host.parameters())
        self.checkpoint_modules = {}

    monkeypatch.setattr(PPO, "__init__", init)
    agent = AuxLossPPO(aux_loss=kind, aux_weight=1)
    expected = {id(p) for p in agent._aux.head.parameters()}
    assert {id(p) for p in agent.optimizer.param_groups[1]["params"]} == expected
    assert agent.checkpoint_modules["aux_head"] is agent._aux.head
    assert not (expected & {id(p) for p in host.parameters()})


@pytest.mark.parametrize("kind", ["eoo_dual", "gwm_dual"])
def test_production_backward_scope_restores_strict_mode(kind):
    """Execute the actual PPO backward block with a CPU nondeterminism sentinel.

    This still detects an accidental fused/strict aux backward on GPU-less CI.
    The optimizer test above additionally uses real grid_sample on CUDA if present.
    """
    import ast
    import inspect
    import textwrap
    from types import SimpleNamespace
    from nett_skrl.brain.aux.ppo_aux import AuxLossPPO
    from nett_skrl.brain.skrl_patches import strict_determinism, relaxed_determinism

    class AuxBackwardGuard(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            assert torch.is_deterministic_algorithms_warn_only_enabled()
            return grad

    class PPOBackwardGuard(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            return grad

    host, aux = build(kind)
    tree = ast.parse(textwrap.dedent(inspect.getsource(AuxLossPPO.update)))
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and isinstance(node.test, ast.BoolOp)
                 and "aux_loss.requires_grad" in ast.unparse(node.test))
    with strict_determinism():
        actual = aux.compute(host, torch.rand(2, 24, 32, 6))
        ns = dict(torch=torch, os=os, relaxed_determinism=relaxed_determinism,
                  aux_loss=AuxBackwardGuard.apply(actual),
                  _ppo_loss=PPOBackwardGuard.apply(host.conv.weight.square().sum()),
                  self=SimpleNamespace(scaler=SimpleNamespace(scale=lambda x: x)))
        exec(compile(ast.Module(body=[block], type_ignores=[]), "ppo_backward", "exec"), ns)
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert host.conv.weight.grad is not None
    assert not torch.equal(host.conv.weight.grad, 2 * host.conv.weight.detach())


@pytest.mark.parametrize("kind", ["eoo", "gwm"])
def test_vendored_reference_numeric_and_architecture_parity(kind, monkeypatch):
    from pathlib import Path
    import sys
    from nett_skrl.brain.aux.dual_stream import DualStreamHead
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    ref = Path('/home/zlaborde/code/isaac/NETT_Global_Workspace/archive/2026-08_campaign/scripts') / kind
    if not ref.exists():
        pytest.skip("Read-only vendored reference not installed on this machine")

    def load(name):
        spec = importlib.util.spec_from_file_location(f"reference_{kind}_{name}", ref / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    models, losses = load("model"), load("losses")
    ours = DualStreamHead(12)
    reference = models.EmergenceOfObjectness() if kind == "eoo" else models.GuessWhatMoves()
    reference.dorsal.load_state_dict(ours.dorsal.state_dict(), strict=True)
    assert sum(p.numel() for p in ours.dorsal.parameters()) == 75906
    a, b = torch.rand(2, 3, 24, 32), torch.rand(2, 3, 24, 32)
    rf = reference.dorsal(a, b)[0] if kind == "eoo" else reference.get_flow(a, b)
    torch.testing.assert_close(ours.dorsal.forward_single(a, b), rf, rtol=0, atol=0)
    masks = ours.get_masks(torch.rand(2, 12, 3, 4), (24, 32)).detach().requires_grad_()
    flow = (torch.rand(2, 2, 24, 32) * .1).requires_grad_()
    if kind == "eoo":
        from nett_skrl.brain.aux.eoo_dual_aux import objectness_loss
        args = (a, b, masks[:, 1:2], flow, -flow)
        expected, actual = losses.unflow_loss(*args), objectness_loss(*args)
    else:
        from nett_skrl.brain.aux.gwm_dual_aux import flow_reconstruction_loss
        expected, actual = losses.flow_reconstruction_loss(masks, flow), flow_reconstruction_loss(masks, flow)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    eg = torch.autograd.grad(expected, (masks, flow), retain_graph=True)
    ag = torch.autograd.grad(actual, (masks, flow))
    for x, y in zip(ag, eg):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["eoo_dual", "gwm_dual"])
def test_real_nature_host_receives_aux_gradient_at_real_eye(kind):
    from nett_skrl.brain.encoders.nature_cnn import NatureCNN
    from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
    host = NatureCNN(gym.spaces.Box(0, 255, (80, 128, 6), dtype=np.uint8),
                     conv_dim=75, input_frames=1)
    aux = AUX_LOSSES[kind](host)
    opt = torch.optim.Adam([*host.parameters(), *aux.head.parameters()], lr=1e-5)
    obs = torch.randint(0, 256, (2, 80, 128, 6), dtype=torch.uint8)
    # Policy gets the last frame; aux gets the same normalized frame without /255 twice.
    policy_before = host(obs).detach()
    prepared = host._prepare_frames(obs)
    spatial_inputs = []
    hook = next(m for m in host.cnn if isinstance(m, nn.Conv2d)).register_forward_pre_hook(
        lambda m, args: spatial_inputs.append(args[0].detach()))
    for step in range(3):
        opt.zero_grad()
        aux.compute(host, obs.flatten(1)).backward()
        grads = [p.grad for p in host.cnn.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        if step >= 1:
            assert sum(g.abs().sum() for g in grads) > 0
        opt.step()
    hook.remove()
    for frame in spatial_inputs:
        torch.testing.assert_close(frame, prepared, rtol=0, atol=0)
    assert not torch.equal(host(obs), policy_before)
    assert not ({id(p) for p in host.parameters()} & {id(p) for p in aux.head.parameters()})


@pytest.mark.parametrize("strict", [False, True])
def test_gwm_forward_solve_has_production_determinism_scope(monkeypatch, strict):
    from nett_skrl.brain.aux import gwm_dual_aux
    from nett_skrl.brain.skrl_patches import strict_determinism
    monkeypatch.setenv("NETT_AUX_STRICT", "1" if strict else "0")
    host, aux = build("gwm_dual")
    actual_solve = gwm_dual_aux.flow_reconstruction_loss
    observed = []

    def guarded_solve(masks, flow):
        observed.append(torch.is_deterministic_algorithms_warn_only_enabled())
        return actual_solve(masks, flow)

    monkeypatch.setattr(gwm_dual_aux, "flow_reconstruction_loss", guarded_solve)
    with strict_determinism():
        aux.compute(host, torch.rand(2, 24, 32, 6))
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert observed == [not strict]
    assert not hasattr(aux.head.ventral, "enc1")
