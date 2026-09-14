"""GWM observation masking, isolated optimization, and scalar logging contracts."""

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from nett_skrl.body import Body
from nett_skrl.body.wrappers.framestack import FrameStack
from nett_skrl.body.wrappers.gwm_seg import GwmSeg
from nett_skrl.body.wrappers.motok_seg import MoTokSeg
from nett_skrl.body.wrappers.segmentation import SegmentationObservationWrapper


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name in list(os.environ):
        if name.startswith("NETT_SEG_"):
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


class Images(gym.Env):
    num_envs = 2
    device = "cpu"

    def __init__(self, channels=6, batched=True):
        shape = (24, 32, channels)
        if batched:
            shape = (self.num_envs, *shape)
        self.observation_space = gym.spaces.Box(0, 255, shape, np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.raw = np.random.default_rng(17).integers(0, 256, shape, dtype=np.uint8)

    def reset(self, **kwargs):
        return self.raw.copy(), {}

    def step(self, action):
        return self.raw.copy(), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), {}


def test_shared_scaffolding_and_registered_arm(monkeypatch):
    assert issubclass(MoTokSeg, SegmentationObservationWrapper)
    assert issubclass(GwmSeg, SegmentationObservationWrapper)
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    import campaign_train as campaign

    spec = campaign.MODELS["CNN2F+GWM-Seg"]
    control = campaign.MODELS["CNN2F"]
    assert spec["cfg"] == control["cfg"] == {"trainable": True, "features_dim": 512, "conv_dim": 75}
    assert spec["encoder"] == control["encoder"] == "nature_cnn"
    assert spec["framestack"] and spec["seg_after"] and not spec.get("aux")
    assert campaign.segmentation_wrappers(spec) == ["framestack", "gwm_seg"]
    assert campaign.segmentation_wrappers(campaign.MODELS["MoTok-Seg2F"]) == ["motok_seg", "framestack"]
    assert campaign.segmentation_wrappers(campaign.MODELS["MoTok-Seg"]) == ["motok_seg"]
    wrapped = Body(wrappers=campaign.segmentation_wrappers(spec)).wrap(Images(3))
    assert isinstance(wrapped.env, GwmSeg)
    assert isinstance(wrapped.env.env, FrameStack)
    obs, _ = wrapped.reset()
    assert obs.shape == (2, 6, 24, 32)


@pytest.mark.parametrize("channels", [1, 3, 5])
def test_requires_framestack(channels):
    env = GwmSeg(Images(channels))
    with pytest.raises(ValueError, match="GwmSeg needs framestack=True and must be ordered AFTER framestack"):
        env.reset()
    assert env._model is None


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("chw", [True, False])
@pytest.mark.parametrize("tensor", [True, False])
def test_layout_and_gradient_isolation(batched, chw, tensor):
    env = GwmSeg(Images(batched=batched))
    raw = env.env.raw
    if chw:
        raw = np.moveaxis(raw, -1, -3)
    if tensor:
        raw = torch.from_numpy(raw)
    env._ensure(3)
    original = env._model.get_masks
    seen = []

    def checked(frame):
        masks = original(frame)
        seen.append(masks.requires_grad)
        return masks

    env._model.get_masks = checked
    result = env.observation({"policy": raw, "critic": "unchanged"})
    out = result["policy"]
    assert result["critic"] == "unchanged"
    assert out.shape == raw.shape and out.dtype == raw.dtype
    assert isinstance(out, torch.Tensor) == tensor
    assert seen == [False, False]
    assert env.last_stats["seg/fg_slot"] in (0, 1)
    assert not np.array_equal(np.asarray(out), np.asarray(raw))
    assert np.asarray(out).max() > 0
    # A real policy backward cannot reach any perception parameter.
    policy = nn.Conv2d(6, 1, 1)
    image = torch.as_tensor(out).float()
    if not batched:
        image = image.unsqueeze(0)
    if not chw:
        image = image.permute(0, 3, 1, 2)
    policy(image).mean().backward()
    assert policy.weight.grad is not None
    assert all(p.grad is None for p in env._model.parameters())
    # Positive control: inference without the wrapper does carry a graph.
    assert original(torch.rand(1, 3, 24, 32)).requires_grad


@pytest.mark.parametrize("cls", [GwmSeg, MoTokSeg])
def test_guard_raises_on_leaked_mask(cls):
    env = cls(Images())
    env._ensure(3)

    def leaking(frame):
        with torch.enable_grad():
            return torch.ones(frame.shape[0], 2, *frame.shape[2:], requires_grad=True)

    env._model.get_masks = leaking
    with pytest.raises(RuntimeError, match="mask carries requires_grad=True"):
        env.reset()


def test_mask_controls_observation_and_buffer_keeps_raw_pairs(monkeypatch):
    monkeypatch.setenv("NETT_SEG_FG_SLOT", "1")
    monkeypatch.setenv("NETT_SEG_BUFFER", "2")
    env = GwmSeg(Images(channels=9))
    env._ensure(3)
    for value in (1.0, 0.0, 0.0):
        def forced(frame):
            masks = torch.zeros(frame.shape[0], 2, *frame.shape[2:])
            masks[:, 1] = value
            masks[:, 0] = 1 - value
            return masks
        env._model.get_masks = forced
        out, _ = env.reset()
        np.testing.assert_array_equal(out, env.env.raw if value else np.zeros_like(out))
        assert env.last_stats["seg/fg_slot"] == 1
    assert len(env._buf) == 2 and env._seen == 3
    raw = torch.from_numpy(env.env.raw).permute(0, 3, 1, 2).float() / 255
    expected = torch.cat((raw[:, :3], raw[:, -3:]), dim=1)
    for sample in env._buf:
        torch.testing.assert_close(sample, expected, rtol=0, atol=0)
        assert not sample.requires_grad


def test_real_step_uses_both_frames_and_two_learning_rates(monkeypatch):
    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "1")
    env = GwmSeg(Images())
    env.reset()
    groups = env._optim.param_groups
    assert isinstance(env._optim, torch.optim.AdamW) and len(groups) == 2
    ventral, dorsal = groups
    assert ventral["lr"] == 1e-4 and dorsal["lr"] == 1e-5
    assert ventral["lr"] / dorsal["lr"] == pytest.approx(10)
    assert {id(p) for p in ventral["params"]} == {id(p) for p in env._model.ventral.parameters()}
    assert {id(p) for p in dorsal["params"]} == {id(p) for p in env._model.dorsal.parameters()}
    assert all(g["weight_decay"] == 1e-4 for g in groups)
    before = {k: p.detach().clone() for k, p in env._model.named_parameters()}
    actual_step = env._optim.step
    actual_flow = env._model.dorsal.forward_single
    actual_loss = __import__("nett_skrl.body.wrappers.gwm_seg", fromlist=["flow_reconstruction_loss"]).flow_reconstruction_loss
    calls, flows, losses = [], [], []

    def step(*args, **kwargs):
        calls.append([g["lr"] for g in env._optim.param_groups])
        return actual_step(*args, **kwargs)

    def flow(prev, curr):
        # Pair must be raw and ordered, including under rollout's no_grad.
        raw = env._buf[0].repeat(2, 1, 1, 1)
        torch.testing.assert_close(prev, raw[:, :3])
        torch.testing.assert_close(curr, raw[:, -3:])
        result = actual_flow(prev, curr)
        assert not torch.allclose(result, actual_flow(curr, prev))
        assert not torch.allclose(result, actual_flow(torch.zeros_like(prev), curr))
        assert not torch.allclose(result, actual_flow(prev, torch.zeros_like(curr)))
        flows.append(result.detach())
        return result

    def loss(masks, flow, reg):
        assert reg == 1e-4
        torch.testing.assert_close(masks, env._model.get_masks(env._buf[0].repeat(2, 1, 1, 1)[:, -3:]))
        result = actual_loss(masks, flow, reg=reg)
        losses.append(result.item())
        return result

    monkeypatch.setattr(env._optim, "step", step)
    monkeypatch.setattr(env._model.dorsal, "forward_single", flow)
    monkeypatch.setattr("nett_skrl.body.wrappers.gwm_seg.flow_reconstruction_loss", loss)
    # Real periodic train_step inside skrl's ambient rollout mode.
    with torch.no_grad():
        env.step(0)
    assert calls == [[1e-4, 1e-5]]
    for stream in ("ventral", "dorsal"):
        assert any(not torch.equal(before[k], p) for k, p in env._model.named_parameters() if k.startswith(stream))
        assert sum(p.grad.abs().sum() for p in getattr(env._model, stream).parameters()) > 0
    assert torch.sqrt(sum(p.grad.square().sum() for p in env._model.parameters())) <= 1.00001
    stats = env.last_stats
    assert stats["seg/train_steps"] == 1 and not env._model.training
    assert stats["seg/recon_loss"] == stats["seg/loss"] == losses[0]
    assert stats["seg/flow_spatial_std"] == pytest.approx(flows[0].std((2, 3), correction=0).mean().item())
    assert stats["seg/flow_absmax"] == flows[0].abs().max().item()
    assert 0 < stats["seg/fg_area"] < 1
    assert all(np.isfinite(v) for v in stats.values())


def test_equal_learning_rates_are_refused(monkeypatch):
    monkeypatch.setenv("NETT_SEG_BACKBONE_LR", "1e-4")
    with pytest.raises(ValueError, match="NETT_SEG_LR / NETT_SEG_BACKBONE_LR = 10"):
        GwmSeg(Images())


def test_constant_flow_collapse_is_visible():
    env = GwmSeg(Images())
    env.reset()
    env.reset()
    # Deliberately different u/v AND batch offsets: global std is nonzero,
    # but each image/component is spatially constant and must log collapse.
    def constant(prev, curr):
        return torch.arange(prev.shape[0] * 2).reshape(-1, 2, 1, 1).float().expand(-1, -1, *prev.shape[2:])
    env._model.dorsal.forward_single = constant
    assert env.train_step() is not None
    assert env.last_stats["seg/flow_spatial_std"] == 0.0
    assert env.last_stats["seg/flow_absmax"] > 0


def test_frame_pairs_reuse_framestack_episode_resets():
    raw = Images(3)
    env = GwmSeg(FrameStack(raw, n_stack=2))
    env.reset()
    previous = raw.raw.copy()
    raw.raw[:] = 123
    raw.step = lambda action: (raw.raw.copy(), np.zeros(2), np.array([True, False]), np.zeros(2, bool), {})
    env.step(0)
    pair = env._buf[-1]
    torch.testing.assert_close(pair[0, :3], pair[0, -3:])
    torch.testing.assert_close(pair[1, :3], torch.from_numpy(previous[1]).permute(2, 0, 1).float() / 255)
    assert torch.all(pair[1, -3:] == 123 / 255)


def test_ventral_matches_read_only_reference():
    from nett_skrl.brain.aux.dual_stream import SmallCNNVentral
    reference = Path("/home/zlaborde/.claude/uploads/934895bd-06a3-4110-a59a-6243fbfa121d/b46532c0-model.py")
    if not reference.exists():
        pytest.skip("Owner's read-only model reference is not installed")
    # Compile the read-only source directly; do not create a __pycache__ there.
    namespace = {}
    exec(compile(reference.read_text(), str(reference), "exec"), namespace)
    ref = namespace["SmallCNNVentral"](num_out_channels=2)
    actual = SmallCNNVentral()
    actual.load_state_dict(ref.state_dict(), strict=True)
    frame = torch.rand(1, 3, 80, 128)
    torch.testing.assert_close(actual(frame), ref(frame), rtol=0, atol=0)


@pytest.mark.parametrize("cls", [GwmSeg, MoTokSeg])
def test_train_step_diagnostics_reach_run_scalar_logs(cls, monkeypatch, tmp_path):
    from collections import defaultdict
    from skrl.agents.torch.base import Agent
    from skrl.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from nett_skrl.body.skrl_adapter import IsaacEnvWrapper
    from nett_skrl.body.wrappers.channels_first import ChannelsFirst
    from nett_skrl.brain.trainer import BrainTrainer

    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "1")
    seg = cls(Images(channels=6 if cls is GwmSeg else 3))
    env = IsaacEnvWrapper(ChannelsFirst(seg), device="cpu")

    class ScalarAgent:
        # Use the actual skrl scalar accumulator and disk writer.
        track_data = Agent.track_data
        write_tracking_data = Agent.write_tracking_data

        def __init__(self, directory):
            self.tracking_data = defaultdict(list)
            self._track_rewards, self._track_timesteps = [], []
            self.writer = SummaryWriter(log_dir=str(directory))

    agents = [ScalarAgent(tmp_path / str(i)) for i in range(2)]

    class UnitSequentialTrainer:
        def __init__(self, *, env, agents, scopes, cfg):
            self.env = env

        def train(self):
            self.env.reset()
            with torch.no_grad():
                self.env.step(torch.zeros(2, 1, dtype=torch.long))

    monkeypatch.setattr("nett_skrl.brain.trainer.SequentialTrainer", UnitSequentialTrainer)
    # Exercise the production trainer's binding, real adapter, real segmenter
    # train_step, skrl.track_data, disk write, and an independent scalar reader.
    BrainTrainer(env, agents, device="cpu")._run_skrl_train(env, timesteps=1)
    assert seg.last_stats["seg/train_steps"] == 1
    for i, agent in enumerate(agents):
        assert dict(agent.tracking_data) == {k: [v] for k, v in seg.last_stats.items()}
        agent.write_tracking_data(timestep=1, timesteps=1)
        agent.writer.close()
        events = EventAccumulator(str(tmp_path / str(i))).Reload()
        for key, value in seg.last_stats.items():
            assert events.Scalars(key)[-1].value == pytest.approx(value)


# ─────────────────────────────────────────────────────────────────────────────
# Keep-mask rule. Consolidated from test_seg_mask_rule.py: one build per slot
# count checks the rule, the background choice, AND the retained content,
# instead of three tests rebuilding the same wrapper to check one facet each.
# ─────────────────────────────────────────────────────────────────────────────

def _slot_masks(queries, monkeypatch, rule=None):
    monkeypatch.setenv("NETT_SEG_QUERIES", str(queries))
    if rule is not None:
        monkeypatch.setenv("NETT_SEG_MASK_RULE", rule)
    env = GwmSeg(Images())
    env._ensure(3)
    with torch.no_grad():
        return env, env._model.get_masks(torch.rand(4, 3, 24, 32))


@pytest.mark.parametrize("queries", [2, 3, 5])
def test_keep_mask_rule_follows_slot_count(queries, monkeypatch):
    """⛔ KEEPING ONE SLOT AT K>2 CAN DELETE A TEST ALTERNATIVE.

    The parsing test shows TWO objects on two monitors plus the chamber. With K>2
    a single-slot rule suppresses every other slot, so if the objects land in
    different slots the agent is shown ONE option in a two-alternative forced
    choice and scores at chance for a reason unrelated to the hypothesis. K=2 must
    stay byte-identical to the reference, or the Unity baseline moves.
    """
    env, masks = _slot_masks(queries, monkeypatch)
    keep = env._keep_mask(masks)
    slot = int(env.last_stats["seg/selected_slot"])
    areas = masks.mean(dim=(0, 2, 3))

    if queries == 2:
        assert not env.last_stats["seg/mask_rule_not_background"]
        torch.testing.assert_close(keep, masks[:, slot:slot + 1], rtol=0, atol=0)
        assert float(areas[slot]) <= float(areas[1 - slot])          # foreground = smaller
    else:
        assert env.last_stats["seg/mask_rule_not_background"]
        assert slot == int(torch.argmax(areas).item())               # background = largest
        others = sum(masks[:, j] for j in range(queries) if j != slot)
        torch.testing.assert_close(keep[:, 0], others, rtol=1e-4, atol=1e-5)
        assert float(keep.mean()) > max(
            float(areas[j]) for j in range(queries) if j != slot
        )
    # legacy keys stay populated whatever the rule -- renaming them once broke 14
    # tests including MoTok's acceptance test, whose logged stats ARE its behaviour.
    assert env.last_stats["seg/fg_slot"] in range(queries)
    assert 0.0 <= env.last_stats["seg/kept_area"] <= 1.0


def test_mask_rule_override_and_rejection(monkeypatch):
    env, masks = _slot_masks(5, monkeypatch, rule="foreground")
    env._keep_mask(masks)
    assert not env.last_stats["seg/mask_rule_not_background"]
    monkeypatch.setenv("NETT_SEG_MASK_RULE", "whatever")
    with pytest.raises(ValueError, match="NETT_SEG_MASK_RULE"):
        GwmSeg(Images())


@pytest.mark.parametrize("queries", [2, 5])
def test_slot_dilution_is_measurable(queries, monkeypatch):
    """Dilution is invisible in the loss. Measured at 150 steps on real frames, no
    slot ever dies but confidently-assigned pixels fall 0.79 -> 0.18 from K=2 to
    K=5 while the closest slot pair reaches 0.906 cosine."""
    env, masks = _slot_masks(queries, monkeypatch)
    stats = env.slot_diagnostics(masks)
    assert set(stats) == {
        "seg/confident_pixels", "seg/slot_pair_cosine_max",
        "seg/slot_occ_min", "seg/slot_occ_max",
    }
    assert 0.0 <= stats["seg/confident_pixels"] <= 1.0
    assert -1.0 <= stats["seg/slot_pair_cosine_max"] <= 1.0 + 1e-6
    assert stats["seg/slot_occ_min"] <= stats["seg/slot_occ_max"]
