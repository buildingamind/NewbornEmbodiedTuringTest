"""I-JEPA / V-JEPA (owner 2026-10-15): encoders/jepa.py + aux/jepa_aux.py.

Pins the parameter counts at the Unity-recipe++ eye, checks the position embeddings against the
reference repos' own functions (copied verbatim below), the mask samplers' invariants, and that
the loss trains the encoder and predictor but never the EMA target. Each guarantee has a mutant
or a positive control next to it.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.aux.jepa_aux import (IJEPA_MIN_KEEP, IJEPA_NPRED, VJEPA_MASKS, JEPAPredictor, JEPATerm,
                                          ijepa_masks, vjepa_masks)
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.jepa import JEPAEncoder, sincos_2d, sincos_3d
from nett_skrl.brain.registry import encoder_mapping

EYE_H, EYE_W = 280, 448          # Unity-recipe++ NETT_EYE_RES 448x280
SMALL_H, SMALL_W = 112, 168      # 8 x 12 tokens at patch 14, for the CPU training tests
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"

IJEPA_CFG = dict(mode="ijepa", pred_dim=48, pred_depth=3, pred_heads=2, ema=0.996, aux_batch=128, loss="smooth_l1")
VJEPA_CFG = dict(mode="vjepa", pred_dim=48, pred_depth=3, pred_heads=2, ema=0.998, aux_batch=128, loss="l1")


def _enc(frames, h=EYE_H, w=EYE_W, depth=None, **jepa):
    cfg = dict(IJEPA_CFG if frames == 1 else VJEPA_CFG, **jepa)
    depth = depth if depth is not None else (3 if frames == 1 else 2)
    probe = "mean_linear" if cfg["mode"] == "ijepa" else "attentive"
    return JEPAEncoder(gym.spaces.Box(0, 255, (h, w, 3 * frames), np.uint8), num_frames=frames, jepa=cfg, probe=probe,
                       tubelet=1, features_dim=128, patch_size=14, embed_dim=96, depth=depth, num_heads=4)


def _count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ── reference functions, verbatim (ijepa src/models/vision_transformer.py; jepa src/models/utils/pos_embs.py)
def _ref_1d(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _ref_2d(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return np.concatenate([_ref_1d(embed_dim // 2, grid[0]), _ref_1d(embed_dim // 2, grid[1])], axis=1)


def _ref_3d(embed_dim, grid_size, grid_depth):
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)
    d = int(np.ceil(embed_dim / 6) * 2)                     # uniform_power=True
    emb = np.concatenate([_ref_1d(d, grid_d), _ref_1d(d, grid_h), _ref_1d(d, grid_w)], axis=1)
    return emb[:, :embed_dim]


# ═════════════════════════════════════════════════════════════════════ sizes
def test_param_counts_at_the_unity_recipe_eye():
    """Owner answer 3: encoder + probe + predictor ~500k, EMA target not counted."""
    i, v = _enc(1), _enc(2)
    assert (i.n_h, i.n_w, i.n_tokens) == (20, 32, 640) and (v.n_t, v.n_tokens) == (2, 1280)
    assert _count(i) == 404_672           # I-JEPA depth 3 + mean-pool linear readout
    assert _count(v) == 404_768           # V-JEPA depth 2 + attentive probe (one cross-attention block + query)
    assert _count(JEPATerm(i, "ijepa").head) == 94_320
    assert _count(JEPATerm(v, "vjepa").head) == 94_368
    assert _count(i) + 94_320 == 498_992 and _count(v) + 94_368 == 499_136
    # tubelet 1 = a per-frame patch embed: the Conv3d has exactly a Conv2d's parameters
    assert _count(v.patch_embed) == _count(i.patch_embed) == 3 * 14 * 14 * 96 + 96
    # the fixed position table is not a parameter
    assert "pos_embed" not in dict(i.named_parameters()) and "pos_embed" not in i.state_dict()


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _label_space(frames):
    return gym.spaces.Box(0, 255, (EYE_H, EYE_W, 3 * frames), np.uint8)


def test_models_labels_build_the_pinned_encoders(campaign):
    assert encoder_mapping["jepa"] is JEPAEncoder
    assert {"ijepa", "vjepa"} <= set(AUX_LOSSES)
    for label, frames, n in (("I-JEPA", 1, 404_672), ("V-JEPA", 2, 404_768)):
        m = campaign.MODELS[label]
        assert m["framestack"] is (frames == 2) and m["aux"] == ("ijepa" if frames == 1 else "vjepa")
        assert "head" not in m                                       # default head, as CNN-UnityRecipe
        kw = EncoderCfg(**m["cfg"]).as_kwargs()
        kw.pop("trainable")
        enc = encoder_mapping[m["encoder"]](_label_space(frames), **kw)
        assert _count(enc) == n, label
        assert type(AUX_LOSSES[m["aux"]](enc)) is JEPATerm



@pytest.mark.parametrize("label,frames,enc_n,pred_n", [("I-JEPA-13M", 1, 10_923_008, 2_372_928),
                                                       ("V-JEPA-13M", 2, 10_923_392, 2_373_120)])
def test_13m_labels_match_cnn_unityrecipe_budget(campaign, label, frames, enc_n, pred_n):
    """Owner answer 3: ~13.2M versions, sized to CNN-UnityRecipe's encoder at 448x280 (measured 13,281,568)."""
    m = campaign.MODELS[label]
    kw = EncoderCfg(**m["cfg"]).as_kwargs()
    kw.pop("trainable")
    enc = encoder_mapping["jepa"](_label_space(frames), **kw)
    pred = AUX_LOSSES[m["aux"]](enc).head
    assert (_count(enc), _count(pred)) == (enc_n, pred_n)
    assert abs(enc_n + pred_n - 13_281_568) / 13_281_568 < 0.002


@pytest.mark.parametrize("label,frames", [("I-JEPA", 1), ("V-JEPA", 2)])
def test_builder_keeps_the_reference_init(campaign, label, frames):
    """Through build_models_for_algorithm as campaign_train assembles a head-less label: the encoder's
    own trunc_normal(0.02) init and the sqrt(2*layer_id) rescale survive (OWNS_INIT), and policy and
    value share it."""
    from nett_skrl.brain.brain import Brain
    from nett_skrl.brain.models.builder import build_models_for_algorithm
    from nett_skrl.brain.models.model_cfg import ModelCfg
    from nett_skrl.brain.registry import algorithm_spec
    spec = campaign.MODELS[label]
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    torch.manual_seed(0)
    models = build_models_for_algorithm(
        algorithm_spec(Brain(encoder="jepa", algorithm="PPO").algorithm),
        encoder_cls=encoder_mapping["jepa"], encoder_kwargs=kw, observation_space=_label_space(frames),
        action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32), device=torch.device("cpu"),
        cfg=ModelCfg(value_bound=None, shared_encoder=True, hidden_sizes=[], clip_actions=False))
    enc = models["policy"].encoder
    assert type(enc) is JEPAEncoder and enc is models["value"].encoder
    assert abs(float(enc.patch_embed.weight.std()) - 0.02) < 0.002
    # torch/timm trunc_normal_ bounds are ABSOLUTE (+-2), so the max is no test; zero biases are: a
    # builder re-init (PyTorch default uniform) would leave every bias non-zero.
    assert all(float(m.bias.abs().max()) == 0.0 for m in enc.modules()
               if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)))
    for i, blk in enumerate(enc.blocks, start=1):
        want = 0.02 / (2.0 * i) ** 0.5
        assert abs(float(blk.proj.weight.std()) - want) < 0.15 * want, (label, i)
        assert abs(float(blk.mlp[2].weight.std()) - want) < 0.15 * want, (label, i)


@pytest.mark.parametrize("aux", ["ijepa", "vjepa"])
def test_auxlossppo_adds_only_the_predictor_and_checkpoints_nothing(monkeypatch, aux):
    from types import SimpleNamespace
    from nett_skrl.brain.aux.ppo_aux import PPO, AuxLossPPO
    enc = _enc(1 if aux == "ijepa" else 2, SMALL_H, SMALL_W)

    def init(self, *args, **kwargs):
        self.policy = SimpleNamespace(encoder=enc)
        self.optimizer = torch.optim.Adam(enc.parameters())
        self.checkpoint_modules = {}
        self.memory = None

    monkeypatch.setattr(PPO, "__init__", init)
    agent = AuxLossPPO(aux_loss=aux, aux_weight=1)
    assert isinstance(agent._aux, JEPATerm)
    head = {id(p) for p in agent._aux.head.parameters()}
    assert {id(p) for p in agent.optimizer.param_groups[1]["params"]} == head
    assert not head & {id(p) for p in enc.parameters()}
    teacher = {id(p) for p in agent._aux.teacher.module.parameters()}
    assert not teacher & {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
    assert "aux_head" not in agent.checkpoint_modules


@pytest.mark.parametrize("frames,mode", [(1, "ijepa"), (2, "vjepa")])
def test_loss_is_repeatable_under_strict_determinism(frames, mode):
    """Same seed -> same masks, same loss, under use_deterministic_algorithms(True). CPU only."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    losses = []
    obs = torch.randint(0, 256, (6, SMALL_H, SMALL_W, 3 * frames), dtype=torch.uint8,
                        generator=torch.Generator().manual_seed(3)).float()
    try:
        torch.use_deterministic_algorithms(True)
        for _ in range(2):
            torch.manual_seed(5)
            enc = _enc(frames, SMALL_H, SMALL_W, aux_batch=4)
            losses.append(JEPATerm(enc, mode).compute(enc, obs))
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
    assert torch.equal(losses[0].detach(), losses[1].detach())


# ═════════════════════════════════════════════════════════════ positions
@pytest.mark.parametrize("dim,g", [(96, 6), (48, 5), (16, 3)])
def test_sincos_2d_is_the_ijepa_table(dim, g):
    np.testing.assert_allclose(sincos_2d(dim, g, g).numpy(), _ref_2d(dim, g), atol=1e-6)


@pytest.mark.parametrize("dim,g,t", [(96, 6, 1), (96, 4, 2), (48, 5, 1)])
def test_sincos_3d_is_the_vjepa_uniform_power_table(dim, g, t):
    np.testing.assert_allclose(sincos_3d(dim, t, g, g).numpy(), _ref_3d(dim, g, t), atol=1e-6)


def test_sincos_rectangular_grid_is_row_major_with_w_first():
    n_h, n_w, dim = 3, 5, 16
    emb = sincos_2d(dim, n_h, n_w).numpy()
    for r in range(n_h):
        for c in range(n_w):
            want = np.concatenate([_ref_1d(dim // 2, np.array([float(c)])), _ref_1d(dim // 2, np.array([float(r)]))], 1)[0]
            np.testing.assert_allclose(emb[r * n_w + c], want, atol=1e-6)
    # mutant: transposing the grid must be caught
    assert not np.allclose(sincos_2d(dim, n_w, n_h).numpy()[:n_w], emb[:n_w])


# ═══════════════════════════════════════════════════════════════════ masks
def test_ijepa_masks_shape_bounds_and_disjointness():
    g = torch.Generator().manual_seed(0)
    n_h, n_w, b = 20, 32, 64
    ctx, tgts = ijepa_masks(g, b, n_h, n_w)
    assert len(tgts) == IJEPA_NPRED and ctx.shape[0] == b and ctx.shape[1] > IJEPA_MIN_KEEP
    n = n_h * n_w
    for t in tgts:
        assert t.shape[0] == b and 0 < t.shape[1] <= int(0.2 * n)
        assert int(t.min()) >= 0 and int(t.max()) < n
    assert int(ctx.min()) >= 0 and int(ctx.max()) < n
    disjoint = sum(not (set(ctx[i].tolist()) & set().union(*(t[i].tolist() for t in tgts))) for i in range(b))
    assert disjoint == b
    # reproducible from the generator, and a different seed draws different masks
    ctx2, tgts2 = ijepa_masks(torch.Generator().manual_seed(0), b, n_h, n_w)
    assert torch.equal(ctx, ctx2) and all(torch.equal(a, c) for a, c in zip(tgts, tgts2))
    ctx3, _ = ijepa_masks(torch.Generator().manual_seed(1), b, n_h, n_w)
    assert not (ctx.shape == ctx3.shape and torch.equal(ctx, ctx3))


@pytest.mark.parametrize("num_blocks,scale", VJEPA_MASKS)
def test_vjepa_masks_partition_the_grid(num_blocks, scale):
    g = torch.Generator().manual_seed(0)
    n_h, n_w, b = 20, 32, 32
    ctx, tgt = vjepa_masks(g, b, n_h, n_w, num_blocks, scale)
    n = n_h * n_w
    assert ctx.shape[0] == tgt.shape[0] == b
    assert ctx.shape[1] + tgt.shape[1] <= n and tgt.shape[1] >= int(n * scale) * 0.5
    for i in range(b):
        assert not (set(ctx[i].tolist()) & set(tgt[i].tolist()))
    # the long-range config masks most of the grid; the short-range one leaves most of it as context
    if scale == 0.7:
        assert tgt.shape[1] > ctx.shape[1]
    else:
        assert ctx.shape[1] > tgt.shape[1] / 3


# ═══════════════════════════════════════════════════════ V-JEPA tubelet order
def test_vjepa_tubelet_1_tokenises_each_frame_oldest_first():
    """Tokens 0..N-1 are the OLDEST frame (t=0), N..2N-1 the newest, each from its own frame only."""
    torch.manual_seed(0)
    enc = _enc(2, SMALL_H, SMALL_W)
    n = enc.n_spatial
    a = torch.randint(0, 256, (2, SMALL_H, SMALL_W, 6), dtype=torch.uint8)
    b = a.clone()
    b[..., 3:] = torch.randint(0, 256, b[..., 3:].shape, dtype=torch.uint8)   # change the NEWEST frame only
    with torch.no_grad():
        ea, eb = enc._embed(enc._prepare_image(a.float())), enc._embed(enc._prepare_image(b.float()))
    assert ea.shape[1] == 2 * n
    assert torch.equal(ea[:, :n], eb[:, :n]) and not torch.allclose(ea[:, n:], eb[:, n:])
    # positions: the two frames share h/w channels and differ in the t channels
    pos = enc.pos_embed[0]
    assert not torch.equal(pos[:n], pos[n:]) and torch.equal(pos[:n, 32:], pos[n:, 32:])


def test_vjepa_masks_repeat_over_time():
    # batch 1: with b > 1 the t-major list is truncated to the batch minimum (as the reference does), so the
    # second frame's indices of a longer row are cut first and the rows are no longer a plain repeat.
    ctx, tgt = vjepa_masks(torch.Generator().manual_seed(0), 1, 8, 12, 8, 0.15, n_t=2)
    ctx1, tgt1 = vjepa_masks(torch.Generator().manual_seed(0), 1, 8, 12, 8, 0.15, n_t=1)
    assert torch.equal(ctx, torch.cat([ctx1, ctx1 + 96], 1)) and torch.equal(tgt, torch.cat([tgt1, tgt1 + 96], 1))


def test_attentive_probe_is_the_reference_classifier():
    """jepa AttentiveClassifier: query trunc_normal, block rescaled at layer 1, the readout reads every token."""
    torch.manual_seed(0)
    enc = _enc(2, SMALL_H, SMALL_W)
    assert enc.probe == "attentive" and enc.probe_query.shape == (1, 1, 96)
    assert abs(float(enc.probe_block.proj.weight.std()) - 0.02 / 2 ** 0.5) < 0.003
    tok = torch.randn(3, enc.n_tokens, 96, requires_grad=True)
    enc._readout(tok).sum().backward()
    assert bool((tok.grad.abs().sum(-1) > 0).all())
    assert _enc(1, SMALL_H, SMALL_W).probe == "mean_linear"
    with pytest.raises(ValueError, match="probe"):
        JEPAEncoder(gym.spaces.Box(0, 255, (SMALL_H, SMALL_W, 3), np.uint8), num_frames=1, jepa=IJEPA_CFG,
                    probe="attentive")


# ═══════════════════════════════════════════════════════════════ training
@pytest.mark.parametrize("frames,mode", [(1, "ijepa"), (2, "vjepa")])
def test_loss_trains_encoder_and_predictor_never_the_target(frames, mode):
    torch.manual_seed(0)
    enc = _enc(frames, SMALL_H, SMALL_W, aux_batch=4)
    term = JEPATerm(enc, mode)
    obs = torch.randint(0, 256, (6, SMALL_H, SMALL_W, 3 * frames), dtype=torch.uint8).float()
    loss = term.compute(enc, obs)
    assert loss.ndim == 0 and torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    assert enc.patch_embed.weight.grad is not None and enc.patch_embed.weight.grad.abs().sum() > 0
    assert enc.blocks[0].qkv.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in term.head.parameters() if p.requires_grad)
    assert all(p.grad is None and not p.requires_grad for p in term.teacher.module.parameters())
    # the teacher is not registered on the term (it must not enter the PPO optimiser)
    assert not any(p is q for p in term.parameters() for q in term.teacher.module.parameters())
    # the readout is not on the JEPA path: the policy trains it, the JEPA loss cannot
    assert enc.head.weight.grad is None
    s = term.last_scalars
    assert s["B"] == 4 and s["n_ctx"] > 0 and s["n_tgt"] > 0 and s["target_std"] > 0
    assert s["ema"] == pytest.approx(0.996 if mode == "ijepa" else 0.998)


@pytest.mark.parametrize("frames,mode", [(1, "ijepa"), (2, "vjepa")])
def test_target_is_the_ema_teacher_not_the_live_encoder(frames, mode):
    """Positive control: after the live encoder moves, a decay-1.0 teacher (frozen at init) and a
    decay-0.0 teacher (= live) must give different losses on the same masks. If the loss read the
    live encoder for its target, both would agree."""
    torch.manual_seed(0)
    enc = _enc(frames, SMALL_H, SMALL_W, aux_batch=4)
    frozen, live = JEPATerm(enc, mode), JEPATerm(enc, mode)
    live.head.load_state_dict(frozen.head.state_dict())
    frozen.teacher.decay, live.teacher.decay = 1.0, 0.0
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(0.05 * torch.randn_like(p))
    obs = torch.randint(0, 256, (4, SMALL_H, SMALL_W, 3 * frames), dtype=torch.uint8).float()
    frozen._gen.manual_seed(7)
    live._gen.manual_seed(7)
    a, b = frozen.compute(enc, obs).item(), live.compute(enc, obs).item()
    assert a != pytest.approx(b, rel=1e-4)
    # and the decay-0 teacher now equals the live encoder; the decay-1 one does not
    assert all(torch.equal(t, s) for t, s in zip(live.teacher.module.parameters(), enc.parameters()))
    assert not all(torch.equal(t, s) for t, s in zip(frozen.teacher.module.parameters(), enc.parameters()))


# ═══════════════════════════════════════════════════════════════ refusals
def test_refusals():
    with pytest.raises(ValueError, match="unknown"):
        _enc(1, SMALL_H, SMALL_W, typo=1)
    bad = dict(IJEPA_CFG)
    del bad["ema"]
    with pytest.raises(ValueError, match="missing"):
        JEPAEncoder(gym.spaces.Box(0, 255, (SMALL_H, SMALL_W, 3), np.uint8), num_frames=1, jepa=bad, probe="mean_linear")
    with pytest.raises(ValueError, match="is ijepa"):
        _enc(1, SMALL_H, SMALL_W, mode="vjepa")
    with pytest.raises(ValueError, match="RGB"):
        JEPAEncoder(gym.spaces.Box(0, 255, (SMALL_H, SMALL_W, 6), np.uint8), num_frames=1, jepa=IJEPA_CFG, probe="mean_linear")
    with pytest.raises(ValueError, match="divide"):
        _enc(1, 100, SMALL_W)
    enc = _enc(1, SMALL_H, SMALL_W)
    with pytest.raises(ValueError):
        JEPATerm(enc, "vjepa")
    with pytest.raises(ValueError, match="loss"):
        JEPATerm(_enc(1, SMALL_H, SMALL_W, loss="l1"), "ijepa")
    with pytest.raises(TypeError):
        JEPATerm(torch.nn.Linear(2, 2), "ijepa")


def test_predictor_output_is_block_major():
    torch.manual_seed(0)
    enc = _enc(1, SMALL_H, SMALL_W)
    pred = JEPAPredictor(enc, 16, 1, 2, num_mask_tokens=1, zero_mask_tokens=False)
    z = torch.randn(3, 5, 96)
    ctx = torch.arange(5).expand(3, 5)
    t1, t2 = torch.arange(10, 14).expand(3, 4), torch.arange(20, 24).expand(3, 4)
    out = pred(z, ctx, [t1, t2], 0)
    assert out.shape == (6, 4, 96)
    torch.testing.assert_close(out[:3], pred(z, ctx, [t1], 0))
    torch.testing.assert_close(out[3:], pred(z, ctx, [t2], 0))
