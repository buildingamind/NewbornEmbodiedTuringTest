"""Owner, workspace DECISIONS §162 add. 3: the ~500K-parameter rungs of the §162 CLTT-Ref labels, and the
parameter table of every §162 label (encoder / policy+value heads / aux, counted separately).

Each count is measured through the PRODUCTION builder (Brain -> build_agents, campaign_train's model block:
shared_encoder True, hidden_sizes []) at obs (6, 80, 128) uint8 -- (12, 80, 128) for the -DVS6 labels -- and
pinned here. Rungs move ONE ladder axis (3DCNN conv_dim, ViViT embed_dim); campaign_train.py asserts that at
import and these tests re-check it. Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
DVS_ENV = {"NETT_DVS_COLOR": "1", "NETT_AUX_CLTT_CHANNELS_PER_FRAME": "6", "NETT_DVS_THRESHOLD": "24",
           "NETT_DVS_BLUR_KERNEL": "5"}

#: label: (encoder, policy+value heads, aux module, total) -- measured, production builder, 128x80
TABLE = {
    "3DCNN-CLTT-Ref":              (695_981, 1_541, 329_216, 1_026_738),
    "3DCNN-CLTT-Ref-T1":           (713_549, 1_541, 329_216, 1_044_306),
    "3DCNN-CLTT-Ref-DVS6":         (697_709, 1_541, 329_216, 1_028_466),
    "3DCNN-CLTT-Ref-T1-DVS6":      (714_413, 1_541, 329_216, 1_045_170),
    "ViViT-CLTT-Ref":              (694_016, 1_541, 329_216, 1_024_773),
    "ViViT-CLTT-Ref-T2":           (804_320, 1_541, 329_216, 1_135_077),
    "ViViT-CLTT-Ref-DVS6":         (804_608, 1_541, 329_216, 1_135_365),
    "ViViT-CLTT-Ref-T2-DVS6":      (1_025_504, 1_541, 329_216, 1_356_261),
    "3DCNN-500K-CLTT-Ref":         (503_063, 1_541, 329_216, 833_820),
    "3DCNN-500K-CLTT-Ref-T1":      (520_631, 1_541, 329_216, 851_388),
    "3DCNN-500K-CLTT-Ref-DVS6":    (504_791, 1_541, 329_216, 835_548),
    "3DCNN-500K-CLTT-Ref-T1-DVS6": (521_495, 1_541, 329_216, 852_252),
    "ViViT-500K-CLTT-Ref":         (509_312, 1_541, 329_216, 840_069),
    "ViViT-500K-CLTT-Ref-T2":      (510_056, 1_541, 329_216, 840_813),
    "ViViT-500K-CLTT-Ref-DVS6":    (510_272, 1_541, 329_216, 841_029),
    "ViViT-500K-CLTT-Ref-T2-DVS6": (500_762, 1_541, 329_216, 831_519),
    "ViT-SlotRecon":               (687_552, 581, 514_948, 1_203_081),
}
#: rung: (full-size label, the one cfg field the ladder moves, its value)
RUNGS = {
    "3DCNN-500K-CLTT-Ref":         ("3DCNN-CLTT-Ref", "conv_dim", 55),
    "3DCNN-500K-CLTT-Ref-T1":      ("3DCNN-CLTT-Ref-T1", "conv_dim", 55),
    "3DCNN-500K-CLTT-Ref-DVS6":    ("3DCNN-CLTT-Ref-DVS6", "conv_dim", 55),
    "3DCNN-500K-CLTT-Ref-T1-DVS6": ("3DCNN-CLTT-Ref-T1-DVS6", "conv_dim", 55),
    "ViViT-500K-CLTT-Ref":         ("ViViT-CLTT-Ref", "embed_dim", 120),
    "ViViT-500K-CLTT-Ref-T2":      ("ViViT-CLTT-Ref-T2", "embed_dim", 108),
    "ViViT-500K-CLTT-Ref-DVS6":    ("ViViT-CLTT-Ref-DVS6", "embed_dim", 108),
    "ViViT-500K-CLTT-Ref-T2-DVS6": ("ViViT-CLTT-Ref-T2-DVS6", "embed_dim", 87),
}
#: depth-1 / single-tubelet sibling of each T1 / T2 rung (same budget within +-10%)
SIBLINGS = {
    "3DCNN-500K-CLTT-Ref-T1": "3DCNN-500K-CLTT-Ref",
    "3DCNN-500K-CLTT-Ref-T1-DVS6": "3DCNN-500K-CLTT-Ref-DVS6",
    "ViViT-500K-CLTT-Ref-T2": "ViViT-500K-CLTT-Ref",
    "ViViT-500K-CLTT-Ref-T2-DVS6": "ViViT-500K-CLTT-Ref-DVS6",
}


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _channels(label):
    return 12 if label.endswith("-DVS6") else 6


def _space(channels):
    return gym.spaces.Box(0, 255, (channels, H, W), np.uint8)


def _n(module):
    return sum(p.numel() for p in module.parameters())


def _build(spec, channels, seed=0):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(channels), **kwargs)


def _agent_counts(monkeypatch, spec, channels):
    """(encoder, heads, aux, total) of the agent the production builder makes for this label."""
    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain

    class _Env:
        num_envs = 1
        num_agents = 1
        observation_space = _space(channels)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        state_space = None

    for k in ("NETT_AUX_WEIGHT_OVERRIDE", "NETT_AMP", "NETT_HIDDEN_SIZES", "NETT_FEATURE_ATTN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NETT_AUX_LOSS", spec["aux"])
    monkeypatch.setenv("NETT_AUX_WEIGHT", str(spec["aux_weight"]))
    if channels == 12:
        for k, v in DVS_ENV.items():
            monkeypatch.setenv(k, v)
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(0)
    brain = Brain(encoder=spec["encoder"], algorithm="PPO", encoder_cfg=kwargs,
                  algorithm_cfg={"rollouts": 16, "mini_batches": 1, "learning_epochs": 1},
                  model={"shared_encoder": True, "hidden_sizes": []})
    agent = build_agents(brain, _Env(), torch.device("cpu"))[0]
    assert agent.value.encoder is agent.policy.encoder
    aux = agent._aux
    aux_mod = aux.decoder if spec["aux"] == "slot_recon" else aux.head
    unique = lambda params: sum(p.numel() for p in {id(p): p for p in params}.values())
    model = [*agent.policy.parameters(), *agent.value.parameters()]
    enc = _n(agent.policy.encoder)
    return enc, unique(model) - enc, _n(aux_mod), unique([*model, *aux.head.parameters(), *aux_mod.parameters()])


# ---------------------------------------------------------------- the rungs

def test_every_rung_is_its_full_size_label_with_one_ladder_axis_moved(campaign):
    for rung, (full, axis, value) in RUNGS.items():
        r, f = campaign.MODELS[rung], campaign.MODELS[full]
        assert rung.replace("-500K", "") == full
        assert {k: v for k, v in r.items() if k != "cfg"} == {k: v for k, v in f.items() if k != "cfg"}, rung
        assert r["cfg"] == {**f["cfg"], axis: value} and f["cfg"][axis] != value, rung
        assert axis == ("conv_dim" if r["encoder"] == "compact_3dcnn" else "embed_dim"), rung
        assert r["aux"] == "cltt_ref" and r["framestack"] is True, rung
    assert campaign.MODELS_S162_3DCNN_REF == campaign.MODELS["3DCNN-CLTT-Ref"]["cfg"]


def test_rungs_are_within_ten_percent_of_500k_and_t1_t2_of_their_siblings():
    for rung in RUNGS:
        assert abs(TABLE[rung][0] / 500_000 - 1) <= 0.10, rung
    for rung, sib in SIBLINGS.items():
        assert abs(TABLE[rung][0] / TABLE[sib][0] - 1) <= 0.10, rung
    for full in {f for f, _, _ in RUNGS.values()}:                    # why the rungs exist
        assert abs(TABLE[full][0] / 500_000 - 1) > 0.10, full


def test_each_ladder_value_is_the_nearest_admissible_one(campaign):
    """3DCNN conv_dim nearest 500K (T1 / DVS6 keep it: within +-10% of the sibling); ViViT embed_dim the multiple
    of 3 nearest 500K (joint, T2-DVS6) or nearest the sibling (T2, DVS6) -- the neighbours on each side are worse."""
    def enc(label, value):
        spec = campaign.MODELS[label]
        full, axis, _ = RUNGS[label]
        return _n(_build({**spec, "cfg": {**spec["cfg"], axis: value}}, _channels(label)))

    def nearest(label, target, step):
        v = RUNGS[label][2]
        here = abs(enc(label, v) - target)
        assert here <= abs(enc(label, v - step) - target) and here <= abs(enc(label, v + step) - target), label

    nearest("3DCNN-500K-CLTT-Ref", 500_000, 1)
    nearest("ViViT-500K-CLTT-Ref", 500_000, 3)
    nearest("ViViT-500K-CLTT-Ref-T2-DVS6", 500_000, 3)
    nearest("ViViT-500K-CLTT-Ref-T2", TABLE["ViViT-500K-CLTT-Ref"][0], 3)
    nearest("ViViT-500K-CLTT-Ref-DVS6", TABLE["ViViT-500K-CLTT-Ref"][0], 3)
    # at the sibling's value the ViViT T2 / DVS6 rungs would leave +-10%, which is why they moved
    assert enc("ViViT-500K-CLTT-Ref-T2", 120) / TABLE["ViViT-500K-CLTT-Ref"][0] - 1 > 0.10
    assert enc("ViViT-500K-CLTT-Ref-DVS6", 120) / TABLE["ViViT-500K-CLTT-Ref"][0] - 1 > 0.10
    # ViT-CLTT-Ref-500K's own embed_dim (108) misses +-10% on the joint ViViT
    assert campaign.VIT_500K_CFG["embed_dim"] == 108
    assert enc("ViViT-500K-CLTT-Ref", 108) / 500_000 - 1 < -0.10


@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung_builds_at_128x80_and_refuses_the_other_stack(campaign, rung):
    spec = campaign.MODELS[rung]
    ch = _channels(rung)
    enc = _build(spec, ch)
    assert _n(enc) == TABLE[rung][0]
    x = torch.randint(0, 256, (2, ch, H, W), dtype=torch.uint8).float()
    assert enc(x).shape == (2, 512)
    with pytest.raises(ValueError):
        _build(spec, 18 - ch)                                         # 6 <-> 12


# ---------------------------------------------------------------- the table, through the production builder

@pytest.mark.parametrize("label", list(TABLE))
def test_parameter_table_through_the_production_builder(campaign, monkeypatch, label):
    spec = campaign.MODELS[label]
    assert _agent_counts(monkeypatch, spec, _channels(label)) == TABLE[label]
    enc, heads, aux, total = TABLE[label]
    assert total == enc + heads + aux


def test_vivit_and_slot_recon_labels_are_patch_16(campaign):
    """§162 add. 3 (2): "patch 16 trunks" for ViViT and ViT-SlotRecon (ViT-SlotRecon on the base ViT-CLTT-Ref trunk)."""
    labels = [k for k in TABLE if campaign.MODELS[k]["encoder"] in ("compact_vivit", "vit_slot_recon")]
    assert len(labels) == 9
    assert all(campaign.MODELS[k]["cfg"]["patch_size"] == 16 for k in labels)
    assert {k: v for k, v in campaign.MODELS["ViT-SlotRecon"]["cfg"].items()
            if k in ("patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio")} == \
        {k: campaign.MODELS["ViT-CLTT-Ref"]["cfg"][k] for k in ("patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio")}
