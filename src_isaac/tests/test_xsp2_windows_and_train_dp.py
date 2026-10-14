"""Workspace DECISIONS §108 knobs, each with a control that can fail.

NETT_XSP2_WINDOWS            -- "slab" (default, the pre-§108 draw, unchanged) | "iid": B independent
                                (t, env) windows, uniform over episode-contiguous pairs only.
NETT_TRAIN_DECISION_PERIOD   -- train build only: screens.decision_period = P; test keeps its own.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.aux.xsp2_aux import WINDOWS, XSP2Term
from nett_skrl.brain.encoders.xsp2 import XSP2Encoder
from nett_skrl.environment.environment import Environment

H, W, CPF = 80, 128, 3
C = 2 * CPF
_SRC = Path(__file__).resolve().parents[1]
_TRAIN = _SRC / "examples" / "campaign_train.py"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NETT_XSP2_BATCH", "NETT_XSP2_PREDICTOR", "NETT_XSP2_HIDDEN", "NETT_XSP2_REFINE_HIDDEN",
                 "NETT_XSP2_LR_SCALE", "NETT_XSP2_SURPRISE", "NETT_XSP2_WINDOWS", "NETT_AUX_CLTT_CHANNELS_PER_FRAME",
                 "NETT_CONV3D_ORTHO_INIT", "NETT_TRAIN_DECISION_PERIOD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_XSP2_BATCH", "64")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _space():
    return gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)


class _CodedMemory:
    """Each frame's pixels ENCODE (t, env), so a drawn window can be decoded back to its source.

    Episode boundaries: env 0 terminates at t=5, env 1 truncates at t=10; the ring is full with the
    write head at memory_index (the seam between memory_index-1 and memory_index).
    """

    def __init__(self, t_max=24, n_env=3, memory_index=17):
        self.t_max = t_max
        obs = torch.zeros(t_max, n_env, H, W, C, dtype=torch.uint8)
        for t in range(t_max):
            for e in range(n_env):
                for f in range(2):                                  # both stacked frames carry the code
                    obs[t, e, ..., f * CPF] = t
                    obs[t, e, ..., f * CPF + 1] = e
        acts = torch.zeros(t_max, n_env, 2)
        acts[..., 0] = torch.arange(t_max).view(t_max, 1) / 100.0 + torch.arange(n_env).view(1, n_env) / 10.0
        term = torch.zeros(t_max, n_env, 1, dtype=torch.bool)
        trunc = torch.zeros(t_max, n_env, 1, dtype=torch.bool)
        term[5, 0] = True
        trunc[10, 1] = True
        self.tensors = {"observations": obs, "terminated": term, "truncated": trunc, "actions": acts}
        self.memory_size, self.filled, self.memory_index = t_max, True, memory_index


def _term(monkeypatch, windows, memory=None):
    if windows is None:
        monkeypatch.delenv("NETT_XSP2_WINDOWS", raising=False)
    else:
        monkeypatch.setenv("NETT_XSP2_WINDOWS", windows)
    monkeypatch.setenv("NETT_XSP2_PREDICTOR", "conv")
    torch.manual_seed(1)
    enc = XSP2Encoder(_space(), features_dim=235, policy_input="fg")
    term = XSP2Term(enc)
    term.attach_memory(memory or _CodedMemory())
    return enc, term


def _decode(obs):
    """(B, H, W, C) or (B, C*H*W)-like raw stack -> per-sample (t, env) of the CURRENT frame."""
    o = obs.reshape(obs.shape[0], H, W, C) if obs.ndim != 4 else obs
    return o[:, 0, 0, CPF].long(), o[:, 0, 0, CPF + 1].long()


def test_windows_default_is_slab_and_iid_is_the_only_alternative(monkeypatch):
    assert WINDOWS == ("slab", "iid")
    _, t0 = _term(monkeypatch, None)
    assert t0.windows == "slab"
    _, t1 = _term(monkeypatch, "iid")
    assert t1.windows == "iid"
    for junk in ("IID ", "random", "1", "slabs"):
        with pytest.raises(Exception):
            _term(monkeypatch, junk)


def test_iid_draw_is_valid_pairs_only_and_spreads_over_envs(monkeypatch):
    enc, term = _term(monkeypatch, "iid")
    seen = set()
    for seed in range(20):
        torch.manual_seed(seed)
        obs_t, obs_tk, a, turn = term._draw(enc)
        assert obs_t.shape[0] == obs_tk.shape[0] == a.shape[0] == 64
        t, e = _decode(obs_t)
        tk, ek = _decode(obs_tk)
        assert torch.equal(ek, e) and torch.equal(tk, t + 1)                 # same env, the next frame
        # never across a done flag (env 0 at 5, env 1 at 10) nor the ring seam (16 -> 17)
        bad = ((e == 0) & (t == 5)) | ((e == 1) & (t == 10)) | (t == 16) | (t == 23)
        assert not bad.any()
        # the action is a_t of THAT (t, env)
        assert torch.allclose(a[:, 0], t.float() / 100 + e.float() / 10)
        assert turn == pytest.approx(float(a[:, 0].abs().mean()))
        seen |= set(zip(t.tolist(), e.tolist()))
    # control: the draw is NOT a slab -- one batch covers several envs and many times
    assert len({env for _, env in seen}) == 3 and len(seen) > 50


def test_slab_default_is_contiguous_and_iid_differs(monkeypatch):
    long = lambda: _CodedMemory(t_max=80, memory_index=70)                  # runs of >= 32 exist
    enc, slab = _term(monkeypatch, None, long())
    torch.manual_seed(3)
    obs_t, *_ = slab._draw(enc)
    t, e = _decode(obs_t)
    assert t.numel() == 64
    # pre-§108 draw: two halves of 32 consecutive frames, one env each
    for half in (slice(0, 32), slice(32, 64)):
        assert e[half].unique().numel() == 1
        assert torch.equal(t[half], torch.arange(int(t[half][0]), int(t[half][0]) + 32))
    enc2, iid = _term(monkeypatch, "iid", long())
    torch.manual_seed(3)
    obs2, *_ = iid._draw(enc2)
    t2, e2 = _decode(obs2)
    assert e2[:32].unique().numel() > 1                                      # control: iid is not one env


def test_iid_runs_a_full_aux_step(monkeypatch):
    enc, term = _term(monkeypatch, "iid")
    term.build_optimizer(1e-3)
    before = {k: v.detach().clone() for k, v in enc.state_dict().items()}
    torch.manual_seed(2)
    term.compute(enc, None)
    assert np.isfinite(term.last_scalars["loss"])
    assert any(not torch.equal(before[k], v) for k, v in enc.state_dict().items())


# ---------------------------------------------------------------- train-only decision period

def _env_obj(**kw):
    env = Environment.__new__(Environment)
    env.train_decision_period = kw.get("train_decision_period")
    return env


def _cfg(dp=1):
    return SimpleNamespace(screens=SimpleNamespace(decision_period=dp))


def test_train_decision_period_overrides_train_only():
    env = _env_obj(train_decision_period=2)
    tr, te, rec = _cfg(), _cfg(), _cfg()
    env._apply_train_decision_period(tr, "train")
    env._apply_train_decision_period(te, "test")
    env._apply_train_decision_period(rec, "record")
    assert tr.screens.decision_period == 2
    assert te.screens.decision_period == 1 and rec.screens.decision_period == 1
    # control: unset leaves the train build alone
    off = _cfg(3)
    _env_obj()._apply_train_decision_period(off, "train")
    assert off.screens.decision_period == 3


def test_train_decision_period_runs_after_the_field_copy_and_before_provenance():
    src = inspect.getsource(Environment._apply_sim_cfg)
    i_copy = src.index("self._copy_env_cfg_fields(cfg)")
    i_tdp = src.index("self._apply_train_decision_period(cfg, config.current_mode)")
    i_print = src.index('print(f"[NETT screens] decision_period=')
    assert i_copy < i_tdp < i_print
    sig = inspect.signature(Environment.__init__).parameters
    assert sig["train_decision_period"].default is None and sig["decision_period"].default == 1


def test_train_decision_period_refuses_below_one_and_schema_accepts_it():
    with pytest.raises(ValueError):
        Environment(train_decision_period=0)
    schema = json.loads((_SRC / "nett_skrl" / "schema.json").read_text())

    def _find(node):
        if isinstance(node, dict):
            if "train_decision_period" in node and "decision_period" in node:
                return node
            for v in node.values():
                r = _find(v)
                if r is not None:
                    return r
        return None

    props = _find(schema)
    assert props is not None
    entry = props["train_decision_period"]
    assert entry["minimum"] == 1 and entry["default"] is None and set(entry["type"]) == {"integer", "null"}


def test_campaign_knob_is_opt_in_and_refuses_junk(campaign, monkeypatch):
    src = _TRAIN.read_text()
    assert '**({"train_decision_period": _train_decision_period()}' in src
    assert 'if os.environ.get("NETT_TRAIN_DECISION_PERIOD") else {})' in src
    monkeypatch.setenv("NETT_TRAIN_DECISION_PERIOD", "2")
    assert campaign._train_decision_period() == 2
    for junk in ("0", "-1", "2.0", "two", "1.5"):
        monkeypatch.setenv("NETT_TRAIN_DECISION_PERIOD", junk)
        with pytest.raises(ValueError):
            campaign._train_decision_period()
