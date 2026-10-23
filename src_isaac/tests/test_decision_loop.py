"""Workspace DECISIONS §114: the agent-side decision loop, the -F3 XSP2 labels and the "frames" aux draw.

OWNER (2026-10-14): one agent decision per 3 env ticks, the video playing on all three, train and test
alike ("1. Same 3-step loop"); an agent that expects all three frames (XSP) receives all three. Every
check below has a control that fails when the property is broken.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.aux.xsp2_aux import WINDOWS, XSP2Term, frames_valid_rows
from nett_skrl.brain.encoders.xsp2 import XSP2Encoder
from nett_skrl.brain.env_wrappers import DECISION_LOOP_ENV, DecisionLoopEnvWrapper, decision_loop_period

H, W, CPF = 80, 128, 3
P = 3
_SRC = Path(__file__).resolve().parents[1]
_TRAIN = _SRC / "examples" / "campaign_train.py"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NETT_XSP2_BATCH", "NETT_XSP2_PREDICTOR", "NETT_XSP2_HIDDEN", "NETT_XSP2_REFINE_HIDDEN",
                 "NETT_XSP2_LR_SCALE", "NETT_XSP2_SURPRISE", "NETT_XSP2_WINDOWS", "NETT_AUX_CLTT_CHANNELS_PER_FRAME",
                 "NETT_CONV3D_ORTHO_INIT", "NETT_TRAIN_DECISION_PERIOD", DECISION_LOOP_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_XSP2_BATCH", "16")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


# ---------------------------------------------------------------- the knob

def test_period_knob_is_opt_in_and_refuses_junk(monkeypatch):
    assert decision_loop_period() is None
    monkeypatch.setenv(DECISION_LOOP_ENV, "")
    assert decision_loop_period() is None
    monkeypatch.setenv(DECISION_LOOP_ENV, "3")
    assert decision_loop_period() == 3
    for junk in ("1", "0", "-3", "3.0", "three", "2.5"):
        monkeypatch.setenv(DECISION_LOOP_ENV, junk)
        with pytest.raises(ValueError, match=DECISION_LOOP_ENV):
            decision_loop_period()


# ---------------------------------------------------------------- the wrapper

class _RawEnv:
    """NETTEnv's counters: the pose moves only on ticks with step_in_episode % P == 0, reward is masked to
    those ticks (repo B ``reward * _last_decision_mask``), an episode is ``ep_ticks`` ticks and resets in the
    same tick it truncates. obs = (tick count since construction, step_in_episode before the tick)."""

    def __init__(self, n_env=2, ep_ticks=4 * P, period=P, start=0):
        self.n_env, self.ep_ticks, self.period = n_env, ep_ticks, period
        self._step_in_episode = torch.full((n_env,), start, dtype=torch.long)
        self.ticks, self.actions_seen = 0, []

    def _decision_period(self):
        return self.period


class _Adapter:
    """The skrl adapter stand-in: step() is ONE env tick of the raw env underneath."""

    def __init__(self, raw):
        self._env = raw
        self.num_envs = raw.n_env

    def reset(self):
        return torch.zeros(self.num_envs, 2), {}

    def step(self, actions):
        raw = self._env
        raw.actions_seen.append(actions.clone())
        before = raw._step_in_episode.clone()
        decision = (before % raw.period == 0).float()
        reward = (10.0 * decision + 1.0 * (1 - decision)).view(-1, 1) * decision.view(-1, 1)
        raw._step_in_episode += 1
        trunc = (raw._step_in_episode >= raw.ep_ticks).view(-1, 1)
        raw._step_in_episode[trunc.view(-1)] = 0
        raw.ticks += 1
        obs = torch.stack([torch.full((raw.n_env,), float(raw.ticks)), before.float()], dim=1)
        return obs, reward, torch.zeros_like(trunc), trunc, {"tick": raw.ticks}


def _loop(**kw):
    raw = _RawEnv(**kw)
    adapter = _Adapter(raw)
    return raw, adapter, DecisionLoopEnvWrapper.wrap(adapter, adapter, P)


def test_unset_is_the_unwrapped_adapter():
    raw = _RawEnv()
    adapter = _Adapter(raw)
    assert DecisionLoopEnvWrapper.wrap(adapter, adapter, None) is adapter


def test_one_agent_step_is_p_ticks_with_the_same_action_last_obs_and_summed_reward():
    raw, _, env = _loop()
    a = torch.tensor([[0.5, -0.25], [1.0, 0.0]])
    obs, rew, term, trunc, info = env.step(a)
    assert raw.ticks == P and len(raw.actions_seen) == P
    assert all(torch.equal(x, a) for x in raw.actions_seen)                 # the one action, all P ticks
    assert torch.equal(obs[:, 0], torch.full((2,), float(P)))              # the LAST tick's observation
    assert torch.equal(obs[:, 1], torch.full((2,), float(P - 1)))
    assert torch.equal(rew, torch.full((2, 1), 10.0))                       # = the decision tick's reward
    assert info == {"tick": P} and not bool((term | trunc).any())
    assert torch.equal(raw._step_in_episode, torch.full((2,), P))           # aligned on the next decision


def test_episode_ends_on_the_last_tick_of_a_decision():
    raw, _, env = _loop()
    dones = []
    for _ in range(8):
        _, _, term, trunc, _ = env.step(torch.zeros(2, 2))
        dones.append(bool((term | trunc).all()))
    assert dones == [False, False, False, True] * 2                         # 4 decisions = 12 ticks
    assert raw.ticks == 8 * P


def test_refuses_a_done_before_the_last_tick():
    _, _, env = _loop(ep_ticks=4 * P - 1)                                   # control: misaligned episode
    with pytest.raises(RuntimeError, match="ended its episode on tick"):
        for _ in range(4):
            env.step(torch.zeros(2, 2))


def test_refuses_an_env_that_drifts_off_the_decision_ticks():
    _, _, env = _loop(start=1)                                              # control: one tick out of phase
    with pytest.raises(RuntimeError, match="multiple of 3"):
        env.step(torch.zeros(2, 2))


def test_wrap_refuses_a_period_mismatch_hold_windows_and_acuity(monkeypatch):
    raw = _RawEnv(period=1)
    adapter = _Adapter(raw)
    with pytest.raises(ValueError, match="decision_period=1"):
        DecisionLoopEnvWrapper.wrap(adapter, adapter, P)
    raw.period = P
    assert isinstance(DecisionLoopEnvWrapper.wrap(adapter, adapter, P), DecisionLoopEnvWrapper)   # control
    monkeypatch.setenv("NETT_XSP2_WINDOWS", "hold")
    with pytest.raises(ValueError, match="hold"):
        DecisionLoopEnvWrapper.wrap(adapter, adapter, P)
    monkeypatch.delenv("NETT_XSP2_WINDOWS")
    from nett_skrl.body.wrappers.acuity import AcuityCurriculum

    fake = AcuityCurriculum.__new__(AcuityCurriculum)
    fake.__dict__["_env"] = adapter
    with pytest.raises(ValueError, match="acuity"):
        DecisionLoopEnvWrapper.wrap(adapter, fake, P)


def test_brain_hook_is_the_only_brain_py_edit():
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR")}
    try:
        added_xsp2 = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--",
                                     "nett_skrl/brain/encoders/xsp2.py"], cwd=_SRC, capture_output=True, text=True,
                                    check=True, env=env).stdout.split()
        diff = subprocess.run(["git", "diff", f"{added_xsp2[-1]}^", "--unified=0", "--",
                               "nett_skrl/brain/brain.py"], cwd=_SRC, capture_output=True, text=True,
                              check=True, env=env).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        pytest.skip(f"git unavailable here: {exc}")
    removed = [ln[1:].strip() for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:].strip() for ln in diff if ln.startswith("+") and not ln.startswith("+++")]
    assert removed == ["return device, IsaacEnvWrapper(envs, device=device)"]
    code = [ln for ln in added if not ln.startswith("#")]
    # DECISIONS §187 add. 1 adds the first-full-stack hook between the adapter and the loop (test_first_full_stack.py).
    assert code == ["wrapped = IsaacEnvWrapper(envs, device=device)",
                    "from .env_wrappers import DecisionLoopEnvWrapper, decision_loop_period",
                    "from .env_wrappers import FirstFullStackEnvWrapper",
                    "wrapped = FirstFullStackEnvWrapper.wrap(wrapped, envs)",
                    "return device, DecisionLoopEnvWrapper.wrap(wrapped, envs, decision_loop_period())"]


# ---------------------------------------------------------------- the campaign kwargs

def test_campaign_sets_the_all_phase_period_and_refuses_the_train_only_one(campaign, monkeypatch):
    src = _TRAIN.read_text()
    assert src.count("**_decision_loop_env_kwargs(),") == 1
    assert campaign._decision_loop_env_kwargs() == {}                       # unset: nothing passed
    monkeypatch.setenv(DECISION_LOOP_ENV, "3")
    assert campaign._decision_loop_env_kwargs() == {"decision_period": 3}
    monkeypatch.setenv("NETT_TRAIN_DECISION_PERIOD", "3")
    with pytest.raises(ValueError, match="Pick one"):
        campaign._decision_loop_env_kwargs()


def test_environment_applies_decision_period_to_test_too():
    """Control for the design: ``decision_period`` (unlike §108's train_decision_period) reaches the test build."""
    from types import SimpleNamespace

    from nett_skrl.environment.environment import Environment

    env = Environment.__new__(Environment)
    env.train_decision_period = None
    te = SimpleNamespace(screens=SimpleNamespace(decision_period=3))       # what _copy_env_cfg_fields wrote
    env._apply_train_decision_period(te, "test")
    assert te.screens.decision_period == 3


# ---------------------------------------------------------------- XSP2 at three frames

def _space(frames):
    return gym.spaces.Box(low=0, high=255, shape=(frames * CPF, H, W), dtype=np.uint8)


def _enc(frames, **kw):
    kw = {"features_dim": 640, "map_stride": 4, **kw}
    torch.manual_seed(3)
    return XSP2Encoder(_space(frames), stack_frames=frames, num_frames=frames, **kw)


def test_f3_labels_are_the_2_frame_labels_plus_stack_depth(campaign):
    for lbl, base in (("XSP2-MAP20x32-F3", "XSP2-MAP20x32"), ("XSP2-MAP40x64-F3", "XSP2-MAP40x64")):
        assert campaign.MODELS[lbl] == {**campaign.MODELS[base], "cfg": {**campaign.MODELS[base]["cfg"], "stack_frames": 3}}
    assert campaign.MODELS["XSP2-MAP40x64"]["cfg"] == {**campaign.XSP2_CFG, "features_dim": 2560, "map_stride": 2}
    assert "policy_input" not in campaign.MODELS["XSP2-MAP40x64"]["cfg"]    # map only (§112)


def test_f3_refuses_a_two_frame_stack(campaign):
    from nett_skrl.brain.config import EncoderCfg

    kw = EncoderCfg(**campaign.MODELS["XSP2-MAP20x32-F3"]["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    assert kw["num_frames"] == 2                                            # default NETT_FRAMESTACK_N
    with pytest.raises(ValueError, match="disagree"):                     # a 3-frame obs, cfg depth 2
        XSP2Encoder(_space(3), **kw)
    with pytest.raises(ValueError, match="NETT_FRAMESTACK_N=3"):            # a 2-frame obs, cfg depth 2
        XSP2Encoder(_space(2), **kw)
    XSP2Encoder(_space(3), **{**kw, "num_frames": 3})                        # control: depth 3 builds


def test_f3_has_the_same_parameters_and_the_policy_reads_the_newest_frame():
    e2, e3 = _enc(2), _enc(3)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(e3) == n(e2) == 166_510
    e3.load_state_dict(e2.state_dict())
    g = torch.Generator().manual_seed(7)
    frames = [torch.randint(0, 256, (2, CPF, H, W), generator=g).float() for _ in range(3)]   # flat obs = CHW
    stack = lambda fs: torch.cat(fs, dim=1).reshape(2, -1)
    stack3, stack2 = stack(frames), stack(frames[1:])
    assert torch.equal(e3(stack3), e2(stack2))                              # same map as (o1, o2) at depth 2
    other = stack([frames[2], frames[0], frames[2]])
    assert torch.equal(e3(other), e3(stack3))                               # o0, o1 do not reach the policy
    swapped = stack([frames[0], frames[2], frames[1]])
    assert not torch.equal(e3(swapped), e3(stack3))                         # control: o2 does


# ---------------------------------------------------------------- the "frames" aux draw

class _FramesMemory:
    """Row (t, env) stores a 3-frame window whose frame f carries the code (t, env, f) in channels 0..2 at
    pixel (0, 0); elsewhere the three frames are IDENTICAL within a row and differ ACROSS rows (a stand-in
    for 'the body held still inside the window and moved between windows'). env 0 terminates at t=5,
    env 1 truncates at t=10; the ring is full with the write head at memory_index."""

    def __init__(self, t_max=24, n_env=3, memory_index=17, within=0):
        obs = torch.zeros(t_max, n_env, H, W, 3 * CPF, dtype=torch.uint8)
        for t in range(t_max):
            for e in range(n_env):
                for f in range(3):
                    obs[t, e, 1:, :, f * CPF:(f + 1) * CPF] = (7 * t + 31 * e) % 170 + within * f
                    obs[t, e, 0, 0, f * CPF:(f + 1) * CPF] = torch.tensor([t, e, f], dtype=torch.uint8)
        acts = torch.full((t_max, n_env, 2), 0.7)
        term = torch.zeros(t_max, n_env, 1, dtype=torch.bool)
        trunc = torch.zeros(t_max, n_env, 1, dtype=torch.bool)
        term[5, 0] = True
        trunc[10, 1] = True
        self.tensors = {"observations": obs, "terminated": term, "truncated": trunc, "actions": acts}
        self.memory_size, self.filled, self.memory_index = t_max, True, memory_index


def _frames_term(monkeypatch, memory=None, frames=3, loop="3", windows="frames"):
    monkeypatch.setenv("NETT_XSP2_WINDOWS", windows)
    monkeypatch.setenv("NETT_XSP2_PREDICTOR", "conv")
    if loop is None:
        monkeypatch.delenv(DECISION_LOOP_ENV, raising=False)
    else:
        monkeypatch.setenv(DECISION_LOOP_ENV, loop)
    enc = _enc(frames)
    term = XSP2Term(enc)
    term.attach_memory(memory or _FramesMemory())
    return enc, term


def _code(obs):
    o = obs.reshape(obs.shape[0], H, W, 3 * CPF)
    return o[:, 0, 0, 0].long(), o[:, 0, 0, 1].long()


def test_frames_is_the_fourth_choice_and_needs_the_3_frame_label_and_the_loop(monkeypatch):
    assert WINDOWS == ("slab", "iid", "hold", "frames")
    _, ok = _frames_term(monkeypatch)
    assert ok.windows == "frames"
    with pytest.raises(ValueError, match="stack_frames=3"):
        _frames_term(monkeypatch, frames=2)
    for loop in (None, "2", "4"):
        with pytest.raises(ValueError, match=DECISION_LOOP_ENV):
            _frames_term(monkeypatch, loop=loop)
    for windows in ("slab", "iid"):                                         # cross-row draws refuse depth 3
        with pytest.raises(ValueError, match="frames"):
            _frames_term(monkeypatch, windows=windows)


def test_valid_rows_drop_the_first_chronological_row_and_rows_after_a_done():
    mem = _FramesMemory()
    valid = frames_valid_rows(mem)
    assert valid.shape == (24, 3)
    bad = {(17, 0), (17, 1), (17, 2), (6, 0), (11, 1)}                      # head 17 = oldest; 5/10 are dones
    for t in range(24):
        for e in range(3):
            assert bool(valid[t, e]) == ((t, e) not in bad), (t, e)
    mem.filled, mem.memory_index = False, 9                                 # partial fill: rows 0..8 only
    valid = frames_valid_rows(mem)
    assert not valid[9:].any() and not valid[0].any() and valid[1:9, 1:].all()
    assert not valid[6, 0] and valid[6, 1]
    mem.memory_index = 0                                                    # nothing written: no row, no crash
    assert not frames_valid_rows(mem).any() and frames_valid_rows(mem).shape == (24, 3)


def test_frames_draw_is_valid_rows_their_predecessors_and_zero_actions(monkeypatch):
    enc, term = _frames_term(monkeypatch)
    valid = frames_valid_rows(term._memory)
    seen = set()
    for seed in range(10):
        torch.manual_seed(seed)
        obs, prev, a, turn = term._draw(enc)
        t, e = _code(obs)
        tp, ep = _code(prev)
        assert bool(valid[t, e].all())
        assert torch.equal(ep, e) and torch.equal(tp, (t - 1) % 24)         # same env, one decision earlier
        assert not bool(a.any()) and a.shape == (16, 2) and turn == 0.0     # the held tick's command: zero
        seen |= set(zip(t.tolist(), e.tolist()))
    assert len({env for _, env in seen}) == 3 and len(seen) > 30            # control: iid, not a slab


def test_frames_step_trains_on_o0_o1_with_target_o2_and_the_move_control_sees_the_move(monkeypatch):
    enc, term = _frames_term(monkeypatch)
    term.build_optimizer(1e-3)
    calls = []
    real = term.step

    def spy(encoder, prepared_t, prepared_tk, actions, turn_state):
        calls.append((prepared_t.clone(), prepared_tk.clone()))
        return real(encoder, prepared_t, prepared_tk, actions, turn_state)

    term.step = spy
    before = {k: v.detach().clone() for k, v in enc.state_dict().items()}
    torch.manual_seed(1)
    term.compute(enc, None)
    (pt, ptk), = calls
    assert pt.shape[1] == ptk.shape[1] == 2 * CPF
    assert torch.equal(pt[:, CPF:], ptk[:, :CPF])                            # o1 is both "current" and dorsal x_t
    s = term.last_scalars
    assert np.isfinite(s["loss"]) and any(not torch.equal(before[k], v) for k, v in enc.state_dict().items())
    # within-window frames are identical (but for the 1-pixel code) and rows differ: the move control fires
    assert s["move_d"] > 50 * max(s["frames_d01"], s["frames_d12"], 1e-9)
    assert 0.0 < s["frames_frac"] < 1.0
    # control: frames that DO change inside the window are seen by the within-window telemetry
    enc2, term2 = _frames_term(monkeypatch, memory=_FramesMemory(within=40))
    term2.build_optimizer(1e-3)
    torch.manual_seed(1)
    term2.compute(enc2, None)
    assert term2.last_scalars["frames_d01"] > 50 * s["frames_d01"]
