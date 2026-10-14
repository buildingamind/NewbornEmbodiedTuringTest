"""Workspace DECISIONS §110: NETT_XSP2_WINDOWS=hold -- the last observation triple of each pose hold.

With NETT_TRAIN_DECISION_PERIOD=P >= 3 the env moves only on step_in_episode % P == 0, so the frames
after steps Pk .. Pk+P-1 share one pose. "hold" draws memory rows t with step_before(t) % P == P-1:
dorsal (o0, o1) = row t's stack, ventral + predictor o1, target o2 = row t+1's newest frame.
step_before is recovered from the done flags (hold_step_before). Every check has a control that fails.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.aux.xsp2_aux import WINDOWS, XSP2Term, hold_step_before
from nett_skrl.brain.encoders.xsp2 import XSP2Encoder

H, W, CPF = 80, 128, 3
C = 2 * CPF


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


class _HoldMemory:
    """A rollout simulated the way NETTEnv steps it, written into a FILLED RING.

    Env e starts mid-episode at step_before ``start[e]`` (unknown to the code under test); an episode
    lasts L = steps * P ticks and truncates when step_in_episode reaches L (repo B _get_dones), so the
    row with step_before L-1 carries the done and the next row is a reset (step_before 0).
    Pixels: channel 0 of each frame = its POSE id (changes only after a move step), channel 1 = env,
    channel 2 = the row's true step_before (current frame; off with sb_pixel=False, so that frames of one
    pose are pixel-identical). Chronological row c is stored at (memory_index + c) % T, so the oldest
    row sits at the write head. T is NOT a multiple of L by default: with T % L == 0 the storage order
    happens to continue step_before across the seam, and the ring-order control could not fail.
    """

    def __init__(self, P=3, steps=4, T=61, n_env=3, memory_index=23, starts=(5, 0, 11), sb_pixel=True):
        L = steps * P
        self.true_sb = torch.full((T, n_env), -9, dtype=torch.long)
        obs = torch.zeros(T, n_env, H, W, C, dtype=torch.uint8)
        trunc = torch.zeros(T, n_env, 1, dtype=torch.bool)
        acts = torch.zeros(T, n_env, 2)
        for e in range(n_env):
            sb, episode, prev_pose = starts[e], 0, None
            for c in range(T):
                row = (memory_index + c) % T
                # the CURRENT frame at step_before sb is the image after step sb-1 (reset image at 0)
                pose = (episode * 50 + (sb - 1) // P) if sb > 0 else (episode * 50 + 40)
                code = pose % 200 + 1
                prev = code if (sb == 0 or prev_pose is None) else prev_pose
                obs[row, e, ..., 0] = prev
                obs[row, e, ..., CPF] = code
                obs[row, e, ..., 1] = obs[row, e, ..., CPF + 1] = e
                if sb_pixel:
                    obs[row, e, ..., CPF + 2] = sb
                self.true_sb[row, e] = sb
                acts[row, e, 0] = (sb % P) / 10.0
                if sb == L - 1:
                    trunc[row, e] = True
                    sb, episode, prev_pose = 0, episode + 1, None
                else:
                    sb, prev_pose = sb + 1, code
        self.tensors = {"observations": obs, "terminated": torch.zeros_like(trunc), "truncated": trunc,
                        "actions": acts}
        self.memory_size, self.filled, self.memory_index = T, True, memory_index
        self.P, self.L, self.T = P, L, T


def _space():
    return gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)


def _term(monkeypatch, memory, dp="3", windows="hold"):
    monkeypatch.setenv("NETT_XSP2_WINDOWS", windows)
    if dp is None:
        monkeypatch.delenv("NETT_TRAIN_DECISION_PERIOD", raising=False)
    else:
        monkeypatch.setenv("NETT_TRAIN_DECISION_PERIOD", dp)
    monkeypatch.setenv("NETT_XSP2_PREDICTOR", "conv")
    torch.manual_seed(1)
    enc = XSP2Encoder(_space(), features_dim=235, policy_input="fg")
    term = XSP2Term(enc)
    term.attach_memory(memory)
    return enc, term


def test_hold_is_the_third_choice_and_refuses_without_a_period_of_at_least_three(monkeypatch):
    assert WINDOWS == ("slab", "iid", "hold")
    mem = _HoldMemory()
    _, ok = _term(monkeypatch, mem)
    assert ok.windows == "hold" and ok.hold_period == 3
    for bad in (None, "1", "2", "3x", "", "-3"):
        with pytest.raises(ValueError, match="hold"):
            _term(monkeypatch, mem, dp=bad)
    # control: the knob is read for "hold" only -- slab with no period still builds
    _, slab = _term(monkeypatch, mem, dp=None, windows="slab")
    assert slab.hold_period == 0


@pytest.mark.parametrize("P,steps,T,memory_index,starts", [(3, 4, 61, 23, (5, 0, 11)), (3, 6, 80, 0, (17, 2, 9)),
                                                           (4, 3, 67, 41, (1, 11, 6))])
def test_step_before_is_recovered_exactly_after_the_first_boundary(P, steps, T, memory_index, starts):
    mem = _HoldMemory(P=P, steps=steps, T=T, memory_index=memory_index, starts=starts)
    assert T % mem.L != 0
    sb = hold_step_before(mem)
    anchored = sb >= 0
    assert torch.equal(sb[anchored], mem.true_sb[anchored])
    # rows before each column's first done are unanchored, and there are some (the exact rule)
    assert (~anchored).any() and anchored.float().mean() > 0.6
    # unanchored = EXACTLY each column's chronological prefix up to and including its first done,
    # and every later done row carries step_before L-1
    order = torch.cat([torch.arange(memory_index, T), torch.arange(0, memory_index)])
    done = mem.tensors["truncated"][..., 0][order]
    first = done.float().argmax(0)
    prefix = torch.arange(T).view(T, 1) <= first.view(1, -1)
    assert torch.equal((~anchored)[order], prefix)
    later_done = done & ~prefix
    assert later_done.any() and (sb[order][later_done] == mem.L - 1).all()
    # CONTROL: reading the ring in STORAGE order (ignoring the write head) gets it wrong
    if memory_index:
        naive = type(mem).__new__(type(mem))
        naive.__dict__.update(mem.__dict__)
        naive.memory_index = 0
        wrong = hold_step_before(naive)
        # it anchors rows the true order cannot (before a column's first chronological done, anchored
        # to a NEWER done across the seam), and gives them a step_before that is false
        stray = (wrong >= 0) & ~anchored
        assert stray.any() and (wrong[stray] != mem.true_sb[stray]).any()


def test_hold_draw_is_last_triple_only_and_frames_share_one_pose(monkeypatch):
    mem = _HoldMemory(T=90, memory_index=37)
    enc, term = _term(monkeypatch, mem)
    seen = set()
    for seed in range(15):
        torch.manual_seed(seed)
        obs_t, obs_tk, a, _ = term._draw(enc)
        assert obs_t.shape[0] == obs_tk.shape[0] == a.shape[0] == 64
        sb = obs_t[:, 0, 0, CPF + 2].long()
        assert (sb % 3 == 2).all()                                           # middle obs of the last triple
        assert torch.equal(obs_tk[:, 0, 0, CPF + 2].long(), sb + 1)           # target = next step, same episode
        assert torch.equal(obs_tk[:, 0, 0, CPF + 1], obs_t[:, 0, 0, CPF + 1])  # same env
        o0, o1, o2 = obs_t[:, 0, 0, 0], obs_t[:, 0, 0, CPF], obs_tk[:, 0, 0, CPF]
        assert torch.equal(o0, o1) and torch.equal(o1, o2)                   # one pose: no ego-motion
        assert torch.allclose(a[:, 0], torch.full((64,), 0.2))               # a_t is the ignored hold sample
        seen |= set(zip(sb.tolist(), obs_t[:, 0, 0, 1].tolist()))
    assert len({e for _, e in seen}) == 3 and len({s for s, _ in seen}) >= 3


def test_off_by_one_phase_would_cross_a_move(monkeypatch):
    """CONTROL for the previous test: the other two phases DO cross a pose change."""
    mem = _HoldMemory(T=90, memory_index=37)
    sb = hold_step_before(mem)
    obs = mem.tensors["observations"]
    for phase in (0, 1):
        rows = ((sb % 3 == phase) & (sb >= 0) & ~mem.tensors["truncated"][..., 0]).nonzero()
        rows = rows[(rows[:, 0] + 1) < mem.T]
        rows = rows[rows[:, 0] + 1 != mem.memory_index]                       # not across the seam
        t, e = rows[:, 0], rows[:, 1]
        o0, o1, o2 = obs[t, e, 0, 0, 0], obs[t, e, 0, 0, CPF], obs[t + 1, e, 0, 0, CPF]
        assert ((o0 != o1) | (o1 != o2)).float().mean() > 0.5


def test_hold_aux_step_runs_and_its_telemetry_is_a_positive_control(monkeypatch):
    mem = _HoldMemory(T=90, memory_index=37, sb_pixel=False)
    enc, term = _term(monkeypatch, mem)
    term.build_optimizer(1e-4)
    g_state = torch.random.get_rng_state()
    out = term.compute(enc, torch.zeros(1))
    assert float(out) == 0.0 and term.n_steps == 1
    sc = term.last_scalars
    assert sc["hold_d01"] == 0.0 and sc["hold_d12"] == 0.0
    assert sc["uniform_d01"] > 0.0 and sc["uniform_d12"] > 0.0
    assert 0.2 < sc["hold_frac"] < 0.45                                       # ~1/3 of anchored valid rows
    assert np.isfinite(sc["loss"]) and sc["B"] == 64.0
    # the comparison draw uses a private generator: the global stream advanced by the hold draw only
    torch.random.set_rng_state(g_state)
    torch.randint(1, (64,))
    expected = torch.random.get_rng_state()
    torch.random.set_rng_state(g_state)
    term._draw(enc)
    assert torch.equal(torch.random.get_rng_state(), expected)
