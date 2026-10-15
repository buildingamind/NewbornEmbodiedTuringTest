"""The PPO minibatches are indexed out of the rollout buffer one at a time, not all up front (owner, workspace
DECISIONS §117 addendum 4: "Code fix to stop duplication of buffer").

skrl's ``sample_by_index`` builds every minibatch before the first gradient step; advanced indexing copies, so
the update held a second, permuted copy of the whole rollout. The NETT_ADV_NORM=minibatch wrapper then listed
them all again, which on the hybrid (CPU) buffer moved the whole rollout onto the GPU.

Known answers, each with a control that can fail:
  1. Every minibatch is BIT-IDENTICAL to skrl's eager one, same seed, same permutation, for every memory class
     NETT builds (GPU uint8, hybrid uint8, hybrid float), names that are absent included.
  2. Nothing is indexed until a minibatch is asked for (the lazy object holds index splits only); one
     minibatch keeps skrl's eager path (it is the whole rollout either way).
  3. The ADV_NORM wrapper gives the same standardised advantages as the old list-building wrapper, lazily.
  4. CUDA (skipped without a card): skrl's eager sample allocates about one rollout of observations; the lazy
     sample allocates nothing until iterated, and iterating holds about one minibatch (positive control:
     the eager path's allocation is measured in the same test).
"""

from __future__ import annotations

from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
from skrl.memories.torch import RandomMemory
from skrl.memories.torch.base import Memory

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))

from nett_skrl.brain import hybrid_memory as hm  # noqa: E402
from nett_skrl.brain import ppo_metrics as pm  # noqa: E402

N_ENVS, ROLL, C, H, W = 4, 16, 3, 28, 44
SPACE = gym.spaces.Box(0, 255, (C, H, W), np.uint8)
NAMES = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]
MB = 8


def _mem(cls, device="cpu", roll=ROLL, **kw):
    m = cls(memory_size=roll, num_envs=N_ENVS, device=device, **kw)
    m.create_tensor("observations", size=SPACE, dtype=torch.float32)
    for n in ("actions", "log_prob", "values", "returns", "advantages"):
        m.create_tensor(n, size=2 if n == "actions" else 1, dtype=torch.float32)
    gen = torch.Generator().manual_seed(3)
    for _ in range(roll):
        m.add_samples(observations=(torch.rand(N_ENVS, C * H * W, generator=gen) * 255).round().to(device),
                      **{n: torch.randn(N_ENVS, 2 if n == "actions" else 1, generator=gen).to(device)
                         for n in ("actions", "log_prob", "values", "returns", "advantages")})
    return m


CLASSES = [
    ("gpu-uint8", hm.Uint8StatesMemory, {}),
    ("hybrid-uint8", hm.Uint8HybridDeviceMemory, {"compute_device": "cpu"}),
    ("hybrid-float", hm.HybridDeviceMemory, {"compute_device": "cpu"}),
]


@pytest.mark.parametrize("label,cls,kw", CLASSES, ids=[c[0] for c in CLASSES])
def test_minibatches_are_skrls_bit_for_bit(label, cls, kw):
    m = _mem(cls, **kw)
    torch.manual_seed(11)
    lazy = m.sample(NAMES, batch_size=len(m), mini_batches=MB)
    eager = Memory.sample_by_index(m, NAMES, indexes=m.sampling_indexes, mini_batches=MB)   # skrl's own
    torch.manual_seed(11)
    ref_idx = torch.randperm(len(m), dtype=torch.long)
    assert torch.equal(m.sampling_indexes, ref_idx)              # the permutation is skrl's, drawn once
    got = list(lazy)
    assert len(lazy) == len(got) == len(eager) == MB
    for g, e in zip(got, eager):
        assert len(g) == len(e) == len(NAMES)
        for gt, et in zip(g, e):
            if et is None:
                assert gt is None                                # "states" is absent: None, as skrl
            else:
                assert gt.dtype == et.dtype and torch.equal(gt.cpu(), et.cpu())
    # control: a DIFFERENT permutation gives different minibatches, so equality above can fail
    torch.manual_seed(12)
    other = list(m.sample(NAMES, batch_size=len(m), mini_batches=MB))
    assert not torch.equal(other[0][0].cpu(), got[0][0].cpu())


def test_nothing_is_indexed_until_asked_and_one_minibatch_stays_eager():
    m = _mem(hm.Uint8StatesMemory)
    lazy = m.sample_by_index(NAMES, indexes=torch.randperm(len(m)), mini_batches=MB)
    assert isinstance(lazy, hm._LazyIndexedBatches)
    assert not any(isinstance(v, torch.Tensor) and v.numel() > len(m) for v in vars(lazy).values())
    assert all(b.numel() == len(m) // MB for b in lazy._batches)  # index splits only
    one = m.sample_by_index(NAMES, indexes=torch.randperm(len(m)), mini_batches=1)
    assert isinstance(one, list) and len(one) == 1                # skrl's eager single batch
    # the plain skrl class is untouched (NETT_UINT8_BUFFER=0 on the GPU keeps skrl's RandomMemory)
    assert not isinstance(RandomMemory.sample_by_index(_mem(RandomMemory), NAMES,
                                                       indexes=torch.randperm(len(m)), mini_batches=MB),
                          hm._LazyIndexedBatches)


class _Agent:
    _tensors_names = NAMES


def _old_adv_norm(batches, k):
    out = []
    for b in batches:
        b = list(b)
        a = b[k]
        if a.numel() > 1:
            b[k] = (a - a.mean()) / (a.std() + 1e-8)
        out.append(b)
    return out


@pytest.mark.parametrize("label,cls,kw", CLASSES, ids=[c[0] for c in CLASSES])
def test_adv_norm_wrapper_is_lazy_and_unchanged(label, cls, kw, monkeypatch):
    monkeypatch.setenv("NETT_ADV_NORM", "minibatch")
    m = _mem(cls, **kw)
    agent = _Agent()
    agent.memory = m
    assert pm.minibatch_advantage_norm(agent) is True
    torch.manual_seed(5)
    wrapped = agent.memory.sample(names=NAMES, batch_size=len(m), mini_batches=MB)
    assert isinstance(wrapped, pm._AdvNormBatches) and len(wrapped) == MB
    got = list(wrapped)
    del agent.memory.sample
    torch.manual_seed(5)
    ref = _old_adv_norm(m.sample(names=NAMES, batch_size=len(m), mini_batches=MB), NAMES.index("advantages"))
    k = NAMES.index("advantages")
    for g, r in zip(got, ref):
        assert torch.equal(g[k], r[k]) and torch.equal(g[0].cpu(), r[0].cpu())
        assert abs(float(g[k].mean())) < 1e-5                    # standardised
    # control: without the wrapper the advantages are NOT standardised per minibatch
    torch.manual_seed(5)
    raw = list(m.sample(names=NAMES, batch_size=len(m), mini_batches=MB))
    assert any(abs(float(b[k].mean())) > 1e-3 for b in raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")
def test_cuda_lazy_sample_does_not_allocate_a_second_rollout():
    dev = torch.device("cuda")
    roll = 64
    m = _mem(hm.Uint8StatesMemory, device=str(dev), roll=roll)
    obs_bytes = m.tensors["observations"].numel()                  # uint8: one byte each
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated(dev)
    torch.manual_seed(1)
    eager = Memory.sample_by_index(m, NAMES, indexes=torch.randperm(len(m)), mini_batches=MB)
    torch.cuda.synchronize()
    eager_bytes = torch.cuda.memory_allocated(dev) - base
    assert eager_bytes >= obs_bytes                                 # positive control: skrl copies the rollout
    del eager
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated(dev)
    torch.manual_seed(1)
    lazy = m.sample(NAMES, batch_size=len(m), mini_batches=MB)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(dev) - base < obs_bytes // MB   # nothing indexed yet
    peak = 0
    for b in lazy:
        torch.cuda.synchronize()
        peak = max(peak, torch.cuda.memory_allocated(dev) - base)
        del b
    assert peak < 2 * obs_bytes // MB + (1 << 20)                  # about one minibatch, never the rollout
