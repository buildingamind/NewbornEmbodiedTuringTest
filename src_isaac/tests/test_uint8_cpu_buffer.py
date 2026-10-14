"""uint8 image observations in the rollout buffer, on the CPU (hybrid) path as on the GPU path (workspace
DECISIONS §103, owner 2026-10-13: "Gpu and cpu both use uint8 for observation buffer"), and the bf16
autocast for the matched encoders (NETT_MATCHED_AMP; opt-in from §103, DEFAULT since §111, owner 2026-10-14:
"uint8 buffer and bf16 should always be active").

Known answers, each with a control that can fail:
  1. The CPU hybrid buffer stores 0-255 image observations as uint8 by default, and float32 under
     NETT_UINT8_BUFFER=0 (the old path); scalars keep float32 either way.
  2. TRANSPARENCY: the encoder input from a uint8-stored minibatch == the input from the float32-stored
     one, bit for bit (prepare_image_tensor), on the same draws.
  3. A minibatch comes back on the compute device (the hybrid contract) and uint8.
  4. A non-integer or out-of-range float write into a uint8 slot REFUSES (copy_ would truncate); the
     integer write of the same shape does not (control).
  5. A Space bounded at 1 keeps float storage on both paths: the float path does not divide by 255 there,
     so uint8 would change the encoder input. Control: the same values under a 0-255 space do go uint8.
  6. agent_factory picks Uint8HybridDeviceMemory / Uint8StatesMemory by default and the float classes
     under "0"; campaign_train no longer pops a declared NETT_UINT8_BUFFER when it moves the buffer to CPU.
  7. NETT_MATCHED_AMP: unset == "bf16" (§111); "off" == the fp32 forward exactly; bf16 (CUDA only) returns
     fp32 features close to fp32 and NOT identical (control that autocast engaged), and unset matches an
     explicit "bf16" bit for bit; on CPU every setting is the same fp32 forward; a typo refuses.
"""

from __future__ import annotations

from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))

from nett_skrl.body.observation import prepare_image_tensor  # noqa: E402
from nett_skrl.brain import hybrid_memory as hm  # noqa: E402
from nett_skrl.brain.encoders.matched import MatchedEncoder, matched_amp_dtype  # noqa: E402

N_ENVS, ROLL, C, H, W = 4, 8, 6, 160, 256
SPACE = gym.spaces.Box(0, 255, (C, H, W), np.uint8)


def _mem(cls, space=SPACE, **kw):
    m = cls(memory_size=ROLL, num_envs=N_ENVS, device="cpu", **kw)
    m.create_tensor("observations", size=space, dtype=torch.float32)
    m.create_tensor("rewards", size=1, dtype=torch.float32)
    return m


def _hybrid(cls, space=SPACE):
    return _mem(cls, space, compute_device="cpu")


def _fill(m, gen, scale=255.0, integer=True):
    for _ in range(ROLL):
        x = torch.rand(N_ENVS, C * H * W, generator=gen) * scale
        if integer:
            x = x.round()
        m.add_samples(observations=x, rewards=torch.rand(N_ENVS, 1, generator=gen))


def test_cpu_hybrid_is_uint8_by_default_and_float_under_0(monkeypatch):
    monkeypatch.delenv("NETT_UINT8_BUFFER", raising=False)
    assert hm.uint8_buffer_enabled()
    m = _hybrid(hm.Uint8HybridDeviceMemory)
    assert m.tensors["observations"].dtype == torch.uint8 and m.tensors["observations"].device.type == "cpu"
    assert m.tensors["rewards"].dtype == torch.float32
    monkeypatch.setenv("NETT_UINT8_BUFFER", "0")
    assert not hm.uint8_buffer_enabled()
    assert _hybrid(hm.HybridDeviceMemory).tensors["observations"].dtype == torch.float32   # control


def test_uint8_buffer_gives_the_float_buffers_encoder_input_bit_for_bit():
    u8, f32 = _hybrid(hm.Uint8HybridDeviceMemory), _hybrid(hm.HybridDeviceMemory)
    _fill(u8, torch.Generator().manual_seed(1)); _fill(f32, torch.Generator().manual_seed(1))
    assert u8.tensors["observations"].dtype == torch.uint8 and f32.tensors["observations"].dtype == torch.float32
    torch.manual_seed(7); bu = list(u8.sample(["observations"], batch_size=ROLL * N_ENVS, mini_batches=4))
    torch.manual_seed(7); bf = list(f32.sample(["observations"], batch_size=ROLL * N_ENVS, mini_batches=4))
    assert len(bu) == len(bf) == 4
    for (xu,), (xf,) in zip(bu, bf):
        assert xu.dtype == torch.uint8 and xf.dtype == torch.float32
        pu, pf = prepare_image_tensor(xu, SPACE), prepare_image_tensor(xf, SPACE)
        assert torch.equal(pu, pf)
        assert pu.min() >= 0 and pu.max() <= 1 and pu.max() > .9      # control: really /255, not raw


def test_host_bytes_are_a_quarter():
    u8, f32 = _hybrid(hm.Uint8HybridDeviceMemory), _hybrid(hm.HybridDeviceMemory)
    nb = lambda m: m.tensors["observations"].numel() * m.tensors["observations"].element_size()
    assert nb(f32) == 4 * nb(u8)


@pytest.mark.parametrize("cls,hybrid", [(hm.Uint8HybridDeviceMemory, True), (hm.Uint8StatesMemory, False)])
def test_lossy_writes_refuse_and_integer_writes_do_not(cls, hybrid):
    m = _hybrid(cls) if hybrid else _mem(cls)
    ok = torch.full((N_ENVS, C * H * W), 200.0)
    m.add_samples(observations=ok, rewards=torch.zeros(N_ENVS, 1))                   # control: accepted
    for bad in (ok * 0 + 0.5, ok * 0 + 256.0, ok * 0 - 1.0):
        with pytest.raises(ValueError, match="not integers in \\[0, 255\\]"):
            m.add_samples(observations=bad, rewards=torch.zeros(N_ENVS, 1))
    m.add_samples(observations=ok.to(torch.uint8), rewards=torch.zeros(N_ENVS, 1))   # uint8 in: no check


@pytest.mark.parametrize("cls,hybrid", [(hm.Uint8HybridDeviceMemory, True), (hm.Uint8StatesMemory, False)])
def test_space_bounded_at_one_keeps_float_storage(cls, hybrid):
    unit = gym.spaces.Box(0.0, 1.0, (C, H, W), np.float32)
    m = _hybrid(cls, unit) if hybrid else _mem(cls, unit)
    assert m.tensors["observations"].dtype == torch.float32
    m.add_samples(observations=torch.full((N_ENVS, C * H * W), 0.5), rewards=torch.zeros(N_ENVS, 1))  # no refusal
    m2 = _hybrid(cls) if hybrid else _mem(cls)                                      # control: 0-255 space
    assert m2.tensors["observations"].dtype == torch.uint8


def test_agent_factory_selects_by_the_knob():
    src = (_SRC / "nett_skrl/brain/agent_factory.py").read_text()
    assert "Uint8HybridDeviceMemory if uint8_buffer_enabled() else HybridDeviceMemory" in src
    assert "Uint8StatesMemory if uint8_buffer_enabled() else RandomMemory" in src
    camp = (_SRC / "examples/campaign_train.py").read_text()
    assert 'os.environ.pop("NETT_UINT8_BUFFER"' not in camp


def _obs(frames, n=2):
    g = torch.Generator().manual_seed(3)
    return torch.randint(0, 256, (n, 3 * frames, H, W), generator=g, dtype=torch.uint8)


def test_matched_amp_default_is_bf16_off_is_fp32_and_typo_refuses(monkeypatch):
    monkeypatch.delenv("NETT_MATCHED_AMP", raising=False)
    assert matched_amp_dtype() is torch.bfloat16              # DECISIONS §111
    for v in ("off", "0", "fp32"):
        monkeypatch.setenv("NETT_MATCHED_AMP", v); assert matched_amp_dtype() is None
    monkeypatch.setenv("NETT_MATCHED_AMP", "bf16"); assert matched_amp_dtype() is torch.bfloat16
    monkeypatch.setenv("NETT_MATCHED_AMP", "bf61")
    with pytest.raises(ValueError):
        matched_amp_dtype()


@pytest.mark.parametrize("variant", ["vivit", "3dcnn"])
def test_matched_amp_off_is_unchanged_on_cpu_even_when_bf16_is_set(monkeypatch, variant):
    """autocast engages only on CUDA tensors: on CPU "bf16" (explicit or the §111 default) must leave the
    forward bit-identical to "off"."""
    enc = MatchedEncoder(gym.spaces.Box(0, 255, (6, H, W), np.uint8), variant=variant).eval()
    x = _obs(2)
    monkeypatch.setenv("NETT_MATCHED_AMP", "off")
    with torch.no_grad():
        ref = enc(x)
    monkeypatch.setenv("NETT_MATCHED_AMP", "bf16")
    with torch.no_grad():
        assert torch.equal(enc(x), ref)
    monkeypatch.delenv("NETT_MATCHED_AMP", raising=False)
    with torch.no_grad():
        assert torch.equal(enc(x), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast engages on CUDA only")
@pytest.mark.parametrize("variant", ["cnn", "vit", "vivit", "3dcnn_g6"])
def test_matched_amp_bf16_on_cuda_is_close_not_identical_and_fp32_out(monkeypatch, variant):
    frames = 2 if variant in ("vivit", "3dcnn_g6") else 1
    enc = MatchedEncoder(gym.spaces.Box(0, 255, (3 * frames, H, W), np.uint8), variant=variant).cuda().eval()
    x = _obs(frames, n=8).cuda()
    monkeypatch.setenv("NETT_MATCHED_AMP", "off")
    with torch.no_grad():
        ref = enc(x)
    monkeypatch.setenv("NETT_MATCHED_AMP", "bf16")
    with torch.no_grad():
        out = enc(x)
    monkeypatch.delenv("NETT_MATCHED_AMP", raising=False)
    with torch.no_grad():
        default = enc(x)
    assert torch.equal(default, out), "unset must be the §111 default, bf16"
    assert out.dtype == torch.float32
    assert not torch.equal(out, ref), "autocast did not engage"
    # LayerNorm(128) output is O(1); bf16 has ~3 significant digits
    assert (out - ref).abs().max() < 0.15 and torch.nn.functional.cosine_similarity(out, ref).min() > 0.995


# --- DECISIONS §113: bf16 stays the default everywhere; pre-Ampere cards get a one-time warning only ---

@pytest.mark.parametrize("capability, warns", [((7, 5), True), ((7, 0), True), ((8, 0), False), ((8, 6), False)])
def test_bf16_pre_ampere_warning_logs_once_and_never_changes_the_dtype(monkeypatch, caplog, capability, warns):
    from nett_skrl.brain.encoders import matched as m
    monkeypatch.setattr(m, "_BF16_SLOW_WARNED", set())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: capability)
    monkeypatch.delenv("NETT_MATCHED_AMP", raising=False)
    with caplog.at_level("WARNING", logger=m.logger.name):
        m.warn_if_bf16_without_tensor_cores(torch.device("cuda", 3), "NETT_MATCHED_AMP")
        m.warn_if_bf16_without_tensor_cores(torch.device("cuda", 3), "NETT_MATCHED_AMP")
    hits = [r for r in caplog.records if "no bf16 tensor-core path" in r.getMessage()]
    assert len(hits) == (1 if warns else 0)          # positive control: (7, 5) DOES log, exactly once
    if warns:
        assert "cuda:3" in hits[0].getMessage() and "NETT_MATCHED_AMP=off" in hits[0].getMessage()
    assert matched_amp_dtype() is torch.bfloat16     # §113: the default is untouched on every host


def test_bf16_warning_is_silent_on_cpu(monkeypatch, caplog):
    from nett_skrl.brain.encoders import matched as m
    monkeypatch.setattr(m, "_BF16_SLOW_WARNED", set())
    def boom(index):
        raise AssertionError("a CPU device must not query CUDA capability")
    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    with caplog.at_level("WARNING", logger=m.logger.name):
        m.warn_if_bf16_without_tensor_cores(torch.device("cpu"), "NETT_AMP")
    assert not caplog.records
