"""LumNorm must survive the DICT observation the real env hands it.

⛔ THE DEFECT THIS PINS, found by seat:insect 2026-09-15 on a live smoke of wave row 03. The
Isaac bridge yields observations as ``{"policy": array}``. ``LumNorm.observation`` called
``np.asarray(obs)`` straight on that, and ``np.asarray({"policy": arr})`` is a **0-d object
array** -- so the wrapper raised ``ValueError: LumNorm expects HWC/NHWC or CHW/NCHW, got shape ()``
during ``validate_tasklist`` -> ``reset``, before a single training step.

⭐ THE FIX IS NOT NEW CODE, IT IS THE CODE NEXT DOOR. ``framestack.py`` already solved this with
``_policy_obs`` / ``_replace_policy_obs``; LumNorm shipped without them. A wrapper that sits in the
same chain as another wrapper must handle the same observation shapes -- the chain is the contract,
not the wrapper's own docstring.

⚠ WHY THIS WAS NOT CAUGHT: there was no lumnorm test at all. The wrapper arrived in 29e927c with a
registry entry and no fixture, and every check that ran against it used a bare array, which is the
one input shape the real environment never produces at reset.

⛔⛔ AND THE FIRST FIX FOR IT WAS ALSO INCOMPLETE, FOR THE SAME REASON ONE LEVEL IN. Every fixture in
the first version of this file was a NUMPY array. The dict unwrap was correct and the suite went
green, but the real payload inside ``{"policy": ...}`` is a **CUDA torch.Tensor**, so the very next
smoke died on ``np.asarray(<cuda tensor>)`` -> "can't convert cuda:0 device type tensor to numpy"
(seat:insect again, on the fix's own sha). ⭐ A RED-THEN-GREEN TEST PROVES THE CODE HANDLES THE INPUT
**THE TEST** SUPPLIES. Going green says nothing about the input the environment supplies, and the
fixture is the thing least likely to be questioned once the bug it was written for is dead. The
tensor cases below exist because the numpy ones could not have failed.
"""
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.lumnorm import LumNorm


class _StubEnv:
    """Minimal stand-in: LumNorm only needs an object gym can wrap."""
    observation_space = None
    action_space = None
    metadata: dict = {}
    render_mode = None
    spec = None

    def reset(self, **kw):
        return {"policy": np.zeros((2, 8, 8, 3), dtype=np.uint8)}, {}


def _wrapper():
    w = object.__new__(LumNorm)
    w.env = _StubEnv()
    w.target_mean = 0.45
    w.target_std = 0.25
    return w


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(2, 8, 8, 3), dtype=np.uint8)


def test_dict_observation_is_unwrapped_and_rewrapped():
    """The regression: a {"policy": array} obs must not reach np.asarray as a dict."""
    w, arr = _wrapper(), _frame()
    out = w.observation({"policy": arr})
    assert isinstance(out, dict), "a dict observation must come back as a dict"
    assert set(out) == {"policy"}
    assert out["policy"].shape == arr.shape
    assert out["policy"].dtype == arr.dtype


def test_dict_with_extra_keys_preserves_them():
    """Only `policy` is normalised; siblings ride through untouched (framestack's contract)."""
    w, arr = _wrapper(), _frame(1)
    critic = np.arange(4)
    out = w.observation({"policy": arr, "critic": critic})
    assert set(out) == {"policy", "critic"}
    assert np.array_equal(out["critic"], critic), "a non-policy key must be passed through unchanged"


def test_bare_array_still_works_unchanged():
    """The pre-existing contract must not regress."""
    w, arr = _wrapper(), _frame(2)
    out = w.observation(arr)
    assert isinstance(out, np.ndarray) and out.shape == arr.shape


def test_dict_and_bare_array_give_the_same_pixels():
    """⛔ The unwrap must be a pure re-wrap: identical input pixels, identical output pixels.
    A fix that unwrapped but normalised a different axis would pass the shape assertions above."""
    w, arr = _wrapper(), _frame(3)
    assert np.array_equal(w.observation({"policy": arr})["policy"], w.observation(arr))


def test_still_rejects_a_genuinely_wrong_rank():
    """The guard must survive the fix -- a 1-d observation is still an error, dict-wrapped or not."""
    w = _wrapper()
    with pytest.raises(ValueError, match="HWC/NHWC or CHW/NCHW"):
        w.observation(np.zeros(5, dtype=np.uint8))
    with pytest.raises(ValueError, match="HWC/NHWC or CHW/NCHW"):
        w.observation({"policy": np.zeros(5, dtype=np.uint8)})


# --------------------------------------------------------------------------------------------
# The payload type. The env hands back a CUDA tensor inside the dict, not a numpy array.
# --------------------------------------------------------------------------------------------

def _tensor_frame(seed=0, device="cpu"):
    return torch.as_tensor(_frame(seed), device=device)


def test_cpu_tensor_payload_is_normalised_not_crashed():
    """The second regression: np.asarray(<tensor>) is the failure, dict or no dict."""
    w, t = _wrapper(), _tensor_frame(4)
    out = w.observation({"policy": t})
    assert isinstance(out, dict)
    assert isinstance(out["policy"], torch.Tensor), (
        "a tensor in must yield a tensor out -- LumNorm is ordered FIRST and must not rely on a "
        "later wrapper to repair the type it emits")
    assert out["policy"].shape == t.shape and out["policy"].dtype == t.dtype


def test_bare_tensor_payload_works_too():
    """Not every chain wraps the observation in a dict; the tensor path must not need one."""
    w, t = _wrapper(), _tensor_frame(5)
    out = w.observation(t)
    assert isinstance(out, torch.Tensor) and out.shape == t.shape and out.dtype == t.dtype


def test_torch_and_numpy_backends_agree_exactly():
    """⛔ THE ONE THAT CATCHES A SILENT PORT ERROR. The two paths are separate implementations of
    one function, so they can drift while both stay green in isolation.

    The concrete trap already avoided: ``np.std`` is the POPULATION estimator (ddof=0) while
    ``torch.std`` defaults to the UNBIASED one (ddof=1). A naive port disagrees by exactly
    sqrt(n/(n-1)).

    ⚠ n IS THE SPATIAL GROUP SIZE H*W, not the array size -- this wrapper reduces per frame and per
    channel. Quote the ratio; a pixel count is data- and resolution-dependent, and quoting one
    without its n is how two people measure two numbers and conclude one of them is wrong.
    Measured over 8 seeds: this 8x8 fixture n=64 -> 1.00790526 and 42.3% of pixels shift by 1 LSB,
    but production 64x64 n=4096 -> 1.00012209 and 0.59%. THE FIXTURE OVERSTATES IT ~70x.

    Still worth pinning: the bias is one-directional (unbiased std is larger, so scale is smaller,
    so contrast is slightly compressed), so it does not average out across frames -- and it is
    undetectable from either backend alone."""
    w = _wrapper()
    for seed in range(4):
        arr = _frame(seed)
        got_np = w.observation(arr)
        got_t = w.observation(torch.as_tensor(arr)).numpy()
        assert np.array_equal(got_np, got_t), (
            f"seed {seed}: numpy and torch backends disagree; max abs diff "
            f"{np.abs(got_np.astype(int) - got_t.astype(int)).max()}")


def test_a_constant_channel_does_not_produce_nan_in_either_backend():
    """std==0 is a divide-by-zero in both implementations, and NaN would poison silently."""
    w = _wrapper()
    flat = np.full((2, 8, 8, 3), 7, dtype=np.uint8)
    assert not np.isnan(w.observation(flat).astype(np.float32)).any()
    assert not torch.isnan(w.observation(torch.as_tensor(flat)).to(torch.float32)).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA on this host")
def test_cuda_tensor_stays_on_its_device():
    """The exact input from the failing smoke: a cuda:0 tensor. It must neither crash nor be
    silently migrated to the host."""
    w, t = _wrapper(), _tensor_frame(6, device="cuda:0")
    out = w.observation({"policy": t})["policy"]
    assert isinstance(out, torch.Tensor)
    assert out.device.type == "cuda", f"observation left the device: {out.device}"
    assert np.array_equal(out.cpu().numpy(), w.observation(_frame(6)))


def test_still_rejects_a_wrong_rank_tensor():
    """The rank guard must hold on the tensor path too, not just the numpy one."""
    w = _wrapper()
    with pytest.raises(ValueError, match="HWC/NHWC or CHW/NCHW"):
        w.observation({"policy": torch.zeros(5, dtype=torch.uint8)})
