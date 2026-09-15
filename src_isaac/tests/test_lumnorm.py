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
"""
import numpy as np
import pytest

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
