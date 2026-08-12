"""Pins for ``examples/probe_frozen_features.py`` (L5b frozen-feature linear probes).

Every test here exists because a specific wrong answer was produced first, on
2026-08-08, and each wrong answer was READABLE AS A FINDING:

1. The first split put the first half of each clip in train and the second half in
   test. But pose variance is ~50x the class signal (imprint frame 0 vs frame 90 =
   21.6 mean grey levels; imprint vs `1Ca` at the SAME pose = 0.50), so the probe
   learned pose and inverted on held-out poses -- `binding` scored **0.02**, i.e.
   almost perfectly wrong, and colour/shape sat at chance. It looked like "RL
   destroys all information".
2. The clips only differ in SOME frames: `O1_1Ca`/`O1_1Sa` are pixel-identical to
   the imprint over frames 90-179. Half the labels carried no signal at all.
3. At full stimulus scale every encoder -- including an untrained one -- scores
   0.93-0.99 on all three probes. A saturated benchmark measures nothing.

None of these needs a GPU, a checkpoint or the .mov files, so they run in the fast
suite. Everything uses synthetic clips whose answer is known by construction.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

_PROBE = Path(__file__).resolve().parents[1] / "examples" / "probe_frozen_features.py"


@pytest.fixture(scope="module")
def mod():
    """Load the driver by path -- examples/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("_frozen_probe", _PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clip(mod, n: int, value: int, differ: slice | None = None,
          other: int = 0) -> np.ndarray:
    """A synthetic clip: constant ``value``, optionally ``other`` inside ``differ``."""
    a = np.full((n, mod.RES, mod.RES, 3), value, dtype=np.uint8)
    if differ is not None:
        a[differ] = other
    return a


# --- the bf16 trap ---------------------------------------------------------

def test_amp_is_forced_off_before_encoders_are_imported(mod):
    """NatureCNN autocasts to bf16 on CUDA, which would quantise the probed features.

    The assignment must also come BEFORE the encoder import, or the module-level
    import order stops protecting anything.
    """
    assert os.environ["NETT_AMP"] == "off"
    src = _PROBE.read_text()
    assert src.index('os.environ["NETT_AMP"]') < src.index("from nett_skrl.brain")


# --- bug 2: frames that carry no signal ------------------------------------

def test_informative_frames_drops_identical_frames(mod):
    n = 40
    cache = {"a": _clip(mod, n, 200),
             "b": _clip(mod, n, 200, differ=slice(0, 10), other=0)}
    idx = mod.informative_frames(cache, "a", "b")
    assert idx.tolist() == list(range(10))


def test_informative_frames_respects_the_floor(mod):
    """A difference below DIFF_FLOOR is noise, not signal, and must be dropped."""
    n = 20
    tiny = _clip(mod, n, 200)
    tiny[:] = 200
    tiny[0:5] = 200 - int(mod.DIFF_FLOOR)      # strictly below the floor
    cache = {"a": _clip(mod, n, 200), "b": tiny}
    assert mod.informative_frames(cache, "a", "b").size == 0


# --- bug 1: pose must not predict the label --------------------------------

def _synthetic_probe_set(mod, monkeypatch, n=60):
    """A probe whose two classes differ on every frame, so selection is not the variable."""
    cache = {"neg": _clip(mod, n, 200), "pos": _clip(mod, n, 40)}
    monkeypatch.setitem(mod.STIMULI, "_t", (["neg"], ["pos"]))
    monkeypatch.setattr(mod, "N_FRAMES", n)
    return mod.build_probe_set(cache, "_t")


def test_classes_are_pose_matched_so_pose_cannot_predict_the_label(mod, monkeypatch):
    """★ THE 0.02 BUG. Both classes must be sampled at the SAME frame indices.

    Reconstructed from the returned arrays: within each split, the multiset of
    per-frame content for class 0 and class 1 must cover the same count, which is
    what makes frame index -- and therefore pose -- uninformative.
    """
    tr_x, tr_y, te_x, te_y = _synthetic_probe_set(mod, monkeypatch)
    for y in (tr_y, te_y):
        assert (y == 0).sum() == (y == 1).sum(), "classes unbalanced within a split"
    assert len(tr_y) > 0 and len(te_y) > 0


def test_train_and_test_use_disjoint_frame_blocks(mod, monkeypatch):
    """Alternating blocks: no frame index may appear in both splits."""
    n = 60
    cache = {"neg": np.stack([np.full((mod.RES, mod.RES, 3), i, np.uint8)
                              for i in range(n)]),
             "pos": np.stack([np.full((mod.RES, mod.RES, 3), 255 - i, np.uint8)
                              for i in range(n)])}
    monkeypatch.setitem(mod.STIMULI, "_t", (["neg"], ["pos"]))
    monkeypatch.setattr(mod, "N_FRAMES", n)
    tr_x, tr_y, te_x, te_y = mod.build_probe_set(cache, "_t")
    # frame index is recoverable from the pixel value for class 0
    tr_idx = {int(x[0, 0, 0]) for x, y in zip(tr_x, tr_y) if y == 0}
    te_idx = {int(x[0, 0, 0]) for x, y in zip(te_x, te_y) if y == 0}
    assert tr_idx and te_idx
    assert not (tr_idx & te_idx), "a frame index leaked across the train/test split"


def test_a_pose_confounded_split_would_be_caught(mod, monkeypatch):
    """The regression guard: if class 1 came only from frames the class 0 side lacks,
    the probe could separate on pose alone. Assert the builder never does that."""
    n = 60
    cache = {"neg": _clip(mod, n, 200), "pos": _clip(mod, n, 40)}
    monkeypatch.setitem(mod.STIMULI, "_t", (["neg"], ["pos"]))
    monkeypatch.setattr(mod, "N_FRAMES", n)
    tr_x, tr_y, _, _ = mod.build_probe_set(cache, "_t")
    assert len(tr_x) == len(tr_y)
    assert 0 < tr_y.mean() < 1


# --- the task-shaped (side) probe ------------------------------------------

def test_side_probe_classes_hold_identical_content_in_exchanged_positions(mod):
    """★ THE INVARIANT THAT MAKES THE SIDE PROBE MEAN ANYTHING.

    Each frame contributes BOTH orderings, so the two classes carry the same two
    images with left/right swapped. Colour, shape, area, pose and luminance are
    therefore matched by construction and only POSITION distinguishes the classes.
    If this ever stops holding, a "side" accuracy could be read off object identity.
    """
    tr_x, tr_y, te_x, te_y = mod.build_side_probe_set(_video_cache(mod))
    for x, y in ((tr_x, tr_y), (te_x, te_y)):
        assert (y == 0).sum() == (y == 1).sum()
        # identical content per class => identical mean luminance, to rounding
        assert abs(x[y == 0].mean() - x[y == 1].mean()) < 0.5


def test_side_probe_splits_are_disjoint_and_nonempty(mod):
    tr_x, tr_y, te_x, te_y = mod.build_side_probe_set(_video_cache(mod))
    assert len(tr_y) > 20 and len(te_y) > 20
    assert len(tr_x) == len(tr_y) and len(te_x) == len(te_y)


def _video_cache(mod):
    """Real clips -- the side probe composites two of them, so synthetic stand-ins
    would not exercise the informative-frame selection it depends on."""
    if not mod.VIDEO_DIR.exists():
        pytest.skip(f"stimulus clips not present at {mod.VIDEO_DIR}")
    neg, (pos,) = mod.STIMULI["binding"][0][0], mod.STIMULI["binding"][1]
    return {c: mod.load_clip(c) for c in (neg, pos)}


# --- bug 3: the difficulty axis --------------------------------------------

def test_rescale_is_identity_at_full_scale(mod):
    imgs = np.random.default_rng(0).integers(0, 255, (4, mod.RES, mod.RES, 3), dtype=np.uint8)
    assert np.array_equal(mod.rescale(imgs, 1.0, seed=0), imgs)


def test_rescale_shrinks_content_and_pads_white(mod):
    imgs = np.zeros((3, mod.RES, mod.RES, 3), dtype=np.uint8)     # all-black frames
    out = mod.rescale(imgs, 0.25, seed=0)
    assert out.shape == imgs.shape
    black = (out == 0).all(axis=3).sum(axis=(1, 2))
    side = int(round(mod.RES * 0.25))
    assert (black <= side * side).all(), "content grew instead of shrinking"
    assert (black >= (side - 2) ** 2).all(), "content vanished"
    assert (out[:, 0, 0] == 255).all(), "pad is not the clips' white background"


def test_rescale_is_deterministic_for_a_seed(mod):
    imgs = np.random.default_rng(1).integers(0, 255, (4, mod.RES, mod.RES, 3), dtype=np.uint8)
    assert np.array_equal(mod.rescale(imgs, 0.5, seed=7), mod.rescale(imgs, 0.5, seed=7))


# --- the probe itself ------------------------------------------------------

def test_linear_probe_solves_a_separable_problem_and_sits_at_chance_on_noise(mod):
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 60)
    signal = torch.from_numpy(
        (rng.normal(size=(120, 16)) + y[:, None] * 4.0).astype(np.float32))
    noise = torch.from_numpy(rng.normal(size=(120, 16)).astype(np.float32))
    assert mod.linear_probe(signal, y, signal, y) > 0.95
    assert 0.2 < mod.linear_probe(noise, y, noise, y) < 0.8


def test_linear_probe_standardises_with_train_statistics_only(mod):
    """A constant offset on the TEST features must not be silently absorbed."""
    y = np.tile([0, 1], 40)
    x = torch.from_numpy((np.tile([[0.0], [1.0]], (40, 8))).astype(np.float32))
    clean = mod.linear_probe(x, y, x, y)
    shifted = mod.linear_probe(x, y, x + 50.0, y)
    assert clean > 0.95 and shifted < clean


# --- architectures ---------------------------------------------------------

@pytest.mark.parametrize("arm", ["CNN", "ViT (cls)", "ViT-Sp (qk)", "ViT-Mixer-Sp"])
def test_random_init_encoder_builds_for_every_arm(mod, arm):
    """Each ARMS entry must name a real campaign model and construct on CPU."""
    label = mod.ARMS[arm][1]
    enc = mod.load_encoder(label, None, torch.device("cpu"), seed=0)
    out = enc(torch.zeros(2, mod.RES, mod.RES, 3, dtype=torch.uint8))
    assert out.shape == (2, mod.FEATURES_DIM)


def test_every_arm_maps_to_a_known_campaign_model(mod):
    from campaign_train import MODELS
    for arm, (_, label) in mod.ARMS.items():
        assert label in MODELS, f"{arm} names unknown model {label!r}"


def test_stimuli_have_one_negative_clip_each(mod):
    """build_probe_set unpacks a single negative clip; keep that contract explicit."""
    for probe, (neg, pos) in mod.STIMULI.items():
        assert len(neg) == 1, f"{probe}: expected one imprint clip, got {neg}"
        assert pos, f"{probe}: no contrast clips"
