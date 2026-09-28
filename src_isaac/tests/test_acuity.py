"""The acuity curriculum must be the published schedule, blur only in training, and refuse to guess.

Every constant below is read from ``rajankita/Visual_Acuity_Curriculum`` @ 79772b08b0,
``train_vac.py`` (see body/wrappers/acuity.py for the line numbers). Fixtures use the input the
real env supplies -- ``{"policy": uint8 NHWC tensor}`` with tensor done flags -- on CUDA when
present (test_lumnorm.py's lesson: a numpy-only fixture cannot fail on the CUDA path).
"""
import csv

import gymnasium as gym

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF

from nett_skrl.body.wrappers import acuity as A

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _StubEnv(gym.Env):
    observation_space = None
    action_space = None
    metadata: dict = {}
    render_mode = None
    spec = None

    def __init__(self, n=4, h=24, w=40, device="cpu", done_at=None):
        self.n, self.h, self.w, self.device = n, h, w, device
        self.done_at = done_at or {}   # step index -> list of env ids that finish on that step
        self.t = 0
        g = torch.Generator().manual_seed(0)
        self.frame = torch.randint(0, 256, (n, h, w, 3), generator=g, dtype=torch.uint8).to(device)

    def reset(self, **kwargs):
        self.t = 0
        return {"policy": self.frame.clone()}, {}

    def step(self, action):
        self.t += 1
        done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for i in self.done_at.get(self.t, []):
            done[i] = True
        return ({"policy": self.frame.clone()}, torch.zeros(self.n, device=self.device), done,
                torch.zeros(self.n, dtype=torch.bool, device=self.device), {})

    def close(self):
        pass


def _bound(cls, tmp_path, phase="train", start=0, total=1000, **kw):
    env = cls(_StubEnv(**kw))
    env.bind_phase(phase, start_step=start, total_steps=total, log_dir=tmp_path, offset=3, seed=[7, 3])
    return env


# ── the schedule is the paper's ─────────────────────────────────────────────────────────────
def test_stage_boundaries_are_cumulative_durations():
    assert np.allclose(A.stage_table(), [13 / 200, 40 / 200, 1.0])
    assert [A.stage_of(p) for p in (0.0, 0.064, 0.065, 0.199, 0.2, 0.9, 1.0)] == [0, 0, 1, 1, 2, 2, 2]


def test_replay_weights_are_normalised_cumulative_boundaries():
    assert np.allclose(A.replay_weights(0), [1.0])
    assert np.allclose(A.replay_weights(1), [13 / 53, 40 / 53])
    assert np.allclose(A.replay_weights(2), [13 / 253, 40 / 253, 200 / 253])


def test_kernel_size_matches_reference_and_is_odd():
    assert [A.kernel_size(s) for s in (1, 2, 14, 28)] == [7, 13, 85, 169]
    assert all(A.kernel_size(s) % 2 == 1 for s in np.linspace(0.3, 30, 57))


def test_sigmas_are_fraction_of_width():
    assert np.allclose(np.array(A.SIGMA_FRACS) * 448, [28, 14, 0])


# ── the blur is torchvision's, without the dense kernel ─────────────────────────────────────
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sigma", [0.7, 1.0, 2.0, 3.3])
def test_separable_blur_equals_torchvision(device, sigma):
    x = torch.rand(2, 3, 30, 44, device=device) * 255
    k = A.kernel_size(sigma)
    ref = TF.gaussian_blur(x, [k, k], [sigma, sigma])
    assert torch.allclose(A.gaussian_blur_nchw(x, sigma), ref, atol=1e-3)


# ── phase handling ───────────────────────────────────────────────────────────────────────────
def test_base_class_refuses():
    with pytest.raises(TypeError):
        A.AcuityCurriculum(_StubEnv())


def test_unbound_wrapper_refuses_to_step():
    env = A.AcuityVAC(_StubEnv())
    with pytest.raises(RuntimeError, match="never bound"):
        env.reset()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("phase", ["test", "record"])
def test_eval_phases_are_sharp_identity(tmp_path, device, phase):
    env = _bound(A.AcuityVAC, tmp_path, phase=phase, device=device, done_at={1: [0, 2]})
    obs, _ = env.reset()
    assert torch.equal(obs["policy"], env.env.frame)
    obs, *_ = env.step(None)
    assert torch.equal(obs["policy"], env.env.frame)


@pytest.mark.parametrize("device", DEVICES)
def test_train_blurs_and_preserves_type(tmp_path, device):
    # norep at progress 0 is stage 0: every env at sigma width/16
    env = _bound(A.AcuityNoReplay, tmp_path, device=device, w=64, h=32)
    obs, _ = env.reset()
    p = obs["policy"]
    assert p.dtype == torch.uint8 and p.device.type == device and p.shape == env.env.frame.shape
    assert np.allclose(env._sigma, 64 / 16)
    assert not torch.equal(p, env.env.frame)
    ref = A.gaussian_blur_nchw(env.env.frame.float().permute(0, 3, 1, 2), 4.0)
    assert torch.equal(p, ref.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8))


# ── per-episode draws on the real flag type ─────────────────────────────────────────────────
@pytest.mark.parametrize("device", DEVICES)
def test_sigma_is_redrawn_only_for_envs_whose_episode_ended(tmp_path, device):
    env = _bound(A.AcuityVAC, tmp_path, start=150, total=1000, device=device, done_at={2: [1, 3]})
    env.reset()
    before = env._sigma.copy()
    env.step(None)
    assert np.array_equal(env._sigma, before)           # no done: nothing redrawn
    # force a visible change for the done envs by moving to stage 0 draws only
    env._start = 0
    env._sigma[:] = -1.0
    env.step(None)                                       # step 2: envs 1 and 3 finish
    assert list(np.where(env._sigma != -1.0)[0]) == [1, 3]
    rows = list(csv.reader(open(tmp_path / "acuity_train_off3.csv")))
    assert rows[0][0] == "global_step" and len(rows) == 1 + 4 + 2


# ── the draws follow the schedule ────────────────────────────────────────────────────────────
def _freqs(cls, tmp_path, progress, width=448, n=40000):
    env = _bound(cls, tmp_path, start=int(progress * 100000), total=100000)
    draws = np.array([env._draw(width)[0] for _ in range(n)])
    return {s: np.mean(np.isclose(draws, s)) for s in (28.0, 14.0, 0.0)}


def test_vac_final_stage_draws_match_replay_weights(tmp_path):
    f = _freqs(A.AcuityVAC, tmp_path, 0.5)
    assert np.allclose([f[28.0], f[14.0], f[0.0]], A.replay_weights(2), atol=0.01)


def test_vac_first_stage_is_all_blur(tmp_path):
    assert _freqs(A.AcuityVAC, tmp_path, 0.01, n=500)[28.0] == 1.0


def test_norep_uses_the_stage_sigma_alone(tmp_path):
    assert _freqs(A.AcuityNoReplay, tmp_path, 0.1, n=500)[14.0] == 1.0
    assert _freqs(A.AcuityNoReplay, tmp_path, 0.5, n=500)[0.0] == 1.0


def test_rev_is_vac_time_reversed(tmp_path):
    f = _freqs(A.AcuityReversed, tmp_path, 0.0)       # reversed start == forward end
    assert np.allclose([f[28.0], f[14.0], f[0.0]], A.replay_weights(2), atol=0.01)
    assert _freqs(A.AcuityReversed, tmp_path, 0.99, n=500)[28.0] == 1.0


def test_a_resumed_chunk_continues_the_schedule(tmp_path):
    env = _bound(A.AcuityNoReplay, tmp_path, start=500, total=1000)
    assert env.progress() == 0.5 and env._draw(448) == (0.0, 2)


# ── wiring: registry and model labels ────────────────────────────────────────────────────────
def test_registry_resolves_each_schedule():
    from nett_skrl.body.wrappers.registry import _load_wrapper
    assert _load_wrapper("acuity_vac") is A.AcuityVAC
    assert _load_wrapper("acuity_norep") is A.AcuityNoReplay
    assert _load_wrapper("acuity_rev") is A.AcuityReversed


def test_model_labels_put_acuity_innermost_and_nothing_else(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train as campaign
    base = campaign.MODELS["CNN-UnityRecipe"]
    for label, name in (("CNN-UnityRecipe+VAC", "acuity_vac"),
                        ("CNN-UnityRecipe+VACnoRep", "acuity_norep"),
                        ("CNN-UnityRecipe+VACrev", "acuity_rev")):
        spec = campaign.MODELS[label]
        assert campaign.segmentation_wrappers(spec) == [name]
        assert {k: v for k, v in spec.items() if k != "pre"} == base


def test_find_acuity_walks_the_chain(tmp_path):
    inner = _bound(A.AcuityVAC, tmp_path)
    outer = gym.Wrapper(inner)
    assert A.find_acuity(outer) == [inner]
