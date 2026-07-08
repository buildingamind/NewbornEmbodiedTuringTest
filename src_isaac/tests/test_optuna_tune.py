"""Unit tests for the Optuna HP-tuning engine (examples/optuna_tune.py).

Tester role in the builder/critic/tester loop. These tests target the PURE
functions only (no Isaac / GPU): score_trajectory, read_trajectory,
sample_config (driven via optuna FixedTrial), and objective (with run_training
and read_trajectory monkeypatched).

Scoring contract: SPEC.md §OBJECTIVE.
    score = clip( P + 0.25*max(I,0) - 0.5*U , [-1, 2] )
      P = mean(reward last quarter)
      I = late_mean - early_mean
      U = std(diff(late)) + 0.5*max_drawdown(late)
    Hard fail -> -1.0 on: NaN/Inf in reward or any loss; T < min_updates(8); crash.
"""

from __future__ import annotations

import math
from pathlib import Path

import optuna
import pytest

import examples.optuna_tune as ot

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def make_traj(reward, losses=None, has_nan=None, crashed=None):
    """Build a trajectory dict shaped like read_trajectory's output."""
    reward = list(reward)
    losses = losses or {}

    def _bad(xs):
        return any(math.isnan(x) or math.isinf(x) for x in xs)

    if has_nan is None:
        has_nan = _bad(reward) or any(_bad(v) for v in losses.values())
    if crashed is None:
        crashed = len(reward) == 0
    return {
        "reward": reward,
        "losses": losses,
        "has_nan": has_nan,
        "crashed": crashed,
        "n": len(reward),
    }


def lerp_series(lo, hi, n, jitter=0.0, seed=0):
    """Steadily-changing series from lo to hi with optional sine jitter."""
    out = []
    for i in range(n):
        base = lo + (hi - lo) * i / max(1, n - 1)
        out.append(base + jitter * math.sin(i * 1.7 + seed))
    return out


# --------------------------------------------------------------------------- #
# 1. score_trajectory — monotonic sanity / ordering
# --------------------------------------------------------------------------- #


def test_rising_beats_flat_beats_declining():
    rising = make_traj(lerp_series(0.1, 0.8, 24, jitter=0.01))
    flat = make_traj([0.3] * 24)
    declining = make_traj(lerp_series(0.8, 0.1, 24, jitter=0.01))

    s_rise, _ = ot.score_trajectory(rising)
    s_flat, _ = ot.score_trajectory(flat)
    s_decl, _ = ot.score_trajectory(declining)

    assert s_rise > s_flat, f"rising({s_rise}) should beat flat({s_flat})"
    assert s_flat > s_decl, f"flat({s_flat}) should beat declining({s_decl})"


def test_stable_beats_oscillating_same_mean():
    # Both centered at 0.6 over the whole series; stable has near-zero jitter,
    # oscillating swings +/-0.15 update-to-update (large diff std + drawdown).
    n = 24
    stable = make_traj([0.6] * n)
    osc = make_traj([0.6 + (0.15 if i % 2 == 0 else -0.15) for i in range(n)])

    # same mean over last quarter
    q = n // 4
    assert abs(sum(stable["reward"][-q:]) - sum(osc["reward"][-q:])) < 1e-9

    s_stable, _ = ot.score_trajectory(stable)
    s_osc, _ = ot.score_trajectory(osc)
    assert s_stable > s_osc, (
        f"stability term should bite: stable({s_stable}) > osc({s_osc})"
    )


def test_nan_in_reward_hard_fails():
    traj = make_traj([0.3] * 12 + [float("nan")] + [0.3] * 11)
    score, _ = ot.score_trajectory(traj)
    assert score == ot.HARD_FAIL_SCORE


def test_inf_in_reward_hard_fails():
    traj = make_traj([0.3] * 12 + [float("inf")] + [0.3] * 11)
    score, _ = ot.score_trajectory(traj)
    assert score == ot.HARD_FAIL_SCORE


def test_nan_in_loss_hard_fails():
    # reward clean, but a loss field has NaN -> has_nan True -> hard fail.
    traj = make_traj(
        [0.3] * 24,
        losses={"Loss / Value loss": [0.1] * 23 + [float("nan")]},
    )
    assert traj["has_nan"] is True
    score, _ = ot.score_trajectory(traj)
    assert score == ot.HARD_FAIL_SCORE


def test_too_short_hard_fails():
    traj = make_traj([0.5] * (ot.MIN_UPDATES - 1))
    score, _ = ot.score_trajectory(traj)
    assert score == ot.HARD_FAIL_SCORE
    # exactly min_updates is allowed (not a hard fail by length)
    ok = make_traj([0.5] * ot.MIN_UPDATES)
    score_ok, _ = ot.score_trajectory(ok)
    assert score_ok != ot.HARD_FAIL_SCORE or score_ok > ot.HARD_FAIL_SCORE


def test_crash_flag_hard_fails():
    traj = make_traj([0.5] * 24, crashed=True)
    score, _ = ot.score_trajectory(traj)
    assert score == ot.HARD_FAIL_SCORE


def test_intermediates_nonempty_and_length_matches():
    n = 20
    traj = make_traj(lerp_series(0.1, 0.7, n))
    _, inter = ot.score_trajectory(traj)
    assert inter, "intermediates must be non-empty for a valid series"
    assert len(inter) == n, "running-mean length must track series length"
    # longer series -> longer intermediates
    _, inter_short = ot.score_trajectory(make_traj(lerp_series(0.1, 0.7, 10)))
    assert len(inter) > len(inter_short)
    # running mean of a constant series is that constant (monotonic-ish sanity)
    _, flat_inter = ot.score_trajectory(make_traj([0.4] * 12))
    assert all(abs(x - 0.4) < 1e-9 for x in flat_inter)


def test_score_finite_and_within_clip_for_valid_inputs():
    cases = [
        lerp_series(0.1, 0.8, 24),
        [0.3] * 24,
        lerp_series(0.9, 0.2, 30),
        [0.0] * 16,
        [1.0] * 16,
        lerp_series(0.2, 0.95, 40, jitter=0.05),
    ]
    for r in cases:
        score, _ = ot.score_trajectory(make_traj(r))
        assert math.isfinite(score)
        assert ot.SCORE_CLIP[0] <= score <= ot.SCORE_CLIP[1], (
            f"score {score} out of clip {ot.SCORE_CLIP} for series start={r[0]}"
        )


# --------------------------------------------------------------------------- #
# 2. read_trajectory — synthetic tfevents + existing smoke files
# --------------------------------------------------------------------------- #


def _write_events(dirpath: Path, reward, losses=None):
    from torch.utils.tensorboard import SummaryWriter

    losses = losses or {}
    w = SummaryWriter(log_dir=str(dirpath))
    for step, r in enumerate(reward):
        w.add_scalar(ot.REWARD_TAG, r, step)
        for tag, vals in losses.items():
            if step < len(vals):
                w.add_scalar(tag, vals[step], step)
    w.flush()
    w.close()


def test_read_trajectory_synthetic_in_order(tmp_path):
    reward = [0.1, 0.2, 0.35, 0.5, 0.62, 0.7]
    losses = {"Loss / Value loss": [0.9, 0.7, 0.5, 0.4, 0.35, 0.3]}
    _write_events(tmp_path, reward, losses)

    events = sorted(tmp_path.glob("events.out.tfevents.*"))
    assert events, "SummaryWriter should have produced a tfevents file"
    traj = ot.read_trajectory(events[0])

    assert traj["n"] == len(reward)
    assert traj["reward"] == pytest.approx(reward, abs=1e-5)
    assert "Loss / Value loss" in traj["losses"]
    assert traj["losses"]["Loss / Value loss"] == pytest.approx(losses["Loss / Value loss"], abs=1e-5)
    assert traj["has_nan"] is False
    assert traj["crashed"] is False


def test_read_trajectory_flags_nan(tmp_path):
    reward = [0.1, 0.2, float("nan"), 0.4, 0.5]
    _write_events(tmp_path, reward)
    events = sorted(tmp_path.glob("events.out.tfevents.*"))
    traj = ot.read_trajectory(events[0])
    assert traj["has_nan"] is True


def test_read_trajectory_on_existing_smoke_runs():
    runs = Path("/home/zlaborde/code/isaac/optuna_tune/runs")
    if not runs.exists():
        pytest.skip("no smoke runs present")
    files = sorted(runs.glob("**/events.out.tfevents.*"))
    if not files:
        pytest.skip("no tfevents under runs/")
    # at least one real file should yield a non-empty reward series
    found = False
    for f in files:
        traj = ot.read_trajectory(f)
        if traj["n"] > 0:
            found = True
            assert traj["reward"], "non-empty n must come with a reward series"
            break
    assert found, "expected at least one smoke tfevents with a reward series"


# --------------------------------------------------------------------------- #
# 3. sample_config validity
# --------------------------------------------------------------------------- #


def _fixed_params(encoder):
    # iter-3 CORRECTED regime: the known-good anchor (rollouts=8000, lr=3e-4,
    # ent=0.01, epochs=10, batch=500) must be representable in the search space.
    params = {
        "rollouts": 8000,
        "learning_rate": 3e-4,
        "mini_batches": 16,
        "learning_epochs": 10,
        "entropy_loss_scale": 0.01,
        "value_loss_scale": 0.5,
        "grad_norm_clip": 0.5,
        "gae_lambda": 0.95,
        "discount_factor": 0.99,
        "ratio_clip": 0.2,
        "kl_threshold": 0.5,
        "hidden_size": 64,
        "features_dim": 512,
    }
    if encoder == "guess_what_moves":
        params["conv_dim"] = 96
    return params


def test_sample_config_nature_cnn_valid():
    trial = optuna.trial.FixedTrial(_fixed_params("nature_cnn"))
    cfg = ot.sample_config(trial, "nature_cnn", target_timesteps=400_000)

    brain = cfg["brain"]
    assert brain["encoder"] == "nature_cnn"
    assert brain["model"]["hidden_sizes"] == [64, 64], "2-layer MLP must be [H,H]"
    assert cfg["body"]["wrappers"] == []
    assert "num_frames" not in brain["encoder_cfg"]
    assert brain["encoder_cfg"]["features_dim"] == 512
    assert brain["wandb"]["mode"] == "disabled"


def test_sample_config_gwm_valid():
    trial = optuna.trial.FixedTrial(_fixed_params("guess_what_moves"))
    cfg = ot.sample_config(trial, "guess_what_moves", target_timesteps=400_000)

    brain = cfg["brain"]
    assert brain["encoder"] == "guess_what_moves"
    assert brain["model"]["hidden_sizes"] == [64, 64]
    assert cfg["body"]["wrappers"] == ["framestack"]
    assert brain["encoder_cfg"]["num_frames"] == 2
    assert brain["encoder_cfg"]["conv_dim"] == 96


def test_sample_config_algorithm_cfg_within_ranges():
    # Sample REAL trials so the suggest() ranges (iter-3 CORRECTED regime) are
    # exercised. Large rollouts (4000/8000), ent/lr/epochs INCLUDE the known-good.
    study = optuna.create_study()
    for _ in range(40):
        trial = study.ask()
        cfg = ot.sample_config(trial, "nature_cnn")
        a = cfg["brain"]["algorithm_cfg"]
        assert 1e-4 <= a["learning_rate"] <= 5e-4, "lr range must include known-good 3e-4"
        assert a["rollouts"] in (4000, 8000), "rollouts must be in the large/known-good regime"
        assert a["mini_batches"] in (8, 16, 32)
        assert a["rollouts"] // a["mini_batches"] >= 100, "batch must stay >= 100"
        assert 5 <= a["learning_epochs"] <= 10
        assert 5e-3 <= a["entropy_loss_scale"] <= 3e-2, "ent range must include known-good 0.01"
        assert 0.25 <= a["value_loss_scale"] <= 0.75
        assert 0.3 <= a["grad_norm_clip"] <= 1.0
        assert 0.9 <= a["lambda"] <= 0.97
        assert 0.95 <= a["discount_factor"] <= 0.999
        assert 0.1 <= a["ratio_clip"] <= 0.3
        assert a["kl_threshold"] in (0.05, 0.2, 0.5)
        assert cfg["brain"]["model"]["hidden_sizes"][0] in (32, 64, 128)
        assert cfg["brain"]["encoder_cfg"]["features_dim"] in (128, 256, 512)
        assert cfg["steps_per_episode"] == 500, "must match known-good episode length"
        assert cfg["brain"]["algorithm"] == "PPO"
        study.tell(trial, 0.0)


def test_search_space_contains_known_good_anchor():
    # GUARD (the flaw in iters 1-2 was the search space EXCLUDING the HPs that
    # train >90% rest). The known-good anchor must produce exactly that config.
    trial = optuna.trial.FixedTrial(_fixed_params("nature_cnn"))
    cfg = ot.sample_config(trial, "nature_cnn", target_timesteps=400_000)
    a = cfg["brain"]["algorithm_cfg"]
    assert a["rollouts"] == 8000
    assert abs(a["learning_rate"] - 3e-4) < 1e-9
    assert abs(a["entropy_loss_scale"] - 0.01) < 1e-9
    assert a["learning_epochs"] == 10
    assert a["rollouts"] // a["mini_batches"] == 500, "batch must equal known-good 500"
    assert cfg["steps_per_episode"] == 500
    assert cfg["episodes"]["train"] * cfg["steps_per_episode"] == 400_000
    assert a["rollouts"] // cfg["steps_per_episode"] == 16, "num_envs must equal known-good 16"


def test_sample_config_unknown_encoder_raises():
    trial = optuna.trial.FixedTrial(_fixed_params("nature_cnn"))
    with pytest.raises(ValueError):
        ot.sample_config(trial, "bogus_encoder")


def test_sample_config_target_timesteps_scales_episodes():
    # episodes.train should scale with target_timesteps, INDEPENDENT of rollouts
    # (the iter-1 confound was length scaling with rollouts).
    t_small = optuna.trial.FixedTrial(_fixed_params("nature_cnn"))
    t_big = optuna.trial.FixedTrial(_fixed_params("nature_cnn"))
    cfg_small = ot.sample_config(t_small, "nature_cnn", target_timesteps=100_000)
    cfg_big = ot.sample_config(t_big, "nature_cnn", target_timesteps=400_000)
    assert cfg_big["episodes"]["train"] > cfg_small["episodes"]["train"]
    # total timesteps == target, regardless of the sampled rollouts
    assert cfg_big["episodes"]["train"] * cfg_big["steps_per_episode"] == 400_000


# --------------------------------------------------------------------------- #
# 4. objective robustness (no Isaac; run_training/read_trajectory patched)
# --------------------------------------------------------------------------- #


def _make_study(encoder="nature_cnn", target_timesteps=40_000):
    study = optuna.create_study(direction="maximize")
    study.set_user_attr("encoder", encoder)
    study.set_user_attr("target_timesteps", target_timesteps)
    return study


def test_objective_normal_trajectory_records_finite_score(monkeypatch):
    reward = lerp_series(0.1, 0.8, 24, jitter=0.01)
    traj = make_traj(reward, losses={"Loss / Value loss": [0.5] * 24})

    monkeypatch.setattr(ot, "run_training", lambda config, gpu, out_dir: Path("/fake/events"))
    monkeypatch.setattr(ot, "read_trajectory", lambda tb_path: traj)

    study = _make_study()
    study.optimize(ot.objective, n_trials=1, catch=())

    t = study.trials[0]
    assert t.state == optuna.trial.TrialState.COMPLETE
    assert t.value is not None and math.isfinite(t.value)
    expected, _ = ot.score_trajectory(traj)
    assert t.value == pytest.approx(expected, abs=1e-9)


def test_objective_swallows_training_crash(monkeypatch):
    def boom(config, gpu, out_dir):
        raise RuntimeError("isaac exploded")

    monkeypatch.setattr(ot, "run_training", boom)
    # read_trajectory should never be reached, but patch it to be safe.
    monkeypatch.setattr(ot, "read_trajectory", lambda tb_path: make_traj([0.5] * 24))

    study = _make_study()
    # catch=() means optimize re-raises anything objective raises; it must NOT.
    study.optimize(ot.objective, n_trials=1, catch=())

    t = study.trials[0]
    assert t.value == ot.HARD_FAIL_SCORE
    assert t.user_attrs.get("failure") == "training_crash"


def test_objective_handles_read_failure(monkeypatch):
    monkeypatch.setattr(ot, "run_training", lambda config, gpu, out_dir: Path("/fake/events"))

    def boom(tb_path):
        raise IOError("corrupt events")

    monkeypatch.setattr(ot, "read_trajectory", boom)

    study = _make_study()
    study.optimize(ot.objective, n_trials=1, catch=())
    t = study.trials[0]
    assert t.value == ot.HARD_FAIL_SCORE
    assert t.user_attrs.get("failure") == "read_failure"


def test_objective_crashed_trajectory_hard_fails(monkeypatch):
    traj = make_traj([], crashed=True)  # empty reward => crashed
    monkeypatch.setattr(ot, "run_training", lambda config, gpu, out_dir: Path("/fake/events"))
    monkeypatch.setattr(ot, "read_trajectory", lambda tb_path: traj)

    study = _make_study()
    study.optimize(ot.objective, n_trials=1, catch=())
    t = study.trials[0]
    assert t.value == ot.HARD_FAIL_SCORE
    assert t.user_attrs.get("failure") in ("crashed", "too_short(n=0)")
