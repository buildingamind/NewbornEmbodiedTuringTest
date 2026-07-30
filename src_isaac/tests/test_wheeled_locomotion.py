"""Wheeled (PhysX-GPU) agent — distribution head, config wiring, source wiring.

Maps to acceptance criteria in workspace/notes/10_wheeled_physics.md.

Three flavors (Isaac Sim is not importable here):
* **Behavior tests** build the MultivariateGaussianActor / Environment for real.
* **Config tests** drive Environment._configure_cfg against a stub cfg.
* **Source-inspection tests** read the in-Isaac wiring files as text and assert
  substrings, following the repo's established pattern (test_camera_mount.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import pytest

from nett_skrl.brain.models import (
    GaussianActor,
    MultivariateGaussianActor,
    ModelCfg,
)
from nett_skrl.brain.models.builder import build_models_for_algorithm
from nett_skrl.environment.environment import Environment


# --- Repo path resolution (robust to checkout location) ---------------------
# repoA is resolved via $NETT_REPO_A or the sibling default; see _repo_paths for
# what that does and does NOT guarantee about the package on PYTHONPATH.
from _repo_paths import SRC_ISAAC as _SRC_ISAAC, nett_isaac_dir, read_source as _read

_ENVIRONMENT_PY = _SRC_ISAAC / "nett_skrl" / "environment" / "environment.py"
_NETT_ISAAC = nett_isaac_dir()
_NETT_ENV_PY = _NETT_ISAAC / "nett_env.py"
_CAMERA_RIG_PY = _NETT_ISAAC / "camera_rig.py"
_NETT_ENV_CFG_PY = _NETT_ISAAC / "nett_env_cfg.py"
_MOTOR_PY = _NETT_ISAAC / "motor_system.py"


# === Distribution head (behavior) ==========================================

class _TinyEncoder(nn.Module):
    def __init__(self, observation_space, **kw):
        super().__init__()
        self.features_dim = 8
        self.net = nn.Linear(int(np.prod(observation_space.shape)), 8)

    def forward(self, x):
        return self.net(x.float().flatten(1))


def _spaces(action_dim=2):
    obs = gym.spaces.Box(low=0, high=255, shape=(4,), dtype=np.float32)
    act = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return obs, act


def _build(action_dim=2, **cfg_kw):
    obs, act = _spaces(action_dim)
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Spec:
        actor_type = "gaussian"
        critic_type = "value"
        model_keys = ("policy", "value")

    return build_models_for_algorithm(
        _Spec(), encoder_cls=_TinyEncoder, encoder_kwargs={},
        observation_space=obs, action_space=act, device="cpu",
        cfg=ModelCfg(**cfg_kw),
    )


def test_mvg_actor_builds_and_acts():
    obs, act = _spaces(2)
    m = MultivariateGaussianActor(
        encoder_cls=_TinyEncoder, encoder_kwargs={}, observation_space=obs,
        action_space=act, device="cpu", cfg=ModelCfg(),
    )
    a, extra = m.act({"states": torch.rand(6, 4)}, role="policy")
    assert tuple(a.shape) == (6, 2)
    assert "log_prob" in extra and tuple(extra["log_prob"].shape) == (6, 1)


def test_builder_defaults_to_mvg_and_gaussian_is_opt_out():
    """The DEFAULT is the correlated actor; the diagonal one is now the opt-out.

    Flipped 2026-07-30 (was the reverse). Explicit ``None`` keeps resolving to the diagonal
    head on purpose -- old configs and checkpoints were built with it, so silently upgrading
    them would change an architecture mid-experiment.
    """
    assert type(_build()["policy"]).__name__ == "MultivariateGaussianActor"
    assert type(_build(actor_distribution="multivariate_gaussian")["policy"]).__name__ == (
        "MultivariateGaussianActor"
    )
    # Both opt-outs -- the explicit name and the legacy None -- give the diagonal actor.
    assert isinstance(_build(actor_distribution="gaussian")["policy"], GaussianActor)
    assert isinstance(_build(actor_distribution=None)["policy"], GaussianActor)
    # Value head is unchanged by the distribution swap.
    assert type(_build(actor_distribution="multivariate_gaussian")["value"]).__name__ == (
        "ValueCritic"
    )


def test_builder_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="actor_distribution"):
        _build(actor_distribution="banana")


def test_mvg_shares_gaussian_params_plus_only_correlation_terms():
    """Encoder + mean head + diagonal log_std match GaussianActor; the ONLY extra
    parameters are the off-diagonal Cholesky (correlation) terms inherent to a
    correlated distribution — n*(n-1)/2 of them (1 for the 2-wheel action)."""
    obs, act = _spaces(2)
    common = dict(encoder_cls=_TinyEncoder, encoder_kwargs={}, observation_space=obs,
                  action_space=act, device="cpu", cfg=ModelCfg())
    g = GaussianActor(**common)
    m = MultivariateGaussianActor(**common)
    g_shapes = sorted(tuple(p.shape) for p in g.parameters())
    m_shapes = sorted(tuple(p.shape) for p in m.parameters())
    # Every Gaussian param shape is present in the MVG actor.
    from collections import Counter
    extra = Counter(m_shapes) - Counter(g_shapes)
    assert list(extra.elements()) == [(1,)], f"unexpected extra params: {extra}"


def test_mvg_is_identical_diagonal_gaussian_at_init():
    """At init (off-diagonal = 0, log_std = 0) the MVG actor's log-prob equals the
    independent-Normal log-prob — a clean A/B baseline identical to GaussianActor."""
    import math
    from torch.distributions import Normal
    obs, act = _spaces(2)
    m = MultivariateGaussianActor(
        encoder_cls=_TinyEncoder, encoder_kwargs={}, observation_space=obs,
        action_space=act, device="cpu", cfg=ModelCfg(),
    )
    with torch.no_grad():
        m.mean_layer.weight.zero_()
        m.mean_layer.bias.zero_()
    taken = torch.tensor([[0.1, -0.2], [0.5, 0.3]])
    out = m.act({"states": torch.zeros(2, 4), "taken_actions": taken})[1]
    indep = Normal(torch.zeros(2, 2), torch.ones(2, 2)).log_prob(taken).sum(-1)
    assert torch.allclose(out["log_prob"].squeeze(-1), indep, atol=1e-5)


def test_mvg_effective_std_is_exp_log_std_not_skrl_bug():
    """Regression for Critic W3: the corrected scale_tril makes the effective std
    exp(log_std) (skrl's stock mixin would give exp(2*log_std))."""
    import math
    obs, act = _spaces(2)
    m = MultivariateGaussianActor(
        encoder_cls=_TinyEncoder, encoder_kwargs={}, observation_space=obs,
        action_space=act, device="cpu", cfg=ModelCfg(),
    )
    with torch.no_grad():
        m.log_std.fill_(math.log(2.0))  # std should be 2.0, not 4.0
    m.act({"states": torch.zeros(2, 4)})
    std0 = float(m._mg_distribution.covariance_matrix[0, 0, 0]) ** 0.5
    assert abs(std0 - 2.0) < 1e-4


def test_mvg_off_diagonal_introduces_correlation():
    """Setting the off-diagonal Cholesky term yields a non-zero action covariance
    (the capability a diagonal Gaussian cannot represent)."""
    obs, act = _spaces(2)
    m = MultivariateGaussianActor(
        encoder_cls=_TinyEncoder, encoder_kwargs={}, observation_space=obs,
        action_space=act, device="cpu", cfg=ModelCfg(),
    )
    with torch.no_grad():
        m.tril_offdiag.fill_(0.7)
    m.act({"states": torch.zeros(2, 4)})
    assert abs(float(m._mg_distribution.covariance_matrix[0, 0, 1])) > 0.1


def test_model_cfg_actor_distribution_defaults_to_mvg():
    """Pins the DEFAULT itself, not just the builder's dispatch on it.

    Worth its own test because this default was documented in blueprint.md as
    multivariate-Gaussian long before the code agreed, so a run made on defaults did not
    match the written configuration. The assertion is now the thing keeping them in step.
    """
    assert ModelCfg().actor_distribution == "multivariate_gaussian"


# === Environment config wiring (behavior) ==================================

def _design_sheet(tmp_path: Path) -> Path:
    csv = tmp_path / "design.csv"
    csv.write_text(
        "ImprintCondition,Phase,TestCondition,TargetVideo,NonTargetVideo,LeftMonitor,RightMonitor\n"
        "Object1,Train,,a.mp4,b.mp4,a.mp4,b.mp4\n"
        "Object1,Test,t1,a.mp4,b.mp4,a.mp4,b.mp4\n"
    )
    return csv


class _Nested:
    pass


class _Cfg:
    def __init__(self):
        self.scene = _Nested()
        self.observation = _Nested()
        self.screens = _Nested()
        self.motor = _Nested()
        self.sim = _Nested()
        self.asset_root = None
        self.scene.num_envs = 1
        self.observation.input_resolution = 64
        self.screens.random_first_frame = False
        self.screens.decision_period = 1
        self.sim.device = "cuda:0"
        self.seed = 0
        self.post_init_calls = 0

    def __post_init__(self):
        self.post_init_calls += 1


class _TaskConfig:
    current_mode = "train"
    condition = "Object1"
    episodes = {"train": 3, "test": 1, "record": 1}
    seed = 123
    dry_run = False
    eval_metrics_only = False
    device = 0
    path: Path


def _env(tmp_path, **kw):
    media_root = tmp_path / "media"
    media_root.mkdir(exist_ok=True)
    return Environment(design_sheet=_design_sheet(tmp_path), media_root=media_root, **kw)


def test_locomotion_defaults_to_wheeled(tmp_path):
    # Default is wheeled, matching schema.json and the validated operating point.
    env = _env(tmp_path)
    assert env.locomotion == "wheeled"


def test_invalid_locomotion_rejected(tmp_path):
    with pytest.raises(ValueError, match="locomotion"):
        _env(tmp_path, locomotion="hover")


def test_kinematic_pins_cpu_physx(tmp_path):
    env = _env(tmp_path, locomotion="kinematic")
    task = _TaskConfig()
    task.path = tmp_path / "run"
    cfg = _Cfg()
    env._configure_cfg(cfg, task)
    assert cfg.motor.locomotion == "kinematic"
    assert cfg.sim.device == "cpu"


def test_wheeled_selects_gpu_physx_and_plumbs_mode(tmp_path):
    env = _env(tmp_path, locomotion="wheeled")
    task = _TaskConfig()
    task.path = tmp_path / "run"
    cfg = _Cfg()
    env._configure_cfg(cfg, task)
    assert cfg.motor.locomotion == "wheeled"
    assert cfg.sim.device == "cuda:0"


def test_nett_sim_device_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("NETT_SIM_DEVICE", "cpu")
    env = _env(tmp_path, locomotion="wheeled")
    task = _TaskConfig()
    task.path = tmp_path / "run"
    cfg = _Cfg()
    env._configure_cfg(cfg, task)
    # Override forces CPU even for the wheeled agent (apples-to-apples compare).
    assert cfg.sim.device == "cpu"


# === Source wiring (in-Isaac, inspected as text) ===========================

def test_env_branches_locomotion_at_all_three_call_sites():
    src = _read(_NETT_ENV_PY)
    assert "_is_wheeled" in src
    # Pre-physics: wheeled drives actuators instead of writing root pose.
    assert "self.rig.drive_wheeled_actuators()" in src
    # Post-physics: wheeled reads the root pose back.
    assert "self.rig.sync_cameras_from_physics()" in src
    # Reset: wheeled teleports just the reset envs.
    assert "self.rig.reset_wheeled_pose(env_ids)" in src


def test_wheeled_drives_joints_and_does_not_write_root_pose():
    src = _read(_CAMERA_RIG_PY)
    # drive_wheeled_actuators uses real wheel actuation for locomotion ...
    drive_start = src.index("def drive_wheeled_actuators(")
    drive_end = src.index("def reset_wheeled_pose(")
    drive_body = src[drive_start:drive_end]
    assert "set_joint_velocity_target(" in drive_body
    # ... and must NOT write the root pose each step (PhysX owns it).
    assert "write_root_link_pose_to_sim" not in drive_body


def test_wheeled_reads_root_pose_back_from_articulation():
    src = _read(_CAMERA_RIG_PY)
    readback_start = src.index("def sync_cameras_from_physics(")
    readback_end = src.index("def sync_chick_and_camera_poses(")
    body = src[readback_start:readback_end]
    assert "root_link_pose_w" in body
    # The physics pose is mirrored back into motor state for logging/rewards
    # (via a local `motor = env.motor` alias).
    assert "motor.x[:]" in body
    assert "motor.yaw_deg[:]" in body


def test_reset_wheeled_teleports_and_zeroes_velocity():
    src = _read(_CAMERA_RIG_PY)
    start = src.index("def reset_wheeled_pose(")
    end = src.index("def sync_cameras_from_physics(")
    body = src[start:end]
    assert "write_root_link_pose_to_sim(" in body
    assert "write_root_link_velocity_to_sim(" in body
    assert "write_joint_state_to_sim(" in body


def test_cfg_enables_replicate_physics_for_wheeled():
    src = _read(_NETT_ENV_CFG_PY)
    assert 'self.motor.locomotion == "wheeled"' in src
    idx = src.index('self.motor.locomotion == "wheeled"')
    assert "replicate_physics = True" in src[idx:idx + 200]


def test_environment_gates_physx_device_on_locomotion():
    src = _read(_ENVIRONMENT_PY)
    # Default fallback is now "wheeled" (matches the constructor + schema default).
    assert 'getattr(self, "locomotion", "wheeled") == "wheeled"' in src
    assert "NETT_SIM_DEVICE" in src


def test_diff_drive_geometry_on_motor_cfg_not_hardcoded():
    src = _read(_MOTOR_PY)
    assert "wheel_radius" in src and "wheel_track" in src
    assert "def wheel_velocity_targets(" in src
    assert "def unicycle_from_wheels(" in src


def test_wheeled_per_step_path_has_no_host_sync():
    """Perf goal (reduce CPU calls): the wheeled per-step path (drive + read-back)
    must contain no GPU->CPU host transfers. The reset path is excluded (episode
    boundary, not per-step)."""
    src = _read(_CAMERA_RIG_PY)
    drive = src[src.index("def drive_wheeled_actuators("):src.index("def reset_wheeled_pose(")]
    sync = src[src.index("def sync_cameras_from_physics("):src.index("def sync_chick_and_camera_poses(")]
    # Strip comments so prose mentioning these tokens doesn't false-positive.
    def _code_only(block: str) -> str:
        return "\n".join(line.split("#", 1)[0] for line in block.splitlines())
    hot = _code_only(drive) + _code_only(sync)
    for tok in (".item(", ".cpu(", ".tolist(", ".any(", ".all(", "bool(", "nonzero"):
        assert tok not in hot, f"per-step host sync `{tok}` found in wheeled hot path"


def test_wheeled_reads_pose_before_rewards_via_get_dones():
    """Critic W1: the post-physics read-back runs in _get_dones (before
    _get_rewards) so wheeled rewards are not one step stale; _get_observations
    guards against a double sync."""
    src = _read(_NETT_ENV_PY)
    dones_idx = src.index("def _get_dones(")
    # The sync + freshness flag are set inside _get_dones for the wheeled path.
    dones_body = src[dones_idx:dones_idx + 500]
    assert "self.rig.sync_cameras_from_physics()" in dones_body
    assert "_wheeled_pose_fresh = True" in dones_body
    obs_idx = src.index("def _get_observations(")
    obs_body = src[obs_idx:obs_idx + 600]
    assert "if not self._wheeled_pose_fresh" in obs_body


def test_reset_clears_freshness_flag_so_first_obs_is_not_stale():
    """Critic N-1: _reset_idx runs after _get_dones set the flag True; it must
    clear the flag so _get_observations re-reads the just-teleported pose instead
    of rendering the pre-reset pose on the new episode's first frame."""
    src = _read(_NETT_ENV_PY)
    reset_idx = src.index("self.rig.reset_wheeled_pose(env_ids)")
    after = src[reset_idx:reset_idx + 400]
    assert "self._wheeled_pose_fresh = False" in after


def test_wheeled_locomotion_gated_off_in_record_phase():
    """Critic N-2: the wheeled velocity drive / read-back must not run in
    RecordPhase (which teleports to fixed grid viewpoints via motor.manual)."""
    src = _read(_NETT_ENV_PY)
    assert "_wheeled_locomotion_active" in src
    prop_idx = src.index("def _wheeled_locomotion_active")
    prop_body = src[prop_idx:prop_idx + 1000]
    assert "not isinstance(self.phase, RecordPhase)" in prop_body
    # All four locomotion call sites use the record-aware gate, not raw _is_wheeled.
    for anchor in ("self.rig.drive_wheeled_actuators()", "self.rig.reset_wheeled_pose(env_ids)"):
        a = src.index(anchor)
        # the nearest preceding `if self.<gate>` should be the record-aware one
        preceding = src.rfind("if self._", 0, a)
        assert "_wheeled_locomotion_active" in src[preceding:a]


def test_wheeled_drive_honors_decision_period_gating():
    """Critic W2: the wheeled velocity drive must use the decision-period-gated
    actions (zero on non-decision sub-steps) so the body holds pose exactly when
    the kinematic body would — not the raw ungated actions."""
    env_src = _read(_NETT_ENV_PY)
    # _apply_motor_actions records the effective (gated) actions in both branches.
    assert "self._effective_actions = self._actions" in env_src
    assert "self._effective_actions = self._gated_actions" in env_src
    rig_src = _read(_CAMERA_RIG_PY)
    drive = rig_src[rig_src.index("def drive_wheeled_actuators("):rig_src.index("def reset_wheeled_pose(")]
    # The drive derives wheel targets from the gated actions, not raw _actions.
    assert "env._effective_actions" in drive
    assert "env._actions" not in drive


def test_wheeled_does_not_log_discarded_kinematic_pose():
    """Critic W2: in wheeled mode motor.apply integrates only the neck DOFs
    (update_pose=False), so x/z/yaw keep the physics read-back value that gets
    logged — not a discarded kinematic guess."""
    src = _read(_NETT_ENV_PY)
    assert "update_pose = not self._wheeled_locomotion_active" in src
    assert "update_pose=update_pose" in src
    motor_src = _read(_MOTOR_PY)
    assert "def apply(self, actions: torch.Tensor, update_pose: bool = True)" in motor_src
    assert "if update_pose:" in motor_src
