"""Unit tests for closeness_reward directional consistency.

Geometry under test
-------------------
* ChamberSpec defaults: half_x=20, monitor_width=20 (y-span ±10), monitor_height=10.
* Correct monitor: right wall, x=+half_x=+20.
* Chick position (env-local): 5 units from the correct monitor face → x=15, y=0 (chamber centre).
* Camera FOV: 45°, 64×64 image.

Yaw convention
--------------
yaw=0° → chick body faces world +Y (scene forward).
Camera orientation is body-yaw composed with the fixed scene→optical roll
(Rx(-90°), see camera_rig._SCENE_TO_OPTICAL_ROLL_RAD) so that the optical
axis points along the body's +Y at yaw=0.

Derived orientations:
  yaw=270° (= -90°) → camera forward = world +X → facing correct monitor (+X wall)
  yaw= 90°          → camera forward = world -X → facing incorrect monitor (-X wall)

Geometry analysis: only at yaw=270° are all four correct-monitor corners in
front of the camera (z_cam=5 for every corner). At every other 45° increment,
at least one corner falls behind the camera (z_cam≤0) so closeness_reward
returns 0.  This means:
  reward[270°] = 1.0  (full-frame or more, clamped to 1)
  reward[all others] = 0.0

No Isaac Sim or GPU is required — the test calls the reward math directly.
"""

from __future__ import annotations

import math

import pytest
import torch

from nett_isaac.lens import LensSpec, resolve_method
from nett_isaac.rewards import closeness_reward
from nett_isaac.utils.geometry import ChamberSpec, monitor_target_bounds

# ── constants ────────────────────────────────────────────────────────────────
_FOV_DEG = 45.0
_IMAGE_SIZE = (64, 64)
_SPEC = ChamberSpec()  # half_x=20, half_y=20, monitor_width=20, monitor_height=10, chick_hover=1.5

_CORRECT_SIDE = "right"   # monitor at x = +half_x
_INCORRECT_SIDE = "left"  # monitor at x = -half_x

_YAW_FACING_CORRECT = 270    # degrees → camera forward = world +X
_YAW_FACING_INCORRECT = 90   # degrees → camera forward = world -X
_YAW_ANGLES = list(range(0, 360, 45))  # [0, 45, 90, 135, 180, 225, 270, 315]

# Env spacing along world Y for parallel envs (mirrors nett_env convention).
_ENV_SPACING = 2.0 * _SPEC.half_y + 8.0  # 48 units


# ── helpers ──────────────────────────────────────────────────────────────────
def _cam_quat_xyzw(yaw_deg: float, B: int) -> torch.Tensor:
    """Camera quaternion (xyzw) for a given body yaw, batched to [B, 4].

    Composes body yaw around world Z with the fixed scene→optical roll
    (Rx(-90°), same as camera_rig._SCENE_TO_OPTICAL_ROLL_RAD):

      body = (cos(θ/2), 0, 0, sin(θ/2))   wxyz, rotation around Z
      optical = (√2/2, -√2/2, 0, 0)        wxyz, Rx(-π/2)
      camera_wxyz = body ⊗ optical

    With x1=y1=0 for the body quaternion the product simplifies to:
      w_cam = cos(θ/2) * √2/2
      x_cam = cos(θ/2) * (-√2/2)
      y_cam = sin(θ/2) * (-√2/2)
      z_cam = sin(θ/2) * √2/2
    """
    theta = math.radians(yaw_deg)
    sq2 = math.sqrt(2.0) / 2.0
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    # xyzw order for rewards.py
    q = torch.tensor([c * (-sq2), s * (-sq2), s * sq2, c * sq2], dtype=torch.float32)
    return q.unsqueeze(0).expand(B, -1).contiguous()


def _env_origins(B: int) -> torch.Tensor:
    """World origins [B, 3] for B parallel envs, centred and spaced along world Y."""
    origins = torch.zeros(B, 3)
    for i in range(B):
        origins[i, 1] = (i - (B - 1) / 2.0) * _ENV_SPACING
    return origins


def _rewards_for_num_envs(num_envs: int) -> dict[int, torch.Tensor]:
    """Compute closeness_reward vs. correct monitor at all 8 yaw angles.

    Returns {yaw_deg: [num_envs] reward tensor}.
    """
    spec = _SPEC
    correct_corners_local = monitor_target_bounds(spec, _CORRECT_SIDE).corners()  # [4, 3]
    origins = _env_origins(num_envs)  # [B, 3]

    # World-space monitor corners per env (relative geometry is identical for each env).
    corners_world = correct_corners_local.unsqueeze(0) + origins.unsqueeze(1)  # [B, 4, 3]

    # Camera position: 5 units from the correct monitor face, centre of chamber in Y.
    cam_local = torch.tensor(
        [spec.half_x - 5.0, 0.0, spec.chick_hover], dtype=torch.float32
    )
    cam_pos = origins + cam_local  # [B, 3]

    # LensSpec replaces the old pinhole K. That matrix built f = (W/2)/tan(fov/2) and
    # the reward path then read a field back out of it via fov_half = cx/f, so the
    # nominal 45 deg actually meant a ~53 deg equidistant field. The lens states the
    # field directly, and being a scalar description it needs no per-env expand.
    lens = LensSpec(method=resolve_method(), fov_h_deg=_FOV_DEG,
                    width=_IMAGE_SIZE[1], height=_IMAGE_SIZE[0])

    return {
        yaw: closeness_reward(
            corners_world,
            cam_pos,
            _cam_quat_xyzw(float(yaw), num_envs),
            lens,
        )
        for yaw in _YAW_ANGLES
    }


# ── tests ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("num_envs", [1, 2, 4])
def test_reward_facing_correct_is_max(num_envs: int) -> None:
    """Facing the correct monitor yields the highest reward (≥ all other angles).

    With near-plane clipping, oblique views of a close monitor can also project
    to an area larger than the image and clamp to 1.0, so this uses ≥ rather
    than strict >.  The key discriminator is that facing the *incorrect* monitor
    (90°) gives 0 while facing the correct monitor gives > 0.
    """
    rewards = _rewards_for_num_envs(num_envs)
    r_correct = rewards[_YAW_FACING_CORRECT][0].item()
    all_rewards = {yaw: rewards[yaw][0].item() for yaw in _YAW_ANGLES}
    other_max = max(v for yaw, v in all_rewards.items() if yaw != _YAW_FACING_CORRECT)
    assert r_correct >= other_max and r_correct > 0.0, (
        f"[num_envs={num_envs}] reward at yaw={_YAW_FACING_CORRECT}° "
        f"({r_correct:.4f}) should be the max and positive; got {all_rewards}"
    )


@pytest.mark.parametrize("num_envs", [1, 2, 4])
def test_reward_facing_incorrect_is_min(num_envs: int) -> None:
    """Facing the incorrect monitor yields reward ≤ every other yaw angle."""
    rewards = _rewards_for_num_envs(num_envs)
    r_incorrect = rewards[_YAW_FACING_INCORRECT][0].item()
    all_rewards = {yaw: rewards[yaw][0].item() for yaw in _YAW_ANGLES}
    other_min = min(v for yaw, v in all_rewards.items() if yaw != _YAW_FACING_INCORRECT)
    assert r_incorrect <= other_min, (
        f"[num_envs={num_envs}] reward at yaw={_YAW_FACING_INCORRECT}° "
        f"({r_incorrect:.4f}) should be ≤ all others; got {all_rewards}"
    )


@pytest.mark.parametrize("num_envs", [1, 2, 4])
def test_reward_uniform_across_batch(num_envs: int) -> None:
    """Every env in a batch receives the same reward at each yaw angle."""
    rewards = _rewards_for_num_envs(num_envs)
    for yaw, r in rewards.items():
        assert torch.allclose(r, r[0].expand_as(r), atol=1e-5), (
            f"[num_envs={num_envs}] rewards differ across envs at yaw={yaw}°: {r.tolist()}"
        )


def test_reward_invariant_to_num_envs() -> None:
    """Each env's reward is the same regardless of total batch size (1, 2, or 4)."""
    results = {n: _rewards_for_num_envs(n) for n in [1, 2, 4]}
    for yaw in _YAW_ANGLES:
        r1 = results[1][yaw][0].item()
        r2 = results[2][yaw][0].item()
        r4 = results[4][yaw][0].item()
        assert math.isclose(r1, r2, abs_tol=1e-5) and math.isclose(r1, r4, abs_tol=1e-5), (
            f"Reward at yaw={yaw}° differs across batch sizes: "
            f"n=1→{r1:.6f}, n=2→{r2:.6f}, n=4→{r4:.6f}"
        )


def test_partial_observation_gives_nonzero_reward() -> None:
    """Near-plane clipping: when only part of the monitor is in front of the
    camera, reward > 0 instead of the old all-or-nothing zero.

    At yaw=45° and yaw=315° the monitor corners with y=+10 have z_cam > 0
    while those with y=-10 have z_cam < 0.  The clipped polygon has positive
    area, so reward > 0.

    Note: with this geometry (5-unit distance, 45° FOV) even a partial view
    projects to an area larger than the image and saturates the reward at 1.0,
    so we only assert > 0.  The strict ordering relative to full-facing is
    verified by the separate directional tests.

    ⚠ THIS TEST WAS ALREADY RED AT THE BASELINE of this branch (assert 0.0 > 0.0),
    before any change here -- it predates the lens work and was stale w.r.t. repoA's
    2026-08-01b front-hemisphere clip, which stopped crediting the folded rear lobe
    this pose relied on. Under the 300 deg equisolid default the pose is genuinely in
    view, so it should now pass on its own terms; if it does not, the geometry (a 45
    deg field at 5 units) no longer straddles the boundary it was written to probe and
    the POSE needs choosing again, not the assertion loosening.
    """
    rewards = _rewards_for_num_envs(1)
    for yaw in (45, 315):
        r = rewards[yaw][0].item()
        assert r > 0.0, (
            f"Partial view at yaw={yaw}° should give reward > 0 after near-plane "
            f"clipping; got {r}"
        )
