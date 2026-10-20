"""Per-step pose change for the STP-E arms, carried losslessly in ONE extra image row (workspace DECISIONS §170).

Labels "STP-E", "STP-E-NoWarp" (``stp_pose_exec``) and "STP-E-Cmd" (``stp_pose_cmd``) in
examples/campaign_train.py; the consumer is brain/encoders/stp_e.py. Nothing else declares these wrappers,
so no other model's observation changes.

WHAT IT CARRIES. Each frame (N, H, W, C) becomes (N, H + 1, W, C): a new row 0 is prepended, and the first
16 bytes of that row's channel 0 hold four little-endian float32 values, everything else in the row is 0:

    executed  (stp_pose_exec):  [fwd, lat, dyaw, valid]   the pose change the body REALLY made over the step,
                                                         after physics (walls, wheel slip, the decision hold):
                                                         fwd/lat in chamber units in the body frame of the
                                                         PREVIOUS pose (fwd along its heading, lat to its
                                                         right), dyaw in degrees wrapped to [-180, 180)
    commanded (stp_pose_cmd):   [move*L_move, 0, turn*L_turn, valid]   the [turn, move] the env applied this
                                                         step (``_effective_actions``: decision gating included,
                                                         clamped to [-1, 1] as MotorSystem.apply does), in the
                                                         same physical units, L_* from the motor cfg

``valid`` is 1.0 for a real transition and 0.0 for the first frame of an episode -- at ``reset()`` and for
every env whose episode ended this step (Isaac auto-resets inside step, so that env's frame is the NEW
episode's first frame). Where valid is 0 the three motion values are exactly 0. The frame stack scrubs the
same envs to [x_new, x_new] (framestack.py ``FrameStack.step``), so a pair never spans two episodes either.

WHY A ROW. The predictor needs the pose change PER STORED SAMPLE (PPO draws minibatches from the rollout
buffer), the only per-sample carrier that already reaches the buffer and the encoder is the observation,
and the buffer stores observations as uint8 (brain/hybrid_memory.py, default since DECISIONS §103). Bytes
survive uint8 storage, float32 storage and ``prepare_image_tensor``'s /255 exactly (round(b/255*255) == b
for every byte; tests/test_stp_e.py round-trips all 256). The encoder strips the row before any conv.

WHERE THE POSE COMES FROM. After ``env.step`` returns, the NETTEnv's ``motor.x``, ``motor.y_pos`` and
``motor.yaw_deg`` hold the post-physics pose (camera_rig ``sync_cameras_from_physics`` on the wheeled body;
MotorSystem.apply on the kinematic one), and for an env that just reset they hold the reset pose (nett_env
``_reset_idx``, re-synced in ``_get_observations``). The delta is taken against the pose at the last frame
THIS wrapper emitted, so (older frame, newer frame, delta) is self-consistent at any stack cadence: with
NETT_DECISION_LOOP the wrapper sees ticks and the stack holds consecutive ticks; with
NETT_TRAIN_DECISION_PERIOD every tick is a step. Repo B is not changed.

ORDER. ``pre``, innermost, before ``framestack`` (the dvs_polarity / lumnorm rule): it needs a raw frame of
<= 3 channels, and framestack then stacks two framed rows. Behind framestack it refuses.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import torch

from ..observation import image_layout
from .framestack import _done_mask, _policy_obs, _replace_policy_obs

#: Rows prepended to each frame.
POSE_ROWS = 1
#: float32 values carried: fwd, lat, dyaw, valid.
POSE_FLOATS = 4
POSE_BYTES = 4 * POSE_FLOATS
#: Index of ``valid`` in the carried vector.
VALID = 3


def pose_to_row(pose: torch.Tensor, width: int, channels: int) -> torch.Tensor:
    """(N, 4) float -> (N, 1, W, C) uint8: the float32 bytes in row 0, channel 0, columns 0..15."""
    if width < POSE_BYTES:
        raise ValueError(f"the STP pose row needs an eye at least {POSE_BYTES} px wide; got {width}.")
    pose = pose.to(torch.float32).contiguous()
    if pose.ndim != 2 or pose.shape[1] != POSE_FLOATS:
        raise ValueError(f"pose must be (N, {POSE_FLOATS}); got {tuple(pose.shape)}")
    row = torch.zeros((pose.shape[0], POSE_ROWS, width, channels), dtype=torch.uint8, device=pose.device)
    row[:, 0, :POSE_BYTES, 0] = pose.view(torch.uint8).view(pose.shape[0], POSE_BYTES)
    return row


def bytes_to_pose(raw_bytes: torch.Tensor) -> torch.Tensor:
    """(..., 16) bytes, as uint8 or as their float value in 0..255 -> (..., 4) float32.

    Float input is rounded to the nearest integer first, so a uint8 buffer, a float32 buffer and the
    prepared [0, 1] tensor (multiply by 255 before calling) all decode to the same bits.
    """
    if raw_bytes.shape[-1] != POSE_BYTES:
        raise ValueError(f"expected {POSE_BYTES} bytes on the last axis; got {tuple(raw_bytes.shape)}")
    b = raw_bytes if raw_bytes.dtype == torch.uint8 else torch.round(raw_bytes.float()).clamp_(0, 255).to(torch.uint8)
    return b.contiguous().view(torch.float32)


def _unwrap_nett(env):
    from ...brain.env_wrappers import _unwrap_nett as unwrap
    return unwrap(env)


class _STPPose(gym.Wrapper):
    """Base: appends the pose row; subclasses say which motion it carries."""

    SOURCE = ""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = self._out_space(env.observation_space)
        self._prev_pose: torch.Tensor | None = None       # (N, 3) x, y, yaw_deg at the last emitted frame
        #: (N, 4) float32, the vector carried by the last emitted frame (diagnostic / tests).
        self.last_pose: torch.Tensor | None = None

    # -- spaces ---------------------------------------------------------------------------
    def _out_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            spaces = dict(space.spaces)
            spaces["policy"] = self._out_space(spaces["policy"])
            return gym.spaces.Dict(spaces)
        shape = tuple(space.shape)
        if len(shape) < 3 or image_layout(shape[-3:]) != "hwc":
            raise ValueError(f"{type(self).__name__} needs an HWC image observation; got shape {shape}.")
        if shape[-1] > 3:
            raise ValueError(
                f"{type(self).__name__} received {shape[-1]} channels; it frames RAW frames (<= 3 channels). "
                "Order it BEFORE framestack in the arm's wrapper list.")
        if shape[-2] < POSE_BYTES:
            raise ValueError(f"{type(self).__name__} needs an eye at least {POSE_BYTES} px wide; got {shape[-2]}.")
        out = shape[:-3] + (shape[-3] + POSE_ROWS,) + shape[-2:]
        return gym.spaces.Box(low=0, high=255, shape=out, dtype=np.uint8)

    # -- pose -----------------------------------------------------------------------------
    def _motor(self):
        raw = _unwrap_nett(self.env)
        motor = getattr(raw, "motor", None)
        if motor is None or not all(hasattr(motor, k) for k in ("x", "y_pos", "yaw_deg")):
            raise RuntimeError(f"{type(self).__name__}: no NETTEnv motor (x, y_pos, yaw_deg) under this wrapper.")
        return raw, motor

    def _pose_now(self) -> torch.Tensor:
        _, motor = self._motor()
        return torch.stack([motor.x, motor.y_pos, motor.yaw_deg], dim=1).to(torch.float64).clone()

    def _motion(self, cur: torch.Tensor) -> torch.Tensor:          # pragma: no cover - abstract
        raise NotImplementedError

    def _frame(self, policy, pose: torch.Tensor):
        as_tensor = isinstance(policy, torch.Tensor)
        t = policy if as_tensor else torch.as_tensor(np.asarray(policy))
        if t.ndim != 4:
            raise ValueError(f"{type(self).__name__} needs a batched (N, H, W, C) frame; got {tuple(t.shape)}.")
        if t.dtype != torch.uint8:
            raise ValueError(f"{type(self).__name__} needs a uint8 frame (the pose bytes are lossless only in "
                             f"uint8); got {t.dtype}.")
        if pose.shape[0] != t.shape[0]:
            raise ValueError(f"{type(self).__name__}: {pose.shape[0]} motor poses for {t.shape[0]} frames.")
        row = pose_to_row(pose.to(t.device), t.shape[2], t.shape[3])
        out = torch.cat([row, t], dim=1)
        self.last_pose = pose.to(torch.float32)
        return out if as_tensor else out.cpu().numpy()

    # -- gym API --------------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        cur = self._pose_now()
        self._prev_pose = cur
        pose = torch.zeros((cur.shape[0], POSE_FLOATS), dtype=torch.float32, device=cur.device)
        return _replace_policy_obs(obs, self._frame(_policy_obs(obs), pose)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cur = self._pose_now()
        if self._prev_pose is None or self._prev_pose.shape != cur.shape:
            raise RuntimeError(f"{type(self).__name__}.step before reset().")
        motion = self._motion(cur).to(torch.float32)                  # (N, 3)
        done = torch.as_tensor(np.asarray(_done_mask(terminated, truncated)).reshape(-1), device=cur.device)
        if done.numel() != cur.shape[0]:
            raise ValueError(f"{type(self).__name__}: {done.numel()} done flags for {cur.shape[0]} envs.")
        valid = (~done.bool()).to(torch.float32)
        pose = torch.cat([motion * valid[:, None], valid[:, None]], dim=1)
        self._prev_pose = cur
        return _replace_policy_obs(obs, self._frame(_policy_obs(obs), pose)), reward, terminated, truncated, info


class STPPoseExecuted(_STPPose):
    """``stp_pose_exec``: the EXECUTED pose change, body frame of the previous pose (module docstring)."""

    SOURCE = "executed"

    def _motion(self, cur: torch.Tensor) -> torch.Tensor:
        prev = self._prev_pose
        dx, dy = cur[:, 0] - prev[:, 0], cur[:, 1] - prev[:, 1]
        yaw0 = torch.deg2rad(prev[:, 2])
        # MotorSystem.apply: forward = (-sin yaw, cos yaw); right = forward rotated -90 deg = (cos yaw, sin yaw).
        fwd = -dx * torch.sin(yaw0) + dy * torch.cos(yaw0)
        lat = dx * torch.cos(yaw0) + dy * torch.sin(yaw0)
        dyaw = torch.remainder(cur[:, 2] - prev[:, 2] + 180.0, 360.0) - 180.0
        return torch.stack([fwd, lat, dyaw], dim=1)


class STPPoseCommanded(_STPPose):
    """``stp_pose_cmd``: the COMMANDED [turn, move] the env applied, in physical units (module docstring)."""

    SOURCE = "commanded"

    def __init__(self, env: gym.Env) -> None:
        # ⛔ The [turn, move] reading of the action is the wheeled default (camera_rig, NETT_WHEEL_TURNMOVE);
        # under "0" the two axes are [left, right] wheel commands and the conversion below would be wrong.
        if os.environ.get("NETT_WHEEL_TURNMOVE", "1") == "0":
            raise ValueError("stp_pose_cmd reads the action as [turn, move]; NETT_WHEEL_TURNMOVE=0 makes it "
                             "[left, right]. Refusing to label a wheel command as a turn.")
        super().__init__(env)

    def _motion(self, cur: torch.Tensor) -> torch.Tensor:
        raw, motor = self._motor()
        eff = getattr(raw, "_effective_actions", None)
        if eff is None:
            raise RuntimeError("stp_pose_cmd: the env has no _effective_actions after step (nett_env "
                               "_apply_motor_actions sets it); refusing to guess the command.")
        cmd = eff.to(torch.float64).clamp(-1.0, 1.0)
        if cmd.ndim != 2 or cmd.shape[0] != cur.shape[0] or cmd.shape[1] < 2:
            raise ValueError(f"stp_pose_cmd: _effective_actions {tuple(cmd.shape)} is not (N >= {cur.shape[0]}, >= 2).")
        cfg = motor.cfg
        turn = cmd[:, 0] * float(cfg.body_turn_speed_limit)
        move = cmd[:, 1] * float(cfg.body_move_speed_limit)
        return torch.stack([move.to(cur.device), torch.zeros_like(move).to(cur.device), turn.to(cur.device)], dim=1)
