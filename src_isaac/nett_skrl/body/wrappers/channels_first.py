"""HWC → CHW conversion wrapper — the single body/brain format boundary."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from ..observation import image_layout, to_chw_space


class ChannelsFirst(gym.Wrapper):
    """Convert HWC image observations to CHW at the end of the body wrapper chain.

    Applied automatically as the last wrapper in Body.wrap() so all downstream
    consumers (skrl models, encoders) always receive CHW/BCHW without needing
    their own permute logic.

    Optional egocentric recording: when enabled via set_recording(True), buffers
    one CHW frame per env per step. Completed episodes are stored in
    completed_episodes and drained by RunRecorder after training to write
    directly to TensorBoard — no additional HWC→CHW conversion needed.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        space = env.observation_space
        if isinstance(space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict(
                {k: to_chw_space(v) for k, v in space.spaces.items()}
            )
        else:
            self.observation_space = to_chw_space(space)

        self._record: bool = False
        self._episode_frames: dict[int, list[np.ndarray]] = {}
        self._episode_counts: dict[int, int] = {}
        self.completed_episodes: list[tuple[str, list[np.ndarray]]] = []

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def set_recording(self, enabled: bool) -> None:
        """Enable or disable egocentric frame buffering."""
        self._record = bool(enabled)
        if not enabled:
            self._episode_frames.clear()

    def drain_completed_episodes(self) -> list[tuple[str, list[np.ndarray]]]:
        """Return and clear all completed (tag, frames) episode pairs."""
        episodes = self.completed_episodes
        self.completed_episodes = []
        return episodes

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._record:
            self._episode_frames.clear()
            for env_id, frame in enumerate(_policy_chw_frames(obs)):
                self._episode_frames[env_id] = [frame]
        return _obs_to_chw(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._record:
            done = _as_bool_array(terminated) | _as_bool_array(truncated)
            for env_id, (frame, is_done) in enumerate(
                zip(_policy_chw_frames(obs), done)
            ):
                self._episode_frames.setdefault(env_id, []).append(frame)
                if is_done:
                    count = self._episode_counts.get(env_id, 0)
                    tag = f"env_{env_id}/episode_{count}"
                    self.completed_episodes.append(
                        (tag, list(self._episode_frames[env_id]))
                    )
                    self._episode_counts[env_id] = count + 1
                    self._episode_frames[env_id] = []
        return _obs_to_chw(obs), reward, terminated, truncated, info

    def observation(self, obs):
        return _obs_to_chw(obs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _obs_to_chw(obs):
    """Recursively convert HWC tensors/arrays in obs to CHW."""
    if isinstance(obs, dict):
        return {k: _obs_to_chw(v) for k, v in obs.items()}
    return _to_chw(obs)


def _to_chw(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 3 and image_layout(x.shape) == "hwc":
            return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))
        if x.ndim == 4 and image_layout(x.shape[1:]) == "hwc":
            return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
    elif isinstance(x, torch.Tensor):
        if x.ndim == 3 and image_layout(tuple(x.shape)) == "hwc":
            return x.permute(2, 0, 1).contiguous()
        if x.ndim == 4 and image_layout(tuple(x.shape[1:])) == "hwc":
            return x.permute(0, 3, 1, 2).contiguous()
    return x


def _policy_chw_frames(obs) -> list[np.ndarray]:
    """Extract per-env CHW uint8 numpy frames from the policy observation."""
    policy = obs.get("policy", obs) if isinstance(obs, dict) else obs
    arr = (
        policy.detach().cpu().numpy()
        if isinstance(policy, torch.Tensor)
        else np.asarray(policy)
    )
    # arr is (B, H, W, C) batched or (H, W, C) single-env
    if arr.ndim == 4:
        return [np.ascontiguousarray(np.transpose(arr[i], (2, 0, 1))) for i in range(arr.shape[0])]
    if arr.ndim == 3 and image_layout(arr.shape) == "hwc":
        return [np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))]
    if arr.ndim == 3:
        return [arr]  # already CHW
    return []


def _as_bool_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.bool().cpu().numpy().ravel()
    return np.asarray(value, dtype=bool).ravel()
