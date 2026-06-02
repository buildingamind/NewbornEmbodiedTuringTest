"""Shared observation/image helpers for NETT Isaac/skrl.

The Isaac env emits image observations in HWC order, skrl flattens spaces for
model input, and several body wrappers alter channel count. Keeping these
rules here avoids small, incompatible shape assumptions spreading through the
brain and wrapper code.
"""

from __future__ import annotations

from collections.abc import Iterable

import gymnasium as gym
import numpy as np
import torch


def image_layout(shape: Iterable[int]) -> str:
    """Return ``"hwc"`` or ``"chw"`` for a 3D image shape."""
    dims = tuple(int(x) for x in shape)
    if len(dims) != 3:
        raise TypeError(f"Image observation must be 3D, got {dims!r}")
    # Isaac HWC images have spatial axes first. Channel-stacked images may
    # have 3, 6, 9, ... channels, so do not special-case only RGB.
    if dims[2] <= dims[0] and dims[2] <= dims[1]:
        return "hwc"
    return "chw"


def image_channels_hw(space_or_shape) -> tuple[int, int, int]:
    """Return ``(channels, height, width)`` for a 3D image space or shape."""
    shape = tuple(getattr(space_or_shape, "shape", space_or_shape))
    layout = image_layout(shape)
    if layout == "hwc":
        return int(shape[2]), int(shape[0]), int(shape[1])
    return int(shape[0]), int(shape[1]), int(shape[2])


def to_chw_space(space: gym.Space) -> gym.Space:
    """Convert HWC image Box spaces to CHW; leave everything else unchanged.

    Handles both 3D (H, W, C) and 4D batched (N, H, W, C) layouts from Isaac
    Lab vectorized environments.
    """
    if not isinstance(space, gym.spaces.Box):
        return space
    if len(space.shape) == 3:
        channels, height, width = image_channels_hw(space)
        if image_layout(space.shape) == "chw":
            return space
        low = np.zeros((channels, height, width), dtype=space.dtype)
        high_value = 255 if np.issubdtype(space.dtype, np.integer) else 1.0
        high = np.full((channels, height, width), high_value, dtype=space.dtype)
        return gym.spaces.Box(low=low, high=high, dtype=space.dtype)
    if len(space.shape) == 4 and image_layout(space.shape[1:]) == "hwc":
        # Isaac Lab batched space: (N, H, W, C) → (N, C, H, W)
        n, h, w, c = space.shape
        low = np.zeros((n, c, h, w), dtype=space.dtype)
        high_value = 255 if np.issubdtype(space.dtype, np.integer) else 1.0
        high = np.full((n, c, h, w), high_value, dtype=space.dtype)
        return gym.spaces.Box(low=low, high=high, dtype=space.dtype)
    return space


def channel_stack_space(space: gym.Space, frames: int) -> gym.Space:
    """Return a Box with frames stacked on the image channel axis."""
    if isinstance(space, gym.spaces.Dict):
        spaces = dict(space.spaces)
        spaces["policy"] = channel_stack_space(spaces["policy"], frames)
        return gym.spaces.Dict(spaces)
    if not isinstance(space, gym.spaces.Box) or len(space.shape) != 3:
        return space
    shape = list(space.shape)
    axis = 2 if image_layout(space.shape) == "hwc" else 0
    shape[axis] *= int(frames)
    low = np.zeros(tuple(shape), dtype=space.dtype)
    high_value = 255 if np.issubdtype(space.dtype, np.integer) else 1.0
    high = np.full(tuple(shape), high_value, dtype=space.dtype)
    return gym.spaces.Box(low=low, high=high, dtype=space.dtype)


def channel_stack_frames(frames: Iterable, base_shape: tuple[int, ...] | None = None):
    """Stack image frames along channel axis, preserving HWC/CHW layout."""
    values = list(frames)
    if not values:
        raise ValueError("channel_stack_frames requires at least one frame")
    shape = base_shape or values[0].shape
    axis = 2 if image_layout(shape) == "hwc" else 0
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values, dim=axis)
    return np.concatenate([np.asarray(frame) for frame in values], axis=axis)


def prepare_image_tensor(
    observations: torch.Tensor,
    observation_space: gym.Space,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Restore flattened CHW image observations and return float BCHW tensors ready for CNNs.

    ChannelsFirst (applied last in Body.wrap) guarantees observations arrive in
    CHW order, so no permute is needed here.
    """
    x = observations
    if device is not None and x.device != device:
        x = x.to(device, non_blocking=True)
    if x.ndim == 2 and isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 3:
        x = x.view(x.shape[0], *observation_space.shape)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        high = getattr(observation_space, "high", None)
        high_value = float(np.max(high)) if high is not None else 1.0
        if high_value > 1.0:
            x = x / 255.0
    return x
