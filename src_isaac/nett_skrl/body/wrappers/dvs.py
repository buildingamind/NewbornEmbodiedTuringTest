"""
Dynamic Vision Sensor (DVS) transformation for gym environments.
"""

import gymnasium as gym
import numpy as np
import cv2
import logging
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DVS(gym.ObservationWrapper):
    """
    A gym observation wrapper that performs Dynamic Vision Sensor (DVS) transformation on the environment observations.

    Args:
        env (gym.Env): The environment to wrap.
        change_threshold (int): The threshold value for detecting changes in pixel intensity.
        kernel_size (tuple): The size of the Gaussian kernel used for blurring.
        sigma (float): The standard deviation of the Gaussian kernel.
        is_color (bool): Whether the observation is in color or grayscale.

    Attributes:
        change_threshold (int): The threshold value for detecting changes in pixel intensity.
        kernel_size (tuple): The size of the Gaussian kernel used for blurring.
        sigma (float): The standard deviation of the Gaussian kernel.
        num_stack (int): The number of frames to stack.
        env (gym.Env): The wrapped environment.
        stack (collections.deque): A deque to store the stacked frames.
        shape (tuple): The shape of the observation space.
        observation_space (gym.spaces.Box): The modified observation space.
        is_color (bool): Whether the observation is in color or grayscale.

    Methods:
        observation(obs): Performs the DVS transformation on the observation.
        threshold(change): Applies a threshold to the change map.
        reset(**kwargs): Resets the environment and returns the initial observation.

    """

    def __init__(
        self,
        env,
        change_threshold=30,
        kernel_size=(3, 3),
        sigma=1,
        is_color=True,
        *args,
        **kwargs,
    ):

        self.change_threshold = change_threshold
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.is_color = is_color

        self.env = gym.wrappers.FrameStackObservation(
            env, stack_size=2, padding_type="zero"
        )
        super().__init__(self.env)

        # FrameStackObservation preserves the dict shape; if the inner space is
        # a Dict, the stacked space is too. Build a matching output space.
        inner = self.env.observation_space
        if isinstance(inner, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({
                key: self._out_box(inner[key].shape) for key in inner.spaces
            })
        else:
            self.observation_space = self._out_box(inner.shape)

    def _out_box(self, stacked_shape: tuple) -> gym.spaces.Box:
        """Per-key output shape after the DVS transform.

        ``stacked_shape`` is the FrameStackObservation-wrapped shape — first
        axis is the stack dim. Output is HWC.
        """
        # Drop the leading stack dim.
        s = stacked_shape[1:]
        if len(s) == 4:
            # legacy stacked input — assume (channels, width, height) order
            channels, width, height = s[0], s[1], s[2]
        else:
            # (H, W, C) input — what NETTEnv returns.
            height, width, channels = s[0], s[1], s[2]
        if not self.is_color:
            channels = 1
        return gym.spaces.Box(shape=(height, width, channels), low=0, high=255, dtype=np.uint8)

    def _create_grayscale(self, image):
        """
        Converts an image to grayscale.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The grayscale image.

        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _gaussianDiff(self, previous, current):
        """
        Computes the difference between two images using Gaussian blur.

        Args:
            previous (numpy.ndarray): The previous image.
            current (numpy.ndarray): The current image.

        Returns:
            numpy.ndarray: The difference map.

        """
        previous = cv2.GaussianBlur(previous, self.kernel_size, self.sigma)

        current = cv2.GaussianBlur(current, self.kernel_size, self.sigma)

        change = current - previous

        return change

    def _transform_pair(self, prev: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Run the DVS pipeline on a (prev, current) HWC frame pair, returning HWC uint8."""
        if not self.is_color:
            prev = self._create_grayscale(prev)
            current = self._create_grayscale(current)

        change = self._gaussianDiff(prev, current)
        dc = self.threshold(change)

        if not self.is_color:
            dc = np.expand_dims(dc, axis=2)  # (H, W) -> (H, W, 1)
        return dc

    def observation(self, obs):
        """Performs the DVS transformation on the observation.

        Supports both raw stacked arrays (legacy gym vec envs) and dict-shaped
        observations (Isaac Lab's ``{"policy": tensor}``). For dict obs we
        operate on each key independently and return a dict with the same keys.
        """
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            out = {}
            for key, stacked in obs.items():
                # FrameStackObservation preserves the dict shape and stacks along
                # the new leading axis; last two entries are (prev, current).
                prev = _as_numpy(stacked[-2])
                current = _as_numpy(stacked[-1])
                out[key] = self._transform_pair(prev, current)
            return out

        # Legacy gym path: obs is a 2-element stack of HWC frames.
        prev = _as_numpy(obs[0])
        current = _as_numpy(obs[1])
        return self._transform_pair(prev, current)

    def threshold(self, change):
        """
        Applies a threshold to the change map.

        Args:
            change (numpy.ndarray): The change map.

        Returns:
            numpy.ndarray: The thresholded change map.

        """
        if not self.is_color:
            ret_frame = np.full(change.shape, 128, dtype=np.uint8)
            ret_frame[change >= self.change_threshold] = 255
            ret_frame[change <= -self.change_threshold] = 0
        else:
            ret_frame = change
            ret_frame[ret_frame < self.change_threshold] = 0

        return ret_frame

    def reset(self, **kwargs):
        """
        Resets the environment and returns the initial observation.

        Args:
            **kwargs: Additional arguments for the reset method.

        Returns:
            numpy.ndarray: The initial observation.
        """
        initial_obs, initial_info = self.env.reset(**kwargs)
        return self.observation(initial_obs), initial_info


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        # DVS is a CPU/NumPy preprocessing wrapper; use Video for GPU-resident frame stacking.
        return value.detach().cpu().numpy()
    return np.asarray(value)
