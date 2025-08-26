"""
Dynamic Vision Sensor (DVS) transformation for gym environments.
"""

import collections
import gymnasium as gym
import numpy as np
import cv2
import logging

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

        try:
            _, channels, width, height = self.env.observation_space.shape  # stack,
            self.shape = (channels, width, height)
            self.observation_space = gym.spaces.Box(
                shape=self.shape, low=0, high=255, dtype=np.uint8
            )
        except Exception as e:
            raise e

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

    def observation(self, obs):
        """
        Performs the DVS transformation on the observation.

        Args:
            obs (list): The list of stacked frames.

        Returns:
            numpy.ndarray: The transformed observation.

        """
        # Avoid transpose by working with channel-first format directly
        prev = np.transpose(obs[0], (1, 2, 0))  # Convert to (H, W, C) format
        current = np.transpose(obs[1], (1, 2, 0))  # Convert to (H, W, C) format

        if not self.is_color:
            prev = self._create_grayscale(prev)
            current = self._create_grayscale(current)

        change = self._gaussianDiff(prev, current)
        dc = self.threshold(change)

        return np.transpose(dc, (2, 0, 1))

    def threshold(self, change):
        """
        Applies a threshold to the change map.

        Args:
            change (numpy.ndarray): The change map.

        Returns:
            numpy.ndarray: The thresholded change map.

        """
        if not self.is_color:
            ret_frame = np.ones(shape=change.shape) * 128
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
