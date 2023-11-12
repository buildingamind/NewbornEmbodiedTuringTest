#!/usr/bin/env python3
import gym
import cv2
from stitching import AffineStitcher
import numpy as np 

class ObservationWrapper(gym.ObservationWrapper):
    """
    Gym env wrapper that transpose visual observations to (H, W, C).
    """
    def __init__(self, env):
        super().__init__(env)

        # Assumes visual observation space with shape (H, W, C)
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        assert len(self.env.observation_space.shape) == 3
        
        channels, width, height = self.env.observation_space.shape
        new_shape = (width, height, channels)
        self.observation_space = gym.spaces.Box(low=0.0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)
    
class StitchVisualObservationsWrapper(gym.ObservationWrapper):
    """
    Gym env wrapper that stitches visual observations from two cameras.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=(3, 64, 64))
        self.stitcher = AffineStitcher(detector="sift", confidence_threshold=0.0)

    def observation(self, observation):
        # stitch and return
        return self.stitch(observation[0], observation[1])

    # implement all stitching logic here and call it from observation()
    def stitch(self, right_eye, left_eye):
        # try to stitch
        try:
            stitched = self.stitcher.stitch([self.swap_channels(right_eye), self.swap_channels(left_eye)])
            stitched = self.swap_channels(cv2.resize(stitched, (64, 64)), channels_first=True)
        # fall back to the first camera in case the stitching fails (doesn't happen often)
        except:
            stitched = right_eye
        return stitched

    # helper to swap channels (cv2 requires channel first)
    def swap_channels(self, image, channels_first=False):
        if channels_first:
            return np.moveaxis(image, 2, 0)
        else:
            return np.moveaxis(image, 0, 2)