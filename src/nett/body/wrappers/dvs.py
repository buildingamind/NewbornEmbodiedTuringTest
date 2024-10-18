"""
Dynamic Vision Sensor (DVS) transformation for gym environments.
"""

import collections
import gym
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

    Methods:
        create_grayscale(image): Converts an image to grayscale.
        gaussianDiff(previous, current): Computes the difference between two images using Gaussian blur.
        observation(obs): Performs the DVS transformation on the observation.
        threshold(change): Applies a threshold to the change map.
        reset(**kwargs): Resets the environment and returns the initial observation.

    """

    def __init__(self, env, change_threshold=60, kernel_size=(3, 3), sigma=1, is_color = True):
        super().__init__(env)
        
        self.change_threshold = change_threshold
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.num_stack = 2 ## default
        self.env = gym.wrappers.FrameStack(env,self.num_stack)
        self.stack = collections.deque(maxlen=self.num_stack)
        self.is_color = is_color
        
        try:
            _, channels, width, height = self.env.observation_space.shape # stack,
            self.shape=(channels, width, height)
            self.observation_space = gym.spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)
            logger.info("In dvs wrapper")
        except Exception as e:
            raise e
        
        
    def create_grayscale(self, image):
        """
        Converts an image to grayscale.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The grayscale image.

        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

    def gaussianDiff(self, previous, current):
        """
        Computes the difference between two images using Gaussian blur.

        Args:
            previous (numpy.ndarray): The previous image.
            current (numpy.ndarray): The current image.

        Returns:
            numpy.ndarray: The difference map.

        """
        previous = cv2.GaussianBlur(previous, self.kernel_size, self.sigma)
        np_previous = np.asarray(previous, dtype=np.int64)
        
        current = cv2.GaussianBlur(current, self.kernel_size, self.sigma)
        np_current = np.asarray(current, dtype=np.int64)
        
        change = np_current - np_previous
        
        return change
    
    def observation(self, obs):
        """
        Performs the DVS transformation on the observation.

        Args:
            obs (list): The list of stacked frames.

        Returns:
            numpy.ndarray: The transformed observation.

        """
        
        if len(obs)>0:
            prev = np.transpose(obs[0], (1, 2, 0))
            current = np.transpose(obs[1], (1, 2, 0))
            
            if not self.is_color:
                prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
                current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
                
            change = self.gaussianDiff(prev, current)
            
            ## threshold
            dc = self.threshold(change)
            
        else:
            obs = np.transpose(obs, (1, 2, 0))
            
            if not self.is_color:
                obs = self.create_grayscale(obs)
            
            obs = np.array(obs, dtype=np.float32) / 255.0
            dc = self.threshold(obs)
        
        # change to channel first, w, h
        dc = np.transpose(dc, (2, 0, 1))
        
        return  dc.astype(np.uint8)

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
            ret_frame = abs(change)
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
        initial_obs = self.env.reset(**kwargs)
        return self.observation(initial_obs)

    def seed(self, seed: int) -> None:
        """
        Seed the environment.

        Args:
            seed (int): The seed value.
        """
        self.env.seed(seed)
