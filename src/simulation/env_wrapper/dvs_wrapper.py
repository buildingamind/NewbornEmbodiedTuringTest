#!/usr/bin/env python3

import collections
import gym
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import pdb
from PIL import Image
import os
import cv2

class DVSWrapper(gym.ObservationWrapper):
    def __init__(self, env, change_threshold=60, kernel_size=(3, 3), sigma=1 ):
        super().__init__(env)
        
        self.change_threshold = change_threshold
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.num_stack = 2
        self.env = gym.wrappers.FrameStack(env,self.num_stack)
        self.stack = collections.deque(maxlen=self.num_stack)
        
        stack, width, height, channels = self.env.observation_space.shape
        self.shape=(1, width, height)
        self.observation_space = gym.spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)
        
        
    def create_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

    def gaussianDiff(self, previous, current):
        previous = cv2.GaussianBlur(previous, self.kernel_size, self.sigma)
        np_previous = np.asarray(previous, dtype=np.int64)
        
        current = cv2.GaussianBlur(current, self.kernel_size, self.sigma)
        np_current = np.asarray(current, dtype=np.int64)
        
        change = np_current - np_previous
        
        return change.reshape(change.shape[0],change.shape[1],1)
    
    
    def observation(self, obs):
        prev = self.create_grayscale(obs[0])
        current = self.create_grayscale(obs[1])
        change = self.gaussianDiff(prev, current)
        
        ## threshold
        dc = self.threshold(change)
          
        return np.swapaxes(dc, 2, 0).astype(np.uint8)

    def threshold(self, change):
        dc = np.ones(shape=change.shape) * 128
        dc[change >= self.change_threshold] = 255
        dc[change <= -self.change_threshold] = 0
        return dc
    
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        frames = []
        [frames.append(obs) for _ in range(self.num_stack)]
        return self.observation(frames)
    
    
        