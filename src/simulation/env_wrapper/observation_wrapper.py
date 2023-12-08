#!/usr/bin/env python3
import gym
import torch
import numpy as np 
import kornia as K

from PIL import Image
from kornia.contrib import ImageStitcher
from torchvision.transforms import Compose, ToTensor, Resize

class ObservationWrapper(gym.ObservationWrapper):
    """
    Gym env wrapper that transpose visual observations to (C,H,W).
    """
    def __init__(self, env):
        super().__init__(env)

        # Assumes visual observation space with shape (H, W, C)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 3

        #h, w, c = env.observation_space.shape
        #self.observation_space = gym.spaces.Box(0, 1, dtype=np.float32, shape=(c, h, w))
        
        width, height, channels = env.observation_space.shape
        new_shape = (channels, width, height)
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
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.stitcher = ImageStitcher(K.feature.LoFTR(pretrained='indoor'), estimator='ransac').to(self.device)
        self.in_transform = Compose([ToTensor()])
        self.out_transform = Compose([Resize((64, 64), antialias=True)])

    def observation(self, observations):
        # stitch
        stitched_image = self.stitch(observations)

        # transform (resize, convert to [0, 255]) and return
        stitched_image = (self.out_transform(stitched_image).cpu().numpy() * 255).astype(np.uint8)
        
        return stitched_image

    # implement all stitching logic here and call it from observation()
    def stitch(self, observations):
        # convert to Tensor
        observations = [self.prepare(observation) for observation in observations]

        # separate (observations are ordered alphabetically, see: https://forum.unity.com/threads/what-is-the-order-of-observations.992467/)
        center_image = observations[0]
        left_image = observations[1]
        right_image = observations[2]

        # attempt to stitch
        try:
            with torch.no_grad():
                stitched_image = self.stitcher(left_image, right_image)[0]
        # fall back to the first camera in case the stitching fails (rare)
        except:
            stitched_image = center_image.squeeze().to(self.device)

        return stitched_image
    
    def prepare(self, image):
        image_channels_first = self.swap_channels(image, channels_first=True)
        prepped_image = torch.unsqueeze(self.in_transform(image_channels_first), dim=0).to(self.device)
        return prepped_image
    
    # helper to swap channels
    def swap_channels(self, image, channels_first=False):
        if channels_first:
            return np.moveaxis(image, 0, 2)
        else:
            return np.moveaxis(image, 2, 0)
        
class ConcatenateVisualObservationsWrapper(gym.ObservationWrapper):
    """
    Gym env wrapper that concatenates visual observations from two cameras.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=(3, 64, 64))

    def observation(self, observations):
        # separate (observations are ordered alphabetically, see: https://forum.unity.com/threads/what-is-the-order-of-observations.992467/)
        left_image = observations[0]
        right_image = observations[1]

        # concatenate
        concatenated_image = self.concatenate(left_image, right_image)

        # resize and return
        return self.resize(concatenated_image)
    
    def concatenate(self, left_image, right_image):
        return np.concatenate((left_image, right_image), axis=2)
    
    def resize(self, image):
        # sinc interpolation preserves sharpness while downsampling
        resized_image = Image.fromarray(self.to_channels_last(image)).resize((64, 64), resample=Image.LANCZOS)
        
        # convert back to numpy, make it channels first, and return
        return self.to_channels_first(np.array(resized_image))
    
    def to_channels_last(self, image):
        return np.moveaxis(image, [0, 1, 2], [2, 0, 1])

    def to_channels_first(self, image):
        return np.moveaxis(image, [0, 1, 2], [1, 2, 0])
