import gymnasium as gym
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Video(gym.ObservationWrapper):

    def __init__(
        self,
        env,
        frames=3,
        *args,
        **kwargs,
    ):
        self.env = gym.wrappers.FrameStackObservation(
            env,
            stack_size=frames,
            padding_type="reset",  # TODO: check padding type is best
        )
        super().__init__(self.env)

        try:
            stack, channels, width, height = self.env.observation_space.shape  # stack,
            self.shape = (channels * stack, width, height)  # TODO: Flip h and w??
            self.observation_space = gym.spaces.Box(
                shape=self.shape, low=0, high=255, dtype=np.uint8
            )
        except Exception as e:
            raise e

    def observation(self, obs):
        # print("Original obs shape:", obs.shape)
        # print("Reformatted obs shape:", self.shape)
        # print("Obs:", obs)
        # obs = np.transpose(obs, (1, 0, 2, 3))  # flip c and f?
        obs = obs.reshape(obs.shape[0] * obs.shape[1], *obs.shape[2:])
        # obs = np.transpose(obs, (1, 0, 2, 3))  # flip c and f?
        return obs

    def reset(self, **kwargs):
        initial_obs, initial_info = self.env.reset(**kwargs)
        # print("Initial Original obs shape:", initial_obs.shape)
        # print("Reformatted obs shape:", self.shape)
        # print("Initial Obs:", initial_obs)
        return self.observation(initial_obs), initial_info
