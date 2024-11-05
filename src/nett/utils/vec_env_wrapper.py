from typing import Optional
from functools import partial
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import supersuit as ss
from supersuit.vector.concat_vec_env import ConcatVecEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

class SafeVecEnv():
    def __init__(self, callback: callable, n_envs: Optional[int] = None, zoo: Optional[bool] = False):
      if zoo:
        env = callback()
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ConcatVecEnv([lambda: env])
        self._vecenv = SB3VecEnvWrapper(env)
      elif n_envs is None:
        self._vecenv = DummyVecEnv([callback])
      else:
        self._vecenv = SubprocVecEnv([partial(callback, rank=i) for i in range(n_envs)])

    def __enter__(self):
      return self._vecenv

    def __exit__(self, *args):
      return self._vecenv.close()
