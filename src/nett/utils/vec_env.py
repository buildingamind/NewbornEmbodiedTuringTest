from typing import Optional
from functools import partial

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class VecEnv():
    def __init__(self, callback: callable, n_envs: Optional[int] = None):
      if n_envs is None:
        self._vecenv = DummyVecEnv([callback])
      else:
        self._vecenv = SubprocVecEnv([partial(callback, rank=i) for i in range(n_envs)])

    def __enter__(self):
      return self._vecenv

    def __exit__(self, *args):
      return self._vecenv.close()
