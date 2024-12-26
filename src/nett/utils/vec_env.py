"""Vector Environment Classes"""

from time import sleep

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import supersuit as ss
from supersuit.vector.concat_vec_env import ConcatVecEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from nett.environment import ZooEnvironment, GymEnvironment
from nett.body import Body
from .task import Task


class SafeEnv:
    def __enter__(self):
        return self._vecenv

    def __exit__(self, *args):
        return self._vecenv.close()


class ZooEnv(SafeEnv):
    def __init__(self, task: Task):
        env = ZooEnvironment(task)
        # env = Body(ZooEnvironment) TODO: Add support for wrapping ZooEnvironments
        # TODO: Add support for recording agents in ZooEnvironments
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ConcatVecEnv([lambda: env])
        self._vecenv = SB3VecEnvWrapper(env)


class SingleEnv(SafeEnv):
    def __init__(self, task: Task):
        def callback():
            env = GymEnvironment(task, False)
            return Body(env, task)

        self._vecenv = DummyVecEnv([callback])


class TestEnv(SafeEnv):
    def __init__(self, task: Task):
        env = GymEnvironment(task, True)
        self._vecenv = Body(env, task)


class MultiEnv(SafeEnv):
    def __init__(self, task: Task, n_envs: int):
        # define callback
        def seed_callback(seed):
            def callback():
                sleep(seed)
                env = GymEnvironment(task, False, seed)
                return Body(env, task)

            return callback

        # create n_envs environments
        self._vecenv = SubprocVecEnv([seed_callback(seed) for seed in range(n_envs)])
