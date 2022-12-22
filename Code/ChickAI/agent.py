#Packages for opening unity package
import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import socket

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid
import os
import hydra
from omegaconf import DictConfig, OmegaConf

test_eps = 1200
ep_steps = 1000
test_steps = ep_steps * test_eps
rest_steps = ep_steps * 100#test_eps
env_path = "../../Env/rearing_chamber"

class Agent:
    def __init__(self, brain=None, path=None):
        #Either read in an existing brain or create a new one
        #Including a rewarder is needed

    def train(self, environment):
        #Run training in environment with reward wrapper is needed

    #Test the agent in the given environment for the set number of steps
    def test(self, env, steps):
        obs = env.reset()
        for i in range(steps):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()

    def save(self, path):
        #Save the brains into a file

    def load(self, path):
        #Load brains from the file


def train(agent_id, useShip=False, sideView=False, steps=1e2):
    global env_path
    env = ChickEnv(agent_id, useShip=useShip, sideView=sideView, path=env_path)
    model = PPO("CnnPolicy", env, verbose=1)
    print("Training")
    model.learn(total_timesteps=steps)
    env.log("Rest Trials")
    obs = env.reset()
    global rest_steps
    print("Rest trials")
    for i in range(rest_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
    env.close()
    return model

def run(agent_id, useShip, sideView):
    model = train(agent_id, useShip, sideView)
    print("Exp 1")
    exp1(model, agent_id, useShip, sideView)
    print("Exp 2")
    exp2(model, agent_id, useShip, sideView)

def intrinsic_train(model,ep_steps,episodes,envs,module):
    _, callback = model._setup_learn(total_timesteps=ep_steps*episodes, eval_env=None)

    for i in range(episodes):
        model.collect_rollouts(
            env=envs,
            rollout_buffer=model.rollout_buffer,
            n_rollout_steps=ep_steps,
            callback=callback
        )
        # Compute intrinsic rewards.
        intrinsic_rewards = module.compute_irs(
            buffer=model.rollout_buffer,
            time_steps=i * ep_steps * len(envs))
        model.rollout_buffer.rewards = intrinsic_rewards
        # Update policy using the currently gathered rollout buffer.
        model.train()

def main():
    run("test_agent", True, True)

if __name__ == '__main__':
    my_app()
    #main()
