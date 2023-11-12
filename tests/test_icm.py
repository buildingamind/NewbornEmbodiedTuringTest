#!/usr/bin/env python3

import pytest
from src.simulation.algorithms.icm import ICM
import torch as th
import os
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from src.simulation.networks.inverse_forward_networks import Encoder, ForwardDynamicsModel,InverseDynamicsModel

from src.simulation.env_wrapper.parsing_env_wrapper import ParsingEnv
from src.simulation.env_wrapper.viewpoint_env_wrapper import ViewpointEnv
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import initialize, compose
'''
@pytest.mark.parametrize("encoder_cls", [Encoder])
@pytest.mark.parametrize("device",["cuda","cpu"])
@pytest.mark.parametrize(
    "env_cls", [ParsingEnv]
)
def test_encoder(encoder_cls, device, env_cls):
    with initialize(version_base=None, config_path="../src/simulation/conf"):
        # config is relative to a module
        config_name = "config_parsing" if env_cls in [ParsingEnv] else "config"
        cfg = compose(config_name=config_name)
     
    if env_cls in [ParsingEnv]:
        env_config = cfg["Environment"]
        with open_dict(env_config):
                object =  "ship" if env_config["use_ship"] else "fork"
                env_config["mode"] = "rest" + "-"+ object +"-"+env_config["background"]
                env_config["random_pos"] = True
                env_config["rewarded"] = True
                env_config["run_id"] = cfg["run_id"] + "_" + "test"
                env_config["rec_path"] = os.path.join(env_config["rec_path"] , f"agent_0/")   
        env = env_cls(**env_config)
    
    e_gen = lambda : env
    env = make_vec_env(env_id=e_gen, n_envs=1)
    
    obs = env.reset()
    print(env.observation_space.shape, env.action_space.shape)
    
    encoder = encoder_cls(obs_shape=env.observation_space.shape,action_dim = env.action_space.shape[0] ,latent_dim=128).to(device)
    
    obs = th.from_numpy(env.observation_space.sample()).to(th.float32)
    
    obs = th.as_tensor(obs, device=device).unsqueeze(0)
    obs = encoder(obs)
    print(obs.shape, encoder.feature_size(device))
    print("Encoder test passed!")


@pytest.mark.parametrize("encoder_cls", [Encoder])
@pytest.mark.parametrize("forward_cls", [ForwardDynamicsModel])
@pytest.mark.parametrize("device",["cuda","cpu"])
@pytest.mark.parametrize(
    "env_cls", [ParsingEnv]
)
def test_forward_network(encoder_cls,forward_cls, device,env_cls):
    with initialize(version_base=None, config_path="../src/simulation/conf"):
        # config is relative to a module
        config_name = "config_parsing" if env_cls in [ParsingEnv] else "config"
        cfg = compose(config_name=config_name)
     
    if env_cls in [ParsingEnv]:
        env_config = cfg["Environment"]
        with open_dict(env_config):
                object =  "ship" if env_config["use_ship"] else "fork"
                env_config["mode"] = "rest" + "-"+ object +"-"+env_config["background"]
                env_config["random_pos"] = True
                env_config["rewarded"] = True
                env_config["run_id"] = cfg["run_id"] + "_" + "test"
                env_config["rec_path"] = os.path.join(env_config["rec_path"] , f"agent_0/")   
        env = env_cls(**env_config)
    
    e_gen = lambda : env
    env = make_vec_env(env_id=e_gen, n_envs=1)
    
    obs = env.reset()
    print(env.observation_space.shape, env.action_space.shape)
    latent_dim = 128
    encoder = encoder_cls(obs_shape=env.observation_space.shape,\
        action_dim = env.action_space.shape[0] ,latent_dim=latent_dim).to(device)
    
    obs = th.from_numpy(env.observation_space.sample()).to(th.float32)
    obs = th.as_tensor(obs, device=device).unsqueeze(0)
    obs = encoder(obs)
    
    sample = env.action_space.sample()
    action = np.array(sample)
    action = th.from_numpy(action)
    forward_net = forward_cls(latent_dim,env.action_space.shape[0]).to(device)
    pred_next_obs = forward_net(obs,action.unsqueeze(0).to(device))
    print(pred_next_obs.shape)
    
    print("Forward network passed")
    
    
@pytest.mark.parametrize("encoder_cls", [Encoder])
@pytest.mark.parametrize("inverse_cls", [InverseDynamicsModel])
@pytest.mark.parametrize("device",["cuda","cpu"])
@pytest.mark.parametrize(
    "env_cls", [ParsingEnv]
)
def test_inverse_network(encoder_cls,inverse_cls, device,env_cls):
    with initialize(version_base=None, config_path="../src/simulation/conf"):
        # config is relative to a module
        config_name = "config_parsing" if env_cls in [ParsingEnv] else "config"
        cfg = compose(config_name=config_name)
     
    if env_cls in [ParsingEnv]:
        env_config = cfg["Environment"]
        with open_dict(env_config):
                object =  "ship" if env_config["use_ship"] else "fork"
                env_config["mode"] = "rest" + "-"+ object +"-"+env_config["background"]
                env_config["random_pos"] = True
                env_config["rewarded"] = True
                env_config["run_id"] = cfg["run_id"] + "_" + "test"
                env_config["rec_path"] = os.path.join(env_config["rec_path"] , f"agent_0/")   
        env = env_cls(**env_config)
    
    e_gen = lambda : env
    env = make_vec_env(env_id=e_gen, n_envs=1)
    
    obs = env.reset()
    print(env.observation_space.shape, env.action_space.shape)
    latent_dim = 128
    
    encoder = encoder_cls(obs_shape=env.observation_space.shape,\
        action_dim = env.action_space.shape[0] ,latent_dim=latent_dim).to(device)
    
    obs = th.from_numpy(env.observation_space.sample()).to(th.float32)
    obs = th.as_tensor(obs, device=device).unsqueeze(0)
    obs1 = encoder(obs)
    
    obs = th.from_numpy(env.observation_space.sample()).to(th.float32)
    obs = th.as_tensor(obs, device=device).unsqueeze(0)
    obs2 = encoder(obs)
    
    inverse_net = inverse_cls(latent_dim,env.action_space.shape[0]).to(device)
    pred_action = inverse_net(obs1, obs2).to(device)
    print(pred_action.shape)
    
    print("Inverse network passed")
'''
@pytest.mark.parametrize(
    "env_cls", [ParsingEnv]
)
@pytest.mark.parametrize(
    "icm_cls", [ICM]
)
@pytest.mark.parametrize("device",["cuda","cpu"])
def test_icm(env_cls, icm_cls,device):
    with initialize(version_base=None, config_path="../src/simulation/conf"):
        # config is relative to a module
        config_name = "config_parsing" if env_cls in [ParsingEnv] else "config"
        cfg = compose(config_name=config_name)
     
    if env_cls in [ParsingEnv]:
        env_config = cfg["Environment"]
        with open_dict(env_config):
                object =  "ship" if env_config["use_ship"] else "fork"
                env_config["mode"] = "rest" + "-"+ object +"-"+env_config["background"]
                env_config["random_pos"] = True
                env_config["rewarded"] = True
                env_config["run_id"] = cfg["run_id"] + "_" + "test"
                env_config["rec_path"] = os.path.join(env_config["rec_path"] , f"agent_0/")   
        env = env_cls(**env_config)
    
    
    e_gen = lambda : env
    env = make_vec_env(env_id=e_gen, n_envs=1)
    
    
    icm = icm_cls(observation_space=env.observation_space, action_space=env.action_space, device=device)
    obs = th.rand(size=(256, 1, *env.observation_space.shape)).to(device)
    action = th.rand(size=(256, 1, env.action_space.shape[0])).to(device)
    
    samples = {
        "observations": obs,
        "actions": action
        
    }

    print(action.shape)
    
    reward = icm.compute_irs(samples)
    print(reward)
    print(f"Intrinsic reward test passed!")

    
    