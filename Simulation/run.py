import os
import socket
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra

from mlagents.trainers.learn import RunOptions, run_cli


import signal
from contextlib import contextmanager


def port_in_use(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
    except socket.error:
        return True
    return False


def process_env_args(env_args_dict):
    env_args_list = []
    for k, v in env_args_dict.items():
        env_args_list.extend([f"--{k}", str(v)])
    return env_args_list


def build_trainer_settings(config):
    trainer_settings = config["trainer"]
    trainer_settings["reward_signals"] = config["reward_signals"]
    if config["checkpoint_settings"]["run_id"].startswith("test"):
        trainer_settings["hyperparameters"].update(
            {"learning_rate": 0, "num_epoch": 0}
        )
        trainer_settings.update(
            {"max_steps": trainer_settings["max_steps"] + 100000}
        )
    return trainer_settings


def build_run_options(config):
    # Convert DictConfig object into mlagents RunOptions.
    config = OmegaConf.to_container(config, resolve=True)

    checkpoint_settings = config["checkpoint_settings"]
    env_settings = config["env_settings"]
    engine_settings = config["engine_settings"]
    trainer_settings = build_trainer_settings(config)

    # Build env_args list.
    env_settings["env_args"] = process_env_args(env_settings["env_args"])

    config_dict = {
        "default_settings": trainer_settings,
        "checkpoint_settings": checkpoint_settings,
        "env_settings": env_settings,
        "engine_settings": engine_settings,
        "torch_settings": {"device": "cuda"}
    }

    run_options = RunOptions.from_dict(config_dict)
    return run_options


@hydra.main(config_path="configs", config_name="main")
def run(config: DictConfig) -> None:
    #print(OmegaConf.to_yaml(config, resolve=True))
    # Find a port that is not in use.
    while port_in_use(config.env_settings.base_port):
        config.env_settings.base_port += 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)
    os.environ["DISPLAY"] = ":0"

    # Convert DictConfig to a python dictionary.
    options = build_run_options(config)

    run_cli(options)

if __name__ == "__main__":
    run()

