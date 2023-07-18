import json
import os

import yaml
from src.simulation.agent.supervised_agent import SupervisedAgent
import src.simulation.common.base_agent as base_agent
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import pdb
import pprint
from src.simulation.common.base_experiment import Experiment
from src.simulation.env_wrapper.parsing_env_wrapper import ParsingEnv
import src.simulation.common.logger as logger


class ParsingExperiment(Experiment):
    def __init__(self, config):
        super().__init__(config)

    def generate_environment(self,env_config):
        env = ParsingEnv(**env_config)
        return env

    def generate_mode_parameter(self,mode, env_config):
        object = "ship" if env_config["use_ship"] else "fork"
        return mode + "-" + object + "-" + env_config["background"]
    
    
    
    def new_agent(self, config):
        return SupervisedAgent(**config)
        
    def generate_log_title(self, env_config):
        """
        Generate log title
        ex.fork_side-agent3_train.csv

        Args:
            mode (str): mode used to indicate the type of the experiment
            run_id (str): run_id entered in the configuration
            agent_id (int): index of the agent
            env_config (dict): environment configuration data

        Returns:
            str: computed log title
        """
        object =  "ship" if env_config["use_ship"] else "fork"
        return "-".join(["_".join([object, env_config["background"]]), env_config["run_id"]])
    
    
@hydra.main(version_base=None, config_path="conf",
            config_name="config")
def run_experiment(cfg: DictConfig):
    ve = ParsingExperiment(cfg)
    pprint.pprint(cfg)
    ve.run()
    # dumps to yaml string
    #yaml_data: str = OmegaConf.to_yaml(cfg)

    # dumps to file:
    #with open("config.yaml", "w") as f:
    #    OmegaConf.save(cfg, f)
    

if __name__ == '__main__':
    run_experiment()
