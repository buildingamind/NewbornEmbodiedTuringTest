import hydra
from omegaconf import DictConfig, open_dict
import pdb

from agent.supervised_agent import SupervisedAgent
from agent.unsupervised_agent import ICMAgent
from common.base_experiment import Experiment
from env_wrapper.identityandview_env_wrapper import IdentityAndViewEnv

class IdentityAndViewExperiment(Experiment):
    def __init__(self, config):
        super().__init__(config)

    def generate_environment(self, env_config):
        env = IdentityAndViewEnv(**env_config)
        return env

    def generate_mode_parameter(self, mode, env_config):
        spec = env_config["object"] + "-" + env_config["rotation"]
        return mode + "-" + spec
    
    def new_agent(self, config):
        if config["reward"].lower() == "supervised":
            return SupervisedAgent(**config)
        return ICMAgent(**config)
        
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
        spec = env_config["object"] + "-" + env_config["rotation"]
        return spec + "-" + env_config["run_id"]
    
    
@hydra.main(version_base=None, config_path="conf",
            config_name="config")
def run_experiment(cfg: DictConfig):
    with open_dict(cfg):
        print(cfg)
    
    ve = IdentityAndViewExperiment(cfg)
    ve.run()
    

if __name__ == '__main__':
    run_experiment()
