import os
from agent.supervised_agent import SupervisedAgent
from agent.unsupervised_agent import ICMAgent
import common.base_agent as base_agent
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from common.base_experiment import Experiment
from env_wrapper.viewpoint_env_wrapper import ViewpointEnv
import common.logger as logger


class ViewpointExperiment(Experiment):
    def __init__(self, config):
        super().__init__(config)

    def generate_environment(self,env_config):
        env = ViewpointEnv(**env_config)
        return env

    def generate_mode_parameter(self, mode, env_config):
        object =  "ship" if env_config["use_ship"] else "fork"
        side_view= "side" if env_config["side_view"] else "front"
        return mode + "-"+ object +"-"+side_view
        
    def new_agent(self, config):
        if config["reward"].lower() == "supervised":
            return SupervisedAgent(**config)
        
        return ICMAgent(**config)
    
    def generate_log_title(self, env_config):
        object =  "ship" if env_config["use_ship"] else "fork"
        side_view= "side" if env_config["side_view"] else "front"
        return "_".join([object, side_view]) + "-" + env_config["run_id"]
    
    #Run the experiment with the specified mode
    def run(self):
        if self.mode == "train": #train agents
            self.train_agents()
        elif self.mode == "test": #run all tests
            self.test_agents("rest")
            self.test_agents("exp1")
            self.test_agents("exp2")
        elif self.mode == "full": #train and run all tests
            self.train_agents()
            self.test_agents("rest")
            self.test_agents("exp1")
            self.test_agents("exp2")
        else:
            self.test_agents(self.mode) #run specified test
        
    
@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig):
    ve = ViewpointExperiment(cfg)
    ve.run()

if __name__ == '__main__':
    run_experiment()
