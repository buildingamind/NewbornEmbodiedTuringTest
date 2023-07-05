import json
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import pdb

from src.simulation.env_wrapper.parsing_env_wrapper import ParsingEnv
from src.simulation.agent.unsupervised_agent import ICMAgent
import src.simulation.common.logger as logger

class ParsingICMExperiment:
    def __init__(self, config):
        #Save configuration for the environment and agent
        self.env_config = config["Environment"]
        agent_config = config["Agent"]
        
        
        #Get experiment configuration
        agent_count = config["agent_count"]
        run_id = config["run_id"]
        self.mode = config["mode"]
        self.log_path = config["log_path"]
        self.test_eps = config["test_eps"]
        self.train_eps = config["train_eps"]
        self.agents = []

        #Create the agents that will be used
        for i in range(agent_count):
            with open_dict(agent_config):
                agent_config.agent_id = f"{run_id}_Agent_{i}"
                agent_config.env_log_path = self.env_config['log_path']
                agent_config.rec_path = os.path.join(self.env_config["rec_path"] , f"Agent_{i}/")
                agent_config.recording_frames = self.env_config["recording_frames"]
            self.agents.append(self.new_agent(agent_config))

    #Run the experiment with the specified mode
    def run(self):
        self.train_agents()
        self.test_agents("exp")
        
        

    #run learn for as many agents as set for the experiment
    def train_agents(self):
        #pdb.set_trace()
        for agent in self.agents:
            env_config = self.env_config
            with open_dict(env_config):
                object =  "ship" if env_config["use_ship"] else "fork"
                env_config["mode"] = "rest" + "-"+ object +"-"+env_config["background"]
                env_config["random_pos"] = True
                env_config["rewarded"] = False
                env_config["run_id"] = agent.id + "_" + "train"
                env_config["rec_path"] = agent.rec_path 
                
            env = self.generate_environment(env_config)
            agent.train(env, self.train_eps)
            #agent.save()
            env.close()
    
    
    def generate_environment(self,env_config):
        env = ParsingEnv(**env_config)
        return env

    def new_agent(self, config):
        return ICMAgent(**config)

@hydra.main(version_base=None, config_path="conf", config_name="config_parsing")
def run_experiment(cfg: DictConfig):
    
    ve = ParsingICMExperiment(cfg)
    print(cfg)
    ve.run()

if __name__ == '__main__':
    run_experiment()
