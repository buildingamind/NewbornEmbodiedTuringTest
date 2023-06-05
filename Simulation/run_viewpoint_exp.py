import Simulation.logger as logger
import os
import agent
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

class ViewpointExperiment:
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
            self.agents.append(self.new_agent(agent_config))

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
        

    #run learn for as many agents as set for the experiment
    def train_agents(self):
        for agent in self.agents:
            env_config = self.env_config
            with open_dict(env_config):
                env_config["mode"] = "rest"
                env_config["random_pos"] = True
                env_config["rewarded"] = True
                env_config["run_id"] = agent.id + "_" + "train"
            env = self.generate_environment(env_config)
            agent.train(env, self.train_eps)
            agent.save()
            env.close()
    
    #Run the specified test trials for every agent
    def test_agents(self, mode):
        for agent in self.agents:
            env_config = self.env_config
            with open_dict(env_config):
                env_config["mode"] = mode
                env_config["run_id"] = agent.id + "_" + mode
                env_config["rewarded"] = True
            env = self.generate_environment(env_config)
            agent.test(env, self.test_eps)
            env.close()
    
    def generate_environment(self, mode="rest"):
        env = logger.ViewpointEnv(**self.env_config)
        return env

    def new_agent(self, config):
        return agent.Agent(**config)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig):
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda)
    #os.environ["DISPLAY"] = ":0"

    ve = ViewpointExperiment(cfg)
    ve.run()

if __name__ == '__main__':
    run_experiment()
