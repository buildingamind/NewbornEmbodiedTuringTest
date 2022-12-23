import chickai_wrapper
import agent
import hydra
from omegaconf import DictConfig, OmegaConf

class ViewpointExperiment:
    def __init__(self, config):
        #Save configuration for the environment and agent
        self.env_config = config["Environment"]
        self.env_config["log_path"] = self.log_path
        agent_config = config["Agent"]
        
        #Get experiment configuration
        agent_count = config["Experiment"]["agent_count"]
        run_id = config["Experiment"]["run_id"]
        self.mode = config["Experiment"]["mode"]
        self.log_path = config["Experiment"]["log_path"]
        self.test_eps = config["Experiment"]["test_eps"]
        self.train_eps = config["Experiment"]["train_eps"]
        self.agents = []

        #Create the agents that will be used
        agent_config["path"] = self.log_path + "/Brains/"
        for i in range(agent_count):
            agent_config["agent_id"] = f"{run_id}_Agent_{i}"
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
            env_config = self.get_env_config
            env_config["mode"] = "rest"
            env_config["run_id"] = agent.id + "_" + "rest"
            env = self.generate_environment(env_config)
            agent.train(env, self.train_eps)
            agent.save(self.log_path)
            env.close()
    
    #Run the specified test trials for every agent
    def test_agents(self, mode):
        for agent in self.agents:
            env_config = self.get_env_config
            env_config["mode"] = mode
            env_config["run_id"] = agent.id + "_" + mode
            env = self.generate_environment(env_config)
            agent.test(env, self.test_eps)
            env.close()
    
    def generate_environment(self, mode="rest"):
        env = chickai_wrapper.ChickEnv(**self.env_config)
        return env

    def new_agent(self, config):
        return agent.Agent(self.log_path**config)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment():
    ve = ViewpointExperiment(config)
    ve.run()

if __name__ == '__main__':
    run_experiment()