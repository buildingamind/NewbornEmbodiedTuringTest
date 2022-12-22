import chickai_wrapper
import hydra
from omegaconf import DictConfig, OmegaConf

class ViewpointExperiment:
    def __init__(self):
        #initialize experiment

    def train_agents(self):
        #run learn for as many agents as set for the experiment

    def test_agents(self):
        #run tests on agents

    def 

@hydra.main(version_base=None, config_path="conf", config_name="config")
def get(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg["ChickAI"]["Experiment"])
    test_fun(**cfg["ChickAI"]["Experiment"])

def run_experiment():


if __name__ == '__main__':
    get()
