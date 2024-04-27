from abc import abstractmethod
import sys
import os
import time
import logging
import argparse
import yaml
from nett import Brain, Body, Environment
from nett import NETT
import pdb
from nett.environment.configs import Binding, Parsing, ViewInvariant
from wrapper.dvs_wrapper import DVSWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_configuration(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class BodyConfiguration:
    def __init__(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class BrainConfiguration:
    def __init__(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class EnvironmentConfiguration:
    def __init__(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Experiment:
    """
    Generic Experiment Class To Run 3 experiments - Parsing, Binding and ViewInvariant
    """

    def __init__(self, **kwargs) -> None:
        ## initialize configurations
        self.brain_config = BrainConfiguration(kwargs.get('Brain'))
        self.body_config = BodyConfiguration(kwargs.get('Body'))
        self.env_config = EnvironmentConfiguration(kwargs.get('Environment'))
        self.base_simclr_checkpoint_path = os.path.join(os.getcwd(), "../data/checkpoints")

        self.encoder_config = {
            "small": {
                "feature_dimensions": 512,  # replace with actual feature dimensions for 'small'
                "encoder": "",
            },
            "medium": {
                "feature_dimensions": 128,  # replace with actual feature dimensions for 'medium'
                "encoder": "resnet10",
            },
            "large": {
                "feature_dimensions": 128,  # replace with actual feature dimensions for 'large'
                "encoder": "resnet18",
            },
            "dinov2": {
                "feature_dimensions": 384,  # replace with actual feature dimensions for 'dinov2'
                "encoder": "dinvo2",
            },
            "dinov1": {
                "feature_dimensions": 384,  # replace with actual feature dimensions for 'dinov1',
                "encoder": "dinov1",
            },
            "simclr": {
                "feature_dimensions": 512,  # replace with actual feature dimensions for 'ego4d'
                "encoder": "frozensimclr",
            },
            "sam": {
                "feature_dimensions": 256,  # replace with actual feature dimensions for 'sam'
                "encoder": "sam",
            }
        }

        ## Environment
        self.env = self.initialize_environment()
        
        ## Body
        self.body = self.initialize_body()

        ## Brain
        self.brain = self.initialize_brain()
        
        ## configuration
        config = kwargs.get('Config')
        self.train_eps = config['train_eps']
        self.test_eps = config['test_eps']
        self.mode = config['mode']
        self.num_brains = config['num_brains']
        self.output_dir = config['output_dir']
        self.run_id  = config['run_id']
        
        print(self.train_eps, self.test_eps, self.mode, self.num_brains, self.output_dir, self.run_id) 
        

    
    def initialize_brain(self):
        """
        Initialize Brain class with the attributes extracted from the brain_config

        Returns:
            _type_: _description_
        """

        # Extract attributes from brain_config
        brain_config_attrs = {attr: getattr(self.brain_config, attr) for attr in dir(self.brain_config) \
                              if not attr.startswith('__')}

        
        ## update encoder attr in brain_config
        if brain_config_attrs['encoder']:
            encoder_config = self.encoder_config[brain_config_attrs['encoder']]
            brain_config_attrs['encoder'] = encoder_config['encoder']
            brain_config_attrs['embedding_dim'] = encoder_config['feature_dimensions']

        
        
        ## Add checkpoint path
        if brain_config_attrs.get('encoder') == 'frozensimclr':
            checkpt_path = self.get_checkpoint_path()
            brain_config_attrs['custom_encoder_args'] = {'checkpoint_path':\
                self.get_checkpoint_path()}

        # Initialize Brain class with the extracted attributes
        brain = Brain(**brain_config_attrs)
        return brain

    def initialize_body(self):
        wrappers = []
        if self.body_config.dvs:
            wrappers = [DVSWrapper]
        
        return Body(type='basic', wrappers=wrappers)

    @abstractmethod
    def initialize_environment(self):
        pass## abstract method to be implemented by the child classes
    
    def run(self):
        benchmarks = NETT(brain=self.brain, body=self.body, environment=self.env)
        print(self.mode)
        benchmarks.run(num_brains=self.num_brains, \
            train_eps=self.train_eps, \
            test_eps=self.train_eps, \
            mode=self.mode, \
            output_dir=self.output_dir,run_id=self.run_id)
        
        #logger.info("Experiment completed successfully")

class ParsingExperiment(Experiment):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def get_checkpoint_path(self):
        ## compute simclr checkpoints
        checkpoint_dict = {
            "ship_a": "ship_A/checkpoints/epoch=97-step=14601.ckpt",
            "ship_b": "ship_B/checkpoints/epoch=97-step=14601.ckpt",
            "ship_c": "ship_C/checkpoints/epoch=96-step=14452.ckpt",
            "fork_b": "fork_B/checkpoints/epoch=95-step=14303.ckpt",
            "fork_a": "fork_A/checkpoints/epoch=97-step=14601.ckpt",
            "fork_c": "fork_C/checkpoints/epoch=97-step=14601.ckpt"
        }
        
        parsing_checkpoint = os.path.join(self.base_simclr_checkpoint_path, 'simclr_parsing')
        checkpoint_key = f"{self.object.lower()}_{self.background.lower()}"
        path = checkpoint_dict.get(checkpoint_key, '')
        return os.path.join(parsing_checkpoint, path)

    def initialize_environment(self):
        """
        Initialize environment class with the attributes extracted from the env_config

        Returns:
            _type_: _description_
        """
        self.object = "ship" if getattr(self.env_config, 'use_ship', False) else "fork"
        self.background = getattr(self.env_config, 'background', '')
        
        # Extract attributes from brain_config
        env_config_attrs = {attr: getattr(self.env_config, attr) for attr in dir(self.env_config) \
                            if not attr.startswith('__')}

        del env_config_attrs['use_ship']
        del env_config_attrs['background']
        
        env_config_attrs['config'] = Parsing(background=self.background, object=self.object)
        return Environment(**env_config_attrs)
    
class BindingExperiment(Experiment):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def get_checkpoint_path(self):
        ## compute simclr checkpoints
        checkpoint_dict = {
            "object_1": "object_1/checkpoints/epoch=97-step=14601.ckpt",
            "object_2": "object_2/checkpoints/epoch=97-step=14601.ckpt"
        }
        
        parsing_checkpoint = os.path.join(self.base_simclr_checkpoint_path, 'simclr_binding')
        checkpoint_key = f"{self.object.lower()}"
        path = checkpoint_dict.get(checkpoint_key, '')
        return os.path.join(parsing_checkpoint, path)

    def initialize_environment(self):
        """
        Initialize environment class with the attributes extracted from the env_config

        Returns:
            _type_: _description_
        """
        # Extract attributes from brain_config
        env_config_attrs = {attr: getattr(self.env_config, attr) for attr in dir(self.env_config) \
                            if not attr.startswith('__')}
        del env_config_attrs['object']
        env_config_attrs['config'] = Binding(object= self.env_config.object)
        return Environment(**env_config_attrs)

class ViewInvariantExperiment(Experiment):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def get_checkpoint_path(self):
        ## compute simclr checkpoints
        checkpoint_dict = {
            "ship_side":"",
            "ship_front":"",
            "fork_side":"",
            "fork_front":""
        }
        
        viewpt_checkpoint = os.path.join(self.base_simclr_checkpoint_path, 'simclr_viewpt')
        checkpoint_key = f"{self.object}_{self.view.lower()}"
        path = checkpoint_dict.get(checkpoint_key, '')
        return os.path.join(viewpt_checkpoint, path)

    def initialize_environment(self):
        """
        Initialize environment class with the attributes extracted from the env_config

        Returns:
            _type_: _description_
        """
        self.object = "ship" if getattr(self.env_config, 'use_ship', False) else "fork"
        self.view =  "side" if getattr(self.env_config, 'side_view', False) else "front"
        
        # Extract attributes from brain_config
        env_config_attrs = {attr: getattr(self.env_config, attr) for attr in dir(self.env_config) \
                            if not attr.startswith('__')}

        del env_config_attrs['use_ship']
        del env_config_attrs['side_view']
        
        env_config_attrs['config'] = ViewInvariant(object=self.object, view=self.view)
        return Environment(**env_config_attrs)

def main():
    args = parse_args()

    if args.exp_name:
        exp_name = args.exp_name
        config_path = f'configuration/{exp_name}.yaml'
        config = load_configuration(config_path)
        
        if exp_name == 'parsing':
            exp = ParsingExperiment(**config)
            exp.run()
            
        
        elif exp_name == 'binding':
            exp = BindingExperiment(**config)
            exp.run()
        
        elif exp_name == 'viewinvariant':
            exp = ViewInvariantExperiment(**config)
            exp.run()
        
        else:
            raise ValueError("Invalid Experiment Name")


            
            
        


def parse_args():
    parser = argparse.ArgumentParser(description='Run the NETT pipeline - NeurIPS 2021 submission')
    parser.add_argument('-exp_name', '--exp_name', type=str, required=True, default="binding",
                        help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    main()
