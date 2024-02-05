import ast
import inspect
import stable_baselines3
import sb3_contrib

from nett.brain import encoders
from pathlib import Path
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# return all available enconders
def list_encoders() -> set[str]:
    encoder_dir = Path.joinpath(Path(__file__).resolve().parent, 'encoders')
    encoders = [encoder.stem for encoder in list(encoder_dir.iterdir()) if "__" not in str(encoder)]
    # set is faster to access than a list
    return set(encoders)
encoders_list = list_encoders()

# return all available policy algorithms
def list_algorithms() -> set[str]:
    sb3_policy_algorithms = [algorithm for algorithm in dir(stable_baselines3) if algorithm[0].isupper()]
    sb3_contrib_policy_algorithms = [algorithm for algorithm in dir(sb3_contrib) if algorithm[0].isupper()]
    available_policy_algorithms = sb3_policy_algorithms + sb3_contrib_policy_algorithms
    # set is faster to access than a list
    return set(available_policy_algorithms)
algorithms = list_algorithms()

# TO DO (v0.3) return all available policy models programmatically
def list_policies() -> set[str]:
    return ['CnnPolicy', 'MlpPolicy', 'MultiInputPolicy', 'MultiInputLstmPolicy', 'CnnLstmPolicy']
policies = list_policies()

# return encoder string to encoder class mapping
# TO DO (v0.3) optimized way to calculate and pass this dict around
def get_encoder_dict():
    encoder_dict = {}
    encoder_dir = Path.joinpath(Path(__file__).resolve().parent, 'encoders')
    # iterate through all files in the directory
    for encoder_path in encoder_dir.iterdir():
        if encoder_path.suffix == '.py' and "__" not in str(encoder_path):
            module_name = encoder_path.stem 
            # read the source
            with open(encoder_path) as f:
                source = f.read()
            # parse it
            module = ast.parse(source)
            # get the first class definition
            encoder_class = [node for node in ast.walk(module) if isinstance(node, ast.ClassDef)][0]
            # add to the dictionary
            encoder_dict[module_name] = encoder_class.name
    return encoder_dict

# TO DO (v0.3) return all available encoder classes programmatically
# def get_encoder_dict():
#     return {'cnnlstm': 'CNNLSTM', 
#             'cotracker': 'CoTracker', 
#             'dinov1': 'DinoV1',
#             'dinov2': 'DinoV2',
#             'vit': 'ViT',
#             'resnet18': 'Resnet18CNN', 
#             'sam': 'SegmentAnything', 
#             'rnd': 'RND'}
encoder_dict = get_encoder_dict()