"""Initializes the brain module."""
import ast
from pathlib import Path

import stable_baselines3
import sb3_contrib
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# NOTE: Import was causing circular import error (nett.brain -> brain)
from nett.brain import encoders

def list_encoders() -> set[str]:
    """
    Return a set of all available encoders.

    :return: Set of encoder names.
    :rtype: set[str]
    """
    encoders_dir = Path.joinpath(Path(__file__).resolve().parent, 'encoders')
    encoders = [encoder.stem for encoder in list(encoders_dir.iterdir()) if "__" not in str(encoder)]
    # set is faster to access than a list
    return set(encoders)

encoders_list = list_encoders()

def list_algorithms() -> set[str]:
    """
    Return a set of all available policy algorithms.

    :return: Set of algorithm names.
    :rtype: set[str]
    """
    sb3_policy_algorithms = [algorithm for algorithm in dir(stable_baselines3) if algorithm[0].isupper()]
    sb3_contrib_policy_algorithms = [algorithm for algorithm in dir(sb3_contrib) if algorithm[0].isupper()]
    available_policy_algorithms = sb3_policy_algorithms + sb3_contrib_policy_algorithms
    # set is faster to access than a list
    return set(available_policy_algorithms)

algorithms = list_algorithms()

# TODO (v0.3) return all available policy models programmatically
def list_policies() -> set[str]:
    """
    Return a set of all available policy models.

    :return: Set of policy model names.
    :rtype: set[str]
    """
    return {'CnnPolicy', 'MlpPolicy', 'MultiInputPolicy', 'MultiInputLstmPolicy', 'CnnLstmPolicy'}

policies = list_policies()

# return encoder string to encoder class mapping
# TODO (v0.3) optimized way to calculate and pass this dict around
def get_encoder_dict():
    """
    Return a dictionary mapping encoder names to encoder class names.

    :return: Dictionary mapping encoder names to encoder class names.
    :rtype: dict[str, str]
    """
    encoders_dict = {}
    encoders_dir = Path.joinpath(Path(__file__).resolve().parent, 'encoders')
    # iterate through all files in the directory
    for encoder_path in encoders_dir.iterdir():
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
            encoders_dict[module_name] = encoder_class.name
    return encoders_dict

# TODO (v0.3) return all available encoder classes programmatically
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
