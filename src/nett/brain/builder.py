"""Module for the Brain class."""

import os
from typing import Any, Optional
from pathlib import Path
import inspect
import torch
import stable_baselines3
import sb3_contrib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common import results_plotter
from nett.brain import algorithms, policies, encoder_dict
from nett.brain import encoders
from nett.utils.callbacks import initialize_callbacks
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# TODO (v0.3): Extend with support for custom policy models
# TODO (v0.3): should we move validation checks to utils under validations.py?

class Brain:
    """Represents the brain of an agent. 

    The brain is made up of an encoder, policy, algorithm, reward function, and the hyperparameters determined for these components such as the batch and buffer sizes. It produces a trained model based on the environment data and the inputs received by the brain through the body.

    Args:
        policy (Any | str): The network used for defining the value and action networks.
        algorithm (str | OnPolicyAlgorithm | OffPolicyAlgorithm): The optimization algorithm used for training the model.
        encoder (Any | str, optional): The network used to extract features from the observations. Defaults to None.
        embedding_dim (int, optional): The dimension of the embedding space of the encoder. Defaults to None.
        reward (str, optional): The type of reward used for training the brain. Defaults to "supervised".
        batch_size (int, optional): The batch size used for training. Defaults to 512.
        buffer_size (int, optional): The buffer size used for training. Defaults to 2048.
        train_encoder (bool, optional): Whether to train the encoder or not. Defaults to True.
        seed (int, optional): The random seed used for training. Defaults to 12.
        custom_encoder_args (dict[str, str], optional): Custom arguments for the encoder. Defaults to {}.

    Example:

        >>> from nett import Brain
        >>> brain = Brain(policy='CnnPolicy', algorithm='PPO')
    """

    def __init__(
        self,
        policy: Any | str,
        algorithm:  str | OnPolicyAlgorithm | OffPolicyAlgorithm,
        encoder: Any | str = None,
        embedding_dim: Optional[int] = None,
        reward: str = "supervised",
        batch_size: int = 512,
        buffer_size: int = 2048,
        learning_rate: float = 0.0003,
        train_encoder: bool = True,
        seed: int = 12,
        custom_encoder_args: dict[str, str]= {},
        custom_policy_arch: Optional[list[int|dict[str,list[int]]]] = None
    ) -> None:
        """Constructor method
        """
        # Initialize logger
        from nett import logger

        self.logger = logger.getChild(__class__.__name__)

        # Set attributes
        self.algorithm = self._validate_algorithm(algorithm)
        self.policy = self._validate_policy(policy)
        self.train_encoder = train_encoder
        self.encoder = self._validate_encoder(encoder) if encoder else None
        self.reward = self._validate_reward(reward) if reward else None

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.custom_encoder_args = custom_encoder_args
        self.custom_policy_arch = custom_policy_arch

    def train(self, env: "gym.Env", job: "Job"):
        """
        Train the brain.

        Args:
            job(Job): The job object containing the environment, paths, and training parameters.

        Raises:
            ValueError: If the environment fails the validation check.
        """
        # importlib.reload(stable_baselines3)
        # validate environment
        env = self._validate_env(env)

        # initialize environment
        envs = make_vec_env(
            env_id=lambda: env, 
            n_envs=1, 
            # seed=self.seed, # Commented out as seed does not work
            monitor_dir=str(job.paths["env_logs"])) #TODO: Switch to multi-processing for parallel environments with vec_envs #TODO: Add custom seed function for seeding env, see https://stackoverflow.com/questions/47331235/how-should-openai-environments-gyms-use-env-seed0

        # build model
        policy_kwargs = {
            "features_extractor_class": self.encoder,
            "features_extractor_kwargs": {
                "features_dim": self.embedding_dim or inspect.signature(self.encoder).parameters["features_dim"].default,
            }
        } if self.encoder is not None else {}

        if len(self.custom_encoder_args) > 0:
            policy_kwargs["features_extractor_kwargs"].update(self.custom_encoder_args)
        
        if self.custom_policy_arch:
            policy_kwargs["net_arch"] = self.custom_policy_arch
            
        self.logger.info(f'Training with {self.algorithm.__name__}')
        try:
            model = self.algorithm(
                self.policy,
                envs,
                batch_size=self.batch_size,
                n_steps=self.buffer_size,
                learning_rate=self.learning_rate,
                verbose=0, #TODO: Incorporate this into options
                policy_kwargs=policy_kwargs,
                device=torch.device("cuda", job.device))
            
        except Exception as e:
            self.logger.exception(f"Failed to initialize model with error: {str(e)}")
            raise e

        # setup tensorboard logger and attach to model
        tb_logger = configure(str(job.paths["logs"]), ["stdout", "csv", "tensorboard"])
        model.set_logger(tb_logger)
        
        self.logger.info(f"Tensorboard logs saved at {str(job.paths['logs'])}")
        # set encoder as eval only if train_encoder is not True
        if not self.train_encoder:
            model = self._set_encoder_as_eval(model)
            self.logger.warning(f"Encoder training is set to {str(self.train_encoder).upper()}")

        # initialize callbacks
        self.logger.info("Initializing Callbacks")
        callback_list = initialize_callbacks(job)

        # train
        self.logger.info(f"Total number of training steps: {job.iterations['train']}")
        model.learn(
            total_timesteps=job.iterations["train"],
            tb_log_name=self.algorithm.__name__,
            progress_bar=False,
            callback=[callback_list])
        self.logger.info("Training Complete")

        # nothing else is needed for memory estimation
        if job.estimate_memory:
            return

        # save
        ## create save directory
        job.paths["model"].mkdir(parents=True, exist_ok=True)
        self.save_encoder_policy_network(model.policy, job.paths["model"])
        print("Saved feature extractor")
        
        save_path = f"{job.paths['model'].joinpath('latest_model.zip')}"
        model.save(save_path)
        self.logger.info(f"Saved model at {save_path}")

        # plot reward graph
        self.plot_results(iterations=job.iterations["train"],
                        model_log_dir=job.paths["env_logs"],
                        plots_dir=job.paths["plots"],
                        name="reward_graph")   

    def test(self, env: "gym.Env", job: "Job"):
        """
        Test the brain.

        Args:
            env (gym.Env): The environment used for testing.
            job (Job): The job object containing the environment, paths, and training parameters.
        """
        # load previously trained model from save_dir, if it exists
        model: OnPolicyAlgorithm | OffPolicyAlgorithm = self.algorithm.load(
            job.paths['model'].joinpath('latest_model.zip'), 
            device=torch.device('cuda', job.device))

        # validate environment
        env = self._validate_env(env)

        # initialize environment
        num_envs = 1
        envs = make_vec_env(
            env_id=lambda: env, 
            n_envs=num_envs, 
            # seed=self.seed # Commented out as seed does not work
            )

        self.logger.info(f'Testing with {self.algorithm.__name__}')

        ## record - test video
        try:
            # vr = VideoRecorder(env=envs,
            # path="{}/agent_{}.mp4".format(job.paths["env_recs"], \
            #     str(index)), enabled=True)
            
            # for when algorithm is RecurrentPPO
            iterations: int = job.iterations["test"]
            self.logger.info(f"Total iterations: {iterations}")
            t = tqdm(total=iterations, desc=f"Condition {job.index}", position=job.index, leave=True)
            record_states: bool = "state" in job.record
            if record_states:
                # change print option for recording obs
                np.set_printoptions(threshold=np.inf)
                # create folder for recording states
                states_path: Path = Path.joinpath(job.paths['env_recs'], 'states')
                states_path.mkdir(parents=True, exist_ok=True)

            if issubclass(self.algorithm, RecurrentPPO):
                for i in range(iterations):
                    #TODO: Does this ever go back into this outer loop after the initial time? Does it come back here between episodes?
                    # cell and hidden state of the LSTM 
                    dones, states = [False], None
                    # episode start signals are used to reset the lstm states
                    episode_starts = np.ones((num_envs,), dtype=bool)
                    obs, info = envs.reset() #TODO: try to use envs. This will return a list of obs, rather than a single obs #see https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html for details on conversion
                    print(f"info: {info}")

                    while not dones[0]: #TODO: Change to support multiple envs
                        action, states = model.predict(
                            obs,
                            state=states,
                            episode_start=episode_starts,
                            deterministic=True)
                        if (record_states and i < job.recording_eps):
                            with open(Path.joinpath(states_path, 'obs.txt'), 'a') as f:
                                f.write(f"{' '.join(map(str, np.array(obs).flatten()))}\n")
                            with open(Path.joinpath(states_path, 'actions.txt'), 'a') as f:
                                f.write(f"{' '.join(map(str, np.array(action).flatten()))}\n")
                            with open(Path.joinpath(states_path, 'states.txt'), 'a') as f:
                                f.write(f"{' '.join(map(str, np.array(states).flatten()))}\n")
                            
                        obs, _, dones, info = envs.step(action) # obs, rewards, dones, info #TODO: try to use envs. This will return a list for each of obs, rewards, done, info rather than single values. Ex: done = [False, False, False, False, False] and not False
                        print(f"info: {info}")
                        t.update(1)
                        episode_starts = dones
                        # envs.render(mode="rgb_array") #TODO: try to use envs. This will return a list of obs, rewards, done, info rather than single values
                        # vr.capture_frame()    

                # vr.close()
                # vr.enabled = False

            # for all other algorithms
            else:
                obs, info = envs.reset()
                print(f"info: {info}")
                for i in range(iterations):
                    action, _ = model.predict(obs, deterministic=True) # action, states
                    if (record_states and i < job.recording_eps*job.steps_per_episode):
                        with open(Path.joinpath(states_path, 'obs.txt'), 'a') as f:
                            f.write(f"{' '.join(map(str, np.array(obs).flatten()))}\n")
                        with open(Path.joinpath(states_path, 'actions.txt'), 'a') as f:
                            f.write(f"{' '.join(map(str, np.array(action).flatten()))}\n")
                    obs, _, dones, info = envs.step(action) # obs, reward, done, info #TODO: try to use envs. This will return a list of obs, rewards, done, info rather than single values
                    print(f"info: {info}")
                    t.update(1)
                    if dones[0]:
                        obs, info = envs.reset()
                    # envs.render(mode="rgb_array")
                    # vr.capture_frame()
        except Exception as e:
            self.logger.exception(f"Failed to test model with error: {str(e)}")
            raise e
        # finally:
            # vr.close()
            # vr.enabled = False
        
        t.close()
    
    @staticmethod
    def save_encoder_policy_network(policy, path: Path):
        """
        Saves the policy and feature extractor of the agent's model.

        This method saves the policy and feature extractor of the agent's model
        to the specified paths. It first checks if the model is loaded, and if not,
        it prints an error message and returns. Otherwise, it saves the policy as
        a pickle file and the feature extractor as a PyTorch state dictionary.

        Returns:
            None
        """        
        ## save policy
        path.mkdir(parents=True, exist_ok=True)
        policy.save(os.path.join(path, "policy.pkl"))
        
        ## save encoder
        encoder = policy.features_extractor.state_dict()
        save_path = os.path.join(path, "feature_extractor.pth")
        torch.save(encoder, save_path)

        return

    @staticmethod
    def plot_results(
        iterations: int,
        model_log_dir: Path,
        plots_dir: Path,
        name: str
    ) -> None:
        """
        Plot the training results.

        Args:
            iterations (int): The number of training iterations.
            model_log_dir (Path): The directory containing the model logs.
            plots_dir (Path): The directory to save the plots.
            name (str): The name of the plot.
        """
        results_plotter.plot_results([str(model_log_dir)],
            iterations,
            results_plotter.X_TIMESTEPS,
            name)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir.joinpath(f"{name}.png"))
        plt.clf()

    @staticmethod
    def _validate_encoder(encoder: Any | str) -> BaseFeaturesExtractor:
        """
        Validate the encoder.

        Args:
            encoder (Any | str): The encoder to validate.

        Returns:
            BaseFeaturesExtractor: The validated encoder.
        """
        # for when encoder is a string
        if isinstance(encoder, str):
            if encoder not in encoder_dict.keys():
                raise ValueError(f"If a string, should be one of: {encoder_dict.keys()}")
            encoder = getattr(encoders, encoder_dict[encoder])

        # for when encoder is a custom PyTorch encoder
        if isinstance(encoder, BaseFeaturesExtractor):
            # TODO (v0.3) pass dummy torch.tensor on "meta" device to validate embedding dim
            pass

        return encoder

    @staticmethod
    def _validate_algorithm(algorithm: str | OnPolicyAlgorithm | OffPolicyAlgorithm) -> OnPolicyAlgorithm | OffPolicyAlgorithm:
        """
        Validate the optimization algorithm.

        Args:
            algorithm (str | OnPolicyAlgorithm | OffPolicyAlgorithm): The algorithm to validate.

        Returns:
            OnPolicyAlgorithm | OffPolicyAlgorithm: The validated algorithm.

        Raises:
            ValueError: If the algorithm is not valid.
        """
        # for when policy is a string
        if isinstance(algorithm, str):
            if algorithm not in algorithms:
                raise ValueError(f"If a string, should be one of: {algorithms}")
            # check for the passed policy in stable_baselines3 as well as sb3-contrib
            # at this point in the code, it is guaranteed to be in either of the two
            try:
                algorithm = getattr(stable_baselines3, algorithm)
                
            except:
                algorithm = getattr(sb3_contrib, algorithm)

        # for when policy algorithm is custom
        elif isinstance(algorithm, OnPolicyAlgorithm) or isinstance(algorithm, OffPolicyAlgorithm):
            # TODO (v0.4) determine appropriate validation checks to be performed before passing
            pass

        else:
            raise ValueError(f"Policy Algorithm should be either one of {algorithms} or a subclass of [{OnPolicyAlgorithm}, {OffPolicyAlgorithm}]")

        
        return algorithm

    @staticmethod
    def _validate_policy(policy: str | BasePolicy) -> str | BasePolicy:
        """
        Validate the policy model.

        Args:
            policy (str | BasePolicy): The policy model to validate.

        Returns:
            str | BasePolicy: The validated policy model.

        Raises:
            ValueError: If the policy is a string and not one of the supported policies.
            ValueError: If the policy is not a string or a subclass of BasePolicy.
        """
        # for when policy is a string
        if isinstance(policy, str):
            # support tested for PPO and RecurrentPPO only
            if policy not in policies:
                raise ValueError(f"If a string, should be one of: {policies}")

        # for when policy is custom
        elif isinstance(policy, BasePolicy):
            # TODO (v0.4) determine appropriate validation checks to be performed before passing
            pass

        else:
            raise ValueError(f"Policy Model should be either one of {policies} or a subclass of [{BasePolicy}]")

        return policy

    @staticmethod
    def _validate_reward(reward: str) -> str:
        """
        Validate the reward type.

        Args:
            reward (str): The reward type to validate.

        Returns:
            str: The validated reward type.

        Raises:
            ValueError: If the reward is a string and not one of the supported reward types.
        """
        # for when reward is a string
        if not isinstance(reward, str) or reward not in ['supervised', 'unsupervised']:
            raise ValueError("If a string, should be one of: ['supervised', 'unsupervised']")
        return reward

    @staticmethod
    def _validate_env(env: "gym.Env") -> "gym.Env":
        """
        Validate the environment.

        Args:
            env (gym.Env): The environment to validate.

        Returns:
            gym.Env: The validated environment.

        Raises:
            ValueError: If the environment fails the validation check.
        """
        try:
            check_env(env)
        except Exception as ex:
            raise ValueError(f"Failed training env check with {str(ex)}")
        return env

    @staticmethod
    def _set_encoder_as_eval(model: OnPolicyAlgorithm | OffPolicyAlgorithm) -> OnPolicyAlgorithm | OffPolicyAlgorithm:
        """
        Set the encoder as evaluation mode and freeze its parameters.

        Args:
            model (OnPolicyAlgorithm | OffPolicyAlgorithm): The model containing the encoder.

        Returns:
            OnPolicyAlgorithm | OffPolicyAlgorithm: The model with the encoder set as evaluation mode.
        """
        model.policy.features_extractor.eval()

        for param in model.policy.features_extractor.parameters():
            param.requires_grad = False
        return model
    
    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return f"{self.__class__.__name__}({attrs!r})"

    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return f"{self.__class__.__name__}({attrs!r})"