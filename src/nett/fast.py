import os
import subprocess
from matplotlib import pyplot as plt
import sb3_contrib
import stable_baselines3
import torch
import yaml
import traceback

from pathlib import Path
from typing import Any, Type

import numpy as np
from gym import Wrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib import RecurrentPPO

from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from nett.brain.rewards.disagreement import Disagreement
from nett.brain.rewards.icm import ICM
from nett.brain.rewards.rnd import RND
from nett.utils.callbacks import IntrinsicRewardWithOnPolicyRL
from nett.utils.environment import Logger

def fast(config):
	with open(config, "r") as file:
		config_text = yaml.safe_load(file)

	# Environment Setup
	executable_path = config_text["Environment"]["executable_path"]
	parent_dir = os.path.dirname(executable_path)
	yaml_files: str = [file for file in os.listdir(parent_dir) if file.endswith(".yaml")]
	yaml_file: str = os.path.join(parent_dir, yaml_files[0])

	# read the yaml file
	with open(yaml_file, "r") as file:
		yaml_data = yaml.safe_load(file)

	num_test_conditions: int = yaml_data["num_test_conditions"]
	imprinting_conditions: list[str] = yaml_data["imprinting_conditions"]
	del yaml_data

	subprocess.run(["chmod", "-R", "755", executable_path], check=True)

	os.environ["DISPLAY"] = str(f":0")

	output_dir = Path(config_text["Run"]["output_dir"])
	output_dir.mkdir(parents=True, exist_ok=True)

	port=config_text["Run"]["base_port"]
	condition = config_text["Run"]["conditions"][0]
	brain_id = 1
	base_path = Path(config_text["Run"]["output_dir"]) / condition /f"brain_{brain_id}"
	reward = config_text["Brain"]["reward"]
	steps_per_episode = config_text["Run"]["steps_per_episode"]

	algorithm = getattr(stable_baselines3, config_text["Brain"]["algorithm"], None) or getattr(sb3_contrib, config_text["Brain"]["algorithm"])

	device = config_text["Run"]["devices"][0]

	# calculate iterations
	iterations: dict[str, int] = {
		"train": config_text["Run"]["train_eps"] * steps_per_episode,
		"test": config_text["Run"]["test_eps"] * num_test_conditions
	}

	if not issubclass(algorithm, RecurrentPPO):
		iterations["test"] *= steps_per_episode

	if config_text["Run"]["mode"] == "full":
			modes = ["train", "test"]
	else: # test or train
			modes = [config_text["Run"]["mode"]]
	
	for mode in modes:
		# actual run
		# run environment
		# can be train or test mode and can be for validation or actual run
		while True:
			try:
				env = Environment(mode, port, executable_path, reward, base_path, condition, steps_per_episode)
				# wrap the body with the environment

				# initialize environment
				envs = make_vec_env(
					env_id=lambda: env, 
					n_envs=1
				)

				if mode == "train":
					train(envs, algorithm, base_path, iterations[mode], device, reward, config_text)
				else:
					test(envs, algorithm, base_path, iterations[mode], device)

				env.close()
				break
			except UnityWorkerInUseException as _:
				print(f"Worker {port} is in use. Trying next port...")
				port += 1
			except Exception as ex:
				print(f"{mode} env failed: {str(ex)}")  
				print(traceback.format_exc()) 
				raise ex

def train(envs: VecEnv, algorithm: Type[BaseAlgorithm], base_path: Path, iterations: dict[str, int], device: int, reward: str, config_text: dict[str, Any]):
	# get callbacks
	callback_list = get_callbacks(envs, reward, device)

	# create model
	model = algorithm(
		config_text["Brain"]["policy"],
		envs,
		batch_size=config_text["Brain"]["batch_size"],
		n_steps=config_text["Brain"]["buffer_size"],
		learning_rate=config_text["Brain"]["learning_rate"],
		ent_coef=config_text["Brain"]["ent_coef"],
		verbose=0,
		device=torch.device("cuda", device)
	)

	# train
	model.learn(
		total_timesteps=iterations,
		tb_log_name=algorithm.__name__,
		progress_bar=False,
		callback=callback_list
	)

	# save model
	save(model, base_path)

	# plot reward graph
	plot_reward_graph(base_path, iterations)

def test(envs: VecEnv, algorithm: Type[BaseAlgorithm], base_path: Path, iterations: dict[str, int], device: int):
	# load previously trained model from save_dir, if it exists
	model: BaseAlgorithm = algorithm.load(
		base_path / 'model' / 'latest_model.zip', 
		device=torch.device('cuda', device)
	)

	# for when algorithm is RecurrentPPO
	if issubclass(algorithm, RecurrentPPO):
		for _ in range(iterations):
			# cell and hidden state of the LSTM 
			dones, states = [False], None
			# episode start signals are used to reset the lstm states
			episode_starts = np.ones((1,), dtype=bool)

			obs = envs.reset()
			while not dones[0]:
				action, states = model.predict(
					obs,
					state=states,
					episode_start=episode_starts,
					deterministic=True)

				obs, _, dones, _ = envs.step(action)
				episode_starts = dones

	# for all other algorithms
	else:
		obs = envs.reset()
		for _ in range(iterations):
			action, _ = model.predict(obs, deterministic=True) # action, states
			obs, _, dones, _ = envs.step(action)
			if dones[0]:
				obs = envs.reset()

def get_callbacks(envs: VecEnv, reward: str, device: int):
	# create reward function
	if reward not in ["supervised", "unsupervised"]:
		match reward.lower():
			case "icm":
				reward_func = ICM(envs, device)
			case "rnd":
				reward_func = RND(envs, device)
			case "disagreement":
				reward_func = Disagreement(envs, device)
			case _:
				raise ValueError(f"Reward type {reward} not recognized")

		return [IntrinsicRewardWithOnPolicyRL(reward_func)]
	else:
		return []

def save(model, base_path: Path):
	# save
	## create save directory
	model_path = base_path / "model"
	model_path.mkdir(parents=True, exist_ok=True)

	## save model
	model.policy.save(str(model_path / "policy.pkl"))

	## save encoder
	torch.save(
		model.policy.features_extractor.state_dict(),
		model_path / "feature_extractor.pth"
	)

	## save model
	model.save(str(model_path / "latest_model.zip"))

def plot_reward_graph(base_path: Path, iterations: dict[str, int]):
	# plot reward graph
	results_plotter.plot_results([str(base_path / "env_logs")],
		iterations["train"],
		results_plotter.X_TIMESTEPS,
		"reward_graph"
	)

	(base_path / "plots").mkdir(parents=True, exist_ok=True)
	plt.savefig(base_path / "plots" / "reward_graph.png")
	plt.clf()
	
class Environment(Wrapper):
	def __init__(self, mode: str, port: int, executable_path: Path, reward: str, base_path: Path, condition: str, steps_per_episode: int):

		# create environment and connect it to logger
		self.env = UnityToGymWrapper(
			UnityEnvironment(
				executable_path, 
				additional_args=[
					"-batchmode",
					"--episode-steps", str(steps_per_episode),
					"--mode", f"{mode}-{condition}",
					"--log-dir", str(base_path / "env_recs") + "/",
					"--rewarded", str(reward == "supervised").lower(),
					"--random-pos", str(mode == "train").lower()
				], 
				side_channels=[Logger(f"{condition.replace('-', '_')}{1}-{mode}", log_dir=f"{base_path / "env_logs"}/")],
				base_port=port
			), 
			uint8_visual=True
		)

		# initialize the parent class (gym.Wrapper)
		super().__init__(self.env)

	def render(self, mode="rgb_array") -> np.ndarray:
			return np.moveaxis(self.env.render(), [0, 1, 2], [2, 0, 1])

