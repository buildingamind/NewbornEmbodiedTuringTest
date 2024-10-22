"""
Callbacks for training the agents.

Classes:
    HParamCallback(BaseCallback)
"""
from pathlib import Path
import sys
import torch as th

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import HParam

# from nett.utils.train import compute_train_performance

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

def initialize_callbacks(job: "Job") -> CallbackList:
    """
    Initialize the callbacks for training.

    Args:
        job (Job): The job for which to initialize the callbacks.
    
    Returns:
        CallbackList: The list of callbacks for training.
    """
    hparam_callback = HParamCallback() # TODO: Are we using the tensorboard that this creates? See https://www.tensorflow.org/tensorboard Appears to be responsible for logs/events.out.. files

    callback_list = [hparam_callback]

    if job.estimate_memory:
        callback_list.extend([
            # creates the parallel progress bars
            multiBarCallback(job.index, "Estimating Memory Usage"), # TODO: Add progress bars to test aswell
            # creates the memory callback for estimation of memory for a single job
            MemoryCallback(job.device, save_path=job.paths["base"])
            ])
    else:
        # creates the parallel progress bars
        callback_list.append(multiBarCallback(job.index, f"{job.condition}-{job.brain_id}", job.iterations["train"])) # TODO: Add progress bars to test aswell

    if job.save_checkpoints:
        callback_list.append(CheckpointCallback(
            save_freq=job.checkpoint_freq, # defaults to 30_000 steps
            save_path=job.paths["checkpoints"],
            save_replay_buffer=True,
            save_vecnormalize=True))
    
    if job.reward not in ["supervised", "unsupervised"]:
        callback_list.append(IntrinsicRewardWithOnPolicyRL(job.reward_func))
        # callback_list.append(IntrinsicRewardWithOffPolicyRL(job.reward_func))

    return CallbackList(callback_list)

# TODO (v0.4): refactor needed, especially logging
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size": self.model.batch_size,
            "n_steps": self.model.n_steps
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class multiBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent using tqdm
    """

    def __init__(self, index: int, label: str, num_steps: int = None) -> None:
        super().__init__()
        # where on the screen the progress bar will be displayed
        self.index = index
        # label to prefix the progress bar
        self.label = label
        # progress bar object
        self.pbar = None
        # number of steps to be done
        self.num_steps = num_steps

    def _on_training_start(self) -> None:
        # if num_steps is None, this means that memory estimation is being done, so the length of a single rollout will be used
        num_steps = self.num_steps if self.num_steps is not None else self.model.n_steps
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=(num_steps), position=self.index, dynamic_ncols=True, desc=self.label, file=sys.stdout, leave=True, mininterval=1.0)
        pass

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        self.pbar.refresh()
        self.pbar.close()
        pass

class MemoryCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    """
    def __init__(self, device: int, save_path: str) -> None:
        super().__init__()
        self.device = device
        self.save_path = save_path
        self.closeEnv = False
        nvmlInit()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.closeEnv:
            # Close the callback
            return False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.closeEnv = True
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Grab the memory being used by the GPU
        used_memory = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(self.device)).used
        # Write the used memory to a file
        with open(Path.joinpath(self.save_path, "mem.txt"), "w") as f:
            f.write(str(used_memory))
        pass

class IntrinsicRewardWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """
    def __init__(self, irs, verbose=0):
        super().__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        # compute the intrinsic rewards
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions, 
                         rewards=rewards, terminateds=dones, 
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #

class IntrinsicRewardWithOffPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and off-policy algorithms from SB3. 
    """
    def __init__(self, irs, verbose=0):
        super().__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.replay_buffer
        

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        device = self.irs.device
        obs = th.as_tensor(self.locals['self']._last_obs, device=device)
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_obs = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(obs, actions, rewards, dones, dones, next_obs)
        # ===================== watch the interaction ===================== #
        
        # ===================== compute the intrinsic rewards ===================== #
        intrinsic_rewards = self.irs.compute(samples={'observations':obs.unsqueeze(0), 
                                            'actions':actions.unsqueeze(0), 
                                            'rewards':rewards.unsqueeze(0),
                                            'terminateds':dones.unsqueeze(0),
                                            'truncateds':dones.unsqueeze(0),
                                            'next_observations':next_obs.unsqueeze(0)}, 
                                            sync=False)
        # ===================== compute the intrinsic rewards ===================== #

        try:
            # add the intrinsic rewards to the original rewards
            self.locals['rewards'] += intrinsic_rewards.cpu().numpy().squeeze()
            # update the intrinsic reward module
            replay_data = self.buffer.sample(batch_size=self.irs.batch_size)
            self.irs.update(samples={'observations': th.as_tensor(replay_data.observations).unsqueeze(1).to(device), # (n_steps, n_envs, *obs_shape)
                                     'actions': th.as_tensor(replay_data.actions).unsqueeze(1).to(device),
                                     'rewards': th.as_tensor(replay_data.rewards).to(device),
                                     'terminateds': th.as_tensor(replay_data.dones).to(device),
                                     'truncateds': th.as_tensor(replay_data.dones).to(device),
                                     'next_observations': th.as_tensor(replay_data.next_observations).unsqueeze(1).to(device)
                                     })
        except:
            pass

        return True

    def _on_rollout_end(self) -> None:
        pass