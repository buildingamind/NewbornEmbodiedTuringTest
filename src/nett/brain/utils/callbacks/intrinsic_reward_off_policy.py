import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm


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
        self.buffer = self.model.replay_buffer  #
        # Set the logger in the intrinsic reward module for tensorboard logging
        if hasattr(self.irs, "set_logger"):
            self.irs.set_logger(self.logger)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        device = self.irs.device  #
        obs = th.as_tensor(self.locals["self"]._last_obs, device=device)  #
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_obs = th.as_tensor(self.locals["new_obs"], device=device)  # ~

        # ===================== watch the interaction ===================== #
        self.irs.watch(obs, actions, rewards, dones, dones, next_obs)  # ~
        # ===================== watch the interaction ===================== #
        ####################################
        # ===================== compute the intrinsic rewards ===================== #
        intrinsic_rewards = self.irs.compute(
            samples={
                "observations": obs.unsqueeze(0),
                "actions": actions.unsqueeze(0),
                "rewards": rewards.unsqueeze(0),
                "terminateds": dones.unsqueeze(0),
                "truncateds": dones.unsqueeze(0),
                "next_observations": next_obs.unsqueeze(0),
            },
            sync=False,
        )
        # ===================== compute the intrinsic rewards ===================== #

        try:
            # add the intrinsic rewards to the original rewards
            self.locals["rewards"] += intrinsic_rewards.cpu().numpy().squeeze()
            # update the intrinsic reward module
            replay_data = self.buffer.sample(batch_size=self.irs.batch_size)
            self.irs.update(
                samples={
                    "observations": th.as_tensor(replay_data.observations)
                    .unsqueeze(1)
                    .to(device),  # (n_steps, n_envs, *obs_shape)
                    "actions": th.as_tensor(replay_data.actions)
                    .unsqueeze(1)
                    .to(device),
                    "rewards": th.as_tensor(replay_data.rewards).to(device),
                    "terminateds": th.as_tensor(replay_data.dones).to(device),
                    "truncateds": th.as_tensor(replay_data.dones).to(device),
                    "next_observations": th.as_tensor(replay_data.next_observations)
                    .unsqueeze(1)
                    .to(device),
                }
            )
        except (RuntimeError, ValueError) as e:
            import logging

            logging.getLogger("nett.Brain").warning(
                f"Failed to update intrinsic rewards: {e}"
            )
        ####################################
        return True

    def _on_rollout_end(self) -> None:
        pass
