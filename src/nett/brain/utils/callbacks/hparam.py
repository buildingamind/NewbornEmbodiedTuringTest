from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


# TODO (v0.4): refactor needed, especially logging
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        lr = self.model.learning_rate
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": lr if isinstance(lr, (int, float)) else str(lr),
            "gamma": getattr(self.model, "gamma", 0.99),
            "batch_size": getattr(self.model, "batch_size", None),
            "n_steps": getattr(self.model, "n_steps", None),
        }
        # Remove None entries (e.g. off-policy algos don't have n_steps)
        hparam_dict = {k: v for k, v in hparam_dict.items() if v is not None}
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
