from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from src.config import ExperimentConfig


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    Adapted from: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-hyperparameters
    """

    def __init__(self, config: ExperimentConfig, verbose: int = 0):
        super().__init__(verbose)
        self.exp_config = config

    def _on_training_start(self) -> None:
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorboard will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0.0,
            "eval/mean_ep_length": 0,
            "eval/mean_reward": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(self.exp_config.to_dict(), metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
