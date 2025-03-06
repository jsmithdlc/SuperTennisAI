from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    Adapted from: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-hyperparameters
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "gamma": self.model.gamma,
            "ent_coef": self.model.ent_coef,
            "batch_size": self.model.batch_size,
            "gae_lambda": self.model.gae_lambda,
            "max_grad_norm": self.model.max_grad_norm,
            "vf_coef": self.model.vf_coef,
            "n_epochs": self.model.n_epochs,
            "n_steps": self.model.n_steps,
        }
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
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
