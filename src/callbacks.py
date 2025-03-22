import os
import pprint
from collections import deque

import gymnasium as gym
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.utils import safe_mean

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


class LogExtraEpisodeStatsCallback(BaseCallback):
    """Used for logging additional episode stats, from the `info` dictionary
    returned at each step, onto the training logger.

    Attributes:
        extra_metric_names (list[str]): names of the metrics, as found in the
        `info` dictionaries, to log
        log_freq (int): frequency, in steps, between logs
        verbose (int): verbosity level
        stats_window_size (int): maximum number of values stored for each metric at
        each collection (FIFO)
    """

    def __init__(
        self,
        extra_metric_names: list[str],
        log_freq: int,
        verbose: int = 0,
        stats_window_size: int = 100,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._extra_buffers = {
            name: deque(maxlen=stats_window_size) for name in extra_metric_names
        }

    def _update_extra_buffers(self):
        """Adds values to buffers if found in info dictionary"""
        for env_id, info in enumerate(self.locals["infos"]):
            if not self.locals["dones"][env_id]:
                continue
            for metric_name in self._extra_buffers:
                val = info.get(metric_name)
                if val is None:
                    raise ValueError(
                        f"Missing metric: {metric_name} from info at episode termination"
                    )
                self._extra_buffers[metric_name].append(val)

    def _dump_extra_metrics(self):
        """Records the mean of each metric, where available, into the logger"""
        for name in self._extra_buffers:
            if len(self._extra_buffers[name]) > 0:
                self.logger.record(
                    f"rollout/{name}", safe_mean(self._extra_buffers[name])
                )

    def _on_step(self) -> bool:
        self._update_extra_buffers()
        if self.num_timesteps % self.log_freq == 0:
            self._dump_extra_metrics()
        return True


def initialize_callbacks(
    eval_env: gym.Env, config: ExperimentConfig, logname: str
) -> list[BaseCallback]:
    """Initializes collection of experiment callbacks

    Args:
        eval_env (gym.Env): evaluation environment for `EvalCallback`
        config (ExperimentConfig): experiment configuration
        logname (str): name of the logging folder

    Returns:
        list[BaseCallback]: experiment callbacks
    """
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join("./logs", logname, "checkpoints"),
        log_path=os.path.join("./logs", logname, "eval_metrics"),
        render=False,
        deterministic=True,
        eval_freq=config.eval_freq // config.n_envs,
        n_eval_episodes=4,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=config.save_freq // config.n_envs,
        save_path=os.path.join("./logs", logname, "checkpoints"),
        name_prefix="ppo_supertennis",
    )
    extra_metric_logger = LogExtraEpisodeStatsCallback(
        [
            "faults",
            "stall_count",
            "ball_returns",
            "player_points",
            "opponent_points",
            "aces",
        ],
        log_freq=config.log_interval * config.n_steps * config.n_envs,
        stats_window_size=config.stats_window_size,
    )
    return [eval_cb, ckpt_callback, HParamCallback(config), extra_metric_logger]
