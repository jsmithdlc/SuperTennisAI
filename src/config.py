from dataclasses import asdict, dataclass
from typing import Any, Literal, Union

import retro
import torch.nn as nn
import yaml

from src.networks.residual_extractor import ResidualCNN


@dataclass
class ExperimentConfig:
    algo_name: str = "Generic"
    seed: int = 23

    # Common Experiment params
    initial_lr: float = 2.5e-4
    batch_size: int = 1024
    gamma: float = 0.99
    n_epochs: int = 4
    n_steps: int = 512
    n_envs: int = 8
    total_timesteps: int = 100_000_000
    max_episode_steps: int = 50_000
    scenario: Union[None, str] = None

    # PREPROCESSING steps to be applied to environment
    skip_animations: bool = False
    clip_rewards: bool = True  # when reward scale matters, this should be set to False.
    sticky_prob: float = 0.25
    n_skip: int = 4

    # REWARD function modifications
    stall_penalty: float = 1.0
    fault_penalty: float = 0.5
    ball_return_reward: float = 0.2

    # Logging parameters
    log_interval: int = 1
    stats_window_size: int = 16
    save_freq: int = 1e6
    eval_freq: int = 1e6

    def get_policy_params(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
        }

    def to_dict(self) -> dict[str, Any]:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class PPOConfig(ExperimentConfig):
    algo_name: str = "PPO"

    # POLICY parameters to be set during creation
    clip_range: float = 0.1
    ent_coef: float = 0.01
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    features_extractor_class: Literal["NatureCNN", "ResidualCNN"] = "NatureCNN"
    features_extractor_dim: int = 1024
    features_extractor_dropout: float = 0.1

    def get_policy_params(self) -> dict[str, Any]:
        base_params = super().get_policy_params()
        base_params.update(
            {
                "ent_coef": self.ent_coef,
                "gae_lambda": self.gae_lambda,
                "max_grad_norm": self.max_grad_norm,
                "vf_coef": self.vf_coef,
            }
        )
        if self.features_extractor_class != "NatureCNN":
            base_params.update(
                {
                    "policy_kwargs": {
                        "features_extractor_class": self._get_feature_extractor_class(
                            self.features_extractor_class
                        ),
                        "features_extractor_kwargs": {
                            "dropout": self.features_extractor_dropout,
                            "features_dim": self.features_extractor_dim,
                        },
                    }
                }
            )
        return base_params

    def _get_feature_extractor_class(self, name: str) -> type:
        if name == "ResidualCNN":
            return ResidualCNN
        else:
            raise NotImplementedError(
                f"Features extractor for: {name} not yet implemented"
            )


def save_to_yaml(config: ExperimentConfig, filepath: str):
    with open(filepath, "w") as outfile:
        data = yaml.safe_dump(vars(config))
        outfile.write(data)


def load_from_yaml(filepath: str) -> ExperimentConfig:
    with open(filepath) as infile:
        data = yaml.safe_load(infile)
        if data["algo_name"] == "PPO":
            return PPOConfig(**data)
