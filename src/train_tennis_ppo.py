"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import retro

# add custom game integration folder path to retro
retro.data.Integrations.add_custom_path(os.path.abspath("./games"))

import pprint
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from src.env_helpers import make_retro, wrap_deepmind_retro

def create_policy_params():
    params = {
        "initial_lr":2e-4,
        "clip_range":0.1,
        "ent_coef":1.4e-5,
        "batch_size":256,
        "gamma":0.999,
        "gae_lambda":0.95,
        "max_grad_norm":2,
        "vf_coef":0.6,
        "n_epochs":10,
        "n_steps":1024,
        "policy_kwargs":{
        "net_arch":{
            "pi": [
                64
            ],
            "vf": [
                64
            ]
            }
        }
    }
    return params


def main():

    render_mode = None
    game = "SuperTennis-Snes"
    state = "SuperTennis.Singles.MattvsBarb.1-set.Hard"
    scenario = None
    log_tensorboard = False
    n_envs = 8

    def make_env():
        env = make_retro(
            game=game, state=state, scenario=scenario, render_mode=render_mode
        )
        env = wrap_deepmind_retro(env)
        return env

    # create training and evaluation environments
    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([make_env] * n_envs), n_stack=4)
    )
    eval_venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]), n_stack=4))

    logname = f"ppo_super_tennis_{datetime.now().strftime('%d_%m_%Y__%H_%M_%S')}"

    # evaluation callback
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=os.path.join("./logs", "checkpoints", logname),
        log_path=os.path.join("./logs", "eval_metrics", logname),
        render=False,
        deterministic=True,
        eval_freq=1_000_000 // n_envs,
        n_eval_episodes=2,
    )

    params = create_policy_params()
    print("Hyperparameters:")
    pprint.pprint(params)
    print("\n")
    initial_lr = params.pop("initial_lr")
    model = PPO(
        policy="CnnPolicy",
        tensorboard_log="./logs/tensorboard/",
        env=venv,
        learning_rate=lambda f: f * initial_lr,
        verbose=1,
        **params
    )
    model.learn(
        total_timesteps=50_000_000,
        callback=eval_cb,
        tb_log_name=logname,
        log_interval=1,
    )
    model.save(os.path.join("./logs", "checkpoints", logname, "last_model"))


if __name__ == "__main__":
    main()
