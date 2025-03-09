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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from src.callbacks import HParamCallback
from src.env_helpers import make_retro, wrap_deepmind_retro


def create_policy_params():
    params = {
        "initial_lr": 2.5e-4,
        "clip_range": 0.1,
        "ent_coef": 0.005,
        "batch_size": 1024,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
        "n_epochs": 4,
        "n_steps": 512,
    }
    return params


def create_logname(saved_model_path, continue_training):
    if saved_model_path is not None and continue_training:
        return os.path.basename(os.path.dirname(saved_model_path))
    return f"ppo_super_tennis_{datetime.now().strftime('%d_%m_%Y__%H_%M_%S')}"


def initialize_model(env):
    params = create_policy_params()
    print("Hyperparameters:")
    pprint.pprint(params)
    print("\n")
    initial_lr = params.pop("initial_lr")
    model = PPO(
        policy="CnnPolicy",
        tensorboard_log="./logs/tensorboard/",
        env=env,
        learning_rate=lambda f: f * initial_lr,
        verbose=1,
        **params,
    )
    return model


def load_saved_model(env, model_path):
    print(f"Load saved model from path: {model_path}")
    model = PPO.load(model_path, tensorboard_log="./logs/tensorboard/")
    model.set_env(env)
    return model


def main():
    render_mode = None
    game = "SuperTennis-Snes"
    states = ["SuperTennis.Singles.MattvsBarb.1-set.Hard.state"]

    continue_training = True
    saved_model_path = "logs/checkpoints/ppo_super_tennis_06_03_2025__09_52_28/ppo_supertennis_100000000_steps_last_second_run.zip"

    save_freq = 1e6
    eval_freq = 1e6
    scenario = None
    n_envs = 8
    total_timesteps = 200_000_000
    max_episode_steps = 5e4

    def make_env():
        env = make_retro(
            game=game,
            states=states,
            scenario=scenario,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        env = wrap_deepmind_retro(env)
        return env

    # create training and evaluation environments
    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([make_env] * n_envs), n_stack=4)
    )
    eval_venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]), n_stack=4))

    logname = create_logname(saved_model_path, continue_training)

    # evaluation callback
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=os.path.join("./logs", "checkpoints", logname),
        log_path=os.path.join("./logs", "eval_metrics", logname),
        render=False,
        deterministic=True,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=2,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=os.path.join("./logs", "checkpoints", logname),
        name_prefix="ppo_supertennis",
    )

    if saved_model_path is not None:
        model = load_saved_model(venv, saved_model_path)
    else:
        model = initialize_model(venv)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, ckpt_callback, HParamCallback()],
        tb_log_name=logname,
        log_interval=1,
        reset_num_timesteps=False if continue_training else True,
    )
    model.save(os.path.join("./logs", "checkpoints", logname, "last_model"))


if __name__ == "__main__":
    main()
