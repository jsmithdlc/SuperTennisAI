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

from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)


class StochasticFrameSkip(gym.Wrapper):
    """
    Stores most previous action and use it over environment only updating it in each step
    based on outcome of stickprob trial. Sort of like StickyAction Wrapper combined with frameskip

    Read: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    """

    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


class TimePenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty_per_step=3e-3):
        super(TimePenaltyWrapper, self).__init__(env)
        self.penalty_per_step = penalty_per_step

    def reward(self, reward):
        return reward - self.penalty_per_step


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, inttype=retro.data.Integrations.ALL, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = TimePenaltyWrapper(env)
    env = ClipRewardEnv(env)
    return env


def main():

    render_mode = None
    game = "SuperTennis-Snes"
    state = "SuperTennis.Singles.MattvsBarb.1-set.Hard"
    scenario = None
    initial_lr = 2.5e-4
    clip_range = 0.1
    ent_coef = 0.01
    batch_size = 128
    gamma = 0.995
    gae_lambda = 0.95
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

    model = PPO(
        policy="CnnPolicy",
        tensorboard_log="./logs/tensorboard/",
        env=venv,
        learning_rate=lambda f: f * initial_lr,
        n_steps=128,
        batch_size=batch_size,
        n_epochs=4,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
    )
    print("Hyperparameters set to:")
    pprint.pprint(
        {
            "initial_lr": initial_lr,
            "batch_size": batch_size,
            "gamma": gamma,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "gae_lambda": gae_lambda,
        }
    )
    print("\n")
    model.learn(
        total_timesteps=10_000_000,
        callback=eval_cb,
        tb_log_name=logname,
        log_interval=10,
    )
    model.save(os.path.join("./logs", "checkpoints", logname, "last_model"))


if __name__ == "__main__":
    main()
