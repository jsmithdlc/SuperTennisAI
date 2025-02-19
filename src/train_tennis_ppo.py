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

from datetime import datetime
import pprint
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
    env = ClipRewardEnv(env)
    return env


def main():

    render_mode = None
    game = "SuperTennis-Snes"
    state = retro.State.DEFAULT
    scenario = None
    initial_lr = 1e-4
    clip_range = 0.1
    ent_coef = 0.005
    batch_size = 128
    gamma = 0.995



    def make_env():
        env = make_retro(
            game=game, state=state, scenario=scenario, render_mode=render_mode
        )
        env = wrap_deepmind_retro(env)
        return env

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4))

    tb_logname = f"ppo_super_tennis_{datetime.now().strftime('%H_%M_%S__%d_%m_%Y')}"
    model = PPO(
        policy="CnnPolicy",
        tensorboard_log="./logs/tensorboard/",
        env=venv,
        learning_rate=lambda f: f * initial_lr,
        n_steps=128,
        batch_size=batch_size,
        n_epochs=4,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
    )
    print("Hyperparameters set to:")
    pprint.pprint({
        "initial_lr":initial_lr,
        "batch_size":batch_size,
        "gamma":gamma,
        "clip_range":clip_range,
        "ent_coef":ent_coef
    })
    print("\n")
    model.learn(total_timesteps=100_000_000, tb_log_name=tb_logname, log_interval=10)
    model.save("./logs/super_tennis_ppo/")


if __name__ == "__main__":
    main()
