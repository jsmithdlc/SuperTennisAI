import os

import retro
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.monitor import Monitor

from src.config import ExperimentConfig
from src.wrappers import (
    FaultPenaltyWrapper,
    FrameSkip,
    InitialStateSetterWrapper,
    ReturnCompensationWrapper,
    SkipAnimationsWrapper,
    StallPenaltyWrapper,
    StickyActionWrapper,
)

N_SKIPPED_FRAMES = 3
STICK_PROB = 0.0


def read_statenames_from_folder(folder):
    statenames = [
        os.path.join(os.path.basename(folder), file)
        for file in os.listdir(folder)
        if file.split(".")[-1] == "state"
    ]
    return statenames


def make_retro(*, game, state, max_episode_steps=4500, **kwargs):
    env = retro.make(game, state, inttype=retro.data.Integrations.ALL, **kwargs)
    env = InitialStateSetterWrapper(env)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def wrap_deepmind_retro(env, config: ExperimentConfig):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    if config.skip_animations:
        env = SkipAnimationsWrapper(env)
    env = StickyActionWrapper(env, config.sticky_prob)
    env = FrameSkip(env, n_skip=config.n_skip)
    env = WarpFrame(env)
    env = StallPenaltyWrapper(
        env,
        base_steps=(
            60 if config.skip_animations else 200
        ),  # must account for animations if not skipped
        penalty=config.stall_penalty,
        skipped_frames=config.n_skip,
    )
    env = FaultPenaltyWrapper(env, penalty=config.fault_penalty)
    env = ReturnCompensationWrapper(env, compensation=config.ball_return_reward)
    if config.clip_rewards:
        env = ClipRewardEnv(env)
    return env
