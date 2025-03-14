import os

import numpy as np
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


def make_retro(*, game, states, max_episode_steps=4500, seed=None, **kwargs):
    env = retro.make(
        game, retro.State.DEFAULT, inttype=retro.data.Integrations.ALL, **kwargs
    )
    env = InitialStateSetterWrapper(env, states=states, seed=seed)
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


def split_initial_states(initial_states: list[str], n_envs: int) -> list[list[str]]:
    """Splits initial states among the different environments, assigning non-overlapping
    states whenever possible (i.e. n_envs < initial_states)

    Args:
        initial_states (list[str]): initial states to distribute among environments
        n_envs (int): number of environments

    Returns:
        list[list[str]]: list of states for each environment
    """
    if len(initial_states) >= n_envs:
        state_splits = np.array_split(initial_states, n_envs)
        splits = []
        for split in state_splits:
            splits.append([str(s) for s in split])
        state_splits = splits
    else:
        state_splits = [[s] for s in initial_states * (n_envs // len(initial_states))]
        for _ in range(n_envs % len(initial_states)):
            state_splits.append(initial_states)
    return state_splits
