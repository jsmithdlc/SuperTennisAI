import retro
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    MaxAndSkipEnv,
    WarpFrame,
)
from stable_baselines3.common.monitor import Monitor

from src.wrappers import (
    FaultPenaltyWrapper,
    FrameSkip,
    ReturnCompensationWrapper,
    StallPenaltyWrapper,
    StickyActionWrapper,
)

N_SKIPPED_FRAMES = 3
STICK_PROB = 0.0


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, inttype=retro.data.Integrations.ALL, **kwargs)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = StickyActionWrapper(env, STICK_PROB)
    env = FrameSkip(env, n_skip=N_SKIPPED_FRAMES)
    env = WarpFrame(env)
    env = StallPenaltyWrapper(env, skipped_frames=N_SKIPPED_FRAMES)
    env = FaultPenaltyWrapper(env)
    # env = ReturnCompensationWrapper(env)
    env = ClipRewardEnv(env)
    return env
