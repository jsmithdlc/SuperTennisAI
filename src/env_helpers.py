import retro
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

from src.wrappers import FaultPenaltyWrapper, StallPenaltyWrapper

N_SKIPPED_FRAMES = 4
STICK_PROB = 0.25


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
    env = StallPenaltyWrapper(env, skipped_frames=N_SKIPPED_FRAMES)
    env = FaultPenaltyWrapper(env)
    env = AtariWrapper(
        env,
        frame_skip=N_SKIPPED_FRAMES,
        clip_reward=True,
        action_repeat_probability=STICK_PROB,
        terminal_on_life_loss=False,
    )
    return env
