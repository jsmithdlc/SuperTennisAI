import gymnasium as gym
import numpy as np


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