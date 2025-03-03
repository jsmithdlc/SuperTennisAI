import pprint
import time

import gymnasium as gym
import numpy as np


class StochasticFrameSkip(gym.Wrapper):
    """
    Skips n frames and applies a sticky action policy, repeating the action with a given probability
    Sort of like StickyAction Wrapper combined with frameskip

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


# TODO: implement this directly into the reward function of the environment as a lua script
class StallPenaltyWrapper(gym.Wrapper):
    """
    Wrapper uses variable 'player_serving' from game data. This variable takes
    many values, depending on the racket action taken by the player.

    It was observed that it is always 1 when player has ball in hand and 17 when
    bouncing it

    IMPORTANT: note that the time to wait until penalty takes into account
    a time that is spent showing the current score and where the variable is
    also 1.

    It was observed that 50 env steps ~ 3 seconds of normal game speed
    """

    stalling_values = {1, 17}

    def __init__(self, env, penalty=1, steps_till_penalty=80, skipped_frames=4):
        super(StallPenaltyWrapper, self).__init__(env)
        self.is_serving_varname = "player_serving"
        self.penalty = penalty
        # must divide by skipped frames to get back to real time
        self.steps_till_penalty = (steps_till_penalty) // skipped_frames
        # 200 base steps must be burned to account for score showing
        self.base_steps = -200 // skipped_frames
        self.in_serving_state = False
        self.step_counter = self.base_steps

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            self.reward(reward, info[self.is_serving_varname]),
            terminated,
            truncated,
            info,
        )

    def reward(self, reward: float, is_serving) -> float:
        """reward modification function

        Args:
            reward (float): reward as given by original environment
            is_serving (bool): variable that indicates if player is serving when == 1

        Returns:
            float: reward with penalization if player has spent more time than
                necessary serving.
        """
        if is_serving in self.stalling_values and not self.in_serving_state:
            self.step_counter = self.base_steps
            self.in_serving_state = True
        elif is_serving in self.stalling_values and self.in_serving_state:
            self.step_counter += 1
            if self.step_counter >= self.steps_till_penalty:
                self.step_counter = 0  # we reset to 0 since game is already in motion
                print("Penalizing agent for stalling")
                return reward - self.penalty
        else:
            self.in_serving_state = False
        return reward
