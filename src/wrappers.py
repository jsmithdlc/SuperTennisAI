import pprint
import random
import time

import gymnasium as gym
import numpy as np
import retro
from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn


class StickyActionWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability
        self._sticky_action = None

    def reset(self, **kwargs) -> AtariResetReturn:
        self._sticky_action = None  # NOOP
        return self.env.reset(**kwargs)

    def step(self, action: int) -> AtariStepReturn:
        if self._sticky_action is None:
            self._sticky_action = action
        elif self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = action
        return self.env.step(self._sticky_action)


class FrameSkip(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env: gym.Env, n_skip: int = 4) -> None:
        super().__init__(env)
        self.n_skip = n_skip

    def step(self, action: int) -> AtariStepReturn:
        """Performs same action over n_skip frames

        Args:
            action (int): action to repeat over n_skip frames

        Returns:
            AtariStepReturn: environment state and reward after frame skipping
        """
        total_reward = 0.0
        for _ in range(self.n_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            if done:
                break
        return obs, total_reward, terminated, truncated, info


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


class FaultPenaltyWrapper(gym.Wrapper):
    """
    Wrapper uses variable 'in_fault' from game data, which indicates if player
    is at fault while serving, to penalize agent for doing faults.
    """

    def __init__(self, env):
        super(FaultPenaltyWrapper, self).__init__(env)
        self.in_fault_varname = "in_fault"
        self.prev_in_fault = False

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            self.reward(reward, info[self.in_fault_varname]),
            terminated,
            truncated,
            info,
        )

    def reward(self, reward, in_fault):
        """reward modification function

        Args:
            reward (float): reward as given by original environment
            is_serving (bool): variable that indicates if player is at fault when == 1

        Returns:
            float: reward with penalization if player has entered fault state
        """
        if in_fault == 1 and not self.prev_in_fault:
            print("Penalizing agent for comitting fault")
            self.prev_in_fault = bool(in_fault)
            return reward - 1
        self.prev_in_fault = bool(in_fault)
        return reward


class ReturnCompensationWrapper(gym.Wrapper):
    """
    Wrapper keeps track of the number of returned balls (successful or not) achieved
    by the agent.

    Note: Assumes 1-set environment where player always serves the first game
    """

    def __init__(self, env, compensation=0.2):
        super(ReturnCompensationWrapper, self).__init__(env)
        self.total_pt_returns_varname = "total_point_returns"
        self.total_games_varname = "total_games"
        self.cur_tot_returns = 0
        self.compensation = compensation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            self.reward(
                reward,
                info[self.total_pt_returns_varname],
                info[self.total_games_varname],
            ),
            terminated,
            truncated,
            info,
        )

    def _is_player_return(self, total_games, total_pt_returns):
        # game where player serves and even total_pt_returns are player returns
        if total_games % 2 == 0 and total_pt_returns % 2 == 0:
            return True
        # game where player is not serving and odd total_pt_returns are player returns
        elif total_games % 2 != 0 and total_pt_returns % 2 != 0:
            return True
        return False

    def reward(self, reward, total_pt_returns, total_games):
        delta_returns = total_pt_returns - self.cur_tot_returns
        self.cur_tot_returns = total_pt_returns
        if delta_returns > 0 and self._is_player_return(total_games, total_pt_returns):
            print("Compensating player returns")
            return reward + self.compensation
        return reward


class RandomInitialStateWrapper(gym.Wrapper):
    """
    Selects initial state for the retro environment from a list of possible states.
    Effective upon environment creation and during reset

    Attributes:
        env (gym.RetroEnv): retro environment
        statenames (list[str]): list of states to sample initial state from. Sould be
        the paths to the .state files relative to the game directory
    """

    def __init__(self, env: retro.RetroEnv, statenames: list[str]):
        assert (
            len(statenames) > 0
        ), "Must select a non-empty list of possible initial states"
        super(RandomInitialStateWrapper, self).__init__(env)
        self.possible_statenames = statenames
        self._set_random_initial_state()

    def _set_random_initial_state(self):
        init_state = random.choice(self.possible_statenames)
        self.unwrapped.load_state(init_state, retro.data.Integrations.ALL)

    def reset(self, seed=None, options=None):
        self._set_random_initial_state()
        return super().reset(seed=seed, options=options)
