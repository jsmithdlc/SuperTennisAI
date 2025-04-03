import gymnasium as gym
import numpy as np
import retro
from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn


class StickyActionWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Repeats previous action with given probability

    Attributes:
        action_repeat_probability (float): probability [0-1] of repeating previous action
    """

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
    """Skips given ammount of frames

    Attributes:
        n_skip (int): skips this number of frames. Defaults to 4
    """

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

    Game Variables:
        player_serving: indicates if player has ball in hand or is bouncing it
    """

    stalling_values = {1, 17}
    episode_stall_varname = "stall_count"

    def __init__(self, env, penalty=1, base_steps=60, skipped_frames=4):
        super(StallPenaltyWrapper, self).__init__(env)
        self.penalty = penalty
        self._n_skip = skipped_frames
        # must divide by skipped frames to get back to real time
        self.frames_till_penalty = 360 // self._n_skip
        self.in_serving_state = False
        # account for some time spent "in-serving" where agent cannot perform action
        # due to text display, etc
        self.base_frames = -base_steps // self._n_skip
        self.frame_counter = self.base_frames
        self.ep_stall_count = 0

    def _evaluate_if_serving(self, info):
        return info["player_serving"] in self.stalling_values

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        new_reward = self.reward(reward, self._evaluate_if_serving(info))
        if terminated or truncated:
            info[self.episode_stall_varname] = self.ep_stall_count
            self.ep_stall_count = 0
        return (
            observation,
            new_reward,
            terminated,
            truncated,
            info,
        )

    def reward(self, reward: float, is_serving) -> float:
        """reward modification function

        Args:
            reward (float): reward as given by original environment
            is_serving (bool): variable that indicates if player is serving

        Returns:
            float: reward with penalization if player has spent more time than
                necessary serving.
        """
        if is_serving and not self.in_serving_state:
            self.frame_counter = self.base_frames
            self.in_serving_state = True
            # frames till penalty is reset to wait for first stall
            self.frames_till_penalty = 360 // self._n_skip
        elif is_serving and self.in_serving_state:
            self.frame_counter += 1
            if self.frame_counter >= self.frames_till_penalty:
                # counter is reset to 0 and steps till penalty
                # reduced to 80 to penalize repeated stalling more
                # quickly
                self.frame_counter = 0
                self.frames_till_penalty = 80 // self._n_skip
                print("Penalizing agent for stalling")
                self.ep_stall_count += 1
                return reward - self.penalty
        else:
            self.in_serving_state = False
        return reward


class FaultPenaltyWrapper(gym.Wrapper):
    """
    Wrapper uses variable 'in_fault' from game data, which indicates if player
    is at fault while serving, to penalize agent for doing faults.

    Game Variables:
        in_fault: is 1 when agent is in fault
        total_games: tracks total number of completed games
    """

    episode_faults_varname = "faults"

    def __init__(self, env, penalty=1):
        super(FaultPenaltyWrapper, self).__init__(env)
        self.prev_in_fault = False
        self.fault_penalty = penalty
        self.ep_faults = 0

    def _evaluate_if_in_fault(self, info):
        # state in fault is activated and player is serving
        return info["in_fault"] == 1 and info["total_games"] % 2 == 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        new_reward = self.reward(reward, self._evaluate_if_in_fault(info))
        if terminated or truncated:
            info[self.episode_faults_varname] = self.ep_faults
            self.ep_faults = 0
        return (
            observation,
            new_reward,
            terminated,
            truncated,
            info,
        )

    def reward(self, reward, in_fault):
        """reward modification function

        Args:
            reward (float): reward as given by original environment
            is_serving (bool): indicates if player is at fault

        Returns:
            float: reward with penalization if player has entered fault state
        """
        if in_fault and not self.prev_in_fault:
            print("Penalizing agent for comitting fault")
            self.ep_faults += 1
            self.prev_in_fault = bool(in_fault)
            return reward - self.fault_penalty
        self.prev_in_fault = bool(in_fault)
        return reward


class ReturnCompensationWrapper(gym.Wrapper):
    """
    Wrapper keeps track of the number of returned balls (successful or not) achieved
    by the agent.

    Note: Assumes 1-set environment where player always serves the first game

    Game Variables:
        total_games: tracks total number of completed games
        total_point_returns: tracks total number of returns in a single point
    """

    episode_ball_returns_varname = "ball_returns"

    def __init__(self, env, compensation=0.2):
        super(ReturnCompensationWrapper, self).__init__(env)
        self.cur_tot_returns = 0
        self.compensation = compensation
        self.ep_ball_returns = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        new_reward = self.reward(
            reward, info["total_point_returns"], self._is_player_return(info)
        )
        if terminated or truncated:
            info[self.episode_ball_returns_varname] = self.ep_ball_returns
            self.ep_ball_returns = 0
        return (
            observation,
            new_reward,
            terminated,
            truncated,
            info,
        )

    def _is_player_return(self, info):
        # game where player serves and even total_pt_returns are player returns
        if info["total_games"] % 2 == 0 and info["total_point_returns"] % 2 == 0:
            return True
        # game where player is not serving and odd total_pt_returns are player returns
        elif info["total_games"] % 2 != 0 and info["total_point_returns"] % 2 != 0:
            return True
        return False

    def reward(self, reward, total_pt_returns, is_player_return):
        delta_returns = total_pt_returns - self.cur_tot_returns
        self.cur_tot_returns = total_pt_returns
        if delta_returns > 0 and is_player_return:
            self.ep_ball_returns += 1
            return reward + self.compensation
        return reward


class SkipAnimationsWrapper(gym.Wrapper):
    """
    Skips environment frames where animation is being displayed and no action is effective.
    Used for accelerating episode length

    Game Variables:
        animation_running: is 1 when animation of change of side or states are being displayed
        text_displayed: is 1 when fault, score, net, etc text is displayed on-screen
    """

    def _evaluate_if_animation(self, info):
        if info["animation_running"] == 1:
            return True
        if info["text_displayed"] == 1:
            return True
        return False

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        in_animation = self._evaluate_if_animation(info)
        while in_animation:
            if terminated or truncated:
                break
            obs, rew, terminated, truncated, info = self.env.step(
                action
            )  # no op in this states
            in_animation = self._evaluate_if_animation(info)
        return obs, rew, terminated, truncated, info


class InitialStateSetterWrapper(gym.Wrapper):
    """
    Sets a given initial state from a list of possible states at reset.
    Can loop through given states or choose at random.

    Param:
        states (list[str]): list of states from where initial state will be sampled at reset
        seed (int): random seed. Defaults to None
        loop_through_states (bool): if True, will loop through states at each reset
    """

    def __init__(
        self, env, states, seed: int = None, loop_through_states: bool = False
    ):
        super(InitialStateSetterWrapper, self).__init__(env)
        self.random_seed = seed
        self.rng = np.random.default_rng(seed)
        self.initial_states = states
        self._loop_through_states = loop_through_states
        self._sampled_state_index = 0

    def reset(self, seed=None, options=None):
        self._sampled_state_index = self.rng.integers(
            low=0, high=len(self.initial_states)
        )
        if self._loop_through_states:
            new_index = self._sampled_state_index + 1
            self._sampled_state_index = (
                new_index if new_index < len(self.initial_states) else 0
            )

        sampled_state = str(self.initial_states[self._sampled_state_index])
        self.unwrapped.load_state(sampled_state, retro.data.Integrations.ALL)
        return super().reset(seed=self.random_seed, options=options)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        info["initial_state"] = self.initial_states[self._sampled_state_index]
        return obs, rew, terminated, truncated, info


# Discretizer from https://github.com/openai/retro/blob/master/retro/examples/discretizer.py#L9
class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class SuperTennisDiscretizer(Discretizer):
    """
    Use SuperTennis specific actions
    based on https://www.retrogames.cz/manualy/SNES/Super_Tennis_-_Manual_-_SNES.pdf and experience playing
    """

    def __init__(self, env):
        shot_options = [["A"], ["B"], ["Y"], ["X"]]
        dir_options = [
            ["UP"],
            ["DOWN"],
            ["LEFT"],
            ["RIGHT"],
            ["LEFT", "DOWN"],
            ["LEFT", "UP"],
            ["RIGHT", "DOWN"],
            ["RIGHT", "UP"],
        ]
        combos = [[]] + shot_options + dir_options  # includes no-op
        combos += self._create_shot_combos(dir_options, shot_options)
        super().__init__(
            env=env,
            combos=combos,
        )

    def _create_shot_combos(self, directions, shots):
        dir_shot_combos = []
        spins = [["L"], ["R"]]
        for d in directions:
            for shot in shots:
                # shots with direction
                dir_shot_combos.append(d + shot)
                # shots with direction and spin
                for spin in spins:
                    dir_shot_combos.append(d + shot + spin)
            # only direction and spin
            for spin in spins:
                dir_shot_combos.append(d + spin)
        # shots without direction but with spin
        for shot in shots:
            for spin in spins:
                dir_shot_combos.append(shot + spin)
        # only spin
        for spin in spins:
            dir_shot_combos.append(spin)
        return dir_shot_combos
