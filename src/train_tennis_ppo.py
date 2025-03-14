"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import os

import retro

# add custom game integration folder path to retro
retro.data.Integrations.add_custom_path(os.path.abspath("./games"))


from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from src.callbacks import HParamCallback
from src.config import PPOConfig, load_from_yaml, save_to_yaml
from src.env_helpers import (
    make_retro,
    read_statenames_from_folder,
    split_initial_states,
    wrap_deepmind_retro,
)


def create_logname(saved_model_path, continue_training, prefix="ppo_super_tennis"):
    if saved_model_path is not None and continue_training:
        return os.path.basename(os.path.dirname(saved_model_path))
    return f"{prefix}_{datetime.now().strftime('%d_%m_%Y__%H_%M_%S')}"


def initialize_model(env, config: PPOConfig):
    model = PPO(
        policy="CnnPolicy",
        tensorboard_log="./logs/",
        env=env,
        learning_rate=lambda f: f * config.initial_lr,
        verbose=1,
        seed=config.seed,
        **config.get_policy_params(),
    )
    return model


def load_saved_model(env, model_path, config):
    print(f"Load saved model from path: {model_path}")
    saved_model = PPO.load(model_path, env=env)
    model = initialize_model(env, config)
    model.policy.load_state_dict(saved_model.policy.state_dict())
    del saved_model
    return model


def main():
    render_mode = None
    game = "SuperTennis-Snes"
    states = read_statenames_from_folder("games/SuperTennis-Snes/working_init_states")

    continue_training = False
    saved_model_path = "logs/checkpoints/ppo_super_tennis_06_03_2025__09_52_28_FIRST_SUCCESSFUL/best_model.zip"
    exp_prefix = "ppo_st_multi_states"

    save_freq = 1e6
    eval_freq = 1e6
    scenario = None
    n_envs = 2
    total_timesteps = 100_000_000
    max_episode_steps = 5e4

    logname = create_logname(saved_model_path, continue_training, prefix=exp_prefix)
    os.makedirs(os.path.join("logs", logname))

    # initialize configuration
    config = PPOConfig(n_skip=3, skip_animations=False, clip_rewards=False)
    if continue_training:
        assert os.path.exists(
            saved_model_path
        ), "Please provide a valid model path to continue training from"
        config = load_from_yaml(os.path.join("logs", logname, "config.yml"))
    else:
        print("Saving configuration file for run ...")
        save_to_yaml(config, os.path.join("logs", logname, "config.yml"))

    def make_env_wrapper(env_states):
        def make_env():
            env = make_retro(
                game=game,
                states=env_states,
                scenario=scenario,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                seed=config.seed,
            )
            env = wrap_deepmind_retro(env, config)
            return env

        return make_env

    state_splits = split_initial_states(states, n_envs)

    # create training and evaluation environments
    venv = VecTransposeImage(
        VecFrameStack(
            SubprocVecEnv([make_env_wrapper(split) for split in state_splits]),
            n_stack=4,
        )
    )
    eval_venv = VecTransposeImage(
        VecFrameStack(
            SubprocVecEnv([make_env_wrapper(states)]),
            n_stack=4,
        )
    )

    # evaluation callback
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=os.path.join("./logs", logname, "checkpoints"),
        log_path=os.path.join("./logs", logname, "eval_metrics"),
        render=False,
        deterministic=True,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=4,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=os.path.join("./logs", logname, "checkpoints"),
        name_prefix="ppo_supertennis",
    )

    if saved_model_path is not None:
        model = load_saved_model(venv, saved_model_path, config)
    else:
        model = initialize_model(venv, config)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, ckpt_callback, HParamCallback(config)],
        tb_log_name=os.path.join(logname, "tensorboard"),
        log_interval=1,
        reset_num_timesteps=False if continue_training else True,
    )
    model.save(os.path.join("./logs", logname, "checkpoints", "last_model"))


if __name__ == "__main__":
    main()
