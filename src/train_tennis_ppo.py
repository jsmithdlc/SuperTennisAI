"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import os

import retro

# add custom game integration folder path to retro
retro.data.Integrations.add_custom_path(os.path.abspath("./games"))


from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)

from src.callbacks import initialize_callbacks
from src.config import PPOConfig, load_from_yaml, save_to_yaml
from src.env_helpers import (
    make_retro,
    read_statenames_from_folder,
    split_initial_states,
    wrap_deepmind_retro,
)
from src.networks.residual_extractor import ResidualCNN


def create_logname(saved_model_path, continue_training, prefix="ppo_super_tennis"):
    if saved_model_path is not None and continue_training:
        return saved_model_path.split("/")[-3]
    return f"{prefix}_{datetime.now().strftime('%d_%m_%Y__%H_%M_%S')}"


def initialize_model(env, config: PPOConfig):
    model = PPO(
        policy="CnnPolicy",
        tensorboard_log="./logs/",
        env=env,
        learning_rate=lambda f: f * config.initial_lr,
        clip_range=lambda f: f * config.clip_range,
        verbose=1,
        seed=config.seed,
        stats_window_size=config.stats_window_size,
        **config.get_policy_params(),
    )
    return model


def load_saved_model(env, model_path, config, continue_training):
    print(f"Load saved model from path: {model_path}")
    if continue_training:
        model = PPO.load(model_path)
        model.set_env(env)
        return model
    saved_model = PPO.load(model_path, env=env)
    model = initialize_model(env, config)
    model.policy.load_state_dict(saved_model.policy.state_dict())
    del saved_model
    return model


def main():
    render_mode = None
    game = "SuperTennis-Snes"
    states = read_statenames_from_folder(
        "games/SuperTennis-Snes/hard-court_easy-opponents_states"
    )

    continue_training = False
    saved_model_path = None
    exp_prefix = "ppo_multi_states_large_network"

    logname = create_logname(saved_model_path, continue_training, prefix=exp_prefix)
    os.makedirs(os.path.join("logs", logname), exist_ok=True)

    # initialize configuration
    config = PPOConfig(
        n_envs=8,
        clip_range=0.2,
        ent_coef=0.01,
        clip_rewards=False,
        stall_penalty=0.5,
        fault_penalty=0.5,
        ball_return_reward=0.2,
        n_steps=256,
        batch_size=1024,
        total_timesteps=400_000_000,
        stats_window_size=32,
        scenario="games/SuperTennis-Snes/scenario.json",
        features_extractor_class="ImpalaCNN",
        features_extractor_dim=256,
    )

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
                scenario=config.scenario,
                render_mode=render_mode,
                max_episode_steps=config.max_episode_steps,
                seed=config.seed,
            )
            env = wrap_deepmind_retro(env, config)
            return env

        return make_env

    state_splits = split_initial_states(states, config.n_envs)

    # create training and evaluation environments
    venv = VecTransposeImage(
        VecFrameStack(
            VecNormalize(
                SubprocVecEnv([make_env_wrapper(split) for split in state_splits]),
                norm_obs=False,
                norm_reward=True,
                gamma=config.gamma,
                clip_reward=10.0,
            ),
            n_stack=4,
        )
    )
    eval_venv = VecTransposeImage(
        VecFrameStack(
            VecNormalize(
                SubprocVecEnv([make_env_wrapper(states)]),
                training=False,
                clip_reward=10.0,
                gamma=config.gamma,
                norm_obs=False,
                norm_reward=True,
            ),
            n_stack=4,
        ),
    )

    # additional callbacks
    callbacks = initialize_callbacks(eval_venv, config, logname)

    if saved_model_path is not None:
        model = load_saved_model(venv, saved_model_path, config, continue_training)
    else:
        model = initialize_model(venv, config)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        tb_log_name=os.path.join(logname, "tensorboard"),
        log_interval=config.log_interval,
        reset_num_timesteps=False if continue_training else True,
    )
    model.save(os.path.join("./logs", logname, "checkpoints", "last_model"))


if __name__ == "__main__":
    main()
