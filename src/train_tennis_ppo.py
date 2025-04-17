"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import os

import retro

# add custom game integration folder path to retro
retro.data.Integrations.add_custom_path(os.path.abspath("./games"))


from datetime import datetime

from stable_baselines3 import PPO

from src.callbacks import initialize_callbacks
from src.config import PPOConfig, load_from_yaml, save_to_yaml
from src.env_helpers import (
    create_vectorized_env,
    read_statenames_from_folder,
    split_initial_states,
)


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
    states = read_statenames_from_folder(
        "games/SuperTennis-Snes/hard-court_easy-opponents_states"
    )

    continue_training = True
    saved_model_path = "logs/ppo_multi_states_resnet_05_04_2025__20_29_57/checkpoints/ppo_supertennis_141000000_steps.zip"
    vec_normalize_path = "logs/ppo_multi_states_resnet_05_04_2025__20_29_57/checkpoints/ppo_supertennis_vecnormalize_141000000_steps.pkl"

    exp_prefix = "ppo_multi_states_resnet"

    logname = create_logname(saved_model_path, continue_training, prefix=exp_prefix)
    os.makedirs(os.path.join("logs", logname), exist_ok=True)

    # initialize configuration
    config = PPOConfig(
        n_envs=8,
        initial_lr=1e-4,
        n_epochs=5,
        clip_range=0.2,
        ent_coef=0.005,
        clip_rewards=False,
        stall_penalty=0.5,
        fault_penalty=0.5,
        gamma=0.995,
        ball_return_reward=0.2,
        n_steps=512,
        batch_size=1024,
        total_timesteps=400_000_000,
        stats_window_size=32,
        scenario="games/SuperTennis-Snes/scenario.json",
        features_extractor_class="ImpalaCNN",
        features_extractor_dim=128,
        vf_coef=0.7,
    )

    if continue_training:
        assert os.path.exists(
            saved_model_path
        ), "Please provide a valid model path to continue training from"
        config = load_from_yaml(os.path.join("logs", logname, "config.yml"))
    else:
        print("Saving configuration file for run ...")
        save_to_yaml(config, os.path.join("logs", logname, "config.yml"))

    state_splits = split_initial_states(states, config.n_envs)

    # create training and evaluation environments
    venv = create_vectorized_env(
        config,
        state_splits,
        render_mode,
        training=True,
        loop_states=False,
        vec_normalize_path=vec_normalize_path,
    )
    eval_venv = create_vectorized_env(
        config, [states], render_mode, training=False, loop_states=True
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
