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


def initialize_model(
    env, config: PPOConfig, tensorboard_log: str = "./logs/", verbose: int = 1
):
    model = PPO(
        policy="CnnPolicy",
        tensorboard_log=tensorboard_log,
        env=env,
        learning_rate=lambda f: f * config.initial_lr,
        clip_range=lambda f: f * config.clip_range,
        verbose=verbose,
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

    continue_training = False
    saved_model_path = None
    vec_normalize_path = None

    exp_prefix = "ppo_v0"

    logname = create_logname(saved_model_path, continue_training, prefix=exp_prefix)
    os.makedirs(os.path.join("logs", logname), exist_ok=True)

    # initialize configuration
    config = PPOConfig(
        n_envs=8,
        initial_lr=1e-4,
        n_epochs=5,
        clip_range=0.2,
        ent_coef=0.005,
        # start training with more exploration, annealed down to ent_coef
        # above (see callbacks.EntropyCoefScheduleCallback)
        ent_coef_initial=0.02,
        clip_rewards=False,
        stall_penalty=0,
        fault_penalty=0,
        gamma=0.995,
        ball_return_reward=0,
        n_steps=512,
        batch_size=1024,
        total_timesteps=10_000_000,
        stats_window_size=32,
        scenario="games/SuperTennis-Snes/scenario_lua.json",
        features_extractor_class="ImpalaCNN",
        features_extractor_dim=128,
        vf_coef=0.7,
        # finer-grained control for reaching/returning balls: less action
        # persistence than the Atari-style defaults (n_skip=4, sticky_prob=0.25)
        n_skip=3,
        sticky_prob=0.1,
        # preserve native 256x224 aspect ratio at higher resolution than the
        # previous hardcoded 84x84 (which also squashed the aspect ratio),
        # so the ball stays visible when far from the player
        frame_width=128,
        frame_height=112,
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
