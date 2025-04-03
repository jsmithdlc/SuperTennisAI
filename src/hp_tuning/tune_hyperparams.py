"""
Script based on https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py, which is based on
https://github.com/DLR-RM/rl-baselines3-zoo

"""

import os

import retro

# add custom game integration folder path to retro
retro.data.Integrations.add_custom_path(os.path.abspath("./games"))

from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.ppo import PPO

from src.callbacks import HParamCallback
from src.config import PPOConfig, save_to_yaml
from src.env_helpers import (
    create_vectorized_env,
    read_statenames_from_folder,
    split_initial_states,
)
from src.hp_tuning.optuna_utils import TrialEvalCallback

N_TRIALS = 100
N_STARTUP_TRIALS = 10
N_EVALUATIONS = 8
LOGS_PATH = "./logs/optuna"
N_TIMESTEPS = 2_000_000
EVAL_FREQ = N_TIMESTEPS // N_EVALUATIONS
N_EVAL_EPISODES = 4


def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    features_extractor_dim = trial.suggest_categorical(
        "features_extractor_dim", [128, 256]
    )

    # TODO: account when using multiple envs
    if batch_size > n_steps * 8:
        batch_size = n_steps

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "initial_lr": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "vf_coef": vf_coef,
        "n_epochs": n_epochs,
        "features_extractor_class": "ImpalaCNN",
        "features_extractor_dim": features_extractor_dim,
    }


def objective(trial: optuna.Trial) -> float:
    # create config with sampled parameters
    config = PPOConfig(**sample_ppo_params(trial))

    # initial_states
    states = read_statenames_from_folder(
        "games/SuperTennis-Snes/hard-court_easy-opponents_states"
    )
    state_splits = split_initial_states(states, config.n_envs)

    # create training and evaluation environments
    # create training and evaluation environments
    venv = create_vectorized_env(
        config, state_splits, render_mode=None, training=True, loop_states=False
    )
    eval_venv = create_vectorized_env(
        config, [states], render_mode=None, training=False, loop_states=True
    )

    trial_path = os.path.join(
        LOGS_PATH, trial.study.study_name, f"trial_{str(trial.number)}"
    )
    os.makedirs(trial_path, exist_ok=True)

    # Create the RL model.
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        verbose=0,
        seed=config.seed,
        tensorboard_log=trial_path,
        learning_rate=lambda f: f * config.initial_lr,
        clip_range=lambda f: f * config.clip_range,
        stats_window_size=config.stats_window_size,
        **config.get_policy_params(),
    )

    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_venv,
        trial,
        best_model_save_path=trial_path,
        log_path=trial_path,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=(EVAL_FREQ) // config.n_envs,
        deterministic=False,
    )

    save_to_yaml(config, os.path.join(trial_path, "config.yml"))

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=[eval_callback, HParamCallback(config)])
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        print(f"Trial: {str(trial.number)} pruned")
        raise optuna.exceptions.TrialPruned()

    # save the best reward achieved in this trial
    return eval_callback.last_mean_reward


if __name__ == "__main__":

    study_name = "ppo_multistates"

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True)

    # do not prune until 2 evaluations are done
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///logs/optuna/{study_name}/study.db",
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
