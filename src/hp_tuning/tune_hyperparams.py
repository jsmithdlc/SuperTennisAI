"""
Script based on https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py, which is based on 
https://github.com/DLR-RM/rl-baselines3-zoo

"""
import retro 
import os

# add custom game integration folder path to retro
retro.data.Integrations.add_custom_path(os.path.abspath("./games"))

import torch
import gymnasium
import optuna
import json
from torch import nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from .optuna_utils import TrialEvalCallback
from typing import Any
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.ppo import PPO
from src.train_tennis_ppo import make_retro, wrap_deepmind_retro



N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(5e6)
STARTING_STATE = "SuperTennis.Singles.MattvsBarb.1-set.Hard"
STUDY_PATH = "./logs/optuna"
EVAL_FREQ = N_TIMESTEPS // N_EVALUATIONS
N_EVAL_EPISODES = 3
TIMEOUT_S = 60 * 60 * 12 # stop optuna study after this number of seconds


DEFAULT_HYPERPARAMS = {
    "policy": "CnnPolicy"
}

def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    # Orthogonal initialization
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]


    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch
        ),
    }

def make_supertennis_env():
    env = make_retro(
        game = "SuperTennis-Snes",
        state= STARTING_STATE,
        scenario = None,
        render_mode = None
    )
    env = wrap_deepmind_retro(env)
    return env

def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))

    # create training and evaluation environments
    env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_supertennis_env] * 8), n_stack=4))
    eval_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_supertennis_env]), n_stack=4))

    trial_path = os.path.join(STUDY_PATH, f"trial_{str(trial.number)}")
    os.makedirs(trial_path, exist_ok = True)

    # Create the RL model.
    model = PPO(
        env = env, 
        verbose = 0, 
        tensorboard_log = trial_path, 
        **kwargs
    )

    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, 
        trial, 
        best_model_save_path = trial_path,
        log_path = trial_path,
        n_eval_episodes=N_EVAL_EPISODES, 
        eval_freq=(EVAL_FREQ) // 8, 
        deterministic=True
    )

    with open(os.path.join(trial_path, "params.json"), "w") as f:
        json.dump(kwargs, f, indent=4)

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        print(f"Trial: {str(trial.number)} pruned")
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(
        study_name = "ppo_supertennis_tuning_v2", 
        sampler=sampler, 
        pruner=pruner, 
        direction="maximize",
        load_if_exists = True)
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout = TIMEOUT_S)
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

