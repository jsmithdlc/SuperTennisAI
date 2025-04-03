import os

import retro

# add custom game integration folder path to retro
retro.data.Integrations.add_custom_path(os.path.abspath("./games"))

import gymnasium as gym
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.ppo import PPO

from src.config import load_from_yaml
from src.env_helpers import create_vectorized_env

MAX_EPISODE_STEPS = None
VIDEO_LENGTH = 10000


def run_episode(model, env: gym.Env):
    episode_over = False
    obs = env.reset()
    tot_reward = 0.0
    while not episode_over:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        tot_reward += reward[0]
        episode_over = terminated
    print(f"Total reward: {float(tot_reward):.1f}")
    env.close()


def record_game(model, env: gym.Env, video_path, video_length=1000):
    # wrap around video recorder
    video_env = VecVideoRecorder(
        env,
        video_path,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"agent_ppo",
    )
    run_episode(model, video_env)


def main():
    states = [
        "all_initial_states/SuperTennis.Singles.PlayerServes.PlayerBot.MattvsRich.1-set.Clay.state"
    ]

    scenario = "games/SuperTennis-Snes/scenario.json"
    render_mode = "human"
    logname = "logs/ppo_multi_states_large_network_03_04_2025__07_53_45"

    model_path = f"{logname}/checkpoints/ppo_supertennis_2000000_steps.zip"
    video_path = os.path.join(logname, "videos")
    config = load_from_yaml(os.path.join(logname, "config.yml"))

    # update config for testing purposes
    config.scenario = scenario

    venv = create_vectorized_env(
        config, [states], render_mode, training=False, loop_states=True
    )
    model = PPO.load(path=model_path, env=venv)
    if render_mode == "rgb_array":
        record_game(model, venv, video_path, video_length=VIDEO_LENGTH)
    else:
        run_episode(model, venv)


if __name__ == "__main__":
    main()
