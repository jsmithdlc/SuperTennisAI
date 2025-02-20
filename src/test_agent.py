import os

import retro

# add custom game integration folder path to retro
retro.data.Integrations.add_custom_path(os.path.abspath("./games"))

from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.ppo import PPO

from train_tennis_ppo import make_retro, wrap_deepmind_retro


def agent_play(model, env):
    episode_over = False
    obs = env.reset()
    while not episode_over:
        action, _ = model.predict(obs)
        obs, reward, terminated, info = env.step(action)
        episode_over = terminated
    env.close()


def main():
    game = "SuperTennis-Snes"
    state = retro.State.DEFAULT
    scenario = None
    render_mode = "human"
    model_path = "./logs/super_tennis_ppo_0.zip"

    def make_env():
        env = make_retro(
            game=game, state=state, scenario=scenario, render_mode=render_mode
        )
        env = wrap_deepmind_retro(env)
        return env

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]), n_stack=4))
    model = PPO.load(path=model_path, env=venv)
    agent_play(model, venv)


if __name__ == "__main__":
    main()
