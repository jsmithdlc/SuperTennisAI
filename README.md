# SuperTennisAI

Project for building a RL agent as an AI for the Super Nintendo game Super Tennis

## Configure

Download the SuperTennis-SNES rom into the `games` directory and import into stable-retro by running:

```bash
python3 -m retro.import games
```

## Code overview (`src`)

The agent is trained with [Stable Baselines3](https://stable-baselines3.readthedocs.io/)'s PPO
implementation on top of a [stable-retro](https://github.com/Farama-Foundation/stable-retro)
emulation of Super Tennis. The modules below are listed roughly in the order you'd read them to
understand the training pipeline.

### `train_tennis_ppo.py`

Entry point for training the PPO agent (`python -m src.train_tennis_ppo`). It:

- Registers the custom `games/` integration folder with `retro` so the `SuperTennis-Snes` game
  and its scenario/state files can be found.
- Builds an `ExperimentConfig`/`PPOConfig` (see `config.py`) describing hyperparameters, reward
  shaping, and the CNN feature extractor to use.
- Loads the initial save states to train on (via `env_helpers.read_statenames_from_folder`) and
  splits them across the parallel environments (`env_helpers.split_initial_states`).
- Creates a vectorized training environment and a separate evaluation environment
  (`env_helpers.create_vectorized_env`).
- Either initializes a fresh `PPO` model (`initialize_model`) or resumes/warm-starts from a saved
  checkpoint (`load_saved_model`), depending on the `continue_training`/`saved_model_path`
  settings at the top of `main()`.
- Wires up logging/checkpointing/eval callbacks (`callbacks.initialize_callbacks`) and calls
  `model.learn(...)`, saving the final model and periodic checkpoints under `logs/<run-name>/`.

Run configuration (which states to use, hyperparameters, whether to continue a previous run) is
currently edited directly in `main()` rather than passed via CLI args.

### `config.py`

Defines the experiment configuration dataclasses:

- `ExperimentConfig` — algorithm-agnostic settings: env/training params (`n_envs`, `n_steps`,
  `total_timesteps`, `gamma`, ...), preprocessing toggles (frame skipping, sticky actions, reward
  clipping), custom reward-shaping coefficients (`stall_penalty`, `fault_penalty`,
  `ball_return_reward`), and logging/checkpoint frequencies.
- `PPOConfig(ExperimentConfig)` — adds PPO-specific hyperparameters (`clip_range`, `ent_coef`,
  `gae_lambda`, `vf_coef`) and selects the CNN feature extractor (`NatureCNN`, `ResidualCNN`, or
  `ImpalaCNN`, see `networks/`).
- `save_to_yaml` / `load_from_yaml` — persist a config alongside each run in `logs/<run-name>/config.yml`
  so runs can be resumed or inspected later.

### `env_helpers.py`

Builds the (vectorized) Gymnasium/retro environments used for training and evaluation:

- `read_statenames_from_folder` — lists `.state` save files in a folder to use as episode starting
  points.
- `split_initial_states` — distributes a list of states across `n_envs` parallel workers.
- `make_retro` — creates a single retro env for `SuperTennis-Snes`, applies the discrete action
  space (`SuperTennisDiscretizer`), random initial-state selection
  (`InitialStateSetterWrapper`), and an episode step limit.
- `wrap_deepmind_retro` — applies the DeepMind-Atari-style preprocessing/reward-shaping stack
  (animation skipping, sticky actions, frame skipping, frame warping, stall/fault penalties, ball
  return bonus, reward clipping) driven by an `ExperimentConfig`, plus `Monitor` for episode stats.
- `create_vectorized_env` — combines the above into a `SubprocVecEnv`, optionally wrapped with
  `VecNormalize` (for reward normalization, loadable from a saved `.pkl`), `VecFrameStack` (4-frame
  stacking), and `VecTransposeImage`.

### `wrappers.py`

Gymnasium wrappers used to shape the raw game into a more learnable RL problem:

- `SuperTennisDiscretizer` (and its base `Discretizer`) — maps the SNES controller's
  `MultiBinary` action space down to a discrete set of meaningful button/direction/spin combos.
- `StickyActionWrapper` — repeats the previous action with some probability, for stochasticity.
- `FrameSkip` — repeats an action for `n_skip` frames and sums the reward.
- `InitialStateSetterWrapper` — loads a randomly-sampled (or looped, for evaluation) save state at
  each `reset()`.
- `SkipAnimationsWrapper` — fast-forwards through non-interactive animation/text frames.
- `StallPenaltyWrapper` — penalizes the agent for holding the ball too long before serving.
- `FaultPenaltyWrapper` — penalizes double faults while serving.
- `ReturnCompensationWrapper` — gives a small reward bonus for successfully returning the ball,
  to counteract the sparsity of point-level rewards.

### `callbacks.py`

Stable Baselines3 callbacks used during training, assembled by `initialize_callbacks`:

- `EvalCallback` — periodically evaluates the current policy on held-out states and saves the best
  model.
- `CheckpointCallback` — periodically saves model + `VecNormalize` checkpoints to
  `logs/<run-name>/checkpoints/`.
- `HParamCallback` — logs the experiment config as TensorBoard hyperparameters.
- `LogExtraEpisodeStatsCallback` — aggregates and logs extra per-episode stats from the env's
  `info` dict (points won ratio, ace ratio, stalls, faults, ball returns), broken down overall and
  per court surface (Clay/Hard/Lawn).

### `test_agent.py`

Loads a trained model/checkpoint and its matching config and `VecNormalize` stats, then either
renders a live episode (`render_mode="human"`) or records one to video via `VecVideoRecorder`
(`render_mode="rgb_array"`). Useful for qualitatively evaluating a trained agent.

### `networks/`

Custom CNN feature extractors usable via `PPOConfig.features_extractor_class`:

- `impala_cnn.py` (`ImpalaCNN64`) — an IMPALA-style CNN with three downsampling
  conv+maxpool blocks, each followed by two residual blocks.
- `residual_extractor.py` (`ResidualCNN`) — a smaller extractor with two initial downsampling
  convolutions followed by two residual blocks and a final convolution.

### `hp_tuning/`

Hyperparameter search using [Optuna](https://optuna.org/):

- `tune_hyperparams.py` — samples PPO hyperparameters and reward-shaping coefficients per trial,
  trains a short PPO run, and reports the best achieved eval reward back to Optuna (with pruning
  of unpromising trials). Studies are persisted to a SQLite DB under `logs/optuna/`.
- `optuna_utils.py` — `TrialEvalCallback`, an `EvalCallback` subclass that reports intermediate
  eval results to Optuna and requests trial pruning when appropriate.
