"""Shared training, checkpoint, and evaluation workflow."""

from pathlib import Path

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from .models import GAMMA

EVAL_SEED = 10_000


def checkpoint_paths(model_path):
    """Accept a model stem or .zip path without stripping other suffixes."""
    path = Path(model_path)
    stem = str(path)[:-4] if path.suffix == ".zip" else str(path)
    return Path(f"{stem}.zip"), Path(f"{stem}_vecnormalize.pkl")


def evaluate(model, env, episodes=5, seed=0):
    """Report raw returns while keeping normalization statistics frozen."""
    training, norm_reward = env.training, env.norm_reward
    env.training = False
    env.norm_reward = False
    try:
        env.seed(seed)
        return evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=True)
    finally:
        env.training, env.norm_reward = training, norm_reward


def train(
    builder,
    env_factory,
    model_path,
    timesteps,
    seed=0,
    device="auto",
    episodes=5,
    eval_seed=EVAL_SEED,
):
    """Save the trained policy and its matching observation statistics."""
    base_env = env_factory()
    env = base_env
    try:
        env = VecNormalize(base_env, gamma=GAMMA, clip_obs=10.0, clip_reward=10.0)
        model = builder(env, seed=seed, device=device)
        model.learn(total_timesteps=timesteps)
        model_path, stats_path = checkpoint_paths(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        env.save(stats_path)
        print(f"Saved model: {model_path}; normalization: {stats_path}")
        return evaluate(model, env, episodes, eval_seed)
    finally:
        env.close()


def replay(
    model_type,
    env_factory,
    model_path,
    seed=EVAL_SEED,
    device="auto",
    episodes=3,
    stats_path=None,
    render_mode="human",
):
    """Evaluate a checkpoint with its saved statistics; optionally render it."""
    model_path, default_stats = checkpoint_paths(model_path)
    stats_path = Path(stats_path) if stats_path is not None else default_stats
    for path in (model_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")

    base_env = env_factory(render_mode=render_mode)
    env = base_env
    try:
        env = VecNormalize.load(stats_path, base_env)
        model = model_type.load(model_path, env=env, device=device)
        return evaluate(model, env, episodes, seed)
    finally:
        env.close()
