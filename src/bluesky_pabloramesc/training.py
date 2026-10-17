"""Shared training, checkpoint, and evaluation workflow."""

import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile
from time import perf_counter
from uuid import uuid4

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from .models import GAMMA
from .reporting import EpisodeHistory, write_metadata

EVAL_SEED = 2026


def checkpoint_paths(model_path):
    """Derive checkpoint paths while preserving dots in the model name.

    Args:
        model_path: Checkpoint stem or path ending in .zip.

    Returns:
        Pair of Paths for the policy archive and normalization statistics.
    """
    path = Path(model_path)
    stem = str(path)[:-4] if path.suffix == ".zip" else str(path)
    return Path(f"{stem}.zip"), Path(f"{stem}_vecnormalize.pkl")


def replay_buffer_path(model_path):
    """Derive the replay-buffer sidecar path for a checkpoint.

    Args:
        model_path: Checkpoint stem or policy archive path.

    Returns:
        Path ending in _replay_buffer.pkl.
    """
    model_path, _ = checkpoint_paths(model_path)
    return model_path.with_name(f"{model_path.stem}_replay_buffer.pkl")


def validate_checkpoint(model_path, stats_path=None):
    """Resolve and validate policy and normalization files.

    Args:
        model_path: Checkpoint stem or policy archive path.
        stats_path: Explicit normalization path, or None to infer it.

    Returns:
        Pair of existing policy and normalization Paths.

    Raises:
        FileNotFoundError: Either required checkpoint file is missing.
    """
    model_path, default_stats = checkpoint_paths(model_path)
    stats_path = Path(stats_path) if stats_path is not None else default_stats
    for path in (model_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
    return model_path, stats_path


def evaluate(model, env, episodes=5, seed=0, return_episode_rewards=False):
    """Evaluate deterministic actions with normalization statistics frozen.

    Args:
        model: SB3-compatible policy to evaluate.
        env: VecNormalize environment; its training flags are restored afterward.
        episodes: Number of evaluation episodes.
        seed: Seed for evaluation scenarios.
        return_episode_rewards: Whether to return individual rewards and lengths.

    Returns:
        Mean and standard deviation of raw returns, or lists of episode returns
        and lengths when return_episode_rewards is True.
    """
    training, norm_reward = env.training, env.norm_reward
    env.training = False
    env.norm_reward = False
    try:
        env.seed(seed)
        return evaluate_policy(
            model,
            env,
            n_eval_episodes=episodes,
            deterministic=True,
            return_episode_rewards=return_episode_rewards,
        )
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
    results_dir="experiments/results",
    resume_from=None,
    model_type=None,
    resume_stats_path=None,
):
    """Train or resume a policy, save its artifacts, and evaluate raw returns.

    Each run keeps episode history, metrics, plots, and matching normalization.
    Off-policy replay buffers are restored when available and saved after learning.

    Args:
        builder: Callable accepting env, seed, and device; unused when resuming.
        env_factory: Callable creating one vector environment in this process.
        model_path: Destination checkpoint stem or policy archive path.
        timesteps: Additional training steps; complete rollouts may exceed the budget.
        seed: Random seed for this training segment.
        device: PyTorch device name, or "auto".
        episodes: Number of final evaluation episodes.
        eval_seed: Scenario seed for final evaluation.
        results_dir: Parent directory for unique run reports and archived checkpoints.
        resume_from: Source checkpoint stem or archive, or None to start fresh.
        model_type: SB3 algorithm class required when resuming.
        resume_stats_path: Source normalization file, or None to infer it.

    Returns:
        Mean and standard deviation of final raw evaluation returns.

    Raises:
        ValueError: Resuming is requested without model_type.
        FileNotFoundError: A required resume checkpoint file is missing.
    """
    if resume_from is not None:
        if model_type is None:
            raise ValueError("model_type is required when resuming training")
        source_model, source_stats = validate_checkpoint(resume_from, resume_stats_path)
    started = datetime.now(timezone.utc)
    model_path, stats_path = checkpoint_paths(model_path)
    run_id = f"{model_path.stem}_{started:%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    directory = Path(results_dir) / run_id
    directory.mkdir(parents=True)
    history = EpisodeHistory(directory)
    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(),
        "seed": seed,
        "evaluation_seed": eval_seed,
        "requested_timesteps": timesteps,
        "evaluation_episodes": episodes,
        "model_path": str(model_path.resolve()),
        "stats_path": str(stats_path.resolve()),
        "versions": {
            name: version(name)
            for name in ("stable-baselines3", "torch", "gymnasium", "numpy")
        },
    }
    write_metadata(directory, metadata)
    env = None
    logger = None
    try:
        env = env_factory()
        if resume_from is None:
            env = VecNormalize(env, gamma=GAMMA, clip_obs=10.0, clip_reward=10.0)
            model = builder(env, seed=seed, device=device)
        else:
            env = VecNormalize.load(source_stats, env)
            env.training = True
            env.norm_reward = True
            model = model_type.load(source_model, env=env, device=device)
            model.set_random_seed(seed)
            metadata["resumed_from"] = str(source_model.resolve())
            metadata["resumed_stats_from"] = str(source_stats.resolve())
            if hasattr(model, "load_replay_buffer"):
                buffer_path = replay_buffer_path(source_model)
                metadata["replay_buffer_restored"] = buffer_path.is_file()
                if buffer_path.is_file():
                    model.load_replay_buffer(buffer_path)
                else:
                    warnings.warn(
                        "No replay buffer found; resuming with an empty buffer and "
                        "a new exploration warmup.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    model.learning_starts += model.num_timesteps
        initial_timesteps = model.num_timesteps
        metadata["initial_timesteps"] = initial_timesteps
        metadata.update(
            {
                "algorithm": type(model).__name__,
                "device": str(model.device),
                "num_envs": env.num_envs,
                "policy": str(model.policy),
                "settings": {
                    name: getattr(model, name)
                    for name in (
                        "learning_rate",
                        "gamma",
                        "batch_size",
                        "buffer_size",
                        "learning_starts",
                        "tau",
                        "train_freq",
                        "gradient_steps",
                        "n_steps",
                        "n_epochs",
                        "gae_lambda",
                        "ent_coef",
                        "policy_delay",
                        "target_policy_noise",
                        "target_noise_clip",
                        "action_noise",
                    )
                    if hasattr(model, name)
                },
            }
        )
        metadata["policy_kwargs"] = model.policy_kwargs
        write_metadata(directory, metadata)
        logger = configure(str(directory), ["stdout", "csv"])
        model.set_logger(logger)
        learning_started = perf_counter()
        model.learn(
            total_timesteps=timesteps,
            callback=history,
            progress_bar=True,
            reset_num_timesteps=resume_from is None,
        )
        metadata["training_seconds"] = perf_counter() - learning_started
        logger.dump(step=model.num_timesteps)
        metadata["added_timesteps"] = model.num_timesteps - initial_timesteps
        metadata["steps_per_second"] = (
            metadata["added_timesteps"] / metadata["training_seconds"]
        )
        metadata["actual_timesteps"] = model.num_timesteps
        metadata["training_episodes"] = len(history.rows)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        env.save(stats_path)
        # Preserve each run even when the convenient default checkpoint is replaced.
        model.save(directory / "policy.zip")
        env.save(directory / "policy_vecnormalize.pkl")
        if hasattr(model, "save_replay_buffer"):
            archived_buffer = replay_buffer_path(directory / "policy")
            model.save_replay_buffer(archived_buffer)
            copyfile(archived_buffer, replay_buffer_path(model_path))
        evaluation_started = perf_counter()
        rewards, lengths = evaluate(model, env, episodes, eval_seed, True)
        mean, std = float(np.mean(rewards)), float(np.std(rewards))
        metadata["evaluation"] = {
            "mean_return": mean,
            "std_return": std,
            "episode_returns": rewards,
            "episode_lengths": lengths,
            "seconds": perf_counter() - evaluation_started,
        }
        metadata["status"] = "completed"
        history.plot()
        print(f"Saved model: {model_path}; normalization: {stats_path}")
        print(f"Run report: {directory}")
        return mean, std
    except BaseException as error:
        metadata["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_metadata(directory, metadata)
        if logger is not None:
            logger.close()
        if env is not None:
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
    """Evaluate a saved policy with frozen normalization and optional rendering.

    Args:
        model_type: SB3 algorithm class used to load the policy.
        env_factory: Callable creating a vector environment with render_mode.
        model_path: Checkpoint stem or policy archive path.
        seed: Seed for evaluation scenarios.
        device: PyTorch device name, or "auto".
        episodes: Number of evaluation episodes.
        stats_path: Explicit normalization file, or None to infer it.
        render_mode: Rendering mode, or None for headless evaluation.

    Returns:
        Mean and standard deviation of raw evaluation returns.

    Raises:
        FileNotFoundError: The policy or normalization file is missing.
    """
    model_path, stats_path = validate_checkpoint(model_path, stats_path)

    base_env = env_factory(render_mode=render_mode)
    env = base_env
    try:
        env = VecNormalize.load(stats_path, base_env)
        model = model_type.load(model_path, env=env, device=device)
        return evaluate(model, env, episodes, seed)
    finally:
        env.close()
