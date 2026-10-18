"""Shared training, checkpoint, and evaluation workflow."""

import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile
from time import perf_counter
from uuid import uuid4

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
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


def architecture_config_path(model_path):
    """Locate the architecture configuration beside a policy."""
    path, _ = checkpoint_paths(model_path)
    return path.with_name(f"{path.stem}_architecture.json")


def load_architecture_config(model_path):
    """Old checkpoints use the flat MLP baseline."""
    path = architecture_config_path(model_path)
    return (
        json.loads(path.read_text())
        if path.is_file()
        else {"architecture": "mlp", "pooling": "mean_max"}
    )


def control_config_path(model_path):
    """Locate the environment configuration saved beside a policy."""
    path, _ = checkpoint_paths(model_path)
    return path.with_name(f"{path.stem}_control.json")


def load_control_config(model_path):
    """Read saved action semantics; checkpoints predating this feature are legacy."""
    from .actions import ControlConfig

    path = control_config_path(model_path)
    return (
        ControlConfig(**json.loads(path.read_text()))
        if path.is_file()
        else ControlConfig()
    )


def validate_control_config(model_path, config):
    """Reject changes to action semantics even when tensor shapes agree."""
    if config is not None and load_control_config(model_path) != config:
        raise ValueError(
            "Checkpoint control configuration does not match requested settings"
        )


def validate_policy_config(model, config):
    """Check the archive copy too, detecting missing or stale sidecars."""
    from .actions import ControlConfig

    if (
        config is not None
        and ControlConfig(**getattr(model, "control_config", config.to_dict()))
        != config
    ):
        raise ValueError("Policy archive control configuration does not match sidecar")


def validate_architecture_config(model, model_path):
    """Detect stale or missing architecture sidecars before using a policy."""
    saved = getattr(model, "architecture_config", None)
    if saved is not None and saved != load_architecture_config(model_path):
        raise ValueError("Policy archive architecture does not match sidecar")


def replay_buffer_path(model_path):
    """Derive the replay-buffer sidecar path for a checkpoint.

    Args:
        model_path: Checkpoint stem or policy archive path.

    Returns:
        Path ending in _replay_buffer.pkl.
    """
    model_path, _ = checkpoint_paths(model_path)
    return model_path.with_name(f"{model_path.stem}_replay_buffer.pkl")


def save_checkpoint(model, env, model_path):
    """Save a policy together with normalization and any off-policy replay buffer.

    Args:
        model: SB3 model to save.
        env: Matching VecNormalize environment.
        model_path: Destination checkpoint stem or archive path.
    """
    model_path, stats_path = checkpoint_paths(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    if getattr(model, "architecture_config", None) is not None:
        architecture_config_path(model_path).write_text(
            json.dumps(model.architecture_config, indent=2) + "\n"
        )
    env.save(stats_path)
    if getattr(model, "control_config", None) is not None:
        control_config_path(model_path).write_text(
            json.dumps(model.control_config, indent=2) + "\n"
        )
    if hasattr(model, "save_replay_buffer"):
        model.save_replay_buffer(replay_buffer_path(model_path))


def save_run_checkpoint(model, env, model_path, directory):
    """Archive the run's state and update the regular checkpoint copies.

    Args:
        model: SB3 model to save.
        env: Matching VecNormalize environment.
        model_path: Destination for the regular checkpoint.
        directory: Run report directory containing the archived policy.
    """
    archived = directory / "policy"
    save_checkpoint(model, env, archived)
    targets = [*checkpoint_paths(model_path)]
    sources = [*checkpoint_paths(archived)]
    if architecture_config_path(archived).is_file():
        sources.append(architecture_config_path(archived))
        targets.append(architecture_config_path(model_path))
    if control_config_path(archived).is_file():
        sources.append(control_config_path(archived))
        targets.append(control_config_path(model_path))
    if hasattr(model, "save_replay_buffer"):
        sources.append(replay_buffer_path(archived))
        targets.append(replay_buffer_path(model_path))
    for source, target in zip(sources, targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            copyfile(source, target)


class PeriodicCheckpoint(BaseCallback):
    """Save complete training state at rollout boundaries after a step interval."""

    def __init__(self, directory, interval, metadata):
        """Configure periodic saves relative to the start of this training segment.

        Args:
            directory: Run report directory for checkpoint subdirectories.
            interval: Additional environment steps between saves.
            metadata: Shared run metadata updated after each complete save.
        """
        super().__init__()
        self.directory = directory
        self.interval = interval
        self.metadata = metadata

    def _on_training_start(self):
        """Set the first save threshold using the model's cumulative step count."""
        self.next_checkpoint = self.num_timesteps + self.interval

    def _on_step(self):
        """Continue collection so the current transition reaches the replay buffer.

        Returns:
            True to continue training.
        """
        return True

    def _on_rollout_end(self):
        """Save due checkpoints after all collected transitions have been stored."""
        if self.num_timesteps < self.next_checkpoint:
            return
        path = self.directory / "checkpoints" / f"step_{self.num_timesteps}" / "policy"
        save_checkpoint(self.model, self.training_env, path)
        self.metadata["latest_checkpoint"] = str(path.with_suffix(".zip").resolve())
        self.metadata["checkpoint_timesteps"] = self.num_timesteps
        write_metadata(self.directory, self.metadata)
        self.next_checkpoint = self.num_timesteps + self.interval
        print(f"Saved periodic checkpoint: {path}")


def record_training_progress(metadata, model, history, learning_started):
    """Record elapsed learning time and cumulative/additional training steps.

    Args:
        metadata: Run report mapping to update.
        model: Current SB3 model.
        history: Episode history for this segment.
        learning_started: Monotonic timestamp at the start of learning.
    """
    metadata.setdefault("training_seconds", perf_counter() - learning_started)
    metadata["added_timesteps"] = model.num_timesteps - metadata["initial_timesteps"]
    metadata["actual_timesteps"] = model.num_timesteps
    metadata["training_episodes"] = len(history.rows)
    metadata["steps_per_second"] = metadata["added_timesteps"] / max(
        metadata["training_seconds"], 1e-9
    )


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


def evaluate_scenarios(model, env, seeds):
    """Evaluate explicit scenario seeds, retaining final metrics for every agent.

    Multi-agent slots complete jointly. Preserve each departing agent's last
    metrics before the fixed-slot wrapper begins returning empty info mappings.
    """
    training, norm_reward = env.training, env.norm_reward
    env.training, env.norm_reward = False, False
    rows = []
    try:
        for seed in seeds:
            env.seed(seed)
            observations = env.reset()
            returns = np.zeros(env.num_envs)
            lengths = np.zeros(env.num_envs, dtype=int)
            final_infos = [{} for _ in range(env.num_envs)]
            finished = np.zeros(env.num_envs, dtype=bool)
            while not finished.all():
                actions, _ = model.predict(observations, deterministic=True)
                observations, rewards, dones, infos = env.step(actions)
                returns[~finished] += rewards[~finished]
                lengths[~finished] += 1
                for index, info in enumerate(infos):
                    if not finished[index] and "flight_time" in info:
                        final_infos[index] = {
                            key: float(value)
                            for key, value in info.items()
                            if isinstance(value, (int, float, np.number))
                        }
                finished |= dones | np.array(
                    [bool(info.get("waypoint_reached", False)) for info in infos]
                )
            for index, info in enumerate(final_infos):
                if "waypoint_reached" in info:
                    info["timed_out"] = float(not info["waypoint_reached"])
                rows.append(
                    {
                        **info,
                        "scenario_seed": int(seed),
                        "agent_index": index,
                        "return": float(returns[index]),
                        "length": int(lengths[index]),
                    }
                )
        return rows
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
    checkpoint_interval=10_000,
    control_config=None,
    evaluation_seeds=None,
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
        checkpoint_interval: Additional steps between saves at rollout boundaries.
        control_config: Optional action/observation settings saved with every checkpoint.
        evaluation_seeds: Explicit scenario seeds; evaluate every agent once per seed.

    Returns:
        Mean and standard deviation of final raw evaluation returns.

    Raises:
        ValueError: A nonpositive interval or resume without model_type is supplied.
        FileNotFoundError: A required resume checkpoint file is missing.
    """
    if evaluation_seeds is not None and not evaluation_seeds:
        raise ValueError("evaluation_seeds must not be empty")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if resume_from is not None:
        if model_type is None:
            raise ValueError("model_type is required when resuming training")
        source_model, source_stats = validate_checkpoint(resume_from, resume_stats_path)
        validate_control_config(source_model, control_config)
    started = datetime.now(timezone.utc)
    model_path, stats_path = checkpoint_paths(model_path)
    run_id = f"{model_path.stem}_{started:%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    directory = Path(results_dir) / run_id
    directory.mkdir(parents=True)
    history = EpisodeHistory(directory)
    metadata = {
        "control_config": control_config.to_dict()
        if control_config is not None
        else None,
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(),
        "seed": seed,
        "evaluation_seed": eval_seed,
        "requested_timesteps": timesteps,
        "checkpoint_interval": checkpoint_interval,
        "evaluation_episodes": episodes,
        "evaluation_scenario_seeds": evaluation_seeds,
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
    model = None
    learning_started = None
    progress = None
    try:
        env = env_factory()
        if resume_from is None:
            interval = control_config.decision_interval if control_config else 10
            from .entities import EntityVecNormalize
            from .temporal import HistoryVecNormalize

            normalizer = (
                HistoryVecNormalize
                if hasattr(env.observation_space, "temporal_frame_space")
                else EntityVecNormalize
                if hasattr(env.observation_space, "entity_layout")
                else VecNormalize
            )
            env = normalizer(
                env, gamma=GAMMA ** (interval / 10), clip_obs=10.0, clip_reward=10.0
            )
            model = builder(env, seed=seed, device=device)
        else:
            env = VecNormalize.load(source_stats, env)
            env.training = True
            env.norm_reward = True
            model = model_type.load(source_model, env=env, device=device)
            validate_policy_config(model, control_config)
            validate_architecture_config(model, source_model)
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
        if control_config is not None:
            model.control_config = control_config.to_dict()
        from .entities import EntityGNN, EntityMLP, EntityTransformer

        extractor = model.policy.features_extractor
        if extractor is None:
            extractor = model.policy.actor.features_extractor
        from .temporal import TemporalFeatures

        temporal_extractor = (
            extractor if isinstance(extractor, TemporalFeatures) else None
        )
        if temporal_extractor is not None:
            extractor = temporal_extractor.frame_encoder
        model.architecture_config = {
            "architecture": "entity_mlp" if isinstance(extractor, EntityMLP) else "mlp",
            "pooling": getattr(extractor, "pooling", "mean_max"),
        }
        if isinstance(extractor, EntityTransformer):
            model.architecture_config["architecture"] = "entity_transformer"
            model.architecture_config["transformer_layers"] = (
                extractor.transformer_layers
            )
        if isinstance(extractor, EntityGNN):
            model.architecture_config.update(
                {
                    "architecture": "entity_gnn",
                    "gnn_type": extractor.gnn_type,
                    "gnn_layers": extractor.gnn_layers,
                    "graph": extractor.graph,
                    "graph_radius_nm": extractor.graph_radius_nm,
                }
            )
        if isinstance(extractor, EntityGNN) and extractor.graph == "knn":
            model.architecture_config["graph_k"] = extractor.graph_k
        if getattr(extractor, "pooling", None) == "additive_attention":
            model.architecture_config["attention_context"] = extractor.attention_context
        if temporal_extractor is not None:
            model.architecture_config.update(
                {
                    "temporal": temporal_extractor.temporal,
                    "history_length": temporal_extractor.history_length,
                }
            )
        metadata["architecture_config"] = model.architecture_config
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
        periodic = PeriodicCheckpoint(directory, checkpoint_interval, metadata)
        progress = ProgressBarCallback()
        learning_started = perf_counter()
        model.learn(
            total_timesteps=timesteps,
            callback=[history, periodic, progress],
            reset_num_timesteps=resume_from is None,
        )
        record_training_progress(metadata, model, history, learning_started)
        logger.dump(step=model.num_timesteps)
        save_run_checkpoint(model, env, model_path, directory)
        evaluation_started = perf_counter()
        if evaluation_seeds is None:
            rewards, lengths = evaluate(model, env, episodes, eval_seed, True)
        else:
            scenario_rows = evaluate_scenarios(model, env, evaluation_seeds)
            metadata["evaluation_scenarios"] = scenario_rows
            metadata["evaluation_episodes"] = len(scenario_rows)
            rewards = [row["return"] for row in scenario_rows]
            lengths = [row["length"] for row in scenario_rows]
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
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        if model is not None and learning_started is not None:
            record_training_progress(metadata, model, history, learning_started)
            save_run_checkpoint(model, env, model_path, directory)
            metadata["interrupted_checkpoint"] = str(
                (directory / "policy.zip").resolve()
            )
            logger.dump(step=model.num_timesteps)
            history.plot()
            print(f"Interrupted training saved: {model_path}; run report: {directory}")
        raise
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_metadata(directory, metadata)
        if progress is not None and hasattr(progress, "pbar"):
            progress.pbar.close()
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
    control_config=None,
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
        control_config: Requested settings, validated against the saved checkpoint.

    Returns:
        Mean and standard deviation of raw evaluation returns.

    Raises:
        FileNotFoundError: The policy or normalization file is missing.
    """
    model_path, stats_path = validate_checkpoint(model_path, stats_path)
    validate_control_config(model_path, control_config)

    base_env = env_factory(render_mode=render_mode)
    env = base_env
    try:
        env = VecNormalize.load(stats_path, base_env)
        model = model_type.load(model_path, env=env, device=device)
        validate_policy_config(model, control_config)
        validate_architecture_config(model, model_path)
        return evaluate(model, env, episodes, seed)
    finally:
        env.close()
