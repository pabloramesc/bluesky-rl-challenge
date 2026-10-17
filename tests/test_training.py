import csv
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import VecNormalize

from bluesky_pabloramesc.models import build_ddpg, build_ppo, build_sac, build_td3
from bluesky_pabloramesc.training import checkpoint_paths, evaluate, replay, train


@pytest.mark.parametrize("name", ["policy", "policy.zip", "policy.v2", "policy.v2.zip"])
def test_checkpoint_names_preserve_dots(name):
    model, stats = checkpoint_paths(Path("models") / name)
    stem = name.removesuffix(".zip")
    assert model == Path(f"models/{stem}.zip")
    assert stats == Path(f"models/{stem}_vecnormalize.pkl")


@pytest.mark.parametrize(
    ("builder", "model_type", "settings"),
    [
        (build_ppo, PPO, {"n_steps": 8, "batch_size": 4, "n_epochs": 1}),
        (build_sac, SAC, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
        (build_td3, TD3, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
        (build_ddpg, DDPG, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
    ],
)
def test_train_and_replay_checkpoint(
    tmp_path, env_factory, builder, model_type, settings, monkeypatch
):
    progress_bars = []
    original_start = ProgressBarCallback._on_training_start

    def record_progress_bar(callback):
        original_start(callback)
        progress_bars.append(callback.pbar)

    monkeypatch.setattr(ProgressBarCallback, "_on_training_start", record_progress_bar)
    path = tmp_path / "models" / "policy.v2"
    builder = partial(builder, verbose=0, **settings)
    mean, _ = train(
        builder,
        env_factory,
        path,
        timesteps=8,
        device="cpu",
        episodes=2,
        results_dir=tmp_path / "results",
    )
    assert mean == 6.0  # Raw rewards, even though training normalizes them.
    reports = list((tmp_path / "results").iterdir())
    assert len(reports) == 1
    report = reports[0]
    metadata = json.loads((report / "run.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["algorithm"] == model_type.__name__
    assert metadata["training_seconds"] > 0
    assert metadata["actual_timesteps"] == 8
    assert metadata["evaluation"]["episode_returns"] == [6.0, 6.0]
    assert metadata["evaluation"]["mean_return"] == 6.0
    assert metadata["settings"]["gamma"] == 0.995
    assert (report / "performance.png").read_bytes().startswith(b"\x89PNG")
    assert (report / "progress.csv").exists()
    with (report / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert [float(row["return"]) for row in rows] == [6.0, 6.0]
    assert [int(row["length"]) for row in rows] == [3, 3]
    assert len(progress_bars) == 1
    assert progress_bars[0].n == 8

    model_path, stats_path = checkpoint_paths(path)
    assert model_path.is_file()
    assert stats_path.is_file()
    env = VecNormalize.load(stats_path, env_factory())
    try:
        model = model_type.load(model_path, env=env, device="cpu")
        assert model.num_timesteps == 8
        assert model._n_updates > 0
        action, _ = model.predict(env.reset(), deterministic=True)
        assert action.shape == (1, 2)
        assert np.isfinite(action).all()
    finally:
        env.close()

    mean, _ = replay(
        model_type, env_factory, report / "policy", device="cpu", episodes=2
    )
    assert mean == 6.0


def test_evaluation_freezes_statistics_and_restores_flags(env_factory):
    env = VecNormalize(env_factory())
    try:
        model = build_ppo(env, device="cpu", verbose=0)
        before = env.obs_rms.count
        assert evaluate(model, env, episodes=2)[0] == 6.0
        assert env.obs_rms.count == before
        assert env.training is True
        assert env.norm_reward is True
    finally:
        env.close()


def test_evaluation_restores_flags_after_failure(env_factory):
    class BrokenPolicy:
        def predict(self, *args, **kwargs):
            raise RuntimeError("prediction failed")

    env = VecNormalize(env_factory())
    try:
        with pytest.raises(RuntimeError, match="prediction failed"):
            evaluate(BrokenPolicy(), env)
        assert env.training is True
        assert env.norm_reward is True
    finally:
        env.close()


def test_missing_checkpoint_does_not_start_simulator(tmp_path):
    def unexpected_env(**kwargs):
        pytest.fail("Environment started before checkpoint validation")

    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        replay(PPO, unexpected_env, tmp_path / "missing")


def test_training_closes_environment_on_failure(env_factory, tmp_path):
    env = env_factory()

    def broken_builder(*args, **kwargs):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        train(
            broken_builder,
            lambda: env,
            tmp_path / "unused",
            timesteps=8,
            results_dir=tmp_path / "results",
        )
    assert env.envs[0].unwrapped.closed


def test_training_uses_separate_evaluation_seed(tmp_path, env_factory):
    seeds = []
    env = env_factory()
    original_seed = env.seed

    def record_seed(seed=None):
        seeds.append(seed)
        return original_seed(seed)

    env.seed = record_seed
    builder = partial(build_ppo, n_steps=8, batch_size=4, n_epochs=1, verbose=0)
    train(
        builder,
        lambda: env,
        tmp_path / "policy",
        timesteps=8,
        seed=7,
        eval_seed=123,
        results_dir=tmp_path / "results",
        device="cpu",
        episodes=2,
    )
    assert seeds == [7, 123]


def test_headless_evaluation_uses_saved_statistics(tmp_path, env_factory):
    path = tmp_path / "policy"
    builder = partial(build_ppo, n_steps=8, batch_size=4, n_epochs=1, verbose=0)
    train(
        builder,
        env_factory,
        path,
        timesteps=8,
        device="cpu",
        episodes=2,
        results_dir=tmp_path / "results",
    )
    _, stats_path = checkpoint_paths(path)
    saved_statistics = stats_path.read_bytes()
    render_modes = []

    def make_env(render_mode=None):
        render_modes.append(render_mode)
        return env_factory()

    assert (
        replay(PPO, make_env, path, device="cpu", episodes=2, render_mode=None)[0]
        == 6.0
    )
    assert render_modes == [None]
    assert stats_path.read_bytes() == saved_statistics


def test_repeated_runs_preserve_reports(tmp_path, env_factory):
    builder = partial(build_ppo, n_steps=8, batch_size=4, n_epochs=1, verbose=0)
    for _ in range(2):
        train(
            builder,
            env_factory,
            tmp_path / "policy",
            8,
            device="cpu",
            episodes=1,
            results_dir=tmp_path / "results",
        )
    reports = list((tmp_path / "results").iterdir())
    assert len(reports) == 2
    assert all((report / "policy.zip").is_file() for report in reports)


def test_failed_run_keeps_metadata(tmp_path, env_factory):
    def broken_builder(*args, **kwargs):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        train(
            broken_builder,
            env_factory,
            tmp_path / "policy",
            8,
            results_dir=tmp_path / "results",
        )
    report = next((tmp_path / "results").iterdir())
    metadata = json.loads((report / "run.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["error"] == "RuntimeError: model failed"
    assert "finished_at" in metadata


def test_report_without_completed_episodes(tmp_path, env_factory):
    builder = partial(build_ppo, n_steps=2, batch_size=2, n_epochs=1, verbose=0)
    train(
        builder,
        env_factory,
        tmp_path / "policy",
        2,
        device="cpu",
        episodes=1,
        results_dir=tmp_path / "results",
    )
    report = next((tmp_path / "results").iterdir())
    with (report / "episodes.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
    assert (report / "performance.png").is_file()
    metadata = json.loads((report / "run.json").read_text())
    assert metadata["training_episodes"] == 0
    assert metadata["evaluation"]["mean_return"] == 6.0


@pytest.mark.parametrize(
    ("builder", "model_type", "settings"),
    [
        (build_ppo, PPO, {"n_steps": 8, "batch_size": 4, "n_epochs": 1}),
        (build_sac, SAC, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
        (build_td3, TD3, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
        (build_ddpg, DDPG, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
    ],
)
def test_resume_adds_steps_and_restores_training_state(
    tmp_path, env_factory, builder, model_type, settings
):
    from bluesky_pabloramesc.training import replay_buffer_path

    source = tmp_path / "original"
    destination = tmp_path / "continued"
    results = tmp_path / "results"
    builder = partial(builder, verbose=0, **settings)
    train(
        builder, env_factory, source, 8, device="cpu", episodes=1, results_dir=results
    )
    source_model, source_stats = checkpoint_paths(source)
    original_bytes = source_model.read_bytes()
    previous = model_type.load(source_model, device="cpu")
    old_env = VecNormalize.load(source_stats, env_factory())
    try:
        old_count = old_env.obs_rms.count
    finally:
        old_env.close()

    def unexpected_builder(*args, **kwargs):
        pytest.fail("Resume must load the existing policy, not construct a new one")

    train(
        unexpected_builder,
        env_factory,
        destination,
        8,
        device="cpu",
        episodes=1,
        results_dir=results,
        resume_from=source,
        model_type=model_type,
    )
    model_path, stats_path = checkpoint_paths(destination)
    model = model_type.load(model_path, device="cpu")
    assert model.num_timesteps == 16
    assert model._n_updates > previous._n_updates
    assert source_model.read_bytes() == original_bytes
    env = VecNormalize.load(stats_path, env_factory())
    try:
        assert env.obs_rms.count > old_count
    finally:
        env.close()
    if model_type != PPO:
        model.load_replay_buffer(replay_buffer_path(destination))
        assert model.replay_buffer.size() == 16
    metadata = [
        json.loads((path / "run.json").read_text()) for path in results.iterdir()
    ]
    resumed = next(row for row in metadata if "resumed_from" in row)
    assert resumed["initial_timesteps"] == 8
    assert resumed["added_timesteps"] == 8
    assert resumed["actual_timesteps"] == 16
    assert resumed["resumed_from"] == str(source_model.resolve())
    if model_type != PPO:
        assert resumed["replay_buffer_restored"] is True


def test_resume_legacy_checkpoint_without_replay_buffer(tmp_path, env_factory):
    from bluesky_pabloramesc.training import replay_buffer_path

    path = tmp_path / "legacy"
    results = tmp_path / "results"
    builder = partial(
        build_td3, learning_starts=2, batch_size=2, buffer_size=32, verbose=0
    )
    train(builder, env_factory, path, 8, device="cpu", episodes=1, results_dir=results)
    replay_buffer_path(path).unlink()
    with pytest.warns(RuntimeWarning, match="empty buffer"):
        train(
            builder,
            env_factory,
            path,
            8,
            device="cpu",
            episodes=1,
            results_dir=results,
            resume_from=path,
            model_type=TD3,
        )
    model = TD3.load(checkpoint_paths(path)[0], device="cpu")
    assert model.num_timesteps == 16
    assert model.learning_starts == 10
    model.load_replay_buffer(replay_buffer_path(path))
    assert model.replay_buffer.size() == 8


def test_resume_missing_checkpoint_does_not_start_simulator(tmp_path):
    def unexpected_env():
        pytest.fail("Environment started before checkpoint validation")

    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        train(
            build_ppo,
            unexpected_env,
            tmp_path / "output",
            8,
            resume_from=tmp_path / "missing",
            model_type=PPO,
        )


@pytest.mark.parametrize(
    ("builder", "model_type", "settings"),
    [
        (build_ppo, PPO, {"n_steps": 8, "batch_size": 4, "n_epochs": 1}),
        (build_sac, SAC, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
        (build_td3, TD3, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
        (build_ddpg, DDPG, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
    ],
)
def test_periodic_checkpoints_can_resume(
    tmp_path, env_factory, builder, model_type, settings
):
    from bluesky_pabloramesc.training import replay_buffer_path

    results = tmp_path / "results"
    builder = partial(builder, verbose=0, **settings)
    train(
        builder,
        env_factory,
        tmp_path / "policy",
        8,
        device="cpu",
        episodes=1,
        results_dir=results,
        checkpoint_interval=4,
    )
    report = next(results.iterdir())
    snapshots = sorted((report / "checkpoints").iterdir())
    expected_steps = [8] if model_type == PPO else [4, 8]
    assert [
        int(path.name.removeprefix("step_")) for path in snapshots
    ] == expected_steps
    for directory, step in zip(snapshots, expected_steps):
        model_path, stats_path = checkpoint_paths(directory / "policy")
        model = model_type.load(model_path, device="cpu")
        assert model.num_timesteps == step
        env = VecNormalize.load(stats_path, env_factory())
        try:
            assert env.obs_rms.count == pytest.approx(step + 1.0001)
        finally:
            env.close()
        if model_type != PPO:
            model.load_replay_buffer(replay_buffer_path(model_path))
            # The current transition is already stored before the periodic save.
            assert model.replay_buffer.size() == step
    metadata = json.loads((report / "run.json").read_text())
    assert metadata["checkpoint_timesteps"] == 8
    assert metadata["latest_checkpoint"] == str(
        (snapshots[-1] / "policy.zip").resolve()
    )
    train(
        builder,
        env_factory,
        tmp_path / "resumed",
        8,
        device="cpu",
        episodes=1,
        results_dir=results,
        resume_from=snapshots[-1] / "policy",
        model_type=model_type,
        checkpoint_interval=4,
    )
    model = model_type.load(checkpoint_paths(tmp_path / "resumed")[0], device="cpu")
    assert model.num_timesteps == 16


@pytest.mark.parametrize(
    ("builder", "model_type", "settings"),
    [
        (build_ppo, PPO, {"n_steps": 8, "batch_size": 4, "n_epochs": 1}),
        (build_sac, SAC, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
        (build_td3, TD3, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
        (build_ddpg, DDPG, {"learning_starts": 2, "batch_size": 2, "buffer_size": 32}),
    ],
)
def test_ctrl_c_saves_state_and_can_resume(
    tmp_path, env_factory, monkeypatch, builder, model_type, settings
):
    from bluesky_pabloramesc.training import replay_buffer_path

    interrupted = False
    original_train = model_type.train

    def interrupt_after_update(model, *args, **kwargs):
        nonlocal interrupted
        original_train(model, *args, **kwargs)
        if not interrupted and model.num_timesteps >= 8:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(model_type, "train", interrupt_after_update)
    env = env_factory()
    path = tmp_path / "policy"
    results = tmp_path / "results"
    builder = partial(builder, verbose=0, **settings)
    with pytest.raises(KeyboardInterrupt):
        train(
            builder,
            lambda: env,
            path,
            32,
            device="cpu",
            episodes=1,
            results_dir=results,
            checkpoint_interval=4,
        )
    assert env.envs[0].unwrapped.closed
    report = next(results.iterdir())
    metadata = json.loads((report / "run.json").read_text())
    assert metadata["status"] == "interrupted"
    assert metadata["actual_timesteps"] == metadata["added_timesteps"] == 8
    assert metadata["training_seconds"] > 0
    assert metadata["training_episodes"] == 2
    assert "finished_at" in metadata
    assert "evaluation" not in metadata
    assert (report / "performance.png").is_file()
    model_path, stats_path = checkpoint_paths(path)
    model = model_type.load(model_path, device="cpu")
    assert model.num_timesteps == 8
    assert model._n_updates > 0
    restored_env = VecNormalize.load(stats_path, env_factory())
    try:
        assert restored_env.obs_rms.count == pytest.approx(9.0001)
    finally:
        restored_env.close()
    if model_type != PPO:
        model.load_replay_buffer(replay_buffer_path(model_path))
        assert model.replay_buffer.size() == 8
    train(
        builder,
        env_factory,
        tmp_path / "resumed",
        8,
        device="cpu",
        episodes=1,
        results_dir=results,
        resume_from=report / "policy",
        model_type=model_type,
        checkpoint_interval=4,
    )
    resumed = model_type.load(checkpoint_paths(tmp_path / "resumed")[0], device="cpu")
    assert resumed.num_timesteps == 16
    assert resumed._n_updates > model._n_updates


def test_periodic_checkpoints_count_all_vector_slots(tmp_path, env_factory):
    builder = partial(build_ppo, n_steps=2, batch_size=10, n_epochs=1, verbose=0)
    train(
        builder,
        lambda: env_factory(n_envs=10),
        tmp_path / "policy",
        40,
        device="cpu",
        episodes=1,
        results_dir=tmp_path / "results",
        checkpoint_interval=15,
    )
    report = next((tmp_path / "results").iterdir())
    checkpoints = report / "checkpoints"
    assert sorted(path.name for path in checkpoints.iterdir()) == ["step_20", "step_40"]
    model = PPO.load(checkpoints / "step_20" / "policy.zip", device="cpu")
    assert model.num_timesteps == 20
    assert model.n_envs == 10


def test_resume_checkpoint_interval_starts_from_loaded_steps(tmp_path, env_factory):
    builder = partial(
        build_td3, learning_starts=2, batch_size=2, buffer_size=32, verbose=0
    )
    results = tmp_path / "results"
    source = tmp_path / "source"
    train(
        builder, env_factory, source, 8, device="cpu", episodes=1, results_dir=results
    )
    train(
        builder,
        env_factory,
        tmp_path / "resumed",
        4,
        device="cpu",
        episodes=1,
        results_dir=results,
        resume_from=source,
        model_type=TD3,
        checkpoint_interval=3,
    )
    report = next(
        path for path in results.iterdir() if path.name.startswith("resumed_")
    )
    assert [path.name for path in (report / "checkpoints").iterdir()] == ["step_11"]


def test_real_sigint_saves_a_loadable_checkpoint(tmp_path):
    import subprocess
    import sys

    script = """
import signal
import sys
from functools import partial
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from bluesky_pabloramesc.models import build_td3
from bluesky_pabloramesc.training import train

torch.set_num_threads(1)
root = Path(sys.argv[1])
original_train = TD3.train

def interrupt_after_update(self, *args, **kwargs):
    original_train(self, *args, **kwargs)
    signal.raise_signal(signal.SIGINT)

TD3.train = interrupt_after_update
builder = partial(build_td3, learning_starts=1, batch_size=2, buffer_size=16, verbose=0)
env_factory = lambda: DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])
try:
    train(builder, env_factory, root / "policy", 16, device="cpu", episodes=1,
          results_dir=root / "results", checkpoint_interval=1)
except KeyboardInterrupt:
    pass
else:
    raise AssertionError("SIGINT did not interrupt training")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = next((tmp_path / "results").iterdir())
    metadata = json.loads((report / "run.json").read_text())
    assert metadata["status"] == "interrupted"
    model = TD3.load(tmp_path / "policy.zip", device="cpu")
    assert model.num_timesteps == 2
    model.load_replay_buffer(tmp_path / "policy_replay_buffer.pkl")
    assert model.replay_buffer.size() == 2
    assert (tmp_path / "policy_vecnormalize.pkl").is_file()


@pytest.mark.parametrize("interval", [0, -1])
def test_invalid_checkpoint_interval_does_not_start_environment(tmp_path, interval):
    def unexpected_env():
        pytest.fail("Environment started before interval validation")

    with pytest.raises(ValueError, match="checkpoint_interval must be positive"):
        train(
            build_ppo,
            unexpected_env,
            tmp_path / "policy",
            8,
            checkpoint_interval=interval,
            results_dir=tmp_path / "results",
        )
    assert not (tmp_path / "results").exists()


def test_interrupt_before_model_initialization_closes_environment(
    tmp_path, env_factory
):
    env = env_factory()

    def interrupted_builder(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train(
            interrupted_builder,
            lambda: env,
            tmp_path / "policy",
            8,
            results_dir=tmp_path / "results",
        )
    assert env.envs[0].unwrapped.closed
    report = next((tmp_path / "results").iterdir())
    assert json.loads((report / "run.json").read_text())["status"] == "interrupted"
    assert not (report / "policy.zip").exists()


def test_control_settings_survive_every_checkpoint(tmp_path, env_factory):
    from bluesky_pabloramesc.actions import ControlConfig
    from bluesky_pabloramesc.training import (
        load_control_config,
        validate_control_config,
    )

    config = ControlConfig(observation_profile="control")
    builder = partial(build_ppo, n_steps=8, batch_size=4, n_epochs=1, verbose=0)
    train(
        builder,
        env_factory,
        tmp_path / "policy",
        8,
        device="cpu",
        episodes=1,
        results_dir=tmp_path / "reports",
        checkpoint_interval=4,
        control_config=config,
    )
    report = next((tmp_path / "reports").iterdir())
    for path in [
        tmp_path / "policy",
        report / "policy",
        report / "checkpoints" / "step_8" / "policy",
    ]:
        assert load_control_config(path) == config
        with pytest.raises(ValueError, match="configuration"):
            validate_control_config(path, ControlConfig())

    def unexpected_env(**kwargs):
        pytest.fail("configuration mismatch must fail before simulator startup")

    with pytest.raises(ValueError, match="configuration"):
        replay(PPO, unexpected_env, tmp_path / "policy", control_config=ControlConfig())


def test_explicit_scenarios_freeze_statistics_and_pair_resets(env_factory):
    from bluesky_pabloramesc.training import evaluate_scenarios

    env = VecNormalize(env_factory())
    try:
        model = build_ppo(env, device="cpu", verbose=0)
        count = env.obs_rms.count
        rows = evaluate_scenarios(model, env, [123, 456])
        assert [row["scenario_seed"] for row in rows] == [123, 456]
        assert [row["return"] for row in rows] == [6, 6]
        assert [row["length"] for row in rows] == [3, 3]
        assert env.obs_rms.count == count
        assert env.training and env.norm_reward
    finally:
        env.close()


def test_scenario_evaluation_preserves_departed_agent_metrics():
    from bluesky_pabloramesc.training import evaluate_scenarios

    class JointEnv:
        num_envs = 2
        training = True
        norm_reward = True

        def seed(self, seed):
            self.scenario_seed = seed

        def reset(self):
            self.steps = 0
            return np.zeros((2, 1))

        def step(self, actions):
            self.steps += 1
            first_info = (
                {"flight_time": 1, "waypoint_reached": 1} if self.steps == 1 else {}
            )
            second_info = {"flight_time": self.steps, "waypoint_reached": 0}
            return (
                np.zeros((2, 1)),
                np.array([2, 3]),
                np.full(2, self.steps == 3),
                [first_info, second_info],
            )

    class Policy:
        def predict(self, observations, deterministic):
            assert deterministic
            return np.zeros((2, 2)), None

    env = JointEnv()
    first, second = evaluate_scenarios(Policy(), env, [42])
    assert first["flight_time"] == 1
    assert first["return"] == 2
    assert first["length"] == 1
    assert first["timed_out"] == 0
    assert second["flight_time"] == 3
    assert second["return"] == 9
    assert second["timed_out"] == 1
    assert env.training and env.norm_reward


def test_policy_config_defaults_encoding_for_older_archives():
    from types import SimpleNamespace

    from bluesky_pabloramesc.actions import ControlConfig
    from bluesky_pabloramesc.training import validate_policy_config

    config = ControlConfig()
    saved = config.to_dict()
    del saved["observation_encoding"]
    model = SimpleNamespace(control_config=saved)
    validate_policy_config(model, config)
    with pytest.raises(ValueError, match="configuration"):
        validate_policy_config(
            model, ControlConfig(observation_encoding="goal_aligned")
        )
