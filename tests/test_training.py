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
