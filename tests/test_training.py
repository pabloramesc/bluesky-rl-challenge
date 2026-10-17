from functools import partial
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize

from bluesky_pabloramesc.models import build_ppo, build_sac
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
    ],
)
def test_train_and_replay_checkpoint(
    tmp_path, env_factory, builder, model_type, settings
):
    path = tmp_path / "models" / "policy.v2"
    builder = partial(builder, verbose=0, **settings)
    mean, _ = train(builder, env_factory, path, timesteps=8, device="cpu", episodes=2)
    assert mean == 6.0  # Raw rewards, even though training normalizes them.

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

    mean, _ = replay(model_type, env_factory, path, device="cpu", episodes=2)
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


def test_training_closes_environment_on_failure(env_factory):
    env = env_factory()

    def broken_builder(*args, **kwargs):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        train(broken_builder, lambda: env, "unused", timesteps=8)
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
        device="cpu",
        episodes=2,
    )
    assert seeds == [7, 123]


def test_headless_evaluation_uses_saved_statistics(tmp_path, env_factory):
    path = tmp_path / "policy"
    builder = partial(build_ppo, n_steps=8, batch_size=4, n_epochs=1, verbose=0)
    train(builder, env_factory, path, timesteps=8, device="cpu", episodes=2)
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
