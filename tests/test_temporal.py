"""History boundaries, temporal semantics, normalization, and RL integration."""

from functools import partial

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from test_entities import TrafficEnv
from test_gnns import GraphTrafficEnv

from bluesky_pabloramesc.models import (
    build_ddpg,
    build_multi_ppo,
    build_multi_sac,
    build_ppo,
    build_sac,
    build_td3,
)
from bluesky_pabloramesc.temporal import (
    HistoryVecNormalize,
    HistoryWindow,
    TemporalFeatures,
)
from bluesky_pabloramesc.training import load_architecture_config, replay, train

METHODS = ["frame_stack", "gru", "lstm", "tcn"]


class CounterEnv(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self, duration=3, inactive_at=None):
        self.duration, self.inactive_at = duration, inactive_at

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.array([self.steps], dtype=np.float32),
            1.0,
            self.steps == self.duration,
            False,
            {"_temporal_inactive": self.steps == self.inactive_at},
        )


def test_history_terminal_observations_resets_and_agent_isolation():
    env = HistoryWindow(DummyVecEnv([lambda: CounterEnv(2), lambda: CounterEnv(4)]), 3)
    try:
        obs = env.reset()
        np.testing.assert_array_equal(obs[:, :, -1], [[0, 0, 1], [0, 0, 1]])
        env.step(np.zeros((2, 1)))
        obs, _, done, infos = env.step(np.zeros((2, 1)))
        assert done.tolist() == [True, False]
        np.testing.assert_array_equal(
            infos[0]["terminal_observation"], [[0, 1], [1, 1], [2, 1]]
        )
        np.testing.assert_array_equal(obs[0], [[0, 0], [0, 0], [0, 1]])
        np.testing.assert_array_equal(obs[1], [[0, 1], [1, 1], [2, 1]])
    finally:
        env.close()


def test_departed_aircraft_history_is_cleared():
    env = HistoryWindow(DummyVecEnv([lambda: CounterEnv(10, inactive_at=2)]), 3)
    try:
        env.reset()
        env.step(np.zeros((1, 1)))
        obs, _, _, _ = env.step(np.zeros((1, 1)))
        assert (obs == 0).all()
    finally:
        env.close()


@pytest.mark.parametrize("factory", [CounterEnv, GraphTrafficEnv])
def test_history_normalization_is_shared_frozen_and_serialized(factory, tmp_path):
    env = HistoryVecNormalize(HistoryWindow(DummyVecEnv([factory]), 4))
    try:
        env.reset()
        obs, _, _, _ = env.step(np.zeros((1, *env.action_space.shape)))
        # Only the latest observations enter statistics, not all overlapping history frames.
        assert env.obs_rms.count == pytest.approx(2.0001)
        np.testing.assert_array_equal(env.obs_rms.mean[0], env.obs_rms.mean[-1])
        np.testing.assert_array_equal(env.obs_rms.var[0], env.obs_rms.var[-1])
        assert (obs[:, :2] == 0).all()
        np.testing.assert_array_equal(obs[:, :, -1], [[0, 0, 1, 1]])
        if factory == GraphTrafficEnv:
            layout = env.observation_space.entity_layout
            np.testing.assert_array_equal(obs[0, -1, layout["mask"]], [1, 0, 1])
            np.testing.assert_array_equal(
                obs[0, -1, layout["graph_edges"]],
                factory().observation[layout["graph_edges"]],
            )
        env.save(tmp_path / "normalizer.pkl")
        restored = VecNormalize.load(
            tmp_path / "normalizer.pkl", HistoryWindow(DummyVecEnv([factory]), 4)
        )
        try:
            restored.training = False
            count = restored.obs_rms.count
            restored.reset()
            restored.step(np.zeros((1, *restored.action_space.shape)))
            assert restored.obs_rms.count == count
            raw = env.get_original_obs()
            np.testing.assert_allclose(restored.normalize_obs(raw), obs)
            np.testing.assert_allclose(env.unnormalize_obs(obs), raw, atol=1e-5)
        finally:
            restored.close()
    finally:
        env.close()


@pytest.mark.parametrize("method", METHODS)
def test_temporal_order_padding_empty_history_and_gradients(method):
    torch.manual_seed(5)
    env = HistoryWindow(DummyVecEnv([CounterEnv]), 4)
    try:
        model = TemporalFeatures(
            env.observation_space,
            env.observation_space.temporal_frame_space,
            FlattenExtractor,
            {},
            method,
            4,
        )
        obs = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        obs.requires_grad_()
        changed = obs.clone()
        changed[:, :2, 0] = float("nan")
        torch.testing.assert_close(model(obs), model(changed))
        reversed_obs = obs.clone()
        reversed_obs[0, 2:, 0] = obs[0, 2:, 0].flip(0)
        assert not torch.allclose(model(obs)[0], model(reversed_obs)[0])
        result = model(obs)
        assert (result[1] == 0).all() and torch.isfinite(result).all()
        # Repeated predictions have no hidden state carried between windows.
        torch.testing.assert_close(result, model(obs))
        result.square().sum().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
    finally:
        env.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "architecture", ["mlp", "entity_mlp", "entity_transformer", "entity_gnn"]
)
def test_spatial_temporal_combinations(method, architecture):
    env = HistoryVecNormalize(HistoryWindow(DummyVecEnv([GraphTrafficEnv]), 4))
    try:
        model = build_ppo(
            env,
            architecture=architecture,
            temporal=method,
            history_length=4,
            graph="knn" if architecture == "entity_gnn" else "fully_connected",
            pooling="mean",
            n_steps=4,
            n_epochs=1,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        model.learn(4)
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert np.isfinite(action).all()
        assert (
            model.policy.pi_features_extractor is not model.policy.vf_features_extractor
        )
        if architecture != "mlp":
            layout = env.observation_space.entity_layout
            permuted = obs.copy()
            aircraft, mask = np.array(layout["aircraft"]), np.array(layout["mask"])
            permuted[..., aircraft] = obs[..., aircraft[[2, 0, 1]]]
            permuted[..., mask] = obs[..., mask[[2, 0, 1]]]
            edges = obs[..., layout["graph_edges"]].reshape(1, 4, 3, 3)
            permuted[..., layout["graph_edges"]] = edges[:, :, [2, 0, 1]][
                :, :, :, [2, 0, 1]
            ].reshape(1, 4, 9)
            np.testing.assert_allclose(
                model.predict(permuted, deterministic=True)[0], action, atol=1e-6
            )
    finally:
        env.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "builder", [build_sac, build_td3, build_ddpg, build_multi_ppo, build_multi_sac]
)
def test_temporal_all_builders_and_roundtrip(method, builder, tmp_path):
    env = HistoryVecNormalize(HistoryWindow(DummyVecEnv([TrafficEnv]), 4))
    try:
        settings = (
            {}
            if builder in (build_multi_ppo, build_multi_sac)
            else {
                "learning_starts": 0,
                "buffer_size": 32,
                "batch_size": 2,
                "verbose": 0,
            }
        )
        model = builder(
            env,
            architecture="entity_mlp",
            pooling="mean",
            temporal=method,
            history_length=4,
            device="cpu",
            **settings,
        )
        if builder in (build_multi_ppo, build_multi_sac):
            model.learning_starts = 0
            if builder == build_multi_ppo:
                model.n_steps, model.batch_size, model.n_epochs = 4, 2, 1
                model.rollout_buffer.buffer_size = 4
                model.rollout_buffer.reset()
            else:
                model.train_freq = type(model.train_freq)(1, model.train_freq.unit)
                model.gradient_steps = 1
        model.learn(4)
        obs = env.reset()
        expected = model.predict(obs, deterministic=True)[0]
        model.save(tmp_path / "policy")
        restored = type(model).load(tmp_path / "policy", env=env, device="cpu")
        np.testing.assert_allclose(
            restored.predict(obs, deterministic=True)[0], expected, atol=1e-6
        )
        if hasattr(model, "replay_buffer"):
            assert (
                model.replay_buffer.observations.shape[2:]
                == env.observation_space.shape
            )
            assert set(np.unique(model.replay_buffer.observations[..., -1])) <= {0, 1}
    finally:
        env.close()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("algorithm", [PPO, SAC])
def test_temporal_training_checkpoint_evaluation_and_resume(
    method, algorithm, tmp_path
):
    factory = lambda render_mode=None: HistoryWindow(DummyVecEnv([CounterEnv]), 4)
    settings = (
        {"n_steps": 4, "n_epochs": 1}
        if algorithm == PPO
        else {"learning_starts": 0, "buffer_size": 32}
    )
    builder = partial(
        build_ppo if algorithm == PPO else build_sac,
        temporal=method,
        history_length=4,
        batch_size=2,
        verbose=0,
        **settings,
    )
    path = tmp_path / "policy"
    train(
        builder,
        factory,
        path,
        4,
        episodes=1,
        device="cpu",
        results_dir=tmp_path / "reports",
    )
    config = load_architecture_config(path)
    assert config == {
        "architecture": "mlp",
        "pooling": "mean_max",
        "temporal": method,
        "history_length": 4,
    }
    assert (
        replay(algorithm, factory, path, episodes=1, device="cpu", render_mode=None)[0]
        == 3
    )
    train(
        None,
        factory,
        tmp_path / "resumed",
        4,
        resume_from=path,
        model_type=algorithm,
        episodes=1,
        device="cpu",
        results_dir=tmp_path / "reports",
    )
    assert load_architecture_config(tmp_path / "resumed") == config


def test_tcn_does_not_use_future_frames():
    torch.manual_seed(9)
    env = HistoryWindow(DummyVecEnv([CounterEnv]), 8)
    try:
        model = TemporalFeatures(
            env.observation_space,
            env.observation_space.temporal_frame_space,
            FlattenExtractor,
            {},
            "tcn",
            8,
        )
        captured = []
        hook = model.convolutions[-1].register_forward_hook(
            lambda module, args, result: captured.append(result.detach().clone())
        )
        obs = torch.ones(1, 8, 2)
        model(obs)
        changed = obs.clone()
        changed[:, 4:, 0] += 10
        model(changed)
        hook.remove()
        torch.testing.assert_close(captured[0][:, :, :4], captured[1][:, :, :4])
        assert not torch.allclose(captured[0][:, :, -1], captured[1][:, :, -1])
    finally:
        env.close()


def test_black_death_marks_departed_agents_for_history():
    from test_multi_agent import DepartingAgents

    from bluesky_pabloramesc.multi_agent import FlattenObs, JointEpisodeBlackDeath

    env = JointEpisodeBlackDeath(FlattenObs(DepartingAgents()), track_history=True)
    env.reset()
    actions = {agent: np.zeros(1, dtype=np.float32) for agent in env.possible_agents}
    _, _, _, _, infos = env.step(actions)
    assert infos["first"]["_temporal_inactive"]
    assert not infos["second"]["_temporal_inactive"]
    env.close()


def test_temporal_builder_rejects_missing_or_mismatched_history():
    raw = DummyVecEnv([CounterEnv])
    history = HistoryWindow(DummyVecEnv([CounterEnv]), 4)
    try:
        with pytest.raises(ValueError, match="HistoryWindow"):
            build_ppo(raw, temporal="gru")
        with pytest.raises(ValueError, match="requires a temporal"):
            build_ppo(history)
        with pytest.raises(ValueError, match="shapes disagree"):
            build_ppo(history, temporal="gru", history_length=8)
    finally:
        raw.close()
        history.close()
