"""Graph encoders preserve set symmetry and implement canonical GCN/GAT updates."""

from functools import partial

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from test_entities import TrafficEnv

from bluesky_pabloramesc.entities import (
    EntityGNN,
    EntityVecNormalize,
    GraphAttention,
    GraphConvolution,
)
from bluesky_pabloramesc.models import (
    build_ddpg,
    build_multi_ppo,
    build_multi_sac,
    build_ppo,
    build_sac,
    build_td3,
)
from bluesky_pabloramesc.observations import distance_adjacency
from bluesky_pabloramesc.training import load_architecture_config, replay, train


class GraphTrafficEnv(TrafficEnv):
    def __init__(self):
        super().__init__()
        space = self.observation_space
        layout = dict(space.entity_layout)
        size = space.shape[0]
        adjacency = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float32)
        self.observation = np.concatenate([self.observation, adjacency.ravel()])
        self.observation_space = spaces.Box(
            np.concatenate([space.low, np.zeros(9)]).astype(np.float32),
            np.concatenate([space.high, np.ones(9)]).astype(np.float32),
        )
        layout["graph_edges"] = list(range(size, size + 9))
        self.observation_space.entity_layout = layout


def test_graph_layer_numerical_examples():
    values = torch.arange(3 * 64, dtype=torch.float32).reshape(1, 3, 64) / 100
    adjacency = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, True]]]
    )
    expected = values.clone()
    expected[:, :2] = values[:, :2].mean(1, keepdim=True)
    for layer in (GraphConvolution(), GraphAttention()):
        with torch.no_grad():
            layer.projection.weight.copy_(torch.eye(64))
            if isinstance(layer, GraphAttention):
                layer.source_score.zero_()
                layer.target_score.zero_()
        torch.testing.assert_close(layer(values, adjacency), expected)
    # Nonuniform degrees distinguish symmetric GCN normalization from row averaging.
    gcn = GraphConvolution()
    with torch.no_grad():
        gcn.projection.weight.copy_(torch.eye(64))
    edges = torch.tensor(
        [[[True, True, False], [True, True, True], [False, True, True]]]
    )
    expected = edges.float() / torch.sqrt(
        torch.tensor([2.0, 3.0, 2.0])[None, :, None]
        * torch.tensor([2.0, 3.0, 2.0])[None, None, :]
    )
    torch.testing.assert_close(gcn(values, edges), expected @ values)


@pytest.mark.parametrize("gnn_type", ["gcn", "gat"])
@pytest.mark.parametrize("graph", ["fully_connected", "distance", "knn"])
@pytest.mark.parametrize("layers", [1, 2, 3])
@pytest.mark.parametrize(
    "pooling,context",
    [
        ("mean", "ownship_goal"),
        ("max", "ownship_goal"),
        ("mean_max", "ownship_goal"),
        ("additive_attention", "ownship_goal"),
        ("additive_attention", "all"),
        ("pma", "ownship_goal"),
    ],
)
def test_gnn_symmetry_masking_and_gradients(gnn_type, graph, layers, pooling, context):
    torch.manual_seed(7)
    env = GraphTrafficEnv()
    layout = env.observation_space.entity_layout
    model = EntityGNN(
        env.observation_space,
        layout,
        pooling,
        context,
        gnn_type=gnn_type,
        gnn_layers=layers,
        graph=graph,
    )
    obs = torch.tensor(np.stack([env.observation] * 2))
    aircraft = np.array(layout["aircraft"])
    obs[0, aircraft[2]] = 3
    obs[1, layout["mask"]] = 0
    permutation = np.array([2, 0, 1])
    changed = obs.clone()
    changed[:, aircraft] = obs[:, aircraft[permutation]]
    changed[:, layout["mask"]] = obs[:, np.array(layout["mask"])[permutation]]
    edges = obs[:, layout["graph_edges"]].reshape(-1, 3, 3)
    changed[:, layout["graph_edges"]] = edges[:, permutation][
        :, :, permutation
    ].reshape(-1, 9)
    torch.testing.assert_close(model(obs), model(changed))
    changed = obs.clone()
    changed[:, aircraft[1]] = float("nan")
    torch.testing.assert_close(model(obs), model(changed))
    result = model(obs)
    assert (result[1, len(layout["context"]) :] == 0).all()
    result.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(obs), model(changed))


def test_distance_adjacency_units_and_padding():
    obs = {
        "x_r": np.array([0, 1852, 10_000, 0]) / 1_000_000,
        "y_r": np.zeros(4),
        "intruder_present": np.array([1, 1, 1, 0]),
    }
    edges = distance_adjacency(obs, 1_000_000, 1)
    np.testing.assert_array_equal(
        edges, [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    )


def test_graph_edges_bypass_normalization():
    env = EntityVecNormalize(DummyVecEnv([GraphTrafficEnv]))
    try:
        obs = env.reset()
        indices = env.obs_rms.layout["graph_edges"]
        np.testing.assert_array_equal(
            obs[:, indices], [GraphTrafficEnv().observation[indices]]
        )
        np.testing.assert_array_equal(
            env.unnormalize_obs(obs)[:, indices], obs[:, indices]
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    "builder",
    [build_ppo, build_sac, build_td3, build_ddpg, build_multi_ppo, build_multi_sac],
)
@pytest.mark.parametrize("gnn_type", ["gcn", "gat"])
@pytest.mark.parametrize("graph", ["distance", "knn"])
def test_gnn_builder_roundtrip(builder, gnn_type, tmp_path, graph):
    env = EntityVecNormalize(DummyVecEnv([GraphTrafficEnv]))
    try:
        model = builder(
            env,
            architecture="entity_gnn",
            gnn_type=gnn_type,
            gnn_layers=2,
            graph=graph,
            graph_k=2,
            pooling="pma",
            device="cpu",
        )
        obs = env.reset()
        expected = model.predict(obs, deterministic=True)[0]
        if hasattr(model.policy, "pi_features_extractor"):
            assert (
                model.policy.pi_features_extractor
                is not model.policy.vf_features_extractor
            )
        else:
            assert model.actor.features_extractor is not model.critic.features_extractor
        model.save(tmp_path / "policy")
        restored = type(model).load(tmp_path / "policy", env=env, device="cpu")
        np.testing.assert_allclose(
            restored.predict(obs, deterministic=True)[0], expected
        )
    finally:
        env.close()


@pytest.mark.parametrize("builder", [build_ppo, build_sac, build_td3, build_ddpg])
@pytest.mark.parametrize("gnn_type", ["gcn", "gat"])
def test_gnn_training(builder, gnn_type):
    env = EntityVecNormalize(DummyVecEnv([TrafficEnv]))
    try:
        settings = (
            {"n_steps": 4, "n_epochs": 1}
            if builder == build_ppo
            else {"buffer_size": 32, "learning_starts": 0}
        )
        model = builder(
            env,
            architecture="entity_gnn",
            gnn_type=gnn_type,
            batch_size=2,
            device="cpu",
            verbose=0,
            **settings,
        )
        model.learn(4)
        assert np.isfinite(model.predict(env.reset(), deterministic=True)[0]).all()
    finally:
        env.close()


@pytest.mark.parametrize("gnn_type", ["gcn", "gat"])
@pytest.mark.parametrize("graph", ["fully_connected", "distance", "knn"])
def test_gnn_checkpoint_resume_and_evaluate(tmp_path, monkeypatch, gnn_type, graph):
    monkeypatch.setattr(
        GraphTrafficEnv,
        "step",
        lambda self, action: (self.observation.copy(), 1.0, True, False, {}),
    )
    factory = lambda render_mode=None: DummyVecEnv([GraphTrafficEnv])
    path = tmp_path / "policy"
    train(
        partial(
            build_ppo,
            architecture="entity_gnn",
            gnn_type=gnn_type,
            gnn_layers=2,
            graph=graph,
            graph_radius_nm=10.0,
            graph_k=2,
            pooling="additive_attention",
            attention_context="all",
            n_steps=4,
            n_epochs=1,
            batch_size=2,
            verbose=0,
        ),
        factory,
        path,
        4,
        episodes=1,
        device="cpu",
        results_dir=tmp_path / "results",
    )
    assert load_architecture_config(path) == {
        "architecture": "entity_gnn",
        "gnn_type": gnn_type,
        "gnn_layers": 2,
        "graph": graph,
        "graph_radius_nm": 10.0,
        **({"graph_k": 2} if graph == "knn" else {}),
        "pooling": "additive_attention",
        "attention_context": "all",
    }
    assert (
        replay(PPO, factory, path, episodes=1, device="cpu", render_mode=None)[0] == 1
    )
    train(
        None,
        factory,
        tmp_path / "resumed",
        4,
        resume_from=path,
        model_type=PPO,
        episodes=1,
        device="cpu",
        results_dir=tmp_path / "results",
    )
    assert load_architecture_config(tmp_path / "resumed") == load_architecture_config(
        path
    )


@pytest.mark.parametrize(
    "encoding",
    [
        "upstream",
        "ego_cartesian",
        "goal_aligned",
        "conflict",
        "ego_cartesian_conflict",
        "goal_aligned_conflict",
    ],
)
@pytest.mark.parametrize("graph", ["distance", "knn"])
def test_distance_graph_hooks_and_observation_layout(encoding, graph):
    from types import SimpleNamespace

    from test_observations import geometry

    from bluesky_pabloramesc.actions import ControlConfig
    from bluesky_pabloramesc.entities import entity_layout
    from bluesky_pabloramesc.observations import ControlHooks, extend_observation_space

    original = geometry()
    original["x_r"] = np.array([0.0, 1852.0]) / 1_000_000
    original["cos_track"][:] = 1
    original["sin_track"][:] = 0
    structured = spaces.Dict(
        {
            key: spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float64)
            for key, value in original.items()
        }
    )

    class BaseObservation:
        def _get_obs(self, ac_id):
            return {key: value.copy() for key, value in original.items()}

    class ControlledObservation(ControlHooks, BaseObservation):
        entity_observations = True
        graph_radius_nm = 2.0 if graph == "distance" else None
        graph_k = 1 if graph == "knn" else None
        intruder_obs = SimpleNamespace(pos_norm=1_000_000, spd_norm=150)
        control_config = ControlConfig(observation_encoding=encoding)

    env = ControlledObservation()
    obs = env._get_obs("agent")
    space = extend_observation_space(
        structured,
        env.control_config,
        True,
        graph_radius_nm=env.graph_radius_nm,
        graph_k=env.graph_k,
    )
    assert space.contains(obs)
    np.testing.assert_array_equal(obs["graph_adjacency"], np.ones((2, 2)))
    layout = entity_layout(space)
    assert len(layout["graph_edges"]) == 4
    assert not set(layout["graph_edges"]) & set(layout["context"])


@pytest.mark.parametrize("gnn_type", ["gcn", "gat"])
def test_distance_graph_connectivity_changes_messages(gnn_type):
    torch.manual_seed(5)
    env = GraphTrafficEnv()
    layout = env.observation_space.entity_layout
    model = EntityGNN(
        env.observation_space,
        layout,
        pooling="max",
        graph="distance",
        gnn_type=gnn_type,
    )
    obs = torch.tensor(env.observation[None])
    obs[:, layout["aircraft"][2]] = 3
    disconnected = obs.clone()
    disconnected[:, layout["graph_edges"]] = torch.eye(3).reshape(1, 9)
    assert not torch.allclose(model(obs), model(disconnected))


def knn_observation(x, present=None):
    return {
        "x_r": np.array(x, dtype=float),
        "y_r": np.zeros(len(x)),
        "intruder_present": np.ones(len(x)) if present is None else np.array(present),
    }


def test_knn_neighbors_are_bidirectional_without_radius():
    from bluesky_pabloramesc.observations import knn_adjacency

    obs = knn_observation([0, 1, 3, 10, 0], [1, 1, 1, 1, 0])
    expected = [
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    np.testing.assert_array_equal(knn_adjacency(obs, 1, 1), expected)
    np.testing.assert_array_equal(knn_adjacency(obs, 1_000_000, 1), expected)


def test_knn_ties_coincidence_and_permutation_invariance():
    from bluesky_pabloramesc.observations import knn_adjacency

    obs = knn_observation([0, 1, -1, 0], [1, 1, 1, 0])
    expected = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(knn_adjacency(obs, 1, 1), expected)
    permutation = [2, 3, 0, 1]
    permuted = {key: value[permutation] for key, value in obs.items()}
    np.testing.assert_array_equal(
        knn_adjacency(permuted, 1, 1), expected[permutation][:, permutation]
    )
    coincident = knn_observation([0, 0, 0, 0], [1, 1, 1, 0])
    np.testing.assert_array_equal(
        knn_adjacency(coincident, 1, 1), np.outer([1, 1, 1, 0], [1, 1, 1, 0])
    )


@pytest.mark.parametrize("present", [[], [0], [1], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
def test_knn_small_traffic_and_large_k(present):
    from bluesky_pabloramesc.observations import knn_adjacency

    obs = knn_observation(range(len(present)), present)
    np.testing.assert_array_equal(knn_adjacency(obs, 1, 20), np.outer(present, present))


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_knn_rejects_invalid_k(k):
    from bluesky_pabloramesc.observations import knn_adjacency

    with pytest.raises(ValueError, match="positive integer"):
        knn_adjacency(knn_observation([0, 1]), 1, k)
