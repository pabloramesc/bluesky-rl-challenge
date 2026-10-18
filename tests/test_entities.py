"""Set encoders preserve slot symmetry throughout training and serialization."""

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bluesky_pabloramesc.entities import (
    EntityMLP,
    EntityTransformer,
    EntityVecNormalize,
    entity_layout,
)
from bluesky_pabloramesc.models import (
    build_ddpg,
    build_multi_ppo,
    build_multi_sac,
    build_ppo,
    build_sac,
    build_td3,
)


class TrafficEnv(Env):
    def __init__(self):
        fields = {
            key: spaces.Box(-10, 10, (3,), dtype=np.float32)
            for key in ("x_r", "y_r", "cos_track", "sin_track", "intruder_distance")
        }
        fields["intruder_present"] = spaces.Box(0, 1, (3,), dtype=np.float32)
        fields["own_cas"] = spaces.Box(-10, 10, (1,), dtype=np.float32)
        fields["obstacle_distance"] = spaces.Box(-10, 10, (1,), dtype=np.float32)
        structured = spaces.Dict(fields)
        self.observation_space = spaces.flatten_space(structured)
        self.observation_space.entity_layout = entity_layout(structured)
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation = spaces.flatten(
            structured,
            {
                key: np.array([1, 0, 1], dtype=np.float32)
                if key == "intruder_present"
                else np.ones(field.shape, dtype=np.float32)
                for key, field in structured.spaces.items()
            },
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation.copy(), {}

    def step(self, action):
        return self.observation.copy(), 1.0, False, False, {}


@pytest.mark.parametrize(
    "encoder_type,layers,pooling,attention_context",
    [
        (EntityMLP, None, pooling, context)
        for pooling, context in [
            ("mean", "ownship_goal"),
            ("max", "ownship_goal"),
            ("mean_max", "ownship_goal"),
            ("additive_attention", "ownship_goal"),
            ("additive_attention", "all"),
        ]
    ]
    + [
        (EntityTransformer, layers, pooling, context)
        for layers in (1, 2, 3)
        for pooling, context in [
            ("mean", "ownship_goal"),
            ("max", "ownship_goal"),
            ("mean_max", "ownship_goal"),
            ("additive_attention", "ownship_goal"),
            ("additive_attention", "all"),
            ("pma", "ownship_goal"),
        ]
    ],
)
def test_pooling_symmetry_padding_and_gradients(
    encoder_type, layers, pooling, attention_context
):
    env = TrafficEnv()
    layout = env.observation_space.entity_layout
    encoder = encoder_type(
        env.observation_space,
        layout,
        pooling,
        attention_context,
        **({"transformer_layers": layers} if layers else {}),
    )
    obs = torch.tensor(np.stack([env.observation] * 2))
    obs[0, layout["aircraft"][2]] = 3
    changed = obs.clone()
    aircraft = np.array(layout["aircraft"])
    changed[:, aircraft] = obs[:, aircraft[[2, 0, 1]]]
    changed[:, layout["mask"]] = obs[:, np.array(layout["mask"])[[2, 0, 1]]]
    torch.testing.assert_close(encoder(obs), encoder(changed))
    changed = obs.clone()
    changed[:, aircraft[1]] = 999
    torch.testing.assert_close(encoder(obs), encoder(changed))
    empty = obs.clone()
    empty[:, layout["mask"]] = 0
    result = encoder(empty)
    assert torch.isfinite(result).all()
    assert (result[:, len(layout["context"]) :] == 0).all()
    (encoder(obs).sum() + result.sum()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )


def test_shared_normalization_and_frozen_roundtrip(tmp_path):
    env = EntityVecNormalize(DummyVecEnv([TrafficEnv]))
    try:
        obs = env.reset()
        layout = env.obs_rms.layout
        aircraft = np.array(layout["aircraft"])
        np.testing.assert_array_equal(obs[:, layout["mask"]], [[1, 0, 1]])
        np.testing.assert_array_equal(
            env.obs_rms.mean[aircraft[0]], env.obs_rms.mean[aircraft[2]]
        )
        # Padded values do not enter the aircraft statistics.
        np.testing.assert_allclose(env.obs_rms.aircraft_rms.mean, 1, atol=1e-4)
        env.save(tmp_path / "stats.pkl")
        restored = VecNormalize.load(tmp_path / "stats.pkl", DummyVecEnv([TrafficEnv]))
        try:
            restored.training = False
            count = restored.obs_rms.aircraft_rms.count
            np.testing.assert_allclose(restored.reset(), obs)
            assert restored.obs_rms.aircraft_rms.count == count
        finally:
            restored.close()
    finally:
        env.close()


@pytest.mark.parametrize(
    "builder",
    [build_ppo, build_sac, build_td3, build_ddpg, build_multi_ppo, build_multi_sac],
)
@pytest.mark.parametrize("attention_context", [None, "ownship_goal", "all"])
@pytest.mark.parametrize("architecture", ["entity_mlp", "entity_transformer"])
def test_builders_and_policy_roundtrip(
    builder, tmp_path, attention_context, architecture
):
    env = EntityVecNormalize(DummyVecEnv([TrafficEnv]))
    try:
        model = builder(
            env,
            architecture=architecture,
            transformer_layers=2,
            pooling="additive_attention"
            if attention_context
            else ("pma" if architecture == "entity_transformer" else "max"),
            attention_context=attention_context or "ownship_goal",
            device="cpu",
        )
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert np.isfinite(action).all()
        if hasattr(model.policy, "pi_features_extractor"):
            assert (
                model.policy.pi_features_extractor
                is not model.policy.vf_features_extractor
            )
        else:
            assert model.actor.features_extractor is not model.critic.features_extractor
        model.save(tmp_path / "policy")
        loaded = type(model).load(tmp_path / "policy", env=env, device="cpu")
        np.testing.assert_allclose(loaded.predict(obs, deterministic=True)[0], action)
    finally:
        env.close()


@pytest.mark.parametrize(
    "pooling,attention_context",
    [
        ("mean", "ownship_goal"),
        ("max", "ownship_goal"),
        ("mean_max", "ownship_goal"),
        ("additive_attention", "ownship_goal"),
        ("additive_attention", "all"),
    ],
)
@pytest.mark.parametrize("builder", [build_ppo, build_sac, build_td3, build_ddpg])
@pytest.mark.parametrize("architecture", ["entity_mlp", "entity_transformer"])
def test_set_policy_can_learn(builder, pooling, attention_context, architecture):
    env = EntityVecNormalize(DummyVecEnv([TrafficEnv]))
    try:
        settings = (
            {"n_steps": 4, "n_epochs": 1}
            if builder == build_ppo
            else {
                "buffer_size": 32,
                "learning_starts": 0,
            }
        )
        model = builder(
            env,
            architecture=architecture,
            transformer_layers=2,
            pooling=pooling,
            attention_context=attention_context,
            batch_size=2,
            device="cpu",
            verbose=0,
            **settings,
        )
        model.learn(4)
        assert model.num_timesteps == 4
        assert np.isfinite(model.predict(env.reset(), deterministic=True)[0]).all()
    finally:
        env.close()


@pytest.mark.parametrize("attention_context", [None, "ownship_goal", "all"])
@pytest.mark.parametrize("architecture", ["entity_mlp", "entity_transformer"])
def test_training_checkpoint_and_evaluation(
    tmp_path, monkeypatch, attention_context, architecture
):
    from functools import partial

    from stable_baselines3 import PPO

    from bluesky_pabloramesc.training import (
        load_architecture_config,
        replay,
        train,
    )

    monkeypatch.setattr(
        TrafficEnv,
        "step",
        lambda self, action: (self.observation.copy(), 1.0, True, False, {}),
    )
    factory = lambda render_mode=None: DummyVecEnv([TrafficEnv])
    path = tmp_path / "policy"
    train(
        partial(
            build_ppo,
            architecture=architecture,
            transformer_layers=3,
            pooling="additive_attention"
            if attention_context
            else ("pma" if architecture == "entity_transformer" else "mean"),
            attention_context=attention_context or "ownship_goal",
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
        "architecture": architecture,
        **({"transformer_layers": 3} if architecture == "entity_transformer" else {}),
        "pooling": "additive_attention"
        if attention_context
        else ("pma" if architecture == "entity_transformer" else "mean"),
        **({"attention_context": attention_context} if attention_context else {}),
    }
    assert (
        replay(PPO, factory, path, episodes=1, device="cpu", render_mode=None)[0] == 1
    )

    if architecture == "entity_transformer":
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
        assert load_architecture_config(
            tmp_path / "resumed"
        ) == load_architecture_config(path)


def test_normalization_commutes_with_aircraft_permutation():
    from bluesky_pabloramesc.entities import EntityRunningMeanStd

    env = TrafficEnv()
    layout = env.observation_space.entity_layout
    aircraft = np.array(layout["aircraft"])
    mask = np.array(layout["mask"])
    obs = np.random.default_rng(7).normal(size=(5, *env.observation_space.shape))
    obs[:, mask] = [1, 0, 1]
    permuted = obs.copy()
    permuted[:, aircraft] = obs[:, aircraft[[2, 0, 1]]]
    permuted[:, mask] = obs[:, mask[[2, 0, 1]]]
    original_stats = EntityRunningMeanStd(env.observation_space.shape, layout)
    permuted_stats = EntityRunningMeanStd(env.observation_space.shape, layout)
    original_stats.update(obs)
    permuted_stats.update(permuted)
    np.testing.assert_allclose(original_stats.mean, permuted_stats.mean)
    np.testing.assert_allclose(original_stats.var, permuted_stats.var)
    assert original_stats.aircraft_rms.count == pytest.approx(10.0001)


@pytest.mark.parametrize("context", ["ownship_goal", "all"])
def test_attention_weights_follow_additive_formula_and_mask(context):
    torch.manual_seed(5)
    env = TrafficEnv()
    layout = env.observation_space.entity_layout
    encoder = EntityMLP(env.observation_space, layout, "additive_attention", context)
    obs = torch.tensor(np.stack([env.observation] * 2))
    obs[0, layout["aircraft"][2]] = 3
    obs[1, layout["mask"]] = 0
    encoded = encoder.encoder(obs[:, encoder.aircraft_indices])
    query = obs[:, encoder.query_indices]
    present = obs[:, encoder.mask_indices] > 0.5
    attention = encoder.attention
    scores = attention.score(
        torch.tanh(
            attention.entity_projection(encoded)
            + attention.context_projection(query).unsqueeze(1)
        )
    ).squeeze(-1)
    expected = torch.softmax(scores[0, [0, 2]], dim=0)
    weights = attention.weights(encoded, query, present)
    torch.testing.assert_close(weights[0, [0, 2]], expected)
    assert weights[0, 1] == 0
    assert (weights[1] == 0).all()
    torch.testing.assert_close(
        encoder(obs)[:, len(layout["context"]) :],
        (encoded * weights.unsqueeze(-1)).sum(1),
    )
    changed = obs.clone()
    obstacle = next(
        index for index in layout["context"] if index not in layout["ownship_goal"]
    )
    changed[:, obstacle] += 2
    changed_query = changed[:, encoder.query_indices]
    changed_weights = attention.weights(encoded, changed_query, present)
    if context == "ownship_goal":
        torch.testing.assert_close(weights, changed_weights)
    else:
        assert not torch.allclose(weights[0], changed_weights[0])
    changed[:, layout["ownship_goal"]] += 4
    ownship_weights = attention.weights(
        encoded, changed[:, encoder.query_indices], present
    )
    assert not torch.allclose(changed_weights[0], ownship_weights[0])


@pytest.mark.parametrize(
    "encoding", ["upstream", "ego_cartesian", "goal_aligned", "goal_aligned_conflict"]
)
def test_attention_query_layout_for_geometry_encodings(encoding):
    from bluesky_pabloramesc.actions import ControlConfig
    from bluesky_pabloramesc.observations import extend_observation_space

    fields = {
        key: spaces.Box(-10, 10, (3,), dtype=np.float64)
        for key in ("x_r", "y_r", "vx_r", "vy_r", "cos_track", "sin_track")
    }
    for key in (
        "cos_drift",
        "sin_drift",
        "waypoint_distance",
        "airspeed",
        "obstacle_distance",
        "obstacle_cos_bearing",
        "obstacle_sin_bearing",
        "sector_point_distance",
        "sector_point_cos_bearing",
        "sector_point_sin_bearing",
    ):
        fields[key] = spaces.Box(-10, 10, (1,), dtype=np.float64)
    structured = extend_observation_space(
        spaces.Dict(fields),
        ControlConfig(observation_encoding=encoding, observation_profile="control"),
        True,
    )
    layout = entity_layout(structured)
    query_fields = set()
    offset = 0
    for key, field in structured.spaces.items():
        indices = set(range(offset, offset + int(np.prod(field.shape))))
        if indices & set(layout["ownship_goal"]):
            query_fields.add(key)
        offset += int(np.prod(field.shape))
    assert "own_cas" in query_fields and "airspeed" in query_fields
    assert not any(key.startswith(("obstacle", "sector")) for key in query_fields)
    if encoding == "ego_cartesian":
        assert {"goal_x", "goal_y"} <= query_fields
    else:
        assert {"cos_drift", "sin_drift", "waypoint_distance"} <= query_fields


def test_transformer_aircraft_interactions_and_independent_layers():
    torch.manual_seed(6)
    env = TrafficEnv()
    layout = env.observation_space.entity_layout
    model = EntityTransformer(env.observation_space, layout, transformer_layers=2)
    entities = torch.randn(1, 3, len(layout["aircraft"][0]))
    present = torch.ones(1, 3, dtype=torch.bool)
    changed = entities.clone()
    changed[:, 2] += 3
    # The first aircraft now incorporates information about the changed third aircraft.
    assert not torch.allclose(
        model.encode_entities(entities, present)[:, 0],
        model.encode_entities(changed, present)[:, 0],
    )
    assert not torch.equal(
        model.transformer[0].linear1.weight, model.transformer[1].linear1.weight
    )


@pytest.mark.parametrize("builder", [build_ppo, build_sac, build_td3, build_ddpg])
def test_pma_policy_can_learn(builder):
    env = EntityVecNormalize(DummyVecEnv([TrafficEnv]))
    try:
        settings = (
            {"n_steps": 4, "n_epochs": 1}
            if builder == build_ppo
            else {"buffer_size": 32, "learning_starts": 0}
        )
        model = builder(
            env,
            architecture="entity_transformer",
            pooling="pma",
            batch_size=2,
            device="cpu",
            verbose=0,
            **settings,
        )
        model.learn(4)
        assert np.isfinite(model.predict(env.reset(), deterministic=True)[0]).all()
    finally:
        env.close()
