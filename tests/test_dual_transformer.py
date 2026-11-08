"""Independent set latents preserve symmetry, normalization, and policy lifecycle."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from bluesky_pabloramesc import cli
from bluesky_pabloramesc.actions import ControlConfig
from bluesky_pabloramesc.architecture_tree import effective_settings, expand_tree
from bluesky_pabloramesc.entities import (
    DualEntityTransformer,
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
from bluesky_pabloramesc.observations import (
    OBSERVATION_ENCODINGS,
    ControlHooks,
    extend_observation_space,
)
from bluesky_pabloramesc.temporal import HistoryVecNormalize, HistoryWindow
from bluesky_pabloramesc.training import load_architecture_config, replay, train

ARCHITECTURE = "dual_entity_transformer"


class RawScene:
    def _get_obs(self, ac_id):
        return {key: value.copy() for key, value in self.raw.items()}


class DualTrafficEnv(ControlHooks, RawScene, Env):
    """Exercise production mask generation and geometry without a simulator."""

    def __init__(self, encoding="goal_aligned_conflict"):
        self.control_config = ControlConfig(observation_encoding=encoding)
        self.entity_observations = True
        self.obstacle_observations = True
        self.intruder_obs = SimpleNamespace(pos_norm=1_000_000, spd_norm=150)
        self.raw = {
            "x_r": np.array([0.01, 0, -0.02]),
            "y_r": np.array([0.02, 0, 0.03]),
            "vx_r": np.array([0.1, 0, -0.1]),
            "vy_r": np.array([-0.1, 0, 0.1]),
            "cos_track": np.array([1.0, 0, -1]),
            "sin_track": np.zeros(3),
            "intruder_distance": np.array([0.02, 0, 0.04]),
            "obstacle_distance": np.array([0.1, 0, 0.3]),
            "obstacle_cos_bearing": np.array([1.0, 0, -1]),
            "obstacle_sin_bearing": np.zeros(3),
            "obstacle_radius": np.array([0.01, 0, 0.02]),
            "sector_point_distance": np.array([0.4, 0.5]),
            "sector_point_cos_bearing": np.array([1.0, -1]),
            "sector_point_sin_bearing": np.zeros(2),
            "cos_drift": np.array([1.0]),
            "sin_drift": np.zeros(1),
            "waypoint_distance": np.array([0.5]),
            "own_cas": np.array([0.2]),
        }
        structured = spaces.Dict(
            {
                key: spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float64)
                for key, value in self.raw.items()
            }
        )
        self.structured = extend_observation_space(
            structured,
            self.control_config,
            entity_observations=True,
            obstacle_observations=True,
        )
        self.observation_space = spaces.flatten_space(self.structured)
        self.observation_space.entity_layout = entity_layout(self.structured)
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation = spaces.flatten(self.structured, self._get_obs("own"))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation.copy(), {}

    def step(self, action):
        return self.observation.copy(), 1.0, True, False, {}


@pytest.mark.parametrize("encoding", OBSERVATION_ENCODINGS)
def test_obstacles_are_tokenized_and_masked_before_geometry(encoding):
    env = DualTrafficEnv(encoding)
    layout = env.observation_space.entity_layout
    obs = env.observation
    np.testing.assert_array_equal(obs[layout["obstacle_mask"]], [1, 0, 1])
    obstacles = np.array(layout["obstacles"])
    assert obstacles.shape == (
        3,
        3 if "cartesian" in encoding or "goal_aligned" in encoding else 4,
    )
    assert not set(obstacles.ravel()) & set(layout["state_context"])
    assert not set(layout["obstacle_mask"]) & set(layout["context"])
    model = DualEntityTransformer(env.observation_space, layout)
    assert torch.isfinite(model(torch.tensor(obs[None], dtype=torch.float32))).all()


@pytest.mark.parametrize("layers", [1, 2])
def test_dual_set_symmetry_branch_isolation_empty_sets_and_gradients(layers):
    torch.manual_seed(11)
    env = DualTrafficEnv()
    layout = env.observation_space.entity_layout
    model = DualEntityTransformer(
        env.observation_space, layout, transformer_layers=layers
    )
    obs = torch.tensor(np.stack([env.observation] * 2), dtype=torch.float32)
    obs[1, layout["mask"] + layout["obstacle_mask"]] = 0
    original = model(obs)
    size = len(layout["state_context"])
    assert original.shape == (2, size + 128)
    assert torch.isfinite(original).all()
    torch.testing.assert_close(original[:, :size], obs[:, layout["state_context"]])
    # Each set can be permuted independently, with its presence mask.
    for field, mask in (("aircraft", "mask"), ("obstacles", "obstacle_mask")):
        indices = np.array(layout[field])
        changed = obs.clone()
        changed[:, indices] = obs[:, indices[[2, 0, 1]]]
        changed[:, layout[mask]] = obs[:, np.array(layout[mask])[[2, 0, 1]]]
        torch.testing.assert_close(original, model(changed))
        changed = obs.clone()
        changed[:, indices[1]] = float("nan")
        changed[1, indices] = 999
        torch.testing.assert_close(original, model(changed))
    changed = obs.clone()
    changed[:, layout["obstacles"][0]] += 2
    updated = model(changed)
    torch.testing.assert_close(original[:, : size + 64], updated[:, : size + 64])
    assert not torch.allclose(original[0, -64:], updated[0, -64:])
    changed = obs.clone()
    changed[:, layout["aircraft"][0]] += 2
    updated = model(changed)
    torch.testing.assert_close(original[:, -64:], updated[:, -64:])
    assert not torch.allclose(
        original[0, size : size + 64], updated[0, size : size + 64]
    )
    changed = obs.clone()
    changed[:, layout["ownship_goal"][0]] += 2
    updated = model(changed)
    assert not torch.allclose(
        original[:, size : size + 64], updated[:, size : size + 64]
    )
    assert not torch.allclose(original[:, -64:], updated[:, -64:])
    (original[:, size:] * torch.randn(2, 128)).sum().backward()
    for branch in (model.traffic_encoder, model.obstacle_encoder):
        assert branch.entity_encoder[0].weight.grad.abs().sum() > 0
        assert branch.context_encoder[0].weight.grad.abs().sum() > 0
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in branch.parameters()
        )


def test_obstacle_normalization_ignores_padding_and_survives_frozen_reload(tmp_path):
    env = EntityVecNormalize(DummyVecEnv([DualTrafficEnv]))
    layout = env.observation_space.entity_layout
    obstacles = np.array(layout["obstacles"])
    raw = env.venv.envs[0].observation.copy()
    raw[obstacles[1]] = 999
    env.venv.envs[0].observation = raw
    try:
        normalized = env.reset()
        np.testing.assert_allclose(
            env.obs_rms.obstacle_rms.mean, raw[obstacles[[0, 2]]].mean(0), atol=1e-4
        )
        assert env.obs_rms.obstacle_rms.count == pytest.approx(2.0001)
        np.testing.assert_array_equal(
            normalized[:, layout["obstacle_mask"]], [[1, 0, 1]]
        )
        assert (env.obs_rms.mean[obstacles[0]] == env.obs_rms.mean[obstacles[2]]).all()
        env.training = False
        expected = env.normalize_obs(raw[None])
        env.save(tmp_path / "normalization.pkl")
        restored = EntityVecNormalize.load(
            tmp_path / "normalization.pkl", DummyVecEnv([DualTrafficEnv])
        )
        restored.venv.envs[0].observation = raw
        try:
            assert not restored.training
            before = restored.obs_rms.obstacle_rms.count
            np.testing.assert_allclose(expected, restored.reset())
            assert restored.obs_rms.obstacle_rms.count == before
        finally:
            restored.close()
    finally:
        env.close()


@pytest.mark.parametrize(
    "builder",
    [build_ppo, build_sac, build_td3, build_ddpg, build_multi_ppo, build_multi_sac],
)
def test_dual_policy_updates_both_branches_and_reloads(
    builder, tmp_path, small_training_budgets
):
    env = EntityVecNormalize(DummyVecEnv([DualTrafficEnv]))
    try:
        model = builder(
            env, architecture=ARCHITECTURE, transformer_layers=2, device="cpu"
        )
        actor = (
            model.policy.pi_features_extractor
            if hasattr(model.policy, "pi_features_extractor")
            else model.actor.features_extractor
        )
        critic = (
            model.policy.vf_features_extractor
            if hasattr(model.policy, "vf_features_extractor")
            else model.critic.features_extractor
        )
        assert actor is not critic
        weights = [
            branch.entity_encoder[0].weight.detach().clone()
            for branch in (actor.traffic_encoder, actor.obstacle_encoder)
        ]
        model.learn(4)
        for before, branch in zip(
            weights, (actor.traffic_encoder, actor.obstacle_encoder)
        ):
            assert not torch.equal(before, branch.entity_encoder[0].weight)
        obs = env.reset()
        action = model.predict(obs, deterministic=True)[0]
        model.save(tmp_path / "policy")
        loaded = type(model).load(tmp_path / "policy", env=env, device="cpu")
        np.testing.assert_allclose(loaded.predict(obs, deterministic=True)[0], action)
    finally:
        env.close()


@pytest.mark.parametrize("temporal", ["none", "gru"])
def test_dual_checkpoint_resume_evaluation_and_temporal_masks(tmp_path, temporal):
    def factory(render_mode=None):
        env = DummyVecEnv([DualTrafficEnv])
        return HistoryWindow(env, 2) if temporal != "none" else env

    path = tmp_path / "policy"
    train(
        partial(
            build_ppo,
            architecture=ARCHITECTURE,
            transformer_layers=2,
            temporal=temporal,
            history_length=2,
            n_steps=2,
            n_epochs=1,
            batch_size=2,
            verbose=0,
        ),
        factory,
        path,
        2,
        episodes=1,
        device="cpu",
        results_dir=tmp_path / "reports",
    )
    saved = load_architecture_config(path)
    assert saved["architecture"] == ARCHITECTURE and saved["transformer_layers"] == 2
    parsed = cli.parse_args("ppo", argv=["evaluate", "--model-path", str(path)])
    assert parsed.architecture == ARCHITECTURE and parsed.pooling == "context_token"
    assert (
        replay(PPO, factory, path, episodes=1, device="cpu", render_mode=None)[0] == 1
    )
    resumed = tmp_path / "resumed"
    train(
        None,
        factory,
        resumed,
        2,
        resume_from=path,
        model_type=PPO,
        episodes=1,
        device="cpu",
        results_dir=tmp_path / "reports",
    )
    assert load_architecture_config(resumed) == saved
    if temporal != "none":
        env = HistoryVecNormalize(factory())
        try:
            obs = env.reset()
            layout = env.observation_space.entity_layout
            np.testing.assert_array_equal(
                obs[:, -1, layout["obstacle_mask"]], [[1, 0, 1]]
            )
        finally:
            env.close()


def test_dual_cli_dispatch_and_tree(monkeypatch):
    options = ["--architecture", ARCHITECTURE, "--transformer-layers", "2"]
    expected = {
        "architecture": ARCHITECTURE,
        "temporal": "none",
        "pooling": "context_token",
        "transformer_layers": 2,
    }
    assert effective_settings(cli.parse_args("ppo", argv=options)) == expected
    assert expand_tree({"name": "dual", "settings": expected})[0].settings == expected
    workflow = Mock(return_value=(1.0, 0.0))
    monkeypatch.setattr(cli, "train", workflow)
    monkeypatch.setattr("sys.argv", ["trainer", *options])
    cli.run_cli("ppo")
    builder, factory = workflow.call_args.args[:2]
    assert builder.keywords["architecture"] == ARCHITECTURE
    assert factory.keywords["entity_observations"]
    assert factory.keywords["obstacle_observations"]


@pytest.mark.parametrize(
    "options",
    [
        ["--pooling", "mean_max"],
        ["--attention-context", "all"],
        ["--transformer-layers", "0"],
    ],
)
def test_dual_cli_rejects_incompatible_options(options):
    with pytest.raises(SystemExit):
        cli.parse_args("ppo", argv=["--architecture", ARCHITECTURE, *options])
