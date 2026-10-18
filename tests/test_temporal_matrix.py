"""Temporal/spatial option matrices using synthetic traffic and short RL updates."""

from functools import partial
from io import BytesIO
from itertools import product

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bluesky_pabloramesc import models
from bluesky_pabloramesc.actions import ControlConfig
from bluesky_pabloramesc.entities import entity_layout
from bluesky_pabloramesc.observations import (
    OBSERVATION_ENCODINGS,
    conflict_features,
    distance_adjacency,
    encode_geometry,
    extend_observation_space,
    knn_adjacency,
    ownship_features,
)
from bluesky_pabloramesc.temporal import HistoryVecNormalize, HistoryWindow

METHODS = ("frame_stack", "gru", "lstm", "tcn")
POOLING = (
    ("mean", "ownship_goal"),
    ("max", "ownship_goal"),
    ("mean_max", "ownship_goal"),
    ("additive_attention", "ownship_goal"),
    ("additive_attention", "all"),
    ("pma", "ownship_goal"),
)
BUILDERS = ("ppo", "sac", "td3", "ddpg", "multi_ppo", "multi_sac")


def spatial_cases(depths=(1,)):
    cases = [("mlp", {"architecture": "mlp", "pooling": "mean_max"})]
    for pooling, context in POOLING:
        base = {"pooling": pooling, "attention_context": context}
        label = f"{pooling}-{context}"
        if pooling != "pma":
            cases.append(
                (f"entity_mlp-{label}", {"architecture": "entity_mlp", **base})
            )
        for depth in depths:
            cases.append(
                (
                    f"transformer{depth}-{label}",
                    {
                        "architecture": "entity_transformer",
                        "transformer_layers": depth,
                        **base,
                    },
                )
            )
            for gnn, graph in product(
                ("gcn", "gat"), ("fully_connected", "distance", "knn")
            ):
                cases.append(
                    (
                        f"{gnn}{depth}-{graph}-{label}",
                        {
                            "architecture": "entity_gnn",
                            "gnn_type": gnn,
                            "gnn_layers": depth,
                            "graph": graph,
                            "graph_radius_nm": 2.0,
                            "graph_k": 1,
                            **base,
                        },
                    )
                )
    return cases


class TemporalTraffic(gym.Env):
    """Vary traffic occupancy, motion and episode boundaries without simulator globals."""

    action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

    def __init__(
        self,
        encoding="goal_aligned_conflict",
        profile="control",
        graph="fully_connected",
        entities=True,
        duration=5,
        truncated=False,
    ):
        self.config = ControlConfig(
            observation_encoding=encoding, observation_profile=profile
        )
        self.graph, self.entities = graph, entities
        self.duration, self.truncated = duration, truncated
        self.steps = 0
        original = self.raw_observation()
        structured = spaces.Dict(
            {
                key: spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float64)
                for key, value in original.items()
            }
        )
        self.structured = extend_observation_space(
            structured,
            self.config,
            entities,
            graph_radius_nm=2.0 if graph == "distance" else None,
            graph_k=1 if graph == "knn" else None,
        )
        self.observation_space = spaces.flatten_space(self.structured)
        if entities:
            self.observation_space.entity_layout = entity_layout(self.structured)

    def raw_observation(self):
        present = np.arange(4) < self.steps % 5
        position = np.array([1.0, 2.0, 5.0, 9.0]) * 1852 / 1_000_000
        return {
            "x_r": position * present,
            "y_r": position[::-1] * present,
            "vx_r": np.array([-0.1, 0.2, -0.3, 0.4]) * present,
            "vy_r": np.array([0.2, -0.4, 0.3, -0.1]) * present,
            "cos_track": present.astype(float),
            "sin_track": np.zeros(4),
            "intruder_distance": np.hypot(position, position[::-1]) * present,
            "cos_drift": np.cos([0.1 * self.steps]),
            "sin_drift": np.sin([0.1 * self.steps]),
            "waypoint_distance": np.array([1.0 - self.steps / 100]),
            "airspeed": np.array([0.3]),
            "obstacle_distance": np.array([0.1, 0.2]),
            "obstacle_cos_bearing": np.array([1.0, 0.0]),
            "obstacle_sin_bearing": np.array([0.0, 1.0]),
            "obstacle_radius": np.array([0.01, 0.02]),
            "sector_point_distance": np.array([0.1, 0.2, 0.3]),
            "sector_point_cos_bearing": np.array([1.0, 0.0, -1.0]),
            "sector_point_sin_bearing": np.array([0.0, 1.0, 0.0]),
            "inside_sector": np.array([1.0]),
        }

    def observation(self):
        raw = self.raw_observation()
        obs = encode_geometry(raw, self.config.observation_encoding)
        if self.config.observation_profile == "control":
            obs.update(ownship_features(90 + self.steps, 300 + self.steps))
        if self.entities:
            obs["intruder_present"] = conflict_features(raw)["intruder_present"]
        if self.graph == "distance":
            obs["graph_adjacency"] = distance_adjacency(obs, 1_000_000, 2.0)
        if self.graph == "knn":
            obs["graph_adjacency"] = knn_adjacency(obs, 1_000_000, 1)
        assert self.structured.contains(obs)
        return spaces.flatten(self.structured, obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        done = self.steps == self.duration
        return (
            self.observation(),
            float(1 - np.square(action).mean()),
            done and not self.truncated,
            done and self.truncated,
            {},
        )


def make_environment(
    case, history, *, encoding="goal_aligned_conflict", profile="control", slots=1
):
    graph = case.get("graph", "fully_connected")
    factory = partial(
        TemporalTraffic,
        encoding,
        profile,
        graph,
        case["architecture"] != "mlp",
        duration=max(5, history + 2),
    )
    return HistoryVecNormalize(HistoryWindow(DummyVecEnv([factory] * slots), history))


def batch_histories(env):
    """One empty, one partially valid, and one full history in each batch."""
    env.reset()
    for _ in range(env.observation_space.shape[0] - 1):
        env.step(np.zeros((env.num_envs, 2)))
    frames = env.get_original_obs()[0]
    result = np.stack([frames, frames, frames])
    result[0] = 0
    result[1, : max(0, frames.shape[0] - 2)] = 0
    return env.normalize_obs(result)


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=name) for name, case in spatial_cases((1, 2, 3))]
)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("history", [1, 2, 8, 17])
def test_every_encoder_pooling_graph_depth_and_history(case, method, history):
    torch.manual_seed(41)
    env = make_environment(case, history)
    try:
        settings = models.architecture_settings(
            env, **case, policy_kwargs={}, temporal=method, history_length=history
        )
        encoder = settings["features_extractor_class"](
            env.observation_space, **settings["features_extractor_kwargs"]
        )
        obs = torch.tensor(batch_histories(env), requires_grad=True)
        output = encoder(obs)
        assert output.shape == (3, encoder.features_dim)
        assert torch.isfinite(output).all() and (output[0] == 0).all()
        changed = obs.detach().clone()
        changed[..., :-1] = changed[..., :-1].masked_fill(
            ~(changed[..., -1:] > 0.5), float("nan")
        )
        torch.testing.assert_close(output, encoder(changed))
        output.square().sum().backward()
        assert torch.isfinite(obs.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in encoder.parameters()
        )
        encoder.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                output.detach(), encoder(obs.detach()), atol=2e-5, rtol=2e-5
            )
        # Each frame may independently reorder aircraft; graph axes move together.
        if case["architecture"] != "mlp":
            layout = env.observation_space.entity_layout
            permutation = [2, 0, 3, 1]
            for time in range(history):
                aircraft = np.array(layout["aircraft"])
                changed[:, time, aircraft] = obs.detach()[
                    :, time, aircraft[permutation]
                ]
                changed[:, time, layout["mask"]] = obs.detach()[
                    :, time, np.array(layout["mask"])[permutation]
                ]
                if layout["graph_edges"]:
                    edges = obs.detach()[:, time, layout["graph_edges"]].reshape(
                        3, 4, 4
                    )
                    changed[:, time, layout["graph_edges"]] = edges[:, permutation][
                        :, :, permutation
                    ].reshape(3, 16)
                permutation = permutation[1:] + permutation[:1]
            with torch.no_grad():
                torch.testing.assert_close(
                    output.detach(), encoder(changed), atol=2e-5, rtol=2e-5
                )
    finally:
        env.close()


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=name) for name, case in spatial_cases()]
)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("builder", BUILDERS)
def test_every_builder_temporal_encoder_and_pooling_trains(
    case, method, builder, small_training_budgets
):
    slots = 10 if builder.startswith("multi_") else 1
    env = make_environment(case, 3, slots=slots)
    try:
        model = getattr(models, f"build_{builder}")(
            env, **case, temporal=method, history_length=3, device="cpu"
        )
        extractor = (
            model.policy.pi_features_extractor
            if isinstance(model, PPO)
            else model.actor.features_extractor
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in extractor.named_parameters()
        }
        model.learn(2 * slots)
        probe = env.reset()
        for _ in range(2):
            probe, _, _, _ = env.step(np.zeros((slots, 2)))
        actions, _ = model.predict(probe, deterministic=True)
        assert actions.shape == (slots, 2) and np.isfinite(actions).all()
        assert all(
            torch.isfinite(parameter).all() for parameter in model.policy.parameters()
        )
        other = (
            model.policy.vf_features_extractor
            if isinstance(model, PPO)
            else model.critic.features_extractor
        )
        assert extractor is not other
        assert not (
            {id(p) for p in extractor.parameters()}
            & {id(p) for p in other.parameters()}
        )
        if before:
            assert any(
                not torch.equal(before[name], parameter)
                for name, parameter in extractor.named_parameters()
            )
        archive = BytesIO()
        model.save(archive)
        archive.seek(0)
        restored = type(model).load(archive, env=env, device="cpu")
        np.testing.assert_allclose(
            restored.predict(probe, deterministic=True)[0], actions, atol=1e-6
        )
        if hasattr(model, "replay_buffer"):
            buffer = model.replay_buffer
            assert buffer.pos == 2 and buffer.observations.shape[1] == slots
            assert set(np.unique(buffer.observations[:2, ..., -1])) <= {0, 1}
    finally:
        env.close()


@pytest.mark.parametrize("encoding", OBSERVATION_ENCODINGS)
@pytest.mark.parametrize("profile", ["legacy", "control"])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "architecture,graph,gnn_type,context",
    [("mlp", "fully_connected", "gat", "ownship_goal")]
    + [
        (architecture, "fully_connected", "gat", context)
        for architecture in ("entity_mlp", "entity_transformer")
        for context in ("ownship_goal", "all")
    ]
    + [
        ("entity_gnn", graph, gnn, context)
        for graph, gnn, context in product(
            ("fully_connected", "distance", "knn"),
            ("gcn", "gat"),
            ("ownship_goal", "all"),
        )
    ],
)
def test_temporal_observation_encoding_and_profile_compatibility(
    encoding, profile, method, architecture, graph, gnn_type, context
):
    case = {
        "architecture": architecture,
        "graph": graph,
        "gnn_type": gnn_type,
        "pooling": "additive_attention" if architecture != "mlp" else "mean_max",
        "attention_context": context,
    }
    env = make_environment(case, 2, encoding=encoding, profile=profile)
    try:
        settings = models.architecture_settings(
            env, **case, policy_kwargs={}, temporal=method, history_length=2
        )
        encoder = settings["features_extractor_class"](
            env.observation_space, **settings["features_extractor_kwargs"]
        )
        output = encoder(torch.tensor(batch_histories(env)))
        assert torch.isfinite(output).all()
        binary = env.obs_rms.binary
        raw = env.get_original_obs()
        np.testing.assert_array_equal(
            env.normalize_obs(raw)[..., binary], raw[..., binary]
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=name) for name, case in spatial_cases((1, 2, 3))]
)
def test_temporal_none_preserves_spatial_baseline(case):
    from stable_baselines3.common.torch_layers import FlattenExtractor

    from bluesky_pabloramesc.entities import EntityVecNormalize
    from bluesky_pabloramesc.temporal import TemporalFeatures

    raw = DummyVecEnv(
        [
            partial(
                TemporalTraffic,
                graph=case.get("graph", "fully_connected"),
                entities=case["architecture"] != "mlp",
            )
        ]
    )
    env = (
        EntityVecNormalize(raw) if case["architecture"] != "mlp" else VecNormalize(raw)
    )
    try:
        settings = models.architecture_settings(
            env, **case, policy_kwargs={}, temporal="none"
        )
        cls = settings.get("features_extractor_class", FlattenExtractor)
        encoder = cls(
            env.observation_space, **settings.get("features_extractor_kwargs", {})
        )
        assert not isinstance(encoder, TemporalFeatures)
        result = encoder(torch.tensor(env.reset()))
        assert (
            result.shape == (1, encoder.features_dim) and torch.isfinite(result).all()
        )
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "case",
    [
        {"architecture": "mlp", "pooling": "mean_max"},
        {
            "architecture": "entity_mlp",
            "pooling": "additive_attention",
            "attention_context": "all",
        },
        {
            "architecture": "entity_transformer",
            "pooling": "pma",
            "transformer_layers": 2,
        },
        *[
            {
                "architecture": "entity_gnn",
                "pooling": "pma",
                "graph": "knn",
                "gnn_type": gnn,
            }
            for gnn in ("gcn", "gat")
        ],
    ],
)
def test_cuda_temporal_forward_backward_and_inference_matches_cpu(method, case):
    from copy import deepcopy

    torch.manual_seed(13)
    env = make_environment(case, 8)
    try:
        settings = models.architecture_settings(
            env, **case, policy_kwargs={}, temporal=method, history_length=8
        )
        cpu = settings["features_extractor_class"](
            env.observation_space, **settings["features_extractor_kwargs"]
        )
        cuda = deepcopy(cpu).cuda()
        obs = torch.tensor(batch_histories(env), requires_grad=True)
        gpu_obs = obs.detach().cuda().requires_grad_()
        expected, actual = cpu(obs), cuda(gpu_obs)
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)
        actual.square().sum().backward()
        assert torch.isfinite(gpu_obs.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in cuda.parameters()
        )
        cpu.eval()
        cuda.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                cuda(gpu_obs).cpu(), cpu(obs), atol=2e-4, rtol=2e-4
            )
    finally:
        env.close()
