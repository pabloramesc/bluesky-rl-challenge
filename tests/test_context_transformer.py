"""Context tokens interact with traffic and survive policy lifecycle operations."""

from functools import partial
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from test_entities import TrafficEnv

from bluesky_pabloramesc import cli
from bluesky_pabloramesc.architecture_tree import effective_settings, expand_tree
from bluesky_pabloramesc.entities import ContextEntityTransformer, EntityVecNormalize
from bluesky_pabloramesc.models import (
    build_ddpg,
    build_multi_ppo,
    build_multi_sac,
    build_ppo,
    build_sac,
    build_td3,
)
from bluesky_pabloramesc.temporal import HistoryVecNormalize, HistoryWindow
from bluesky_pabloramesc.training import load_architecture_config, replay, train

ARCHITECTURE = "context_entity_transformer"


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_context_token_symmetry_masking_and_empty_traffic(layers):
    torch.manual_seed(11)
    env = TrafficEnv()
    layout = env.observation_space.entity_layout
    model = ContextEntityTransformer(
        env.observation_space, layout, transformer_layers=layers
    )
    obs = torch.tensor(np.stack([env.observation] * 2))
    obs[0, layout["aircraft"][2]] = 3
    obs[1, layout["mask"]] = 0
    original = model(obs)
    assert original.shape == (2, len(layout["context"]) + 64)
    assert torch.isfinite(original).all()
    assert original[1, len(layout["context"]) :].abs().sum() > 0
    torch.testing.assert_close(
        original[:, : len(layout["context"])], obs[:, layout["context"]]
    )
    changed = obs.clone()
    changed[1, layout["context"][0]] += 4
    assert not torch.allclose(
        original[1, len(layout["context"]) :],
        model(changed)[1, len(layout["context"]) :],
    )

    aircraft = np.array(layout["aircraft"])
    changed = obs.clone()
    changed[:, aircraft] = obs[:, aircraft[[2, 0, 1]]]
    changed[:, layout["mask"]] = obs[:, np.array(layout["mask"])[[2, 0, 1]]]
    torch.testing.assert_close(original, model(changed))

    changed = obs.clone()
    changed[:, aircraft[1]] = 999
    changed[1, aircraft] = -999
    torch.testing.assert_close(original, model(changed))
    # Exercise gradients for both empty and populated rows without LayerNorm's
    # constant sum making the loss insensitive to attention parameters.
    loss = (original[:, len(layout["context"]) :] * torch.randn(2, 64)).sum()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.context_encoder[0].weight.grad.abs().sum() > 0
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert all(
        layer.norm_first and layer.self_attn.num_heads == 4
        for layer in model.transformer
    )
    if layers > 1:
        assert not torch.equal(
            model.transformer[0].linear1.weight, model.transformer[1].linear1.weight
        )


def test_context_and_aircraft_interact_before_readout():
    torch.manual_seed(11)
    env = TrafficEnv()
    layout = env.observation_space.entity_layout
    model = ContextEntityTransformer(env.observation_space, layout)
    obs = torch.tensor(env.observation[None])
    tokens = model.encode_tokens(obs)
    # Both ownship and environmental context affect aircraft representations,
    # rather than only changing the raw context concatenated after attention.
    for index in layout["context"]:
        changed = obs.clone()
        changed[:, index] += 4
        assert not torch.allclose(tokens[:, 1], model.encode_tokens(changed)[:, 1])
        assert not torch.allclose(tokens[:, 0], model.encode_tokens(changed)[:, 0])
    changed = obs.clone()
    changed[:, layout["aircraft"][2]] += 4
    updated = model.encode_tokens(changed)
    assert not torch.allclose(tokens[:, 0], updated[:, 0])
    assert not torch.allclose(tokens[:, 1], updated[:, 1])


@pytest.mark.parametrize("layers", [0, -1, True])
def test_context_transformer_rejects_invalid_depth(layers):
    env = TrafficEnv()
    with pytest.raises(ValueError, match="positive integer"):
        ContextEntityTransformer(
            env.observation_space,
            env.observation_space.entity_layout,
            transformer_layers=layers,
        )


def test_context_transformer_requires_context():
    env = TrafficEnv()
    layout = {**env.observation_space.entity_layout, "context": []}
    with pytest.raises(ValueError, match="context"):
        ContextEntityTransformer(env.observation_space, layout)


@pytest.mark.parametrize(
    "builder",
    [build_ppo, build_sac, build_td3, build_ddpg, build_multi_ppo, build_multi_sac],
)
def test_context_policy_learns_and_reloads(builder, tmp_path, small_training_budgets):
    env = EntityVecNormalize(DummyVecEnv([TrafficEnv]))
    try:
        model = builder(env, architecture=ARCHITECTURE, device="cpu")
        if hasattr(model.policy, "pi_features_extractor"):
            assert (
                model.policy.pi_features_extractor
                is not model.policy.vf_features_extractor
            )
        else:
            assert model.actor.features_extractor is not model.critic.features_extractor
        extractor = (
            model.policy.pi_features_extractor
            if hasattr(model.policy, "pi_features_extractor")
            else model.actor.features_extractor
        )
        before = extractor.context_encoder[0].weight.detach().clone()
        model.learn(4)
        assert not torch.equal(before, extractor.context_encoder[0].weight)
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert np.isfinite(action).all()
        model.save(tmp_path / "policy")
        loaded = type(model).load(tmp_path / "policy", env=env, device="cpu")
        np.testing.assert_allclose(loaded.predict(obs, deterministic=True)[0], action)
    finally:
        env.close()


@pytest.mark.parametrize("temporal", ["none", "gru"])
def test_context_checkpoint_evaluation_and_resume(tmp_path, monkeypatch, temporal):
    monkeypatch.setattr(
        TrafficEnv,
        "step",
        lambda self, action: (self.observation.copy(), 1.0, True, False, {}),
    )

    def factory(render_mode=None):
        env = DummyVecEnv([TrafficEnv])
        return HistoryWindow(env, 2) if temporal != "none" else env

    path = tmp_path / "policy"
    train(
        partial(
            build_ppo,
            architecture=ARCHITECTURE,
            transformer_layers=2,
            temporal=temporal,
            history_length=2,
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
    expected = {
        "architecture": ARCHITECTURE,
        "pooling": "context_token",
        "transformer_layers": 2,
    }
    if temporal != "none":
        expected.update(temporal=temporal, history_length=2)
    assert load_architecture_config(path) == expected
    restored = cli.parse_args("ppo", argv=["evaluate", "--model-path", str(path)])
    assert restored.architecture == ARCHITECTURE
    assert restored.pooling == "context_token" and restored.transformer_layers == 2
    assert (
        replay(PPO, factory, path, episodes=1, device="cpu", render_mode=None)[0] == 1
    )
    resumed = tmp_path / "resumed"
    train(
        None,
        factory,
        resumed,
        4,
        resume_from=path,
        model_type=PPO,
        episodes=1,
        device="cpu",
        results_dir=tmp_path / "results",
    )
    assert load_architecture_config(resumed) == expected
    with pytest.raises(SystemExit):
        cli.parse_args(
            "ppo",
            argv=["resume", "--model-path", str(path), "--transformer-layers", "3"],
        )


@pytest.mark.parametrize("temporal", ["frame_stack", "gru", "lstm", "tcn"])
def test_context_temporal_policy_updates(temporal, small_training_budgets):
    env = HistoryVecNormalize(HistoryWindow(DummyVecEnv([TrafficEnv]), 2))
    try:
        model = build_ppo(
            env,
            architecture=ARCHITECTURE,
            temporal=temporal,
            history_length=2,
            device="cpu",
        )
        model.learn(4)
        assert np.isfinite(model.predict(env.reset(), deterministic=True)[0]).all()
    finally:
        env.close()


def test_context_cli_dispatch_and_custom_comparison_tree(monkeypatch):
    options = ["--architecture", ARCHITECTURE, "--transformer-layers", "2"]
    args = cli.parse_args("ppo", argv=options)
    expected = {
        "architecture": ARCHITECTURE,
        "temporal": "none",
        "pooling": "context_token",
        "transformer_layers": 2,
    }
    assert effective_settings(args) == expected
    tree = {
        "name": "context",
        "settings": {"architecture": ARCHITECTURE, "transformer_layers": 2},
    }
    assert expand_tree(tree)[0].settings == expected
    workflow = Mock(return_value=(1.0, 0.0))
    monkeypatch.setattr(cli, "train", workflow)
    monkeypatch.setattr("sys.argv", ["trainer", *options])
    cli.run_cli("ppo")
    builder, factory = workflow.call_args.args[:2]
    assert builder.keywords["architecture"] == ARCHITECTURE
    assert builder.keywords["pooling"] == "context_token"
    assert factory.keywords["entity_observations"]


@pytest.mark.parametrize(
    "options",
    [
        ["--architecture", ARCHITECTURE, "--pooling", "mean_max"],
        ["--architecture", "entity_transformer", "--pooling", "context_token"],
        ["--architecture", ARCHITECTURE, "--attention-context", "ownship_goal"],
    ],
)
def test_context_cli_rejects_incompatible_readout(options):
    with pytest.raises(SystemExit):
        cli.parse_args("ppo", argv=options)
