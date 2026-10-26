"""Ego queries read masked traffic sets and survive training/checkpoint reloads."""

from functools import partial

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from test_entities import TrafficEnv

from bluesky_pabloramesc.entities import EntityMLP, EntityVecNormalize
from bluesky_pabloramesc.models import (
    build_ddpg,
    build_multi_ppo,
    build_multi_sac,
    build_ppo,
    build_sac,
    build_td3,
)
from bluesky_pabloramesc.training import load_architecture_config, replay, train


@pytest.mark.parametrize("context", ["ownship_goal", "all"])
def test_ego_query_is_conditioned_on_context_and_masks_neighbours(context):
    torch.manual_seed(11)
    env = TrafficEnv()
    layout = env.observation_space.entity_layout
    encoder = EntityMLP(env.observation_space, layout, "ego_attention", context)
    obs = torch.tensor(np.stack([env.observation] * 2))
    obs[:, layout["aircraft"][2]] = 3
    obs[1, layout["mask"]] = 0
    calls = []

    def capture(module, args, kwargs):
        calls.append((args, kwargs))

    handle = encoder.attention.attention.register_forward_pre_hook(
        capture, with_kwargs=True
    )
    try:
        original = encoder(obs)
    finally:
        handle.remove()
    (query, keys, values), kwargs = calls[0]
    assert query.shape == (2, 1, 64)
    assert keys.shape == values.shape == (2, 3, 64)
    assert encoder.attention.attention.num_heads == 4
    _, weights = encoder.attention.attention(
        query, keys, values, **{**kwargs, "need_weights": True}
    )
    assert weights[0, 0, 1] == 0
    torch.testing.assert_close(weights[0].sum(), torch.tensor(1.0))

    changed = obs.clone()
    obstacle = next(i for i in layout["context"] if i not in layout["ownship_goal"])
    changed[:, obstacle] += 4
    summary = slice(len(layout["context"]), None)
    if context == "ownship_goal":
        torch.testing.assert_close(original[:, summary], encoder(changed)[:, summary])
    else:
        assert not torch.allclose(original[0, summary], encoder(changed)[0, summary])
    changed[:, layout["ownship_goal"]] += 4
    new_query = encoder.attention.query_encoder(changed[:, encoder.query_indices])
    _, new_weights = encoder.attention.attention(
        new_query.unsqueeze(1), keys, values, **{**kwargs, "need_weights": True}
    )
    assert not torch.allclose(weights[0], new_weights[0])
    assert not torch.allclose(original[0, summary], encoder(changed)[0, summary])
    assert (original[1, summary] == 0).all()


@pytest.mark.parametrize(
    "builder",
    [build_ppo, build_sac, build_td3, build_ddpg, build_multi_ppo, build_multi_sac],
)
def test_ego_attention_builders_learn_and_reload(
    builder, tmp_path, small_training_budgets
):
    env = EntityVecNormalize(DummyVecEnv([TrafficEnv]))
    try:
        model = builder(
            env,
            architecture="entity_mlp",
            pooling="ego_attention",
            attention_context="all",
            device="cpu",
        )
        model.learn(4)
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert np.isfinite(action).all()
        model.save(tmp_path / "policy")
        loaded = type(model).load(tmp_path / "policy", env=env, device="cpu")
        np.testing.assert_allclose(loaded.predict(obs, deterministic=True)[0], action)
    finally:
        env.close()


@pytest.mark.parametrize("context", ["ownship_goal", "all"])
def test_ego_checkpoint_metadata_evaluation_and_resume(tmp_path, monkeypatch, context):
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
            architecture="entity_mlp",
            pooling="ego_attention",
            attention_context=context,
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
        "architecture": "entity_mlp",
        "pooling": "ego_attention",
        "attention_context": context,
    }
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
    assert load_architecture_config(resumed) == load_architecture_config(path)


def test_ego_attention_requires_query_features():
    env = TrafficEnv()
    layout = {**env.observation_space.entity_layout, "ownship_goal": []}
    with pytest.raises(ValueError, match="context"):
        EntityMLP(env.observation_space, layout, "ego_attention")
