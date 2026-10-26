"""Attention locality, masking, optimization, and joint PPO checkpoint workflows."""

import json

import numpy as np
import pytest
import supersuit as ss
import torch
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from test_entities import TrafficEnv
from test_mappo import JointTraffic

from bluesky_pabloramesc.entities import EntityVecNormalize, entity_layout
from bluesky_pabloramesc.ippo import IPPO, SharedPPO
from bluesky_pabloramesc.mappo import MAPPO
from bluesky_pabloramesc.models import build_ippo, build_mappo, build_shared_ppo
from bluesky_pabloramesc.multi_agent import (
    FlattenObs,
    JointEpisodeBlackDeath,
    SeedableSB3VecEnvWrapper,
)
from bluesky_pabloramesc.multi_agent_attention import (
    AdditiveAttentionActor,
    CentralizedAdditiveAttentionCritic,
    LocalAdditiveAttentionCritic,
)
from bluesky_pabloramesc.training import (
    architecture_config_path,
    checkpoint_paths,
    evaluate_scenarios,
    replay,
    train,
)


class EntityJointTraffic(JointTraffic):
    def observation_space(self, agent):
        from gymnasium import spaces

        return spaces.Dict(
            {
                "x_r": spaces.Box(-10, 10, (3,), np.float32),
                "y_r": spaces.Box(-10, 10, (3,), np.float32),
                "intruder_present": spaces.Box(0, 1, (3,), np.float32),
                "own_cas": spaces.Box(-10, 10, (1,), np.float32),
            }
        )

    def observations(self, agents):
        return {
            agent: {
                "x_r": np.array([self.steps + 1, 0, 2], np.float32),
                "y_r": np.array([2, 0, 1], np.float32),
                "intruder_present": np.array([1, 0, 1], np.float32),
                "own_cas": np.array([index + self.steps / 10], np.float32),
            }
            for index, agent in enumerate(self.possible_agents)
            if agent in agents
        }


def make_attention_env(render_mode=None):
    traffic = EntityJointTraffic()
    layout = entity_layout(traffic.observation_space("early"))
    env = JointEpisodeBlackDeath(FlattenObs(traffic), track_agents=True)
    env = SeedableSB3VecEnvWrapper(ss.pettingzoo_env_to_vec_env_v1(env))
    env = VecMonitor(env)
    env.observation_space.entity_layout = layout
    return env


def test_local_networks_preserve_aircraft_symmetry_and_ignore_absent_slots():
    torch.manual_seed(3)
    env = TrafficEnv()
    layout = env.observation_space.entity_layout
    size = env.observation_space.shape[0]
    actor = AdditiveAttentionActor(size, 2, layout, hidden_sizes=(8,))
    critic = LocalAdditiveAttentionCritic(size, 2, layout, hidden_sizes=(8,))
    observations = torch.tensor(env.observation).repeat(2, 2, 1)
    observations[:, 1, layout["context"]] += 2
    active = torch.tensor([[True, True], [True, False]])
    means = actor(observations).detach()
    values = critic(observations, active).detach()
    # Reordering local traffic must leave the policy and local value unchanged.
    permuted = observations.clone()
    permuted[..., layout["aircraft"]] = observations[
        ..., np.array(layout["aircraft"])[[2, 0, 1]]
    ]
    permuted[..., layout["mask"]] = observations[
        ..., np.array(layout["mask"])[[2, 0, 1]]
    ]
    torch.testing.assert_close(actor(permuted), means)
    torch.testing.assert_close(critic(permuted, active), values)
    # Absent traffic can contain arbitrary padding, including nonfinite values.
    observations[..., layout["aircraft"][1]] = float("nan")
    observations[1, 1] = float("nan")
    torch.testing.assert_close(actor(observations[0]), means[0])
    torch.testing.assert_close(critic(observations, active), values)
    observations[0, 1] = 5
    torch.testing.assert_close(actor(observations[0, 0]), means[0, 0])
    torch.testing.assert_close(critic(observations, active)[0, 0], values[0, 0])
    # Actor and critic must learn independent encoders.
    assert not {id(p) for p in actor.parameters()} & {
        id(p) for p in critic.parameters()
    }


def test_centralized_attention_uses_peers_and_masks_departed_agents():
    torch.manual_seed(4)
    env = TrafficEnv()
    layout = env.observation_space.entity_layout
    critic = CentralizedAdditiveAttentionCritic(
        env.observation_space.shape[0], 3, layout, hidden_sizes=(8,)
    )
    observations = torch.tensor(env.observation).repeat(2, 3, 1)
    observations[0, 1, layout["context"]] += 2
    active = torch.tensor([[True, True, False], [False, False, False]])
    before = critic(observations, active).detach()
    changed = observations.clone()
    changed[0, 1, layout["context"]] += 4
    assert not torch.allclose(critic(changed, active)[0, 0], before[0, 0])
    order = [2, 0, 1]
    torch.testing.assert_close(
        critic(observations[:, order], active[:, order]), before[:, order]
    )
    observations[~active] = float("nan")
    observations.requires_grad_()
    values = critic(observations, active)
    torch.testing.assert_close(values, before)
    assert (values[~active] == 0).all()
    values.sum().backward()
    assert (observations.grad[~active] == 0).all()
    assert torch.isfinite(observations.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in critic.parameters())


@pytest.mark.parametrize(
    "builder,algorithm",
    [(build_shared_ppo, SharedPPO), (build_ippo, IPPO), (build_mappo, MAPPO)],
)
def test_attention_training_checkpoints_and_frozen_evaluation(
    tmp_path, builder, algorithm
):
    def small_builder(env, **kwargs):
        return builder(
            env,
            architecture="entity_mlp",
            attention_context="all",
            n_steps=2,
            batch_size=2,
            n_epochs=2,
            verbose=0,
            policy_kwargs={
                "actor_kwargs": {"hidden_sizes": [8]},
                "critic_kwargs": {"hidden_sizes": [8]},
            },
            **kwargs,
        )

    env = EntityVecNormalize(make_attention_env())
    try:
        model = small_builder(env, device="cpu")
        before = {
            name: p.detach().clone() for name, p in model.policy.named_parameters()
        }
        model.learn(8)
        for role in ("actor", "critic"):
            assert any(
                not torch.equal(before[name], p)
                for name, p in model.policy.named_parameters()
                if f"{role}.features_extractor.attention" in name
            )
        if algorithm is MAPPO:
            assert any(
                not torch.equal(before[name], p)
                for name, p in model.policy.named_parameters()
                if name.startswith("critic.attention")
            )
        observation = env.reset()
        expected = model.predict(observation, deterministic=True)[0]
        count = env.obs_rms.count
        assert len(evaluate_scenarios(model, env, [2026])) == 2
        assert env.obs_rms.count == count
        model.save(tmp_path / "model")
        restored = algorithm.load(tmp_path / "model", device="cpu")
        np.testing.assert_array_equal(
            restored.predict(observation, deterministic=True)[0], expected
        )
        learners = restored.policy.learners if algorithm is IPPO else [restored.policy]
        assert all(learner.optimizer.state_dict()["state"] for learner in learners)
        restored.set_env(env)
        restored.learn(4, reset_num_timesteps=False)
        assert restored.num_timesteps == 12
    finally:
        env.close()

    # Exercise metadata inference and the actual training/checkpoint workflow too.
    path = tmp_path / "policy"
    train(
        small_builder,
        make_attention_env,
        path,
        4,
        device="cpu",
        episodes=2,
        results_dir=tmp_path / "reports",
    )
    config = json.loads(architecture_config_path(path).read_text())
    assert config == {
        "architecture": "entity_mlp",
        "pooling": "additive_attention",
        "attention_context": "all",
    }
    normalization = VecNormalize.load(checkpoint_paths(path)[1], make_attention_env())
    assert isinstance(normalization, EntityVecNormalize)
    normalization.close()
    assert (
        replay(
            algorithm,
            make_attention_env,
            path,
            device="cpu",
            episodes=2,
            render_mode=None,
        )[0]
        == 1.5
    )
    train(
        small_builder,
        make_attention_env,
        tmp_path / "resumed",
        4,
        device="cpu",
        episodes=2,
        results_dir=tmp_path / "reports",
        resume_from=path,
        model_type=algorithm,
    )
    assert algorithm.load(tmp_path / "resumed", device="cpu").num_timesteps == 8


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_attention_cli_and_checkpoint_configuration(algorithm, tmp_path):
    from bluesky_pabloramesc.cli import parse_args

    options = [
        "--architecture",
        "entity_mlp",
        "--pooling",
        "additive_attention",
        "--attention-context",
        "all",
    ]
    args = parse_args(algorithm, multi_agent=True, argv=options)
    assert args.attention_context == "all"
    path = tmp_path / "policy"
    architecture_config_path(path).write_text(
        json.dumps(
            {
                "architecture": "entity_mlp",
                "pooling": "additive_attention",
                "attention_context": "all",
            }
        )
    )
    args = parse_args(
        algorithm, multi_agent=True, argv=["evaluate", "--model-path", str(path)]
    )
    assert args.architecture == "entity_mlp" and args.attention_context == "all"
    with pytest.raises(SystemExit):
        parse_args(
            algorithm,
            multi_agent=True,
            argv=[
                "resume",
                "--model-path",
                str(path),
                "--attention-context",
                "ownship_goal",
            ],
        )


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_training_cli_wires_attention_builder_and_entity_environment(
    algorithm, monkeypatch
):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli
    from bluesky_pabloramesc.multi_agent import make_mappo_env

    monkeypatch.setattr(
        "sys.argv",
        [
            "trainer",
            "--architecture",
            "entity_mlp",
            "--pooling",
            "additive_attention",
            "--attention-context",
            "all",
        ],
    )
    workflow = Mock(return_value=(1.5, 0))
    monkeypatch.setattr(cli, "train", workflow)
    cli.run_cli(algorithm, multi_agent=True)
    builder, factory = workflow.call_args.args[:2]
    assert builder.func is {"ippo": build_ippo, "mappo": build_mappo}[algorithm]
    assert builder.keywords == {
        "architecture": "entity_mlp",
        "pooling": "additive_attention",
        "attention_context": "all",
    }
    assert factory.func is make_mappo_env
    assert factory.keywords == {"entity_observations": True}
