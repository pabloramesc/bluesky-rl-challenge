"""SAC locality, gradients, replay boundaries, and complete checkpoint workflows."""

import json

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from test_entities import TrafficEnv
from test_multi_agent_attention import make_attention_env

from bluesky_pabloramesc.entities import EntityVecNormalize
from bluesky_pabloramesc.models import build_isac, build_masac, build_shared_sac
from bluesky_pabloramesc.multi_agent_sac import (
    ISAC,
    MASAC,
    AttentionSACActor,
    CentralizedAttentionQ,
    JointReplayBuffer,
    LocalAttentionQ,
    SharedSAC,
    soft_targets,
)
from bluesky_pabloramesc.training import (
    architecture_config_path,
    checkpoint_paths,
    evaluate_scenarios,
    replay,
    replay_buffer_path,
    train,
)

BUILDERS = [(build_shared_sac, SharedSAC), (build_isac, ISAC), (build_masac, MASAC)]


def small_builder(builder):
    def build(env, **kwargs):
        return builder(
            env,
            learning_starts=0,
            buffer_size=16,
            batch_size=4,
            train_freq=2,
            gradient_steps=2,
            verbose=0,
            policy_kwargs={
                "actor_kwargs": {"hidden_sizes": [8]},
                "critic_kwargs": {"hidden_sizes": [8]},
            },
            **kwargs,
        )

    return build


def observations():
    env = TrafficEnv()
    return env.observation_space, torch.tensor(env.observation).repeat(2, 3, 1)


def test_attention_q_locality_peer_actions_and_focal_actor_gradient():
    torch.manual_seed(4)
    space, obs = observations()
    active = torch.tensor([[True, True, False], [False, False, False]])
    actions = torch.zeros(2, 3, 2)
    kwargs = {"layout": space.entity_layout, "hidden_sizes": (8,)}
    local = LocalAttentionQ(space.shape[0], 2, 3, **kwargs)
    central = CentralizedAttentionQ(space.shape[0], 2, 3, **kwargs)
    original_local = local(obs, actions, active).detach()
    original_central = central(obs, actions, active).detach()
    changed = actions.clone()
    changed[0, 1] = 0.9
    torch.testing.assert_close(local(obs, changed, active)[0, 0], original_local[0, 0])
    assert not torch.allclose(
        central(obs, changed, active)[0, 0], original_central[0, 0]
    )
    changed_obs = obs.clone()
    changed_obs[0, 1, space.entity_layout["context"]] += 5
    torch.testing.assert_close(
        local(changed_obs, actions, active)[0, 0], original_local[0, 0]
    )
    assert not torch.allclose(
        central(changed_obs, actions, active)[0, 0], original_central[0, 0]
    )
    permutation = [2, 0, 1]
    torch.testing.assert_close(
        central(obs[:, permutation], actions[:, permutation], active[:, permutation]),
        original_central[:, permutation],
    )
    obs[~active] = float("nan")
    changed[~active] = float("nan")
    torch.testing.assert_close(
        local(obs, changed, active),
        local(obs.nan_to_num(), changed.nan_to_num(), active),
    )
    torch.testing.assert_close(
        central(obs, changed, active),
        central(obs.nan_to_num(), changed.nan_to_num(), active),
    )
    changed.requires_grad_()
    focal = central(obs, changed, active, detach_peers=True)[0, 0]
    focal.backward()
    assert changed.grad[0, 0].abs().sum() > 0
    assert (changed.grad[0, 1:] == 0).all()
    assert torch.isfinite(changed.grad).all()
    assert (central(obs, changed, active)[~active] == 0).all()


def test_squashed_gaussian_log_probability_matches_reference_at_saturation():
    space, obs = observations()
    actor = AttentionSACActor(space.shape[0], 2, space.entity_layout, hidden_sizes=())
    with torch.no_grad():
        actor.network[-1].weight.zero_()
        actor.network[-1].bias.copy_(torch.tensor([20.0, -20.0, -1.0, -1.0]))
    actions, log_prob = actor.sample(obs, deterministic=True)
    assert torch.isfinite(log_prob).all()
    assert (actions.abs() <= 1).all()
    log_prob.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters())
    with torch.no_grad():
        actor.network[-1].bias.copy_(torch.tensor([0.2, -0.3, -1.0, -1.0]))
    actions, log_prob = actor.sample(obs, deterministic=True)
    mean = torch.tensor([0.2, -0.3])
    expected = (
        torch.distributions.Normal(mean, torch.exp(torch.tensor(-1.0))).log_prob(mean)
        - torch.log(1 - actions.square())
    ).sum(-1)
    torch.testing.assert_close(log_prob, expected)


def test_soft_targets_bootstrap_truncations_and_ignore_terminated_values():
    target = soft_targets(
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[True, False, False]]),
        torch.tensor([[False, True, False]]),
        torch.tensor([[float("nan"), 4.0, float("nan")]]),
        torch.tensor([[float("nan"), -2.0, float("nan")]]),
        0.1,
        0.9,
    )
    torch.testing.assert_close(target, torch.tensor([[1.0, 5.78, 3.0]]))


@pytest.mark.parametrize("builder,algorithm", BUILDERS)
def test_raw_replay_masks_final_states_and_normalizes_when_sampled(builder, algorithm):
    env = EntityVecNormalize(make_attention_env(), clip_obs=0.01)
    try:
        model = small_builder(builder)(env, device="cpu")
        model.learn(4)
        data = model.replay_buffer.data
        np.testing.assert_array_equal(data["active"][:2], [[True, True], [False, True]])
        np.testing.assert_array_equal(
            data["next_active"][:2], [[False, True], [False, True]]
        )
        np.testing.assert_array_equal(
            data["terminated"][:2], [[True, False], [False, False]]
        )
        # Final raw own_cas is 1.2, not the auto-reset 1.0 or clipped .01.
        context = env.observation_space.entity_layout["context"]
        assert data["next_observations"][1, 1, context[0]] == pytest.approx(1.2)
        np.testing.assert_array_equal(data["rewards"][:2], [[1, 1], [0, 1]])
        sample = model.replay_buffer.sample(8, "cpu", env)
        assert sample["observations"][..., context].abs().max() <= 0.010001
        assert model.active_timesteps == 3
        assert model._active.all()
    finally:
        env.close()


@pytest.mark.parametrize("builder,algorithm", BUILDERS)
def test_training_updates_encoders_targets_temperatures_and_restores_optimizers(
    tmp_path, builder, algorithm
):
    env = EntityVecNormalize(make_attention_env())
    try:
        model = small_builder(builder)(env, device="cpu")
        before = {
            name: p.detach().clone() for name, p in model.policy.named_parameters()
        }
        model.learn(16)
        for role in (
            "actor.features_extractor.attention",
            "critic.0.features_extractor.attention",
            "critic_target.0",
            "log_ent_coef",
        ):
            assert any(
                not torch.equal(before[name], p)
                for name, p in model.policy.named_parameters()
                if role in name
            )
        if algorithm is MASAC:
            assert any(
                not torch.equal(before[name], p)
                for name, p in model.policy.named_parameters()
                if "critic.0.attention" in name
            )
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        learners = model.policy.learners if algorithm is ISAC else [model.policy]
        for learner in learners:
            assert not {id(p) for p in learner.actor.parameters()} & {
                id(p) for p in learner.critic.parameters()
            }
            assert all(not p.requires_grad for p in learner.critic_target.parameters())
        if algorithm is ISAC:
            assert not {id(p) for p in learners[0].parameters()} & {
                id(p) for p in learners[1].parameters()
            }
            before_peer = [p.detach().clone() for p in learners[1].parameters()]
            batch = model.replay_buffer.sample(4, "cpu", env)
            # Use a known active transition for an isolated learner update.
            own = {name: value[:1, 1:2] for name, value in batch.items()}
            own["active"][:] = True
            model._update_learner(learners[0], own)
            for previous, current in zip(before_peer, learners[1].parameters()):
                torch.testing.assert_close(previous, current)
        observation = env.reset()
        expected = model.predict(observation, deterministic=True)[0]
        count = env.obs_rms.count
        assert len(evaluate_scenarios(model, env, [2026])) == 2
        assert env.obs_rms.count == count
        model.save(tmp_path / "model")
        model.save_replay_buffer(tmp_path / "replay.pkl")
        restored = algorithm.load(tmp_path / "model", device="cpu")
        for name, value in model.policy.state_dict().items():
            torch.testing.assert_close(restored.policy.state_dict()[name], value)
        np.testing.assert_array_equal(
            restored.predict(observation, deterministic=True)[0], expected
        )
        restored.load_replay_buffer(tmp_path / "replay.pkl")
        assert len(restored.replay_buffer) == 8
        restored_learners = (
            restored.policy.learners if algorithm is ISAC else [restored.policy]
        )
        for learner in restored_learners:
            for role in ("actor", "critic", "ent_coef"):
                assert getattr(learner, f"{role}_optimizer").state_dict()["state"]
        restored.set_env(env)
        restored.learn(4, reset_num_timesteps=False)
        assert restored.num_timesteps == 20
    finally:
        env.close()


@pytest.mark.parametrize("builder,algorithm", BUILDERS)
def test_train_evaluate_periodic_checkpoint_and_resume(tmp_path, builder, algorithm):
    path = tmp_path / "policy"
    train(
        small_builder(builder),
        make_attention_env,
        path,
        8,
        device="cpu",
        episodes=2,
        checkpoint_interval=4,
        results_dir=tmp_path / "reports",
    )
    config = json.loads(architecture_config_path(path).read_text())
    assert config == {
        "architecture": "entity_mlp",
        "pooling": "additive_attention",
        "attention_context": "all",
    }
    report = next((tmp_path / "reports").glob("*/run.json"))
    metadata = json.loads(report.read_text())
    assert metadata["algorithm"] == algorithm.__name__
    assert metadata["joint_steps"] == 4
    assert metadata["active_timesteps"] == 6
    assert metadata["gradient_updates"] == 4
    assert (report.parent / "checkpoints/step_4/policy.zip").is_file()
    assert replay_buffer_path(path).is_file()
    normalization = EntityVecNormalize.load(
        checkpoint_paths(path)[1], make_attention_env()
    )
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
        small_builder(builder),
        make_attention_env,
        tmp_path / "resumed",
        4,
        device="cpu",
        episodes=2,
        results_dir=tmp_path / "reports",
        resume_from=path,
        model_type=algorithm,
    )
    restored = algorithm.load(tmp_path / "resumed", device="cpu")
    assert restored.num_timesteps == 12
    resumed_report = next(
        p for p in (tmp_path / "reports").glob("*/run.json") if p != report
    )
    assert json.loads(resumed_report.read_text())["replay_buffer_restored"]


def test_replay_capacity_and_restore_validation(tmp_path):
    buffer = JointReplayBuffer(2, 2, 3, 1)
    for index in range(5):
        buffer.add(rewards=np.full(2, index, np.float32))
    assert len(buffer) == 2
    assert set(buffer.data["rewards"][:, 0]) == {3, 4}
    env = EntityVecNormalize(make_attention_env())
    try:
        model = small_builder(build_shared_sac)(env, device="cpu")
        from stable_baselines3.common.save_util import save_to_pkl

        save_to_pkl(tmp_path / "wrong.pkl", buffer)
        with pytest.raises(ValueError, match="Replay buffer"):
            model.load_replay_buffer(tmp_path / "wrong.pkl")
    finally:
        env.close()


def test_callback_cancellation_and_missing_boundary_metadata(env_factory):
    class Stop(BaseCallback):
        def _on_step(self):
            return False

    env = EntityVecNormalize(make_attention_env())
    try:
        model = small_builder(build_shared_sac)(env, device="cpu")
        model.learn(8, callback=Stop())
        assert model.num_timesteps == 2 and model._n_updates == 0
        model._last_obs = np.zeros((2, model.observation_space.shape[0]), np.float32)
        with pytest.raises(ValueError, match="agent boundary metadata"):
            model._store_transition(
                model._last_obs,
                np.ones(2),
                np.zeros(2, bool),
                [{}, {}],
                np.zeros((2, 1)),
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"buffer_size": 0},
        {"learning_starts": -1},
        {"tau": 2},
        {"gamma": float("nan")},
        {"train_freq": 0},
        {"gradient_steps": 0},
        {"policy_kwargs": {"initial_alpha": 0}},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        SharedSAC(**settings)


@pytest.mark.parametrize("algorithm", ["isac", "shared_sac", "masac"])
@pytest.mark.parametrize("mode", ["train", "resume", "evaluate", "replay"])
def test_cli_defaults_dispatch_and_architecture_restrictions(
    algorithm, mode, monkeypatch, tmp_path
):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli

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
    monkeypatch.setattr("sys.argv", ["trainer", mode, "--model-path", str(path)])
    workflow = Mock(return_value=(1.5, 0))
    monkeypatch.setattr(
        cli, "replay" if mode in ("evaluate", "replay") else "train", workflow
    )
    cli.run_cli(algorithm, multi_agent=True)
    assert workflow.call_args.args[1].keywords["entity_observations"]
    if mode in ("evaluate", "replay"):
        assert (
            workflow.call_args.args[0]
            is {"isac": ISAC, "shared_sac": SharedSAC, "masac": MASAC}[algorithm]
        )
    else:
        assert workflow.call_args.args[0].keywords["attention_context"] == "all"
    with pytest.raises(SystemExit):
        cli.parse_args(algorithm, multi_agent=True, argv=["--temporal", "gru"])
    with pytest.raises(SystemExit):
        cli.parse_args(algorithm, multi_agent=False, argv=[])
