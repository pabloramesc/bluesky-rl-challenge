"""Numerical MAPPO references and one-joint-world training/checkpoint workflows."""

import json

import numpy as np
import pytest
import supersuit as ss
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from torch import nn

from bluesky_pabloramesc.mappo import (
    MAPPO,
    JointRolloutBuffer,
    clipped_losses,
)
from bluesky_pabloramesc.models import build_mappo
from bluesky_pabloramesc.multi_agent import (
    FlattenObs,
    JointEpisodeBlackDeath,
    SeedableSB3VecEnvWrapper,
)
from bluesky_pabloramesc.training import checkpoint_paths, evaluate, replay, train


class JointTraffic(ParallelEnv):
    """One agent reaches its goal; the other hits the joint time limit."""

    render_mode = None

    def __init__(self):
        self.metadata = {"render_modes": []}
        self.possible_agents = ["early", "late"]

    def observation_space(self, agent):
        return spaces.Dict({"state": spaces.Box(-1, 1, (2,), np.float32)})

    def action_space(self, agent):
        return spaces.Box(-1, 1, (1,), np.float32)

    def observations(self, agents):
        return {
            agent: {"state": np.full(2, self.steps / 10 + index / 10, np.float32)}
            for index, agent in enumerate(self.possible_agents)
            if agent in agents
        }

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.agents = self.possible_agents.copy()
        return self.observations(self.agents), {agent: {} for agent in self.agents}

    def step(self, actions):
        assert set(actions) == set(self.agents)
        assert all(
            self.action_space(agent).contains(action)
            for agent, action in actions.items()
        )
        previous = self.agents.copy()
        self.steps += 1
        self.agents = ["late"] if self.steps == 1 else []
        return (
            self.observations(previous),
            {agent: 1.0 for agent in previous},
            {agent: agent == "early" for agent in previous},
            {agent: self.steps == 2 for agent in previous},
            {agent: {} for agent in previous},
        )


def make_joint_env(render_mode=None):
    env = JointEpisodeBlackDeath(FlattenObs(JointTraffic()), track_agents=True)
    env = SeedableSB3VecEnvWrapper(ss.pettingzoo_env_to_vec_env_v1(env))
    return VecMonitor(env)


def small_mappo(env, **kwargs):
    return build_mappo(
        env,
        n_steps=2,
        batch_size=1,
        n_epochs=2,
        verbose=0,
        policy_kwargs={
            "actor_kwargs": {"hidden_sizes": (16,)},
            "critic_kwargs": {"hidden_sizes": (16,)},
        },
        **kwargs,
    )


def test_gae_stops_at_individual_boundaries_and_excludes_padding():
    buffer = JointRolloutBuffer("cpu")
    for reward, active, values, next_values, continuation in (
        ([1, 2], [True, True], [0.5, 1], [0, 2], [False, True]),
        ([999, 3], [False, True], [100, 2], [100, 4], [False, False]),
        ([10, 10], [True, True], [0, 0], [0, 0], [False, False]),
    ):
        buffer.add(
            observations=np.zeros((2, 3), np.float32),
            rewards=np.array(reward, np.float32),
            active=np.array(active),
            old_values=np.array(values, np.float32),
            next_values=np.array(next_values, np.float32),
            continuations=np.array(continuation),
        )
    buffer.compute_returns(gamma=0.9, gae_lambda=0.8)
    # The time-limit value (4) bootstraps, but the following episode (10) does not.
    expected = torch.tensor([[0.5, 1 + 1.8 + 0.72 * 4.6], [0, 4.6], [10, 10]])
    torch.testing.assert_close(buffer.data["advantages"], expected)
    torch.testing.assert_close(
        buffer.data["returns"], expected + buffer.data["old_values"]
    )
    batches = list(buffer.batches(2))
    assert [batch["observations"].shape for batch in batches] == [(2, 2, 3), (1, 2, 3)]


def test_clipped_losses_match_reference_and_have_no_inactive_gradient():
    batch = {
        "active": torch.tensor([[True, True, False]]),
        "old_log_probs": torch.zeros(1, 3),
        "advantages": torch.tensor([[2.0, -2.0, 999.0]]),
        "old_values": torch.zeros(1, 3),
        "returns": torch.tensor([[1.0, 1.0, 999.0]]),
    }
    values = torch.tensor([[0.5, 2.0, 100.0]], requires_grad=True)
    log_probs = torch.tensor([[np.log(1.5), np.log(0.5), 100.0]], requires_grad=True)
    entropy = torch.tensor([[1.0, 3.0, 999.0]], requires_grad=True)
    losses = clipped_losses(batch, values, log_probs, entropy, 0.2, 0.2)
    assert losses["policy_loss"].item() == pytest.approx(-0.4)
    assert losses["value_loss"].item() == pytest.approx(0.5 * (0.8**2 + 1) / 2)
    assert losses["entropy_loss"].item() == -2
    assert losses["clip_fraction"].item() == 1
    sum(losses.values()).backward()
    assert values.grad[0, 2] == log_probs.grad[0, 2] == entropy.grad[0, 2] == 0


def test_actor_is_local_critic_sees_peers_and_masks_departed_slots():
    env = make_joint_env()
    try:
        model = small_mappo(env, device="cpu")
        observations = torch.zeros(1, 2, 2)
        active = torch.ones(1, 2, dtype=torch.bool)
        actor_before = model.policy.actor(observations).detach()
        critic_before = model.policy.critic(observations, active).detach()
        observations[0, 1] = 1
        torch.testing.assert_close(
            model.policy.actor(observations)[0, 0], actor_before[0, 0]
        )
        assert not torch.allclose(
            model.policy.critic(observations, active), critic_before
        )
        active[0, 1] = False
        masked_before = model.policy.critic(observations, active).detach()
        observations[0, 1] = float("nan")
        torch.testing.assert_close(
            model.policy.critic(observations, active), masked_before
        )
    finally:
        env.close()


def test_rollout_uses_final_timeout_state_and_retains_unclipped_actions(monkeypatch):
    env = make_joint_env()
    try:
        model = small_mappo(env, device="cpu")
        monkeypatch.setattr(
            model, "_values", lambda observations, active: observations[:, 0].copy()
        )
        with torch.no_grad():
            model.policy.actor.network[-1].bias.fill_(10)
        model.learn(4)
        data = model.rollout_buffer.data
        torch.testing.assert_close(
            data["active"], torch.tensor([[True, True], [False, True]])
        )
        torch.testing.assert_close(
            data["next_values"], torch.tensor([[0.0, 0.2], [0.0, 0.3]])
        )
        torch.testing.assert_close(
            data["continuations"], torch.tensor([[False, True], [False, False]])
        )
        assert (
            data["actions"] > 1
        ).all()  # Environment assertions checked executed bounds.
        assert data["rewards"][0, 0] == 1  # Keep the departing agent's terminal reward.
        assert model._active.all()  # Joint auto-reset restores every agent.
    finally:
        env.close()


def test_real_updates_serialization_and_decentralized_prediction(tmp_path):
    env = make_joint_env()
    try:
        model = small_mappo(env, device="cpu")
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.policy.named_parameters()
        }
        model.learn(8)
        for module in ("actor", "critic"):
            assert any(
                not torch.equal(before[name], parameter)
                for name, parameter in model.policy.named_parameters()
                if name.startswith(module)
            )
        assert all(
            torch.isfinite(parameter).all() for parameter in model.policy.parameters()
        )
        observation = np.array([0.2, 0.3], np.float32)
        expected, _ = model.predict(observation, deterministic=True)
        model.save(tmp_path / "mappo")
        restored = MAPPO.load(tmp_path / "mappo", device="cpu")
        actual, _ = restored.predict(observation, deterministic=True)
        np.testing.assert_array_equal(actual, expected)
        assert restored.policy.optimizer.state_dict()["state"]
        assert restored.num_timesteps == 8
        restored.set_env(env)
        restored.learn(4, reset_num_timesteps=False)
        assert restored.num_timesteps == 12
    finally:
        env.close()


def test_train_evaluate_resume_and_periodic_checkpoints(tmp_path):
    results = tmp_path / "results"
    path = tmp_path / "policy"
    mean, _ = train(
        small_mappo,
        make_joint_env,
        path,
        8,
        device="cpu",
        episodes=2,
        checkpoint_interval=4,
        results_dir=results,
    )
    assert mean == 1.5
    report = next(results.iterdir())
    metadata = json.loads((report / "run.json").read_text())
    assert metadata["algorithm"] == "MAPPO"
    assert metadata["status"] == "completed"
    assert (report / "checkpoints" / "step_4" / "policy.zip").is_file()
    env = VecNormalize.load(checkpoint_paths(path)[1], make_joint_env())
    try:
        model = MAPPO.load(checkpoint_paths(path)[0], env=env, device="cpu")
        count = env.obs_rms.count
        assert evaluate(model, env, episodes=2)[0] == 1.5
        assert env.obs_rms.count == count
        assert env.training and env.norm_reward
    finally:
        env.close()
    assert (
        replay(MAPPO, make_joint_env, path, device="cpu", episodes=2, render_mode=None)[
            0
        ]
        == 1.5
    )
    resumed_path = tmp_path / "resumed"
    train(
        small_mappo,
        make_joint_env,
        resumed_path,
        4,
        device="cpu",
        episodes=2,
        results_dir=results,
        resume_from=report / "checkpoints" / "step_4" / "policy",
        model_type=MAPPO,
    )
    restored = MAPPO.load(checkpoint_paths(resumed_path)[0], device="cpu")
    assert restored.num_timesteps == 8


def test_interrupt_saves_custom_optimizer_and_can_resume(tmp_path, monkeypatch):
    original_train = MAPPO.train

    def interrupt_after_update(model):
        original_train(model)
        raise KeyboardInterrupt

    path = tmp_path / "interrupted"
    monkeypatch.setattr(MAPPO, "train", interrupt_after_update)
    with pytest.raises(KeyboardInterrupt):
        train(
            small_mappo,
            make_joint_env,
            path,
            8,
            device="cpu",
            episodes=2,
            results_dir=tmp_path / "results",
        )
    restored = MAPPO.load(checkpoint_paths(path)[0], device="cpu")
    assert restored.num_timesteps == 4
    assert restored.policy.optimizer.state_dict()["state"]
    monkeypatch.setattr(MAPPO, "train", original_train)
    train(
        small_mappo,
        make_joint_env,
        tmp_path / "resumed",
        4,
        device="cpu",
        episodes=2,
        results_dir=tmp_path / "results",
        resume_from=path,
        model_type=MAPPO,
    )
    assert MAPPO.load(tmp_path / "resumed", device="cpu").num_timesteps == 8


def test_normalized_timeout_bootstrap_uses_normalized_terminal_observation(monkeypatch):
    class CheckFinalState(BaseCallback):
        checked = False

        def _on_step(self):
            if self.locals["dones"].all():
                transition = self.model.rollout_buffer.transitions[-1]
                expected = self.locals["infos"][1]["terminal_observation"][0]
                assert transition["next_values"][1] == pytest.approx(expected)
                assert expected != pytest.approx(self.locals["new_obs"][1, 0])
                self.checked = True
            return True

    env = VecNormalize(make_joint_env())
    try:
        model = small_mappo(env, device="cpu")
        monkeypatch.setattr(
            model, "_values", lambda observations, active: observations[:, 0].copy()
        )
        callback = CheckFinalState()
        model.learn(4, callback=callback)
        assert callback.checked
    finally:
        env.close()


def test_target_kl_stops_before_an_excessive_update():
    class ForceLargeKL(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_end(self):
            self.model.rollout_buffer.data["old_log_probs"] += 10

    env = make_joint_env()
    try:
        model = small_mappo(env, device="cpu", target_kl=0.001)
        before = [parameter.detach().clone() for parameter in model.policy.parameters()]
        model.learn(4, callback=ForceLargeKL())
        for expected, actual in zip(before, model.policy.parameters()):
            torch.testing.assert_close(actual, expected)
        assert model._n_updates == 1
    finally:
        env.close()


class AlternativeActor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.output = nn.Linear(observation_dim, action_dim)

    def forward(self, observations):
        return self.output(observations)


class AlternativeCritic(nn.Module):
    def __init__(self, observation_dim, n_agents):
        super().__init__()
        self.output = nn.Linear(observation_dim, 1)

    def forward(self, observations, active):
        shared = (observations * active.unsqueeze(-1)).sum(1, keepdim=True)
        return self.output(observations + shared).squeeze(-1)


def test_architecture_modules_can_be_replaced_and_restored(tmp_path):
    env = make_joint_env()
    try:
        model = build_mappo(
            env,
            n_steps=2,
            n_epochs=1,
            device="cpu",
            verbose=0,
            policy_kwargs={
                "actor_class": AlternativeActor,
                "critic_class": AlternativeCritic,
            },
        )
        model.learn(4)
        model.save(tmp_path / "custom")
        restored = MAPPO.load(tmp_path / "custom", device="cpu")
        assert isinstance(restored.policy.actor, AlternativeActor)
        assert isinstance(restored.policy.critic, AlternativeCritic)
    finally:
        env.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"n_steps": 0},
        {"n_epochs": -1},
        {"batch_size": 0},
        {"gamma": 2},
        {"target_kl": -1},
    ],
)
def test_invalid_hyperparameters(settings):
    with pytest.raises(ValueError):
        MAPPO(**settings)


def test_independent_vector_environments_are_rejected(env_factory):
    env = env_factory(n_envs=2)
    try:
        model = small_mappo(env, device="cpu")
        with pytest.raises(ValueError, match="agent boundary metadata"):
            model.learn(4)
    finally:
        env.close()


def test_agent_count_mismatch_rejected_on_checkpoint_load(tmp_path, env_factory):
    env = make_joint_env()
    try:
        model = small_mappo(env, device="cpu")
        model.save(tmp_path / "policy")
    finally:
        env.close()
    env = env_factory(n_envs=3)
    try:
        with pytest.raises(ValueError, match="agent count"):
            # Match spaces so the joint-agent count is the failing constraint.
            env.observation_space = spaces.Box(-1, 1, (2,), np.float32)
            env.action_space = spaces.Box(-1, 1, (1,), np.float32)
            MAPPO.load(tmp_path / "policy", env=env, device="cpu")
    finally:
        env.close()


class StopAfterStep(BaseCallback):
    def _on_step(self):
        return False


def test_callback_cancellation_stops_before_update():
    env = make_joint_env()
    try:
        model = small_mappo(env, device="cpu")
        model.learn(8, callback=StopAfterStep())
        assert model.num_timesteps == 2
        assert model._n_updates == 0
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["train", "resume", "evaluate", "replay"])
def test_mappo_cli_dispatch(mode, monkeypatch):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli, multi_agent

    monkeypatch.setattr("sys.argv", ["trainer", mode])
    workflow = Mock(return_value=(1.5, 0))
    monkeypatch.setattr(
        cli, "replay" if mode in ("evaluate", "replay") else "train", workflow
    )
    cli.run_cli("mappo", multi_agent=True)
    args, kwargs = workflow.call_args
    assert args[1] is multi_agent.make_mappo_env
    assert args[0] is (MAPPO if mode in ("evaluate", "replay") else build_mappo)
    if mode == "resume":
        assert kwargs["model_type"] is MAPPO


@pytest.mark.parametrize(
    "options", [["--architecture", "entity_mlp"], ["--temporal", "gru"]]
)
def test_mappo_cli_rejects_unimplemented_architectures(options):
    from bluesky_pabloramesc.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args("mappo", multi_agent=True, argv=options)
