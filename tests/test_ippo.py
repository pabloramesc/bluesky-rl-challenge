"""Independent networks, masked rollouts, and complete IPPO checkpoints."""

import json
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from test_mappo import make_joint_env

from bluesky_pabloramesc.ippo import IPPO, LocalMLPCritic
from bluesky_pabloramesc.models import build_ippo, build_mappo, build_shared_ppo
from bluesky_pabloramesc.training import checkpoint_paths, evaluate, replay, train


def small_ippo(env, **kwargs):
    return build_ippo(
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=2,
        verbose=0,
        policy_kwargs={
            "actor_kwargs": {"hidden_sizes": (8,)},
            "critic_kwargs": {"hidden_sizes": (8,)},
        },
        **kwargs,
    )


def test_separate_parameters_and_optimizers_with_local_inference():
    env = make_joint_env()
    try:
        model = small_ippo(env, device="cpu")
        left, right = model.policy.learners
        assert not {id(p) for p in left.parameters()} & {
            id(p) for p in right.parameters()
        }
        assert left.optimizer is not right.optimizer
        observations = torch.zeros(1, 2, 2)
        active = torch.ones(1, 2, dtype=torch.bool)
        means = model.policy.distribution(observations).mean.detach()
        values = model.policy.critic(observations, active).detach()
        observations[0, 1] = 1
        torch.testing.assert_close(
            model.policy.distribution(observations).mean[0, 0], means[0, 0]
        )
        torch.testing.assert_close(
            model.policy.critic(observations, active)[0, 0], values[0, 0]
        )
        for index, learner in enumerate(model.policy.learners):
            with torch.no_grad():
                learner.actor.network[-1].bias.fill_(index * 0.25)
                learner.actor.network[-1].weight.zero_()
        actions, _ = model.predict(np.zeros((2, 2), np.float32), deterministic=True)
        np.testing.assert_array_equal(actions[:, 0], [0, 0.25])
        local, _ = right.predict(np.zeros(2, np.float32), deterministic=True)
        np.testing.assert_array_equal(local, actions[1])
        with pytest.raises(ValueError, match="all agent slots"):
            model.predict(np.zeros(2, np.float32))
    finally:
        env.close()


def test_updating_one_learner_cannot_change_peers_or_use_peer_samples():
    env = make_joint_env()
    try:
        model = small_ippo(env, device="cpu")
        model.learn(4)
        left, right = model.policy.learners
        before_left = [p.detach().clone() for p in left.parameters()]
        before_right = [p.detach().clone() for p in right.parameters()]
        model.rollout_buffer.data["rewards"][:, 1] = float("nan")
        model.rollout_buffer.data["returns"][:, 1] = float("nan")
        model.rollout_buffer.data["advantages"][:, 1] = float("nan")
        model._train_agent(0)
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before_left, left.parameters())
        )
        for old, new in zip(before_right, right.parameters()):
            torch.testing.assert_close(old, new)
        assert all(torch.isfinite(p).all() for p in left.parameters())
    finally:
        env.close()


def test_individual_boundaries_and_timeout_bootstrap(monkeypatch):
    env = make_joint_env()
    try:
        model = small_ippo(env, device="cpu")
        monkeypatch.setattr(
            model, "_values", lambda observations, active: observations[:, 0].copy()
        )
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
        assert data["rewards"][0, 0] == 1
        assert model.active_timesteps == 3
        assert model._active.all()
        # Completely inactive learners must not advance Adam or change parameters.
        data["active"][:, 0] = False
        before = [p.detach().clone() for p in model.policy.learners[0].parameters()]
        assert model._train_agent(0) == ([], 0)
        for expected, actual in zip(before, model.policy.learners[0].parameters()):
            torch.testing.assert_close(actual, expected)
    finally:
        env.close()


def test_kl_stopping_is_independent_per_learner():
    class ExcessiveFirstAgentKL(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_end(self):
            self.model.rollout_buffer.data["old_log_probs"][:, 0] += 10

    env = make_joint_env()
    try:
        model = small_ippo(env, device="cpu", target_kl=0.01)
        before = [
            [p.detach().clone() for p in learner.parameters()]
            for learner in model.policy.learners
        ]
        model.learn(4, callback=ExcessiveFirstAgentKL())
        for expected, actual in zip(before[0], model.policy.learners[0].parameters()):
            torch.testing.assert_close(actual, expected)
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before[1], model.policy.learners[1].parameters())
        )
    finally:
        env.close()


def test_independent_checkpoints_preserve_every_optimizer_and_slot(tmp_path):
    env = make_joint_env()
    try:
        model = small_ippo(env, device="cpu")
        model.learn(8)
        observation = np.array([[0.2, 0.3], [0.4, 0.5]], np.float32)
        expected, _ = model.predict(observation, deterministic=True)
        model.save(tmp_path / "ippo")
        restored = IPPO.load(tmp_path / "ippo", device="cpu")
        actual, _ = restored.predict(observation, deterministic=True)
        np.testing.assert_array_equal(actual, expected)
        for original, loaded in zip(model.policy.learners, restored.policy.learners):
            old, new = original.optimizer.state_dict(), loaded.optimizer.state_dict()
            assert old["state"] and old["param_groups"] == new["param_groups"]
            for parameter, state in old["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(new["state"][parameter][name], value)
        assert restored.active_timesteps == 6
        restored.set_env(env)
        restored.learn(4, reset_num_timesteps=False)
        assert restored.num_timesteps == 12
        assert restored.active_timesteps == 9
    finally:
        env.close()


def test_train_evaluate_resume_workflow_and_frozen_normalization(tmp_path):
    path = tmp_path / "policy"
    assert (
        train(
            small_ippo,
            make_joint_env,
            path,
            4,
            device="cpu",
            episodes=2,
            checkpoint_interval=4,
            results_dir=tmp_path / "reports",
        )[0]
        == 1.5
    )
    report = next((tmp_path / "reports").glob("*/run.json"))
    metadata = json.loads(report.read_text())
    assert metadata["algorithm"] == "IPPO"
    assert metadata["joint_steps"] == 2
    assert metadata["active_timesteps"] == 3
    env = VecNormalize.load(checkpoint_paths(path)[1], make_joint_env())
    try:
        model = IPPO.load(path, env=env, device="cpu")
        count = env.obs_rms.count
        assert evaluate(model, env, episodes=2)[0] == 1.5
        assert env.obs_rms.count == count
    finally:
        env.close()
    assert (
        replay(IPPO, make_joint_env, path, device="cpu", episodes=2, render_mode=None)[
            0
        ]
        == 1.5
    )
    train(
        small_ippo,
        make_joint_env,
        tmp_path / "resumed",
        4,
        device="cpu",
        episodes=2,
        results_dir=tmp_path / "reports",
        resume_from=path,
        model_type=IPPO,
    )
    assert IPPO.load(tmp_path / "resumed", device="cpu").num_timesteps == 8


@pytest.mark.parametrize("builder", [build_ippo, build_shared_ppo, build_mappo])
def test_matched_builders_train_under_identical_budgets(builder):
    env = make_joint_env()
    try:
        model = builder(
            env, device="cpu", n_steps=2, batch_size=2, n_epochs=1, verbose=0
        )
        model.learn(5)  # Complete rollouts round up to eight slot transitions.
        assert model.num_timesteps == 8
        assert model.active_timesteps == 6
        assert model.predict(env.reset(), deterministic=True)[0].shape == (2, 1)
        if builder is build_shared_ppo:
            assert isinstance(model.policy.critic, LocalMLPCritic)
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["train", "resume", "evaluate", "replay"])
def test_ippo_cli_dispatch(mode, monkeypatch):
    from bluesky_pabloramesc import cli, multi_agent

    monkeypatch.setattr("sys.argv", ["trainer", mode])
    workflow = Mock(return_value=(1.5, 0))
    monkeypatch.setattr(
        cli, "replay" if mode in ("evaluate", "replay") else "train", workflow
    )
    cli.run_cli("ippo", multi_agent=True)
    args, kwargs = workflow.call_args
    assert args[1] is multi_agent.make_mappo_env
    assert args[0] is (IPPO if mode in ("evaluate", "replay") else build_ippo)
    if mode == "resume":
        assert kwargs["model_type"] is IPPO


@pytest.mark.parametrize(
    "options", [["--architecture", "entity_mlp"], ["--temporal", "gru"]]
)
def test_ippo_cli_rejects_unimplemented_architectures(options):
    from bluesky_pabloramesc.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args("ippo", multi_agent=True, argv=options)


def test_scenario_evaluation_accepts_supersuit_uint8_done_flags(monkeypatch):
    from bluesky_pabloramesc.training import evaluate_scenarios

    env = VecNormalize(make_joint_env())
    try:
        model = small_ippo(env, device="cpu")
        model.learn(4)
        original_step = env.step

        def uint8_step(actions):
            observations, rewards, dones, infos = original_step(actions)
            return observations, rewards, dones.astype(np.uint8), infos

        monkeypatch.setattr(env, "step", uint8_step)
        count = env.obs_rms.count
        rows = evaluate_scenarios(model, env, [2026, 2027])
        assert [row["return"] for row in rows] == [1, 2, 1, 2]
        assert env.obs_rms.count == count
        assert env.training and env.norm_reward
    finally:
        env.close()


def test_scenario_metrics_survive_supersuit_auto_reset(monkeypatch):
    from test_mappo import JointTraffic

    from bluesky_pabloramesc.training import evaluate_scenarios

    class TrafficWithMetrics(JointTraffic):
        def reset(self, seed=None, options=None):
            observations, infos = super().reset(seed=seed, options=options)
            for info in infos.values():
                info.update(flight_time=0, total_reward=0, waypoint_reached=False)
            return observations, infos

        def step(self, actions):
            observations, rewards, terminated, truncated, infos = super().step(actions)
            for agent, info in infos.items():
                info.update(
                    flight_time=self.steps,
                    total_reward=self.steps,
                    waypoint_reached=agent == "early",
                )
            return observations, rewards, terminated, truncated, infos

    monkeypatch.setattr("test_mappo.JointTraffic", TrafficWithMetrics)
    env = VecNormalize(make_joint_env())
    try:
        model = small_ippo(env, device="cpu")
        rows = evaluate_scenarios(model, env, [2026])
        assert [row["flight_time"] for row in rows] == [1, 2]
        assert [row["total_reward"] for row in rows] == [1, 2]
        assert [row["return"] for row in rows] == [1, 2]
        assert [row["waypoint_reached"] for row in rows] == [1, 0]
    finally:
        env.close()
