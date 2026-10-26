"""Reward CLI, reporting, checkpoint restoration, and compatibility checks."""

import csv
import json
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bluesky_pabloramesc.cli import parse_args
from bluesky_pabloramesc.models import build_ppo
from bluesky_pabloramesc.reporting import EpisodeHistory
from bluesky_pabloramesc.rewards import COMPONENTS, RewardConfig
from bluesky_pabloramesc.training import (
    checkpoint_reward_config,
    evaluate_scenarios,
    load_reward_config,
    replay,
    reward_config_path,
    train,
    validate_policy_reward_config,
)


def test_cli_reward_options_and_checkpoint_restoration(tmp_path):
    path = tmp_path / "policy.v2.zip"
    config = RewardConfig(
        progress_weight=2, time_weight=0.01, separation_distance_m=12000
    )
    parsed = parse_args(
        "ppo",
        argv=[
            "--reward-progress-weight",
            "2",
            "--reward-time-weight",
            "0.01",
            "--reward-separation-distance-m",
            "12000",
        ],
    )
    assert parsed.reward_config == config
    reward_config_path(path).write_text(json.dumps(config.to_dict()))
    for mode in ("resume", "evaluate", "replay"):
        parsed = parse_args("ppo", argv=[mode, "--model-path", str(path)])
        assert parsed.reward_config == config
        with pytest.raises(SystemExit):
            parse_args(
                "ppo",
                argv=[mode, "--model-path", str(path), "--reward-progress-weight", "0"],
            )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("progress-weight", "-1"),
        ("time-weight", "nan"),
        ("prediction-horizon-s", "0"),
        ("speed-scale-knots", "inf"),
    ],
)
def test_cli_rejects_invalid_rewards(flag, value):
    with pytest.raises(SystemExit):
        parse_args("ppo", argv=[f"--reward-{flag}", value])


def test_cli_passes_rewards_to_workflow(monkeypatch):
    from bluesky_pabloramesc import cli

    workflow = Mock(return_value=(0, 0))
    monkeypatch.setattr(cli, "train", workflow)
    monkeypatch.setattr(sys, "argv", ["trainer", "--reward-time-weight", "0.1"])
    cli.run_cli("ppo")
    assert workflow.call_args.kwargs["reward_config"] == RewardConfig(time_weight=0.1)


def test_legacy_checkpoints_and_archive_sidecar_mismatch(tmp_path):
    path = tmp_path / "old"
    assert load_reward_config(path) == RewardConfig()
    validate_policy_reward_config(SimpleNamespace(), RewardConfig())
    config = RewardConfig(time_weight=0.01)
    reward_config_path(path).write_text(json.dumps(config.to_dict()))
    assert checkpoint_reward_config(path) == config
    with pytest.raises(ValueError, match="requested settings"):
        checkpoint_reward_config(path, RewardConfig())
    with pytest.raises(ValueError, match="does not match sidecar"):
        validate_policy_reward_config(
            SimpleNamespace(reward_config=config.to_dict()), RewardConfig()
        )


def test_train_resume_and_replay_restore_rewards(tmp_path, env_factory):
    seen = []
    config = RewardConfig(progress_weight=1, predictive_weight=0.1)

    def factory(render_mode=None, reward_config=None):
        seen.append(reward_config)
        return env_factory(render_mode=render_mode)

    path = tmp_path / "policy"
    builder = partial(build_ppo, n_steps=4, batch_size=4, n_epochs=1, verbose=0)
    options = {
        "timesteps": 4,
        "episodes": 1,
        "device": "cpu",
        "results_dir": tmp_path / "results",
        "checkpoint_interval": 1,
    }
    assert train(builder, factory, path, reward_config=config, **options)[0] == 6
    assert load_reward_config(path) == config
    archive = PPO.load(path, device="cpu")
    assert archive.reward_config == config.to_dict()
    report = next((tmp_path / "results").iterdir())
    assert (
        json.loads((report / "run.json").read_text())["reward_config"]
        == config.to_dict()
    )
    assert load_reward_config(report / "policy") == config
    periodic = next((report / "checkpoints").glob("*/policy_reward.json"))
    assert json.loads(periodic.read_text()) == config.to_dict()
    assert (
        replay(PPO, factory, path, device="cpu", episodes=1, render_mode=None)[0] == 6
    )
    # Omitted configuration is restored automatically on resume.
    assert (
        train(
            builder,
            factory,
            tmp_path / "resumed",
            resume_from=path,
            model_type=PPO,
            **options,
        )[0]
        == 6
    )
    assert load_reward_config(tmp_path / "resumed") == config
    assert seen == [config, config, config]
    with pytest.raises(ValueError, match="requested settings"):
        replay(PPO, factory, path, reward_config=RewardConfig())
    assert len(seen) == 3  # Fail before creating an environment.
    reward_config_path(path).unlink()
    with pytest.raises(ValueError, match="does not match sidecar"):
        replay(PPO, factory, path, device="cpu")


def test_episode_history_records_component_totals(tmp_path):
    history = EpisodeHistory(tmp_path)
    history.num_timesteps = 10
    totals = {name: float(index) for index, name in enumerate(COMPONENTS)}
    history.locals = {
        "infos": [{"episode": {"r": 3, "l": 2}, "reward_component_totals": totals}]
    }
    assert history._on_step()
    with (tmp_path / "episodes.csv").open() as stream:
        row = next(csv.DictReader(stream))
    for name, value in totals.items():
        assert float(row[f"reward_{name}"]) == value


def test_scenario_evaluation_retains_raw_component_totals():
    class Episode(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.zeros(1, np.float32), {}

        def step(self, action):
            self.steps += 1
            done = self.steps == 2
            info = {
                "flight_time": self.steps,
                "waypoint_reached": int(done),
                "reward_component_totals": {
                    "progress": self.steps * 0.5,
                    "time": -self.steps * 0.1,
                },
            }
            return np.zeros(1, np.float32), 2, done, False, info

    model = SimpleNamespace(
        predict=lambda observations, **kwargs: (np.zeros((1, 2)), None)
    )
    env = VecNormalize(DummyVecEnv([Episode]))
    try:
        rows = evaluate_scenarios(model, env, [123])
        assert rows[0]["return"] == 4
        assert rows[0]["reward_progress"] == 1
        assert rows[0]["reward_time"] == -0.2
        assert env.training and env.norm_reward
    finally:
        env.close()


def test_episode_history_keeps_departing_agent_totals_until_joint_completion(tmp_path):
    history = EpisodeHistory(tmp_path)
    history.num_timesteps = 10
    history.locals = {
        "infos": [
            {"reward_component_totals": {"time": -3}},
            {"reward_component_totals": {"time": -3}},
        ]
    }
    history._on_step()
    history.locals = {"infos": [{}, {"reward_component_totals": {"time": -10}}]}
    history._on_step()
    history.locals = {
        "infos": [
            {"episode": {"r": -2, "l": 10}},
            {"episode": {"r": -9, "l": 10}, "reward_component_totals": {"time": -10}},
        ]
    }
    history._on_step()
    assert [row["reward_time"] for row in history.rows] == [-3, -10]
    assert history.reward_totals == {}
    # A later unshaped episode must not inherit the previous episode's totals.
    history.locals = {"infos": [{"episode": {"r": 1, "l": 2}}]}
    history._on_step()
    assert "reward_time" not in history.rows[-1]
