"""Decision timing, reward units, and paired comparison budgets."""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from bluesky_pabloramesc.actions import ControlConfig
from bluesky_pabloramesc.cli import parse_args, run_comparison_cli


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_interval_requires_positive_integer(value):
    with pytest.raises(ValueError, match="decision interval"):
        ControlConfig(decision_interval=value)


def test_interval_checkpoint_restoration_and_override(tmp_path):
    path = tmp_path / "policy"
    path.with_name("policy_control.json").write_text(
        json.dumps(ControlConfig(decision_interval=5).to_dict())
    )
    args = parse_args("ppo", argv=["evaluate", "--model-path", str(path)])
    assert args.control_config.decision_interval == 5
    with pytest.raises(SystemExit):
        parse_args(
            "ppo",
            argv=["evaluate", "--model-path", str(path), "--decision-interval", "20"],
        )


@pytest.mark.parametrize("interval,elapsed", [(5, 5), (10, 10), (20, 20), (20, 3)])
def test_environment_passes_interval_and_scales_only_drift(
    monkeypatch, interval, elapsed
):
    from bluesky_pabloramesc import observations, single_agent

    class SyntheticCompetition(gym.Env):
        def __init__(self, render_mode, action_frequency):
            self.action_frequency = action_frequency
            self.observation_space = gym.spaces.Dict(
                {"value": gym.spaces.Box(-1, 1, (1,), dtype=np.float32)}
            )
            self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
            self.metrics = {"agent": {"flight_time": elapsed}}
            self._reward_baseline = {"flight_time": 0}
            self.drift_penalty = -2

        def _get_reward(self, ac_id):
            return 100 + self.drift_penalty * 3

    monkeypatch.setitem(
        sys.modules,
        "bluesky_gym.envs.competition_env",
        SimpleNamespace(CompetitionEnv=SyntheticCompetition),
    )
    monkeypatch.setattr(
        observations,
        "extend_observation_space",
        lambda space, config, entity_observations=False: space,
    )
    env = single_agent.make_env(
        control_config=ControlConfig(decision_interval=interval)
    )
    try:
        base = env.envs[0].unwrapped
        assert base.action_frequency == interval
        assert base._get_reward("agent") == pytest.approx(100 - 6 * elapsed / 10)
        assert base.drift_penalty == -2
    finally:
        env.close()


def test_frequency_comparison_budgets_pairing_and_reuse(tmp_path, monkeypatch):
    commands = []

    def run(command, check):
        commands.append(command)
        root = Path(command[command.index("--results-dir") + 1]) / "run"
        root.mkdir(parents=True)
        interval = int(command[command.index("--decision-interval") + 1])
        (root / "run.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "evaluation_scenarios": [
                        {"scenario_seed": 100, "return": interval}
                    ],
                }
            )
        )

    monkeypatch.setattr("bluesky_pabloramesc.comparison.subprocess.run", run)
    arguments = [
        "--output-dir",
        str(tmp_path / "compare"),
        "--timesteps",
        "100",
        "--eval-seeds",
        "100",
        "--device",
        "cpu",
    ]
    run_comparison_cli(arguments, frequency=True)
    assert len(commands) == 3
    for command, interval, budget in zip(commands, [5, 10, 20], [200, 100, 50]):
        for flag, expected in [
            ("--decision-interval", str(interval)),
            ("--timesteps", str(budget)),
            ("--heading-action", "relative"),
            ("--speed-action", "absolute"),
            ("--observation-encoding", "goal_aligned_conflict"),
        ]:
            assert command[command.index(flag) + 1] == expected
    with (tmp_path / "compare" / "paired.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["difference_from_interval_10s"]) for row in rows] == [-5, 0, 10]
    commands.clear()
    run_comparison_cli([*arguments, "--resume"], frequency=True)
    assert commands == []
    with pytest.raises(SystemExit):
        run_comparison_cli(
            [*arguments, "--resume", "--intervals", "5", "10", "30"], frequency=True
        )
    assert commands == []


def test_cli_adjusts_policy_discount_for_interval(monkeypatch):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli
    from bluesky_pabloramesc.models import GAMMA

    workflow = Mock(return_value=(0, 0))
    monkeypatch.setattr(cli, "train", workflow)
    monkeypatch.setattr(sys, "argv", ["trainer", "--decision-interval", "5"])
    cli.run_cli("ppo")
    builder = workflow.call_args.args[0]
    assert builder.keywords["gamma"] == pytest.approx(GAMMA**0.5)
    assert workflow.call_args.kwargs["control_config"].decision_interval == 5


def test_frequency_comparison_defaults_to_100k_steps(monkeypatch):
    from unittest.mock import Mock

    from bluesky_pabloramesc import comparison

    workflow = Mock()
    monkeypatch.setattr(comparison, "run_comparison", workflow)
    run_comparison_cli([], frequency=True)
    assert workflow.call_args.args[0].timesteps == 100_000
