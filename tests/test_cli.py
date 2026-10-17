from pathlib import Path

import pytest

from bluesky_pabloramesc.cli import parse_args
from bluesky_pabloramesc.training import EVAL_SEED


@pytest.mark.parametrize("algorithm", ["ppo", "sac", "td3", "ddpg"])
def test_single_agent_defaults(algorithm):
    args = parse_args(algorithm, argv=[])
    assert args.mode == "train"
    assert args.timesteps == 100_000
    assert args.checkpoint_interval == 10_000
    assert args.seed == 42
    assert args.model_path == Path(f"experiments/models/{algorithm}_single_mlp")


@pytest.mark.parametrize("observation", [False, True])
@pytest.mark.parametrize("seeds", [None, [7], [0, 1, 2]])
def test_comparisons_use_one_training_seed_unless_explicit(
    observation, seeds, tmp_path, monkeypatch
):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli, comparison

    workflow = Mock()
    monkeypatch.setattr(comparison, "run_comparison", workflow)
    arguments = ["--output-dir", str(tmp_path / "comparison")]
    if seeds is not None:
        arguments += ["--seeds", *map(str, seeds)]
    cli.run_comparison_cli(arguments, observation=observation)
    assert workflow.call_args.args[0].seeds == ([0] if seeds is None else seeds)
    assert workflow.call_args.kwargs["observation"] == observation


def test_legacy_multi_agent_commands():
    args = parse_args("ppo", multi_agent=True, argv=["1000"])
    assert args.timesteps == 1000
    args = parse_args(
        "sac", multi_agent=True, argv=["replay", "model.zip", "stats.pkl"]
    )
    assert args.mode == "replay"
    assert args.model_path == Path("model.zip")
    assert args.stats_path == Path("stats.pkl")


@pytest.mark.parametrize("observation", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_comparison_output_directory_defaults_and_override(
    observation, explicit, monkeypatch
):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli, comparison

    workflow = Mock()
    monkeypatch.setattr(comparison, "run_comparison", workflow)
    arguments = ["--output-dir", "custom/results"] if explicit else []
    cli.run_comparison_cli(arguments, observation=observation)
    expected = (
        Path("custom/results")
        if explicit
        else Path("experiments/results")
        / ("observation_comparison" if observation else "action_comparison")
    )
    assert workflow.call_args.args[0].output_dir == expected


@pytest.mark.parametrize(
    "option", ["--episodes", "--timesteps", "--checkpoint-interval"]
)
@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_invalid_counts(option, value):
    with pytest.raises(SystemExit):
        parse_args("ppo", argv=[option, value])


def test_evaluation_seed_is_independent_of_training_seed():
    args = parse_args("ppo", argv=["--seed", "42"])
    assert args.seed == 42
    assert args.eval_seed == EVAL_SEED
    args = parse_args("sac", argv=["evaluate", "--eval-seed", "123"])
    assert args.mode == "evaluate"
    assert args.eval_seed == 123


@pytest.mark.parametrize("algorithm", ["td3", "ddpg"])
@pytest.mark.parametrize("mode", ["train", "evaluate", "replay"])
def test_deterministic_cli_dispatch(algorithm, mode, monkeypatch):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli, single_agent

    monkeypatch.setattr("sys.argv", ["trainer", mode])
    workflow = Mock(return_value=(6.0, 0.0))
    monkeypatch.setattr(cli, "train" if mode == "train" else "replay", workflow)
    cli.run_cli(algorithm)
    args, kwargs = workflow.call_args
    if mode == "train":
        assert args[0] is getattr(cli, f"build_{algorithm}")
    else:
        assert args[0] is getattr(cli, algorithm.upper())
        assert kwargs["render_mode"] == ("human" if mode == "replay" else None)
    assert args[1] is single_agent.make_env
    assert args[2] == Path(f"experiments/models/{algorithm}_single_mlp")


@pytest.mark.parametrize("algorithm", ["ppo", "sac", "td3", "ddpg"])
def test_resume_cli_dispatch(algorithm, monkeypatch):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "trainer",
            "resume",
            "--model-path",
            "old",
            "--output-path",
            "new",
            "--timesteps",
            "1000",
            "--stats-path",
            "stats.pkl",
        ],
    )
    workflow = Mock(return_value=(6.0, 0.0))
    monkeypatch.setattr(cli, "train", workflow)
    cli.run_cli(algorithm)
    args, kwargs = workflow.call_args
    assert args[2] == Path("new")
    assert args[3] == 1000
    assert kwargs["resume_from"] == Path("old")
    assert kwargs["resume_stats_path"] == Path("stats.pkl")
    assert kwargs["model_type"] is getattr(cli, algorithm.upper())


def test_checkpoint_interval_cli_dispatch(monkeypatch):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli

    monkeypatch.setattr("sys.argv", ["trainer", "--checkpoint-interval", "500"])
    workflow = Mock(return_value=(6.0, 0.0))
    monkeypatch.setattr(cli, "train", workflow)
    cli.run_cli("ppo")
    assert workflow.call_args.kwargs["checkpoint_interval"] == 500


def test_ctrl_c_cli_exits_cleanly(monkeypatch, capsys):
    from unittest.mock import Mock

    from bluesky_pabloramesc import cli

    monkeypatch.setattr("sys.argv", ["trainer"])
    monkeypatch.setattr(cli, "train", Mock(side_effect=KeyboardInterrupt))
    cli.run_cli("ppo")
    assert "Training interrupted." in capsys.readouterr().out


def test_control_cli_validation_and_saved_defaults(tmp_path):
    import json

    from bluesky_pabloramesc.actions import ControlConfig
    from bluesky_pabloramesc.training import control_config_path

    with pytest.raises(SystemExit):
        parse_args("ppo", argv=["--heading-action", "absolute"])
    with pytest.raises(SystemExit):
        parse_args("ppo", argv=["--speed-action", "absolute"])
    config = ControlConfig(
        heading_action="absolute",
        speed_action="absolute",
        observation_profile="control",
        speed_min=100,
        speed_max=500,
    )
    path = tmp_path / "policy"
    control_config_path(path).write_text(json.dumps(config.to_dict()))
    args = parse_args("ppo", argv=["evaluate", "--model-path", str(path)])
    assert args.control_config == config
    with pytest.raises(SystemExit):
        parse_args(
            "ppo",
            argv=["resume", "--model-path", str(path), "--heading-action", "relative"],
        )


def test_observation_encoding_restored_and_cannot_be_overridden(tmp_path):
    import json

    from bluesky_pabloramesc.actions import ControlConfig
    from bluesky_pabloramesc.training import control_config_path

    path = tmp_path / "policy"
    config = ControlConfig(observation_encoding="goal_aligned")
    control_config_path(path).write_text(json.dumps(config.to_dict()))
    args = parse_args("ppo", argv=["evaluate", "--model-path", str(path)])
    assert args.control_config == config
    with pytest.raises(SystemExit):
        parse_args(
            "ppo",
            argv=[
                "evaluate",
                "--model-path",
                str(path),
                "--observation-encoding",
                "ego_cartesian",
            ],
        )
