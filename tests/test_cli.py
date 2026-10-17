from pathlib import Path

import pytest

from bluesky_pabloramesc.cli import parse_args
from bluesky_pabloramesc.training import EVAL_SEED


@pytest.mark.parametrize("algorithm", ["ppo", "sac", "td3", "ddpg"])
def test_single_agent_defaults(algorithm):
    args = parse_args(algorithm, argv=[])
    assert args.mode == "train"
    assert args.model_path == Path(f"experiments/models/{algorithm}_single_mlp")


def test_legacy_multi_agent_commands():
    args = parse_args("ppo", multi_agent=True, argv=["1000"])
    assert args.timesteps == 1000
    args = parse_args(
        "sac", multi_agent=True, argv=["replay", "model.zip", "stats.pkl"]
    )
    assert args.mode == "replay"
    assert args.model_path == Path("model.zip")
    assert args.stats_path == Path("stats.pkl")


@pytest.mark.parametrize("option", ["--episodes", "--timesteps"])
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
