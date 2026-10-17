from pathlib import Path

import pytest

from bluesky_pabloramesc.cli import parse_args


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
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
    assert args.eval_seed == 10_000
    args = parse_args("sac", argv=["evaluate", "--eval-seed", "123"])
    assert args.mode == "evaluate"
    assert args.eval_seed == 123
