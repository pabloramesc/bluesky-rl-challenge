"""Command-line interface shared by the experiment entry points."""

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO, SAC

from .models import build_multi_ppo, build_multi_sac, build_ppo, build_sac
from .training import EVAL_SEED, replay, train


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(algorithm, multi_agent=False, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Train or replay a {algorithm.upper()} MLP"
    )
    parser.add_argument(
        "mode", nargs="?", choices=["train", "evaluate", "replay"], default="train"
    )
    parser.add_argument("--timesteps", type=positive_int, default=2_000_000)
    parser.add_argument(
        "--episodes", type=positive_int, default=12 if multi_agent else 5
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=EVAL_SEED,
        help="evaluation scenario seed, independent of the training seed",
    )
    parser.add_argument("--device", default="auto")
    suffix = "marl_competition" if multi_agent else "single_mlp"
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("experiments/models") / f"{algorithm}_{suffix}",
    )
    parser.add_argument("--stats-path", type=Path)

    args = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original multi-agent commands: STEP_COUNT or replay MODEL STATS.
    if multi_agent and args and args[0].isdigit():
        args = ["--timesteps", args[0], *args[1:]]
    if multi_agent and args[:1] == ["replay"]:
        position = 1
        for flag in ("--model-path", "--stats-path"):
            if len(args) > position and not args[position].startswith("-"):
                args.insert(position, flag)
                position += 2
    return parser.parse_args(args)


def run_cli(algorithm, multi_agent=False):
    args = parse_args(algorithm, multi_agent)
    if multi_agent:
        from .multi_agent import make_vec_env as env_factory

        builder = {"ppo": build_multi_ppo, "sac": build_multi_sac}[algorithm]
    else:
        from .single_agent import make_env as env_factory

        builder = {"ppo": build_ppo, "sac": build_sac}[algorithm]

    options = {"device": args.device, "episodes": args.episodes}
    if args.mode in ("evaluate", "replay"):
        result = replay(
            {"ppo": PPO, "sac": SAC}[algorithm],
            env_factory,
            args.model_path,
            stats_path=args.stats_path,
            seed=args.eval_seed,
            render_mode="human" if args.mode == "replay" else None,
            **options,
        )
    else:
        result = train(
            builder,
            env_factory,
            args.model_path,
            args.timesteps,
            seed=args.seed,
            eval_seed=args.eval_seed,
            **options,
        )
    mean, std = result
    print(f"Episode return: {mean:.2f} +/- {std:.2f}")
