"""Command-line interface shared by the experiment entry points."""

import argparse
import sys
from functools import partial
from pathlib import Path

from stable_baselines3 import DDPG, PPO, SAC, TD3

from .models import (
    build_ddpg,
    build_multi_ppo,
    build_multi_sac,
    build_ppo,
    build_sac,
    build_td3,
)
from .training import EVAL_SEED, replay, train


def positive_int(value):
    """Parse a positive integer for a command-line option.

    Args:
        value: String supplied by argparse.

    Returns:
        Parsed integer greater than zero.

    Raises:
        ValueError: The value cannot be converted to an integer.
        argparse.ArgumentTypeError: The parsed integer is zero or negative.
    """
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(algorithm, multi_agent=False, argv=None):
    """Parse experiment options, including legacy multi-agent arguments.

    Args:
        algorithm: Algorithm name used for help text and checkpoint defaults.
        multi_agent: Whether to use shared-policy defaults and legacy syntax.
        argv: Argument list, or None to read sys.argv.

    Returns:
        Namespace containing validated command-line options.

    Raises:
        SystemExit: Help is requested or arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description=f"Train or replay a {algorithm.upper()} policy"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["train", "resume", "evaluate", "replay"],
        default="train",
    )
    parser.add_argument("--timesteps", type=positive_int, default=100_000)
    parser.add_argument(
        "--episodes", type=positive_int, default=12 if multi_agent else 5
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=EVAL_SEED,
        help="evaluation scenario seed, independent of the training seed",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--architecture",
        choices=["mlp", "entity_mlp", "entity_transformer", "entity_gnn"],
    )
    from .entities import ATTENTION_CONTEXTS, GNN_TYPES, GRAPH_TYPES, POOLING_METHODS

    parser.add_argument("--pooling", choices=POOLING_METHODS)
    parser.add_argument("--attention-context", choices=ATTENTION_CONTEXTS)
    parser.add_argument("--transformer-layers", type=positive_int)
    parser.add_argument("--gnn-type", choices=GNN_TYPES)
    parser.add_argument("--gnn-layers", type=positive_int)
    parser.add_argument("--graph", choices=GRAPH_TYPES)
    parser.add_argument("--graph-radius-nm", type=float)
    parser.add_argument("--graph-k", type=positive_int)
    from .temporal import TEMPORAL_METHODS

    parser.add_argument("--temporal", choices=TEMPORAL_METHODS)
    parser.add_argument("--history-length", type=positive_int)
    suffix = "marl_competition" if multi_agent else "single_mlp"
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("experiments/models") / f"{algorithm}_{suffix}",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=positive_int,
        default=10_000,
        help="additional steps between checkpoints, rounded up to a rollout boundary",
    )
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument(
        "--output-path",
        type=Path,
        help="save resumed training to a different checkpoint path",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))

    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        help="explicit paired scenario seeds for final training evaluation",
    )
    parser.add_argument("--decision-interval", type=positive_int)
    parser.add_argument("--heading-action", choices=["relative", "absolute"])
    parser.add_argument("--speed-action", choices=["relative", "absolute"])
    parser.add_argument("--observation-profile", choices=["legacy", "control"])
    from .observations import OBSERVATION_ENCODINGS

    parser.add_argument("--observation-encoding", choices=OBSERVATION_ENCODINGS)
    parser.add_argument(
        "--speed-min", type=float, help="minimum commanded CAS in knots"
    )
    parser.add_argument(
        "--speed-max", type=float, help="maximum commanded CAS in knots"
    )

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
    parsed = parser.parse_args(args)
    from .training import load_architecture_config

    saved_architecture = (
        load_architecture_config(parsed.model_path)
        if parsed.mode != "train"
        else {"architecture": "mlp", "pooling": "mean_max"}
    )
    if (
        parsed.attention_context is not None
        and (parsed.pooling or saved_architecture["pooling"]) != "additive_attention"
    ):
        parser.error("--attention-context requires --pooling additive_attention")
    if saved_architecture["pooling"] == "additive_attention":
        saved_architecture.setdefault("attention_context", "ownship_goal")
    else:
        saved_architecture["attention_context"] = "ownship_goal"
    if (
        parsed.transformer_layers is not None
        and (parsed.architecture or saved_architecture["architecture"])
        != "entity_transformer"
    ):
        parser.error("--transformer-layers requires --architecture entity_transformer")
    import math

    gnn_defaults = {
        "gnn_type": "gat",
        "gnn_layers": 1,
        "graph": "fully_connected",
        "graph_radius_nm": 25.0,
        "graph_k": 3,
    }
    requested_architecture = parsed.architecture or saved_architecture["architecture"]
    for key, default in gnn_defaults.items():
        if getattr(parsed, key) is not None and requested_architecture != "entity_gnn":
            parser.error(
                f"--{key.replace('_', '-')} requires --architecture entity_gnn"
            )
        saved_architecture.setdefault(key, default)
    requested_graph = parsed.graph or saved_architecture["graph"]
    if parsed.graph_radius_nm is not None and requested_graph != "distance":
        parser.error("--graph-radius-nm requires --graph distance")
    if parsed.graph_k is not None and requested_graph != "knn":
        parser.error("--graph-k requires --graph knn")
    radius = (
        parsed.graph_radius_nm
        if parsed.graph_radius_nm is not None
        else saved_architecture["graph_radius_nm"]
    )
    if not math.isfinite(radius) or radius <= 0:
        parser.error("--graph-radius-nm must be finite and positive")
    saved_architecture.setdefault("temporal", "none")
    saved_architecture.setdefault("history_length", 8)
    if (
        parsed.history_length is not None
        and (parsed.temporal or saved_architecture["temporal"]) == "none"
    ):
        parser.error("--history-length requires a temporal encoder")
    saved_architecture.setdefault("transformer_layers", 1)
    for key in (
        "architecture",
        "pooling",
        "attention_context",
        "transformer_layers",
        "temporal",
        "history_length",
        *gnn_defaults,
    ):
        value = getattr(parsed, key)
        if (
            parsed.mode != "train"
            and value is not None
            and value != saved_architecture[key]
        ):
            parser.error("checkpoint architecture settings cannot be overridden")
        setattr(parsed, key, value if value is not None else saved_architecture[key])
    if parsed.architecture == "mlp" and parsed.pooling != "mean_max":
        parser.error("--pooling requires an entity architecture")
    if parsed.pooling == "pma" and parsed.architecture not in (
        "entity_transformer",
        "entity_gnn",
    ):
        parser.error("--pooling pma requires entity_transformer or entity_gnn")

    if parsed.output_path is not None and parsed.mode != "resume":
        parser.error("--output-path is only supported in resume mode")
    if parsed.eval_seeds is not None and parsed.mode in ("evaluate", "replay"):
        parser.error("--eval-seeds is supported for train and resume")
    from .actions import ControlConfig
    from .training import load_control_config

    saved = (
        load_control_config(parsed.model_path)
        if parsed.mode != "train"
        else ControlConfig()
    )
    values = saved.to_dict()
    for key in (
        "decision_interval",
        "heading_action",
        "speed_action",
        "observation_profile",
        "observation_encoding",
        "speed_min",
        "speed_max",
    ):
        if getattr(parsed, key) is not None:
            values[key] = getattr(parsed, key)
    try:
        parsed.control_config = ControlConfig(**values)
        if parsed.mode != "train" and parsed.control_config != saved:
            parser.error("checkpoint control settings cannot be overridden")
    except ValueError as error:
        parser.error(str(error))
    for key in (
        "decision_interval",
        "heading_action",
        "speed_action",
        "observation_profile",
        "observation_encoding",
        "speed_min",
        "speed_max",
    ):
        setattr(parsed, key, getattr(parsed.control_config, key))
    return parsed


def run_cli(algorithm, multi_agent=False):
    """Run the requested workflow and print the final episode return.

    Args:
        algorithm: One of "ppo", "sac", "td3", or "ddpg".
        multi_agent: Whether to use the shared PPO/SAC environment and builders.
    """
    args = parse_args(algorithm, multi_agent)
    if multi_agent:
        from .multi_agent import make_vec_env as env_factory

        builder = {"ppo": build_multi_ppo, "sac": build_multi_sac}[algorithm]
    else:
        from .single_agent import make_env as env_factory

        builder = {
            "ppo": build_ppo,
            "sac": build_sac,
            "td3": build_td3,
            "ddpg": build_ddpg,
        }[algorithm]

    if multi_agent and args.decision_interval != 10:
        raise ValueError("decision interval tuning supports single-agent runs only")
    if not multi_agent and args.decision_interval != 10:
        from .models import GAMMA

        builder = partial(builder, gamma=GAMMA ** (args.decision_interval / 10))

    from .actions import ControlConfig

    if args.control_config != ControlConfig():
        env_factory = partial(env_factory, control_config=args.control_config)

    if args.architecture in ("entity_mlp", "entity_transformer", "entity_gnn"):
        env_factory = partial(env_factory, entity_observations=True)
        if args.architecture == "entity_gnn" and args.graph == "distance":
            env_factory = partial(env_factory, graph_radius_nm=args.graph_radius_nm)
        if args.architecture == "entity_gnn" and args.graph == "knn":
            env_factory = partial(env_factory, graph_k=args.graph_k)
        builder = partial(
            builder,
            architecture=args.architecture,
            pooling=args.pooling,
            attention_context=args.attention_context,
            transformer_layers=args.transformer_layers,
            **(
                {
                    "gnn_type": args.gnn_type,
                    "gnn_layers": args.gnn_layers,
                    "graph": args.graph,
                    "graph_radius_nm": args.graph_radius_nm,
                    "graph_k": args.graph_k,
                }
                if args.architecture == "entity_gnn"
                else {}
            ),
        )

    if args.temporal != "none":
        env_factory = partial(env_factory, history_length=args.history_length)
        builder = partial(
            builder, temporal=args.temporal, history_length=args.history_length
        )

    model_type = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}[algorithm]
    options = {
        "device": args.device,
        "episodes": args.episodes,
        "control_config": args.control_config,
    }
    if args.mode in ("evaluate", "replay"):
        result = replay(
            model_type,
            env_factory,
            args.model_path,
            stats_path=args.stats_path,
            seed=args.eval_seed,
            render_mode="human" if args.mode == "replay" else None,
            **options,
        )
    else:
        resume_options = {}
        if args.mode == "resume":
            resume_options = {
                "resume_from": args.model_path,
                "model_type": model_type,
                "resume_stats_path": args.stats_path,
            }
        try:
            result = train(
                builder,
                env_factory,
                args.output_path or args.model_path,
                args.timesteps,
                seed=args.seed,
                eval_seed=args.eval_seed,
                results_dir=args.results_dir,
                checkpoint_interval=args.checkpoint_interval,
                evaluation_seeds=args.eval_seeds,
                **resume_options,
                **options,
            )
        except KeyboardInterrupt:
            print("Training interrupted.")
            return
    mean, std = result
    print(f"Episode return: {mean:.2f} +/- {std:.2f}")


def run_comparison_cli(argv=None, *, observation=False, frequency=False):
    """Parse comparison settings and launch isolated runs sequentially."""
    from .comparison import run_comparison

    parser = argparse.ArgumentParser(
        description=(
            "Compare single-agent PPO decisions every 5, 10, and 20 seconds"
            if frequency
            else "Compare geometry encodings and predictive conflict observations"
            if observation
            else "Compare relative/absolute heading and speed"
        )
    )
    parser.add_argument(
        "--algorithm", choices=["ppo", "sac", "td3", "ddpg"], default="ppo"
    )
    parser.add_argument("--multi-agent", action="store_true")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="training seeds (default: 0; one training per variant and seed)",
    )
    parser.add_argument(
        "--eval-seeds", nargs="+", type=int, default=list(range(2026, 2036))
    )
    parser.add_argument("--timesteps", type=positive_int, default=100_000)
    parser.add_argument("--speed-min", type=float, default=100)
    parser.add_argument("--speed-max", type=float, default=500)
    parser.add_argument("--device", default="auto")
    default_output = Path("experiments/results") / (
        "decision_interval_comparison"
        if frequency
        else "observation_comparison"
        if observation
        else "action_comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"comparison output directory (default: {default_output})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed runs in an existing comparison directory",
    )
    if frequency:
        parser.add_argument(
            "--intervals", nargs="+", type=positive_int, default=[5, 10, 20]
        )
        parser.add_argument(
            "--observation-encoding",
            default="goal_aligned_conflict",
            choices=["goal_aligned_conflict"],
        )
    if observation:
        from .observations import DEFAULT_OBSERVATION_ENCODINGS, OBSERVATION_ENCODINGS

        parser.add_argument(
            "--encodings",
            nargs="+",
            choices=OBSERVATION_ENCODINGS,
            default=list(DEFAULT_OBSERVATION_ENCODINGS),
            help="observation variants to compare; the first is the paired baseline",
        )
    if observation or frequency:
        parser.add_argument(
            "--heading-action", choices=["relative", "absolute"], default="relative"
        )
        parser.add_argument(
            "--speed-action",
            choices=["relative", "absolute"],
            default="absolute" if frequency else "relative",
        )
        parser.add_argument(
            "--observation-profile",
            choices=["legacy", "control"],
            default="control" if frequency else "legacy",
        )
    args = parser.parse_args(argv)
    variant_count = (
        len(args.intervals) if frequency else len(args.encodings) if observation else 4
    )
    print(
        f"Comparison: {len(args.seeds)} training seed(s) × {variant_count} variants; completed runs are reused with --resume.",
        flush=True,
    )
    try:
        if frequency:
            run_comparison(args, frequency=True)
        else:
            run_comparison(args, observation=observation)
    except ValueError as error:
        parser.error(str(error))
