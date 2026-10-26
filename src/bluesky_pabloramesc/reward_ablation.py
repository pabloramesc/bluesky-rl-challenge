"""Baseline plus one new reward at a time, under a fixed single-agent PPO policy."""

import json
import signal
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .actions import ControlConfig
from .architecture_comparison import (
    _comparison_lock,
    _evaluation,
    _interrupt_handler,
    _prepare_runs,
    _schedule,
    _write_json,
)
from .architecture_tree import ArchitectureVariant, setting_arguments
from .comparison import comparison_command, write_csv
from .rewards import COMPONENTS, RewardConfig

INITIAL_WEIGHTS = {
    "progress": 0.125,
    "predictive": 0.5,
    "obstacle": 0.1,
    "sector": 0.1,
    "time": 0.001,
    "smoothness": 0.001,
}
BEHAVIOR_METRICS = (
    "waypoint_reached",
    "timed_out",
    "intrusion_events",
    "intrusion_time",
    "restricted_area_events",
    "time_in_restricted_area",
    "sector_exit_events",
    "time_outside_sector",
    "flight_time",
    "successful_flight_time",
    "base_return",
)


def ablation_variants(weights):
    """Create seven unique configurations, each with at most one positive weight."""
    if set(weights) != set(COMPONENTS):
        raise ValueError("Supply exactly the six reward component weights")
    nominal = RewardConfig(
        **{f"{name}_weight": value for name, value in weights.items()}
    )
    if any(getattr(nominal, f"{name}_weight") <= 0 for name in COMPONENTS):
        raise ValueError("Ablation weights must be positive")
    return [ArchitectureVariant(("baseline",), RewardConfig().to_dict())] + [
        ArchitectureVariant(
            (name,), RewardConfig(**{f"{name}_weight": weights[name]}).to_dict()
        )
        for name in COMPONENTS
    ]


def ablation_plan(args):
    """Validate the fixed architecture and matched protocol before starting runs."""
    for key in ("timesteps", "jobs", "threads_per_job"):
        if type(getattr(args, key)) is not int or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seeds", "eval_seeds"):
        values = getattr(args, key)
        if (
            not values
            or len(set(values)) != len(values)
            or any(type(value) is not int or value < 0 for value in values)
        ):
            raise ValueError(f"{key} must contain unique nonnegative integers")
    if (
        not args.devices
        or len(args.devices) > args.jobs
        or any(not value.strip() for value in args.devices)
    ):
        raise ValueError(
            "Provide nonempty device names and at least one job slot per device"
        )
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    variants = ablation_variants(args.weights)
    control = ControlConfig(
        heading_action="relative",
        speed_action="absolute",
        speed_min=100,
        speed_max=500,
        observation_profile="control",
        observation_encoding="goal_aligned_conflict",
    )
    architecture = {
        "architecture": "entity_mlp",
        "pooling": "additive_attention",
        "attention_context": "all",
        "temporal": "none",
    }
    protocol = {
        "algorithm": "ppo",
        "multi_agent": False,
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "eval_seeds": args.eval_seeds,
        "control_config": control.to_dict(),
        "architecture_config": architecture,
        "variants": [
            {"name": variant.name, "reward_config": variant.settings}
            for variant in variants
        ],
    }
    return variants, protocol


def ablation_command(args, variant, seed, directory, device):
    """Pass the complete reward configuration, preventing inherited weight changes."""
    command = comparison_command(
        "ppo",
        False,
        "relative",
        "absolute",
        seed,
        args.eval_seeds,
        directory,
        args.timesteps,
        100,
        500,
        device,
        observation_encoding="goal_aligned_conflict",
        decision_interval=10,
    )
    command.extend(
        setting_arguments(
            {
                "architecture": "entity_mlp",
                "pooling": "additive_attention",
                "attention_context": "all",
                "temporal": "none",
            }
        )
    )
    for name, value in variant.settings.items():
        if name != "schema_version":
            command.extend([f"--reward-{name.replace('_', '-')}", str(value)])
    return command


def ablation_evaluation(record, variant, protocol):
    """Validate saved rewards and objective metrics, then derive a common base score."""
    architecture = ArchitectureVariant(variant.path, protocol["architecture_config"])
    rows = _evaluation(record, architecture, protocol)
    if rows is None:
        return None
    report = json.loads(Path(record["report_path"]).read_text())
    if report.get("reward_config") != variant.settings:
        raise ValueError(f"Reward configuration mismatch in {record['report_path']}")
    required = set(BEHAVIOR_METRICS) - {"successful_flight_time", "base_return"}
    config = RewardConfig(**variant.settings)
    for row in rows:
        if not required <= row.keys() or "total_reward" not in row:
            raise ValueError("Missing objective metrics in evaluation")
        for component in COMPONENTS:
            key = f"reward_{component}"
            if config.enabled and key not in row:
                raise ValueError(f"Missing component total: {key}")
            row.setdefault(key, 0.0)
        row["base_return"] = row["total_reward"] - sum(
            row[f"reward_{name}"] for name in COMPONENTS
        )
    return rows


def ablation_summary(episodes, records, variants, seeds):
    """Pair objective metrics within training seeds; shaped return is diagnostic only."""
    summary, paired = [], []
    metrics = (*BEHAVIOR_METRICS, "return", *(f"reward_{name}" for name in COMPONENTS))
    grouped = {
        (variant.name, seed): [
            row
            for row in episodes
            if row["variant"] == variant.name and row["training_seed"] == seed
        ]
        for variant in variants
        for seed in seeds
    }
    for variant in variants:
        available = [seed for seed in seeds if grouped[variant.name, seed]]
        result = {
            "variant": variant.name,
            "status": "completed"
            if len(available) == len(seeds)
            else "partial"
            if available
            else "pending",
            "completed_seeds": len(available),
            "requested_seeds": len(seeds),
            "weight": 0
            if variant.name == "baseline"
            else variant.settings[f"{variant.name}_weight"],
        }
        for metric in metrics:
            seed_means = []
            for seed in available:
                current = grouped[variant.name, seed]
                reference = grouped["baseline", seed]
                if metric == "successful_flight_time":
                    values = [
                        row["flight_time"] for row in current if row["waypoint_reached"]
                    ]
                else:
                    values = [row[metric] for row in current]
                if values:
                    mean = float(np.mean(values))
                    seed_means.append(mean)
                if (
                    metric in BEHAVIOR_METRICS
                    and metric != "successful_flight_time"
                    and reference
                ):
                    baseline = {
                        (row["scenario_seed"], row["agent_index"]): row[metric]
                        for row in reference
                    }
                    if set(baseline) != {
                        (row["scenario_seed"], row["agent_index"]) for row in current
                    }:
                        raise ValueError("Paired scenarios do not match")
                    difference = np.mean(
                        [
                            row[metric]
                            - baseline[row["scenario_seed"], row["agent_index"]]
                            for row in current
                        ]
                    )
                    paired.append(
                        {
                            "variant": variant.name,
                            "training_seed": seed,
                            "metric": metric,
                            "difference_from_baseline": float(difference),
                        }
                    )
            if seed_means:
                result[f"mean_{metric}"] = float(np.mean(seed_means))
                result[f"std_{metric}"] = (
                    float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
                )
                result[f"seeds_with_{metric}"] = len(seed_means)
        runs = [record for record in records if record["variant"] == variant.name]
        for metric in ("training_seconds", "actual_timesteps"):
            values = [
                record[metric]
                for record in runs
                if record["status"] == "completed" and metric in record
            ]
            if values:
                result[f"mean_{metric}"] = float(np.mean(values))
        if not available and any(
            record["status"] in ("failed", "interrupted") for record in runs
        ):
            result["status"] = "failed"
        summary.append(result)
    return summary, paired


def write_ablation_report(directory, episodes, records, variants, seeds):
    """Save behavior-based reports without ranking incompatible shaped returns."""
    summary, paired = ablation_summary(episodes, records, variants, seeds)
    for name, rows in (
        ("episodes", episodes),
        ("runs", records),
        ("summary", summary),
        ("paired", paired),
    ):
        write_csv(directory / f"{name}.csv", rows, fields=None if rows else ["variant"])
    lines = [
        "# One-term reward ablation",
        "",
        "Each variant adds one new term to the unchanged base reward. Compare safety, completion, and efficiency; shaped returns use different objectives and are diagnostic only.",
        "",
        f"Training seeds: {seeds}. Scenarios are paired across variants. Means weight training seeds equally. A single seed provides no estimate of training uncertainty.",
        "",
        "| Variant | Weight | Status | Completion | Intrusion s | Restricted s | Outside s | Successful flight s | Base return |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        values = [
            f"{row[f'mean_{metric}']:.3f}" if f"mean_{metric}" in row else "—"
            for metric in (
                "waypoint_reached",
                "intrusion_time",
                "time_in_restricted_area",
                "time_outside_sector",
                "successful_flight_time",
                "base_return",
            )
        ]
        lines.append(
            f"| {row['variant']} | {row['weight']:g} | {row['status']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "![Behavior metrics](comparison.png)",
        "",
        "See comparison.json for every weight, scale, architecture, and scenario setting; summary.csv for component totals and training budgets; paired.csv for objective differences against baseline. Failed/partial runs are retained. Successful flight time excludes failures and must be read alongside completion.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))
    figure = Figure(figsize=(14, 8), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, metric in zip(
        figure.subplots(2, 3).flat,
        (
            "intrusion_time",
            "time_in_restricted_area",
            "time_outside_sector",
            "waypoint_reached",
            "successful_flight_time",
            "base_return",
        ),
    ):
        plotted = [row for row in summary if f"mean_{metric}" in row]
        axis.bar(
            [row["variant"] for row in plotted],
            [row[f"mean_{metric}"] for row in plotted],
        )
        axis.set_title(metric.replace("_", " "))
        axis.tick_params(axis="x", labelrotation=45)
    figure.savefig(directory / "comparison.png", dpi=150)


def run_reward_ablation(args):
    """Reuse the bounded subprocess scheduler with reward-specific validation."""
    variants, protocol = ablation_plan(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": protocol,
                    "commands": [
                        ablation_command(
                            args,
                            variant,
                            seed,
                            args.output_dir / variant.name / f"seed_{seed}",
                            args.devices[0],
                        )
                        for variant in variants
                        for seed in args.seeds
                    ],
                },
                indent=2,
            )
        )
        return 0
    if args.resume:
        if not (args.output_dir / "comparison.json").is_file():
            raise ValueError("No reward ablation manifest found")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    with _comparison_lock(args.output_dir):
        manifest = args.output_dir / "comparison.json"
        if args.resume and json.loads(manifest.read_text())["protocol"] != protocol:
            raise ValueError("Cannot change reward ablation settings on resume")
        _write_json(manifest, {"protocol": protocol})
        state = args.output_dir / "runs.json"
        previous = json.loads(state.read_text()) if state.is_file() else []
        records, episodes = _prepare_runs(
            args, variants, protocol, previous, evaluation_reader=ablation_evaluation
        )
        _write_json(
            args.output_dir / "execution.json",
            {
                "jobs": args.jobs,
                "devices": args.devices,
                "threads_per_job": args.threads_per_job,
            },
        )
        handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _interrupt_handler)
        try:
            interrupted = _schedule(
                args,
                variants,
                protocol,
                records,
                episodes,
                command_builder=ablation_command,
                evaluation_reader=ablation_evaluation,
            )
        finally:
            signal.signal(signal.SIGTERM, handler)
            write_ablation_report(
                args.output_dir, episodes, records, variants, args.seeds
            )
    print(f"Reward ablation report: {args.output_dir / 'report.md'}", flush=True)
    return (
        130
        if interrupted
        else int(any(record["status"] != "completed" for record in records))
    )
