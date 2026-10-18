"""Run a paired 2x2 action ablation in isolated simulator processes."""

import csv
import json
import subprocess
import sys
from itertools import product

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .actions import ControlConfig

VARIANTS = tuple(product(("relative", "absolute"), repeat=2))


def comparison_command(
    algorithm,
    multi_agent,
    heading,
    speed,
    seed,
    eval_seeds,
    directory,
    timesteps,
    speed_min,
    speed_max,
    device,
    observation_encoding="upstream",
    observation_profile="control",
    decision_interval=None,
):
    """Build a standard training command, using identical observations and bounds."""
    module = f"experiments.train_{'zoo' if multi_agent else 'single'}_{algorithm}"
    return [
        sys.executable,
        "-m",
        module,
        "--heading-action",
        heading,
        "--speed-action",
        speed,
        "--observation-profile",
        observation_profile,
        "--observation-encoding",
        observation_encoding,
        "--speed-min",
        str(speed_min),
        "--speed-max",
        str(speed_max),
        "--seed",
        str(seed),
        "--timesteps",
        str(timesteps),
        "--model-path",
        str(directory / "policy"),
        "--results-dir",
        str(directory / "reports"),
        "--device",
        device,
        *(["--decision-interval", str(decision_interval)] if decision_interval else []),
        "--eval-seeds",
        *map(str, eval_seeds),
    ]


def write_csv(path, rows, fields=None):
    """Write rows with a union of fields, including optional simulator metrics."""
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, baseline_variant="relative_relative"):
    """Aggregate within training seeds before measuring across-seed variation."""
    summary = []
    paired = []
    seeds = sorted({row["training_seed"] for row in rows})
    baseline = {
        seed: np.mean(
            [
                r["return"]
                for r in rows
                if r["variant"] == baseline_variant and r["training_seed"] == seed
            ]
        )
        for seed in seeds
    }
    for variant in dict.fromkeys(row["variant"] for row in rows):
        selected = [row for row in rows if row["variant"] == variant]
        seed_means = []
        for seed in seeds:
            mean = float(
                np.mean(
                    [row["return"] for row in selected if row["training_seed"] == seed]
                )
            )
            seed_means.append(mean)
            paired.append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "mean_return": mean,
                    f"difference_from_{baseline_variant}": mean - baseline[seed],
                }
            )
        result = {
            "variant": variant,
            "training_seeds": len(seeds),
            "mean_return": float(np.mean(seed_means)),
            "std_across_seed_means": float(np.std(seed_means, ddof=1))
            if len(seeds) > 1
            else 0.0,
        }
        for metric in (
            "waypoint_reached",
            "intrusion_events",
            "intrusion_time",
            "restricted_area_events",
            "time_in_restricted_area",
            "sector_exit_events",
            "flight_time",
            "time_outside_sector",
            "timed_out",
        ):
            values = [row[metric] for row in selected if metric in row]
            if values:
                result[f"mean_{metric}"] = float(np.mean(values))
        summary.append(result)
    return summary, paired


def completed_evaluation(directory, eval_seeds):
    """Reuse a finished run, selecting requested scenarios from its saved metrics."""
    reports = sorted((directory / "reports").glob("*/run.json"))
    if not reports:
        return None
    for path in reversed(reports):
        report = json.loads(path.read_text())
        if report["status"] != "completed":
            continue
        rows = report.get("evaluation_scenarios", [])
        available = {row["scenario_seed"] for row in rows}
        if not set(eval_seeds) <= available:
            raise ValueError(
                f"Requested evaluation seeds are missing from {path}; use a subset of the original seeds"
            )
        return [row for row in rows if row["scenario_seed"] in eval_seeds]
    raise ValueError(
        f"Unfinished run in {directory}; its saved checkpoint is preserved. Select completed training seeds or resume that training separately."
    )


def prepare_comparison(args, variants=None, observation=False, frequency=False):
    """Validate reuse before starting any additional simulator processes."""
    manifest = args.output_dir / "comparison.json"
    settings = {**vars(args), "variants": variants or VARIANTS}
    settings["observation_profile"] = (
        args.observation_profile if observation else "control"
    )
    if getattr(args, "resume", False):
        if not manifest.is_file():
            raise ValueError(f"No comparison manifest found: {manifest}")
        previous = json.loads(manifest.read_text())
        keys = ["algorithm", "multi_agent", "timesteps", "speed_min", "speed_max"]
        if frequency:
            keys += [
                "heading_action",
                "speed_action",
                "observation_encoding",
                "intervals",
            ]
        if observation:
            keys += ["heading_action", "speed_action", "observation_profile"]
            if previous.get("variants") != list(variants):
                raise ValueError(
                    "Cannot change observation variants when reusing results"
                )
        for key in keys:
            if previous[key] != getattr(args, key):
                raise ValueError(f"Cannot change {key} when reusing comparison results")
        # Check every cached run before launching any missing variants.
        for variant in variants or [f"{h}_{s}" for h, s in VARIANTS]:
            for seed in args.seeds:
                completed_evaluation(
                    args.output_dir / variant / f"seed_{seed}",
                    args.eval_seeds,
                )
        (args.output_dir / "selection.json").write_text(
            json.dumps(settings, default=str, indent=2) + "\n"
        )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest.write_text(json.dumps(settings, default=str, indent=2) + "\n")


def run_comparison(args, observation=False, frequency=False):
    """Train every variant/seed once, then save raw and paired comparison reports."""
    fixed_actions = observation or frequency
    if frequency and (args.multi_agent or args.algorithm != "ppo"):
        raise ValueError("decision interval comparison requires single-agent PPO")
    if frequency and (
        not args.intervals
        or len(set(args.intervals)) != len(args.intervals)
        or any(type(value) is not int or value <= 0 for value in args.intervals)
        or 10 not in args.intervals
    ):
        raise ValueError(
            "intervals must be unique positive integers including baseline 10"
        )
    if args.multi_agent and args.algorithm not in ("ppo", "sac"):
        raise ValueError("multi-agent comparison supports PPO and SAC")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.eval_seeds)) != len(
        args.eval_seeds
    ):
        raise ValueError("training and evaluation seeds must each be unique")
    ControlConfig(
        heading_action=args.heading_action if fixed_actions else "relative",
        speed_action=args.speed_action if fixed_actions else "absolute",
        observation_profile=args.observation_profile if fixed_actions else "control",
        speed_min=args.speed_min,
        speed_max=args.speed_max,
    )
    from .observations import DEFAULT_OBSERVATION_ENCODINGS, OBSERVATION_ENCODINGS

    encodings = getattr(args, "encodings", DEFAULT_OBSERVATION_ENCODINGS)
    if observation and (
        not encodings
        or len(set(encodings)) != len(encodings)
        or any(name not in OBSERVATION_ENCODINGS for name in encodings)
    ):
        raise ValueError(
            "observation encodings must be nonempty, unique, and supported"
        )
    variants = (
        [(name, args.heading_action, args.speed_action, name) for name in encodings]
        if observation
        else [(f"{h}_{s}", h, s, "upstream") for h, s in VARIANTS]
    )
    if frequency:
        variants = [
            (
                f"interval_{interval}s",
                args.heading_action,
                args.speed_action,
                args.observation_encoding,
            )
            for interval in args.intervals
        ]
    prepare_comparison(
        args,
        [v[0] for v in variants] if fixed_actions else None,
        observation=fixed_actions,
        frequency=frequency,
    )
    rows = []
    for variant, heading, speed, encoding in variants:
        interval = (
            int(variant.removeprefix("interval_").removesuffix("s"))
            if frequency
            else None
        )
        timesteps = (
            max(1, round(args.timesteps * 10 / interval))
            if frequency
            else args.timesteps
        )
        for seed in args.seeds:
            directory = args.output_dir / variant / f"seed_{seed}"
            evaluation = (
                completed_evaluation(directory, args.eval_seeds)
                if getattr(args, "resume", False)
                else None
            )
            if evaluation is not None:
                print(f"Reusing {variant}, seed {seed}", flush=True)
            else:
                print(f"Training {variant}, seed {seed}", flush=True)
                subprocess.run(
                    comparison_command(
                        args.algorithm,
                        args.multi_agent,
                        heading,
                        speed,
                        seed,
                        args.eval_seeds,
                        directory,
                        timesteps,
                        args.speed_min,
                        args.speed_max,
                        args.device,
                        observation_encoding=encoding,
                        observation_profile=args.observation_profile
                        if fixed_actions
                        else "control",
                        decision_interval=interval,
                    ),
                    check=True,
                )
                reports = list((directory / "reports").glob("*/run.json"))
                if len(reports) != 1:
                    raise RuntimeError(f"Expected one run report in {directory}")
                report = json.loads(reports[0].read_text())
                if report["status"] != "completed":
                    raise RuntimeError(f"Incomplete comparison run: {reports[0]}")
                evaluation = report["evaluation_scenarios"]
            rows.extend(
                {**row, "variant": variant, "training_seed": seed} for row in evaluation
            )
            write_csv(args.output_dir / "episodes.csv", rows)
    summary, paired = summarize(
        rows,
        "interval_10s"
        if frequency
        else encodings[0]
        if observation
        else "relative_relative",
    )
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired.csv", paired)
    figure = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.bar(
        [r["variant"] for r in summary],
        [r["mean_return"] for r in summary],
        yerr=[r["std_across_seed_means"] for r in summary],
        capsize=4,
    )
    axis.set_ylabel("Mean raw return ± SD across training seeds")
    axis.tick_params(axis="x", labelrotation=15)
    figure.savefig(args.output_dir / "comparison.png", dpi=150)
    print(f"Comparison saved: {args.output_dir}")
