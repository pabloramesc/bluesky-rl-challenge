"""Aggregate paired architecture evaluations without treating agents as seeds."""

from collections import defaultdict
from numbers import Real

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .comparison import write_csv


def _number(row, key):
    return f"{row[key]:.3f}" if key in row else "—"


def architecture_summary(episodes, runs, variants, seeds):
    """Weight training seeds equally and pair returns against the first leaf."""
    grouped = defaultdict(list)
    for row in episodes:
        grouped[row["variant"], row["training_seed"]].append(row)
    by_variant = defaultdict(list)
    for row in runs:
        by_variant[row["variant"]].append(row)
    baseline = variants[0].name
    summary, paired = [], []
    excluded = {"training_seed", "scenario_seed", "agent_index", "variant"}
    for variant in variants:
        selected = by_variant[variant.name]
        available = [seed for seed in seeds if (variant.name, seed) in grouped]
        complete = len(available) == len(seeds)
        status = "completed" if complete else "partial" if available else "failed"
        if (
            not available
            and selected
            and all(row["status"] == "queued" for row in selected)
        ):
            status = "pending"
        elif not available and any(row["status"] == "interrupted" for row in selected):
            status = "interrupted"
        result = {
            "variant": variant.name,
            **variant.settings,
            "status": status,
            "completed_seeds": len(available),
            "requested_seeds": len(seeds),
            "evaluation_episodes": sum(
                len(grouped[variant.name, seed]) for seed in available
            ),
        }
        metrics = sorted(
            {
                key
                for seed in available
                for row in grouped[variant.name, seed]
                for key, value in row.items()
                if key not in excluded and isinstance(value, Real)
            }
        )
        for metric in metrics:
            seed_means = [
                np.mean([row[metric] for row in grouped[variant.name, seed]])
                for seed in available
                if all(metric in row for row in grouped[variant.name, seed])
            ]
            # Never mix different metric availability across seeds or agents.
            if len(seed_means) == len(available):
                result[f"mean_{metric}"] = float(np.mean(seed_means))
                result[f"std_{metric}"] = (
                    float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
                )
        for metric in (
            "training_seconds",
            "actual_timesteps",
            "steps_per_second",
            "wall_seconds",
        ):
            values = [
                row[metric]
                for row in selected
                if row["status"] == "completed" and metric in row
            ]
            if values:
                result[f"mean_{metric}"] = float(np.mean(values))
        for seed in available:
            reference = grouped.get((baseline, seed))
            if not reference:
                continue
            current = {
                (row["scenario_seed"], row["agent_index"]): row["return"]
                for row in grouped[variant.name, seed]
            }
            previous = {
                (row["scenario_seed"], row["agent_index"]): row["return"]
                for row in reference
            }
            if current.keys() != previous.keys():
                continue
            paired.append(
                {
                    "variant": variant.name,
                    "training_seed": seed,
                    "baseline": baseline,
                    "mean_return": float(np.mean(list(current.values()))),
                    "paired_difference": float(
                        np.mean([current[key] - previous[key] for key in current])
                    ),
                }
            )
        summary.append(result)
    ranked = sorted(
        (row for row in summary if row["status"] == "completed"),
        key=lambda row: row["mean_return"],
        reverse=True,
    )
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return summary, paired


def write_architecture_report(directory, episodes, runs, variants, seeds):
    """Write raw records, a ranked table and a return plot, including partial status."""
    summary, paired = architecture_summary(episodes, runs, variants, seeds)
    empty_fields = {
        "episodes.csv": [
            "variant",
            "training_seed",
            "scenario_seed",
            "agent_index",
            "return",
            "length",
        ],
        "paired.csv": [
            "variant",
            "training_seed",
            "baseline",
            "mean_return",
            "paired_difference",
        ],
    }
    for filename, rows in (
        ("episodes.csv", episodes),
        ("runs.csv", runs),
        ("summary.csv", summary),
        ("paired.csv", paired),
    ):
        write_csv(
            directory / filename,
            rows,
            fields=empty_fields.get(filename) if not rows else None,
        )
    ordered = sorted(summary, key=lambda row: row.get("rank", float("inf")))
    lines = [
        "# Architecture comparison",
        "",
        "Returns use identical evaluation scenarios. Means weight training seeds equally; error bars show SD across seed means.",
        "Only variants completing every requested training seed receive a rank. Rank uses raw return; inspect safety and completion metrics separately.",
        "",
        f"Paired baseline: `{variants[0].name}`. SD is zero for a single seed and does not estimate uncertainty.",
        "",
        "| Rank | Variant | Status | Seeds | Mean return | SD | Training seconds |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row.get('rank', '—')} | {row['variant']} | {row['status']} | {row['completed_seeds']}/{row['requested_seeds']} | {_number(row, 'mean_return')} | {_number(row, 'std_return')} | {_number(row, 'mean_training_seconds')} |"
        )
    lines += [
        "",
        "![Mean evaluation return](comparison.png)",
        "",
        "See `summary.csv` for all recorded safety/completion metrics, `paired.csv` for seed-paired differences, and `runs.csv` for failures, actual budgets, timings and log paths.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))
    plotted = [row for row in ordered if "mean_return" in row]
    figure = Figure(figsize=(12, max(3, len(plotted) * 0.32)), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    if plotted:
        axis.barh(
            [
                row["variant"] + (" [partial]" if row["status"] != "completed" else "")
                for row in plotted
            ],
            [row["mean_return"] for row in plotted],
            xerr=[row["std_return"] for row in plotted],
            capsize=3,
            color=[
                "steelblue" if row["status"] == "completed" else "gray"
                for row in plotted
            ],
        )
        axis.invert_yaxis()
    else:
        axis.text(
            0.5, 0.5, "No completed evaluations", ha="center", transform=axis.transAxes
        )
    axis.set_xlabel("Mean raw return ± SD across training seeds")
    figure.savefig(directory / "comparison.png", dpi=150)
    return summary
