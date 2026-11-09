"""Matched additive-attention ISAC, shared SAC, and MASAC experiments."""

import json
import math
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

from .architecture_comparison import _evaluation
from .architecture_reporting import write_architecture_report
from .architecture_tree import ArchitectureVariant
from .models import GAMMA, build_isac, build_masac, build_shared_sac
from .multi_agent import N_AGENTS, make_mappo_env
from .multi_agent_comparison import (
    comparison_control,
    run_multi_agent_comparison,
    validate_joint_comparison,
)
from .multi_agent_sac import CentralizedAttentionQ, LocalAttentionQ
from .rewards import RewardConfig
from .training import train

BUILDERS = {"shared_sac": build_shared_sac, "isac": build_isac, "masac": build_masac}
ALGORITHM_CLASSES = {"shared_sac": "SharedSAC", "isac": "ISAC", "masac": "MASAC"}


def sac_comparison_settings(args):
    """Use equal joint replay budgets and per-learner update schedules."""
    return {
        "learning_rate": args.learning_rate,
        "gamma": GAMMA,
        "tau": args.tau,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts * N_AGENTS,
        "batch_size": args.batch_size,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "target_entropy": -2.0,
        "policy_kwargs": {
            "actor_kwargs": {"hidden_sizes": list(args.hidden_sizes)},
            "critic_kwargs": {"hidden_sizes": list(args.hidden_sizes)},
            "initial_alpha": args.initial_alpha,
        },
    }


def sac_comparison_plan(args):
    """Fix the PPO benchmark interface, reward, and paired evaluation protocol."""
    validate_joint_comparison(args)
    for name in ("buffer_size", "train_freq", "gradient_steps"):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(args.learning_starts) is not int or args.learning_starts < 0:
        raise ValueError("learning_starts must be a nonnegative integer")
    for name in ("learning_rate", "initial_alpha", "tau"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.tau > 1:
        raise ValueError("tau must be at most one")
    args.timesteps = args.joint_steps * N_AGENTS
    architecture = {
        "architecture": "entity_mlp",
        "pooling": "additive_attention",
        "attention_context": args.attention_context,
        "temporal": "none",
    }
    variants = [ArchitectureVariant((name,), architecture.copy()) for name in BUILDERS]
    return variants, {
        "algorithm": "multi_agent_sac",
        "multi_agent": True,
        "timesteps": args.timesteps,
        "joint_steps": args.joint_steps,
        "n_agents": N_AGENTS,
        "checkpoint_interval": max(1, math.ceil(args.joint_steps / 4)) * N_AGENTS,
        "seeds": args.seeds,
        "eval_seeds": args.eval_seeds,
        "control_config": comparison_control().to_dict(),
        "reward_config": RewardConfig().to_dict(),
        "settings": sac_comparison_settings(args),
        "architecture_config": architecture,
        "variants": list(BUILDERS),
        "normalization": "raw joint replay; shared normalization per run; frozen evaluation",
        "actor_gradient": "focal action only; peer actions detached in centralized actor loss",
    }


def sac_comparison_command(args, variant, seed, directory, device):
    """Create one isolated worker per method and training seed."""
    command = [
        sys.executable,
        "-m",
        "experiments.compare_multi_agent_sac",
        "--worker",
        variant.name,
    ]
    for name in (
        "joint_steps",
        "batch_size",
        "buffer_size",
        "learning_starts",
        "train_freq",
        "gradient_steps",
        "learning_rate",
        "tau",
        "initial_alpha",
        "attention_context",
    ):
        command += [f"--{name.replace('_', '-')}", str(getattr(args, name))]
    return command + [
        "--hidden-sizes",
        *map(str, args.hidden_sizes),
        "--seeds",
        str(seed),
        "--eval-seeds",
        *map(str, args.eval_seeds),
        "--devices",
        device,
        "--output-dir",
        str(directory),
    ]


def run_sac_comparison_worker(args):
    return train(
        partial(
            BUILDERS[args.worker],
            attention_context=args.attention_context,
            **sac_comparison_settings(args),
        ),
        partial(
            make_mappo_env,
            control_config=comparison_control(),
            entity_observations=True,
        ),
        args.output_dir / "policy",
        args.joint_steps * N_AGENTS,
        seed=args.seeds[0],
        device=args.devices[0],
        results_dir=args.output_dir / "reports",
        evaluation_seeds=args.eval_seeds,
        control_config=comparison_control(),
        checkpoint_interval=max(1, math.ceil(args.joint_steps / 4)) * N_AGENTS,
    )


def sac_comparison_evaluation(record, variant, protocol):
    """Reject mismatched methods, representations, budgets, or update settings."""
    rows = _evaluation(
        record,
        variant,
        {**protocol, "algorithm_class": ALGORITHM_CLASSES[variant.name]},
    )
    if rows is None:
        return None
    report = json.loads(Path(record["report_path"]).read_text())
    if report.get("reward_config") != protocol["reward_config"]:
        raise ValueError("Reward configuration mismatch")
    if report.get("checkpoint_interval") != protocol["checkpoint_interval"]:
        raise ValueError("Checkpoint interval mismatch")
    required = {
        "total_reward",
        "flight_time",
        "waypoint_reached",
        "intrusion_events",
        "intrusion_time",
        "restricted_area_events",
        "time_in_restricted_area",
        "sector_exit_events",
        "time_outside_sector",
        "timed_out",
        "length",
    }
    if any(not required <= row.keys() for row in rows):
        raise ValueError("Missing safety or completion metrics")
    settings = protocol["settings"]
    if any(
        report["settings"].get(key) != value
        for key, value in settings.items()
        if key != "policy_kwargs"
    ):
        raise ValueError("SAC settings mismatch")
    policy = report["policy_kwargs"]
    expected = deepcopy(settings["policy_kwargs"])
    layout = policy.get("actor_kwargs", {}).get("layout")
    if not layout or policy.get("critic_kwargs", {}).get("layout") != layout:
        raise ValueError("Attention entity layouts mismatch")
    for role in ("actor_kwargs", "critic_kwargs"):
        expected[role].update(
            layout=layout, attention_context=variant.settings["attention_context"]
        )
    expected["critic_class"] = str(
        CentralizedAttentionQ if variant.name == "masac" else LocalAttentionQ
    )
    if policy != expected:
        raise ValueError("SAC policy settings mismatch")
    expected_steps = (
        math.ceil(protocol["joint_steps"] / settings["train_freq"])
        * settings["train_freq"]
    )
    if (
        report.get("joint_steps") != expected_steps
        or report["actual_timesteps"] != expected_steps * N_AGENTS
    ):
        raise ValueError("Joint simulator budget mismatch")
    for key in (
        "joint_steps",
        "active_timesteps",
        "parameter_count",
        "gradient_updates",
    ):
        value = report.get(key)
        if type(value) is not int or value < (0 if key == "gradient_updates" else 1):
            raise ValueError(f"Missing or invalid {key}")
        record[key] = value
    collections = expected_steps // settings["train_freq"]
    first_update = max(
        1,
        math.ceil(settings["learning_starts"] / (N_AGENTS * settings["train_freq"])),
    )
    expected_updates = (
        max(0, collections - first_update + 1) * settings["gradient_steps"]
    )
    if report["gradient_updates"] != expected_updates:
        raise ValueError("SAC gradient update budget mismatch")
    if report["active_timesteps"] > report["actual_timesteps"]:
        raise ValueError("Invalid active transition count")
    return rows


def write_sac_comparison_report(directory, episodes, records, variants, seeds):
    summary = write_architecture_report(directory, episodes, records, variants, seeds)
    path = directory / "report.md"
    body = path.read_text().replace(
        "# Architecture comparison", "# Multi-agent SAC comparison", 1
    )
    lines = [
        "",
        "## Protocol",
        "",
        "All methods use separate additive-attention actor and twin Q encoders, individual rewards, raw joint replay, and identical SAC settings. ISAC owns independent networks, targets, optimizers, and temperatures per aircraft. Shared SAC shares local networks and one temperature. MASAC shares a local actor and one temperature; each centralized Q critic attends over active agents' observation/action embeddings, conditioned on the focal agent. Actor gradients pass only through the focal action. Targets use each aircraft's own entropy term.",
        "",
        "Replay capacity, minibatch size, warmup in this comparison CLI, and collection frequency count joint decisions. Each gradient step updates every ISAC learner once (using its valid sampled transitions), or the shared learner once. ISAC therefore performs more optimizer steps; parameter counts and compute differ. Normalization is pooled within each run, applied to raw replay at sampling time, and frozen during evaluation. True terminations stop bootstrapping; time limits bootstrap from final joint observations.",
        "",
        "## Safety, completion, and cost",
        "",
        "| Method | Return | Completion | Intrusion s | Restricted s | Outside s | Joint steps | Valid transitions | Parameters | Updates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        runs = [
            r
            for r in records
            if r["variant"] == row["variant"] and r["status"] == "completed"
        ]
        values = [
            f"{row[f'mean_{metric}']:.3f}" if f"mean_{metric}" in row else "—"
            for metric in (
                "return",
                "waypoint_reached",
                "intrusion_time",
                "time_in_restricted_area",
                "time_outside_sector",
            )
        ]
        for metric in (
            "joint_steps",
            "active_timesteps",
            "parameter_count",
            "gradient_updates",
        ):
            values.append(
                f"{sum(r[metric] for r in runs) / len(runs):.0f}" if runs else "—"
            )
        lines.append(f"| {row['variant']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Updates count training iterations per learner, not summed optimizer calls across ISAC learners. The paired reference is shared_sac. Compare against the PPO table only with matched actual joint budgets, control/reward settings, training seeds, and scenario seeds. Evaluation aircraft within a scenario are not independent trials. No PPO reference is imported or retrained automatically.",
        "",
    ]
    path.write_text(body + "\n".join(lines))


def run_sac_comparison(args):
    return run_multi_agent_comparison(
        args,
        plan=sac_comparison_plan,
        command_builder=sac_comparison_command,
        evaluation_reader=sac_comparison_evaluation,
        report_writer=write_sac_comparison_report,
    )
