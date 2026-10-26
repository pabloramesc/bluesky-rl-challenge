"""Matched flat MLP or additive-attention IPPO, shared PPO, and MAPPO experiments."""

import json
import math
import signal
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

from .actions import ControlConfig
from .architecture_comparison import (
    _comparison_lock,
    _evaluation,
    _interrupt_handler,
    _prepare_runs,
    _schedule,
    _write_json,
)
from .architecture_reporting import write_architecture_report
from .architecture_tree import ArchitectureVariant
from .models import GAMMA, build_ippo, build_mappo, build_shared_ppo
from .multi_agent import N_AGENTS, make_mappo_env
from .rewards import RewardConfig
from .training import train

BUILDERS = {
    "shared_ppo": build_shared_ppo,
    "ippo": build_ippo,
    "mappo": build_mappo,
}
ALGORITHM_CLASSES = {"shared_ppo": "SharedPPO", "ippo": "IPPO", "mappo": "MAPPO"}


def comparison_settings(args):
    """Use identical actor/critic hidden layers and PPO hyperparameters."""
    return {
        "n_steps": args.rollout_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.epochs,
        "learning_rate": 3e-4,
        "gamma": GAMMA,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "value_clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "actor_kwargs": {"hidden_sizes": list(args.hidden_sizes)},
            "critic_kwargs": {"hidden_sizes": list(args.hidden_sizes)},
            "log_std_init": -0.5,
        },
    }


def comparison_control():
    """Fix the action/observation interface across all algorithms."""
    return ControlConfig(
        heading_action="relative",
        speed_action="absolute",
        speed_min=100,
        speed_max=500,
        observation_profile="control",
        observation_encoding="goal_aligned_conflict",
    )


def multi_agent_plan(args):
    """Validate the complete protocol before starting simulator processes."""
    for name in (
        "joint_steps",
        "rollout_steps",
        "batch_size",
        "epochs",
        "jobs",
        "threads_per_job",
    ):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not args.hidden_sizes or any(
        type(size) is not int or size <= 0 for size in args.hidden_sizes
    ):
        raise ValueError("hidden_sizes must contain positive integers")
    for name in ("seeds", "eval_seeds"):
        values = getattr(args, name)
        if (
            not values
            or len(set(values)) != len(values)
            or any(type(value) is not int or value < 0 for value in values)
        ):
            raise ValueError(f"{name} must contain unique nonnegative integers")
    if (
        not args.devices
        or len(args.devices) > args.jobs
        or any(not device.strip() for device in args.devices)
    ):
        raise ValueError(
            "Provide nonempty devices and at least one job slot per device"
        )
    if args.timeout is not None and (
        not math.isfinite(args.timeout) or args.timeout <= 0
    ):
        raise ValueError("timeout must be finite and positive")
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    args.timesteps = args.joint_steps * N_AGENTS
    # Keep the shared-policy method first as the paired reference.
    architecture = {"architecture": args.architecture, "temporal": "none"}
    if args.architecture == "entity_mlp":
        architecture.update(
            pooling="additive_attention", attention_context=args.attention_context
        )
    variants = [ArchitectureVariant((name,), architecture.copy()) for name in BUILDERS]
    return variants, {
        "algorithm": "multi_agent_ppo",
        "multi_agent": True,
        "timesteps": args.timesteps,
        "joint_steps": args.joint_steps,
        "n_agents": N_AGENTS,
        "seeds": args.seeds,
        "eval_seeds": args.eval_seeds,
        "control_config": comparison_control().to_dict(),
        "reward_config": RewardConfig().to_dict(),
        "settings": comparison_settings(args),
        "normalization": "shared VecNormalize statistics per run; frozen during evaluation",
        "variants": list(BUILDERS),
        **({"architecture_config": architecture} if args.architecture != "mlp" else {}),
    }


def multi_agent_command(args, variant, seed, directory, device):
    """Launch one algorithm/seed in a process with its own BlueSky singleton."""
    return [
        sys.executable,
        "-m",
        "experiments.compare_multi_agent",
        "--worker",
        variant.name,
        "--architecture",
        args.architecture,
        *(
            ["--attention-context", args.attention_context]
            if args.architecture == "entity_mlp"
            else []
        ),
        "--joint-steps",
        str(args.joint_steps),
        "--rollout-steps",
        str(args.rollout_steps),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
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


def run_multi_agent_worker(args):
    """Use the common checkpoint and frozen-scenario evaluation workflow."""
    settings = comparison_settings(args)
    return train(
        partial(
            BUILDERS[args.worker],
            architecture=args.architecture,
            attention_context=args.attention_context or "all",
            **settings,
        ),
        partial(
            make_mappo_env,
            control_config=comparison_control(),
            entity_observations=args.architecture == "entity_mlp",
        ),
        args.output_dir / "policy",
        args.joint_steps * N_AGENTS,
        seed=args.seeds[0],
        device=args.devices[0],
        results_dir=args.output_dir / "reports",
        evaluation_seeds=args.eval_seeds,
        control_config=comparison_control(),
    )


def multi_agent_evaluation(record, variant, protocol):
    """Reject unmatched algorithms, PPO settings, budgets, or evaluation scenarios."""
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
    required_metrics = {
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
    if any(not required_metrics <= row.keys() for row in rows):
        raise ValueError("Missing safety or completion metrics")
    settings = protocol["settings"]
    if any(
        report["settings"].get(key) != value
        for key, value in settings.items()
        if key != "policy_kwargs"
    ):
        raise ValueError("PPO settings mismatch")
    expected_policy = settings["policy_kwargs"]
    policy = report["policy_kwargs"]
    if variant.settings["architecture"] == "entity_mlp":
        from .multi_agent_attention import (
            AdditiveAttentionActor,
            CentralizedAdditiveAttentionCritic,
            LocalAdditiveAttentionCritic,
        )

        critic = (
            CentralizedAdditiveAttentionCritic
            if variant.name == "mappo"
            else LocalAdditiveAttentionCritic
        )
        if policy.get("actor_class") != str(AdditiveAttentionActor) or policy.get(
            "critic_class"
        ) != str(critic):
            raise ValueError("Attention network classes mismatch")
        actor_layout = policy.get("actor_kwargs", {}).get("layout")
        if (
            not actor_layout
            or policy.get("critic_kwargs", {}).get("layout") != actor_layout
        ):
            raise ValueError("Attention entity layouts mismatch")
        expected_policy = deepcopy(expected_policy)
        for role in ("actor_kwargs", "critic_kwargs"):
            expected_policy[role].update(
                layout=actor_layout,
                attention_context=variant.settings["attention_context"],
            )
    if any(policy.get(key) != value for key, value in expected_policy.items()):
        raise ValueError("Policy settings mismatch")
    expected_steps = (
        math.ceil(protocol["joint_steps"] / settings["n_steps"]) * settings["n_steps"]
    )
    if (
        report.get("joint_steps") != expected_steps
        or report["actual_timesteps"] != expected_steps * N_AGENTS
    ):
        raise ValueError("Joint simulator budget mismatch")
    for key in ("joint_steps", "active_timesteps", "parameter_count"):
        value = report.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Missing or invalid {key}")
        record[key] = value
    return rows


def write_multi_agent_report(directory, episodes, records, variants, seeds):
    """Retain common paired summaries and expose safety, completion, and cost."""
    summary = write_architecture_report(directory, episodes, records, variants, seeds)
    path = directory / "report.md"
    body = path.read_text().replace(
        "# Architecture comparison", "# Multi-agent PPO comparison", 1
    )
    representation = (
        "All methods use independent actor and critic entity encoders with masked additive attention over locally visible aircraft, "
        f"queried by {variants[0].settings['attention_context']} context. MAPPO additionally uses focal-agent-conditioned additive attention over all active agents (including itself) in its centralized critic. "
        if variants[0].settings["architecture"] == "entity_mlp"
        else "All methods use MLP actors and critics. "
    )
    lines = [
        "",
        "## Protocol",
        "",
        representation
        + "All methods use identical PPO settings, joint rollout lengths, rewards, and paired scenario seeds. IPPO has separate networks and Adam optimizers per aircraft; shared PPO has a shared actor and local critic; MAPPO has a shared actor and centralized critic. All use the same masked rollout collector. Normalization statistics are pooled across aircraft within each run and frozen for evaluation.",
        "",
        "The shared PPO here is the masked local-critic control; the existing SB3 train_zoo_ppo command retains its original collector and defaults. Minibatch size counts joint timesteps for shared PPO/MAPPO and own valid transitions for each IPPO learner. IPPO normalizes advantages separately per learner. Total parameter counts differ despite matched hidden widths.",
        "",
        "## Safety, completion, and training cost",
        "",
        "| Method | Completion | Intrusion s | Restricted s | Outside s | Flight s | Joint steps | Valid transitions | Parameters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                "waypoint_reached",
                "intrusion_time",
                "time_in_restricted_area",
                "time_outside_sector",
                "flight_time",
            )
        ]
        for metric in ("joint_steps", "active_timesteps", "parameter_count"):
            values.append(
                f"{sum(r[metric] for r in runs) / len(runs):.0f}" if runs else "—"
            )
        lines.append(f"| {row['variant']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Read safety and completion alongside return. Flight time includes unsuccessful flights. Joint budgets match exactly after rounding up to complete rollouts; valid transition counts can differ as policies finish flights at different times.",
        "",
    ]
    path.write_text(body + "\n".join(lines))


def run_multi_agent_comparison(args):
    """Schedule isolated runs with resumable manifests and paired reporting."""
    variants, protocol = multi_agent_plan(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": protocol,
                    "commands": [
                        multi_agent_command(
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
            raise ValueError("No multi-agent comparison manifest found")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    with _comparison_lock(args.output_dir):
        manifest = args.output_dir / "comparison.json"
        if args.resume and json.loads(manifest.read_text())["protocol"] != protocol:
            raise ValueError("Cannot change multi-agent comparison settings on resume")
        _write_json(manifest, {"protocol": protocol})
        state = args.output_dir / "runs.json"
        previous = json.loads(state.read_text()) if state.is_file() else []
        records, episodes = _prepare_runs(
            args, variants, protocol, previous, evaluation_reader=multi_agent_evaluation
        )
        _write_json(
            args.output_dir / "execution.json",
            {
                "jobs": args.jobs,
                "devices": args.devices,
                "threads_per_job": args.threads_per_job,
                "timeout": args.timeout,
            },
        )
        handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _interrupt_handler)
        interrupted = False
        try:
            interrupted = _schedule(
                args,
                variants,
                protocol,
                records,
                episodes,
                command_builder=multi_agent_command,
                evaluation_reader=multi_agent_evaluation,
            )
        finally:
            signal.signal(signal.SIGTERM, handler)
            write_multi_agent_report(
                args.output_dir, episodes, records, variants, args.seeds
            )
    print(f"Multi-agent report: {args.output_dir / 'report.md'}", flush=True)
    return (
        130
        if interrupted
        else int(any(record["status"] != "completed" for record in records))
    )
