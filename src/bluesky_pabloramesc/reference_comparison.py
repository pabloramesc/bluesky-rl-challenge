"""Train one new architecture against immutable, protocol-matched reference runs."""

import json
import signal
from argparse import Namespace
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from .architecture_comparison import (
    _comparison_lock,
    _evaluation,
    _interrupt_handler,
    _prepare_runs,
    _schedule,
    _write_json,
    architecture_command,
)
from .architecture_reporting import write_architecture_report
from .architecture_tree import expand_tree
from .rewards import RewardConfig


def matched_evaluation(record, variant, protocol):
    """Require the original PPO settings, actual budget and unshaped reward."""
    rows = _evaluation(record, variant, protocol)
    if rows is None:
        return None
    report = json.loads(Path(record["report_path"]).read_text())
    settings = report.get("settings", {})
    reference_settings = protocol["training_settings"]
    if (
        not isinstance(settings, dict)
        or not isinstance(reference_settings, dict)
        or not reference_settings
    ):
        raise ValueError(f"Missing PPO settings: {record['report_path']}")
    # Older reference reports record fewer fields than current training reports.
    if any(settings.get(key) != value for key, value in reference_settings.items()):
        raise ValueError(f"PPO settings differ from reference: {record['report_path']}")
    if report["actual_timesteps"] != protocol["actual_timesteps"]:
        raise ValueError(
            f"Actual budget differs from reference: {record['report_path']}"
        )
    if (
        report.get("reward_config", RewardConfig().to_dict())
        != RewardConfig().to_dict()
    ):
        raise ValueError(
            f"Reference comparison requires unshaped rewards: {record['report_path']}"
        )
    return rows


def reference_comparison_plan(args, new_variant, reference_tree=None):
    """Validate all reference evidence before allowing any new training."""
    source = args.reference_dir.resolve()
    output = args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            "Reference and output directories must be separate, non-nested paths"
        )
    manifest_path = source / "comparison.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["protocol"]
    reference_tree = reference_tree or manifest["tree"]
    references = expand_tree(reference_tree)
    tree = deepcopy(reference_tree)
    if not tree.get("children"):
        raise ValueError("Reference tree must have architecture children")
    tree["children"].append(new_variant)
    expected = [{"name": v.name, "settings": v.settings} for v in references]
    if (
        protocol["algorithm"] != "ppo"
        or protocol["multi_agent"] is not False
        or protocol["variants"] != expected
    ):
        raise ValueError(
            "Reference must be a matching single-agent PPO architecture comparison"
        )
    seeds = protocol["seeds"]
    for values in (seeds, protocol["eval_seeds"]):
        if (
            not values
            or len(set(values)) != len(values)
            or any(type(v) is not int or v < 0 for v in values)
        ):
            raise ValueError("Reference seeds must be unique nonnegative integers")
    if type(protocol["timesteps"]) is not int or protocol["timesteps"] <= 0:
        raise ValueError("Reference timesteps must be a positive integer")
    records = json.loads((source / "runs.json").read_text())
    indexed = {(r["variant"], r["training_seed"]): r for r in records}
    required = {(v.name, seed) for v in references for seed in seeds}
    if len(indexed) != len(records) or set(indexed) != required:
        raise ValueError(
            "Reference needs exactly one completed run per variant and training seed"
        )
    episodes, hashes = [], {}
    training_settings, actual_timesteps = None, None
    for variant in references:
        for seed in seeds:
            record = indexed[variant.name, seed]
            if record["status"] != "completed" or record.get("returncode") != 0:
                raise ValueError(
                    f"Reference run is incomplete: {variant.name}, seed {seed}"
                )
            rows = _evaluation(record, variant, protocol)
            if rows is None:
                raise ValueError(
                    f"Reference evaluation is missing: {variant.name}, seed {seed}"
                )
            report_path = Path(record["report_path"])
            report = json.loads(report_path.read_text())
            if training_settings is None:
                training_settings = report["settings"]
                actual_timesteps = report["actual_timesteps"]
            matched = {
                **protocol,
                "training_settings": training_settings,
                "actual_timesteps": actual_timesteps,
            }
            matched_evaluation(record, variant, matched)
            episodes.extend(rows)
            record.update(reused=True, reference=True)
            hashes[str(report_path.resolve())] = sha256(
                report_path.read_bytes()
            ).hexdigest()
    devices = {record["device"] for record in records}
    if len(devices) != 1:
        raise ValueError("Reference runs must use one common device")
    execution = json.loads((source / "execution.json").read_text())
    threads = execution["threads_per_job"]
    if type(threads) is not int or threads <= 0:
        raise ValueError("Reference threads_per_job must be a positive integer")
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    if args.timeout is not None and (not 0 < args.timeout < float("inf")):
        raise ValueError("timeout must be finite and positive")
    variants = expand_tree(tree)
    protocol = {
        **protocol,
        "variants": [{"name": v.name, "settings": v.settings} for v in variants],
        "training_settings": training_settings,
        "actual_timesteps": actual_timesteps,
    }
    provenance = {
        "directory": str(source),
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "execution_sha256": sha256(
            (source / "execution.json").read_bytes()
        ).hexdigest(),
        "reports_sha256": hashes,
    }
    training = Namespace(
        **vars(args),
        algorithm="ppo",
        multi_agent=False,
        timesteps=protocol["timesteps"],
        seeds=seeds,
        eval_seeds=protocol["eval_seeds"],
        jobs=1,
        devices=list(devices),
        threads_per_job=threads,
        **{
            key: protocol["control_config"][key]
            for key in (
                "heading_action",
                "speed_action",
                "observation_encoding",
                "observation_profile",
                "speed_min",
                "speed_max",
                "decision_interval",
            )
        },
    )
    training.output_dir = output
    return training, variants, protocol, provenance, records, episodes, tree


def run_reference_comparison(args, new_variant, *, reference_tree=None, label):
    """Only the new variant can enter preparation, retry and subprocess scheduling."""
    training, variants, protocol, source, references, episodes, tree = (
        reference_comparison_plan(args, new_variant, reference_tree)
    )
    new_variants = [
        variant
        for variant in variants
        if variant.name not in {record["variant"] for record in references}
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "source": source,
                    "protocol": protocol,
                    "reused_runs": [
                        {
                            "variant": r["variant"],
                            "training_seed": r["training_seed"],
                            "report_path": r["report_path"],
                        }
                        for r in references
                    ],
                    "commands": [
                        architecture_command(
                            training,
                            variant,
                            seed,
                            training.output_dir
                            / "runs"
                            / Path(*variant.path)
                            / f"seed_{seed}"
                            / "attempt_1",
                            training.devices[0],
                        )
                        for variant in new_variants
                        for seed in training.seeds
                    ],
                },
                indent=2,
            )
        )
        return 0
    if args.resume:
        if not (training.output_dir / "comparison.json").is_file():
            raise ValueError("No reference comparison manifest found")
    else:
        training.output_dir.mkdir(parents=True, exist_ok=False)
    with _comparison_lock(training.output_dir):
        manifest = training.output_dir / "comparison.json"
        configuration = {
            "tree": tree,
            "protocol": protocol,
            "source": source,
        }
        if args.resume and json.loads(manifest.read_text()) != configuration:
            raise ValueError("Cannot change reference evidence or protocol on resume")
        state = training.output_dir / "runs.json"
        previous = json.loads(state.read_text()) if state.is_file() else []
        records, new_episodes = _prepare_runs(
            training,
            new_variants,
            protocol,
            previous,
            evaluation_reader=matched_evaluation,
        )
        _write_json(manifest, configuration)
        _write_json(
            training.output_dir / "execution.json",
            {
                "jobs": training.jobs,
                "devices": training.devices,
                "threads_per_job": training.threads_per_job,
                "timeout": training.timeout,
            },
        )
        handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _interrupt_handler)
        try:
            interrupted = _schedule(
                training,
                new_variants,
                protocol,
                records,
                new_episodes,
                evaluation_reader=matched_evaluation,
            )
        finally:
            signal.signal(signal.SIGTERM, handler)
            _write_json(state, [*references, *records])
            write_architecture_report(
                training.output_dir,
                [*episodes, *new_episodes],
                [*references, *records],
                variants,
                training.seeds,
            )
    print(
        f"{label} comparison report: {training.output_dir / 'report.md'}",
        flush=True,
    )
    return 130 if interrupted else int(any(r["status"] != "completed" for r in records))
