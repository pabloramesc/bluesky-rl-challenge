"""Compare geometry changes against the completed additive-attention PPO baseline."""

import json
from argparse import Namespace
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .actions import ControlConfig
from .architecture_tree import ArchitectureVariant, setting_arguments
from .comparison import comparison_command
from .reference_comparison import matched_evaluation, run_reference_comparison

ATTENTION_ARCHITECTURE = {
    "architecture": "entity_mlp",
    "pooling": "additive_attention",
    "attention_context": "all",
    "temporal": "none",
}


def attention_geometry_plan(args, new_variant, reference_tree=None):
    """Read only the selected baseline and inherit its complete paired protocol."""
    source = args.reference_dir.resolve()
    output = args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            "Reference and output directories must be separate, non-nested paths"
        )
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    if args.timeout is not None and not 0 < args.timeout < float("inf"):
        raise ValueError("timeout must be finite and positive")
    manifest_path = source / "comparison.json"
    manifest = json.loads(manifest_path.read_text())
    original = manifest["protocol"]
    control = ControlConfig(
        heading_action="relative",
        speed_action="absolute",
        observation_encoding="goal_aligned_conflict",
        observation_profile="control",
        speed_min=100,
        speed_max=500,
    )
    if (
        original["algorithm"] != "ppo"
        or original["multi_agent"] is not False
        or ControlConfig(**original["control_config"]) != control
    ):
        raise ValueError(
            "Reference must use the selected single-agent PPO control configuration"
        )
    for name in ("seeds", "eval_seeds"):
        values = original[name]
        if (
            not values
            or len(set(values)) != len(values)
            or any(type(v) is not int or v < 0 for v in values)
        ):
            raise ValueError(
                f"Reference {name} must contain unique nonnegative integers"
            )
    if type(original["timesteps"]) is not int or original["timesteps"] <= 0:
        raise ValueError("Reference timesteps must be a positive integer")
    selected = [
        v for v in original["variants"] if v["settings"] == ATTENTION_ARCHITECTURE
    ]
    if len(selected) != 1:
        raise ValueError(
            "Reference must contain exactly one all-context additive-attention no-history PPO variant"
        )
    reference = ArchitectureVariant(
        tuple(selected[0]["name"].split("/")), selected[0]["settings"]
    )
    baseline = ArchitectureVariant(("current",), dict(ATTENTION_ARCHITECTURE))
    representation = new_variant["name"]
    if representation not in ("nearest_edge", "current_rays"):
        raise ValueError("Only nearest_edge and current_rays follow-ups are supported")
    candidate = ArchitectureVariant((representation,), dict(ATTENTION_ARCHITECTURE))
    variants = [baseline, candidate]
    controls = {
        "current": control.to_dict(),
        representation: replace(
            control, geometry_representation=representation
        ).to_dict(),
    }
    records = [
        r
        for r in json.loads((source / "runs.json").read_text())
        if r["variant"] == reference.name
    ]
    if len(records) != len(original["seeds"]) or {
        r["training_seed"] for r in records
    } != set(original["seeds"]):
        raise ValueError("Reference needs exactly one baseline run per training seed")
    episodes, hashes = [], {}
    training_settings, actual_timesteps = None, None
    for record in records:
        if record["status"] != "completed" or record.get("returncode") != 0:
            raise ValueError("The additive-attention baseline must be completed")
        report_path = Path(record["directory"]) / "reports"
        paths = sorted(report_path.glob("*/run.json"))
        if len(paths) != 1:
            raise ValueError("Expected one completed report for each baseline run")
        report = json.loads(paths[0].read_text())
        if training_settings is None:
            training_settings = report["settings"]
            actual_timesteps = report["actual_timesteps"]
        rows = matched_evaluation(
            record,
            reference,
            {
                **original,
                "training_settings": training_settings,
                "actual_timesteps": actual_timesteps,
            },
        )
        if rows is None:
            raise ValueError("Baseline evaluation is missing or incomplete")
        episodes.extend({**row, "variant": "current"} for row in rows)
        record.update(
            variant="current",
            source_variant=reference.name,
            reference=True,
            reused=True,
        )
        hashes[str(paths[0].resolve())] = sha256(paths[0].read_bytes()).hexdigest()
    execution_path = source / "execution.json"
    execution = json.loads(execution_path.read_text())
    threads = execution["threads_per_job"]
    devices = {r["device"] for r in records}
    if type(threads) is not int or threads <= 0 or len(devices) != 1:
        raise ValueError(
            "Baseline must use one common device and positive thread count"
        )
    protocol = {
        **original,
        "variants": [{"name": v.name, "settings": v.settings} for v in variants],
        "control_configs": controls,
        "training_settings": training_settings,
        "actual_timesteps": actual_timesteps,
    }
    provenance = {
        "directory": str(source),
        "variant": reference.name,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "execution_sha256": sha256(execution_path.read_bytes()).hexdigest(),
        "reports_sha256": hashes,
    }
    training = Namespace(
        **vars(args),
        algorithm="ppo",
        multi_agent=False,
        timesteps=original["timesteps"],
        seeds=original["seeds"],
        eval_seeds=original["eval_seeds"],
        jobs=1,
        devices=list(devices),
        threads_per_job=threads,
    )
    training.output_dir = output
    tree = {
        "name": "attention_geometry",
        "children": [{"name": v.name, "settings": v.settings} for v in variants],
    }
    return training, variants, protocol, provenance, records, episodes, tree


def attention_geometry_command(args, variant, seed, directory, device):
    """Only new geometry variants are eligible for training."""
    if variant.name not in ("nearest_edge", "current_rays"):
        raise ValueError(
            "The completed default-geometry baseline must never be retrained"
        )
    return [
        *comparison_command(
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
            observation_profile="control",
            decision_interval=10,
            geometry_representation=variant.name,
        ),
        *setting_arguments(ATTENTION_ARCHITECTURE),
    ]


def attention_geometry_evaluation(record, variant, protocol):
    """Validate identical PPO settings, rewards, scenarios, and geometry selection."""
    return matched_evaluation(
        record,
        variant,
        {
            **protocol,
            "control_config": protocol["control_configs"][variant.name],
        },
    )


def run_attention_geometry_comparison(args):
    """Reuse default-geometry results and schedule only nearest-edge training."""
    return run_reference_comparison(
        args,
        {"name": "nearest_edge", "settings": ATTENTION_ARCHITECTURE},
        label="Additive-attention geometry",
        plan_builder=attention_geometry_plan,
        command_builder=attention_geometry_command,
        evaluation_reader=attention_geometry_evaluation,
    )


def run_attention_ray_comparison(args):
    """Append rays to default geometry while reusing the completed baseline."""
    return run_reference_comparison(
        args,
        {"name": "current_rays", "settings": ATTENTION_ARCHITECTURE},
        label="Default geometry plus rays",
        plan_builder=attention_geometry_plan,
        command_builder=attention_geometry_command,
        evaluation_reader=attention_geometry_evaluation,
    )
