"""One-term experimental design, shared scheduler, validation and paired reports."""

import json
import sys
from argparse import Namespace
from dataclasses import replace

import pytest

from bluesky_pabloramesc.cli import parse_args, run_reward_ablation_cli
from bluesky_pabloramesc.reward_ablation import (
    INITIAL_WEIGHTS,
    ablation_command,
    ablation_evaluation,
    ablation_plan,
    ablation_summary,
    ablation_variants,
    run_reward_ablation,
)
from bluesky_pabloramesc.rewards import COMPONENTS


def arguments(tmp_path, **overrides):
    return Namespace(
        **{
            "timesteps": 4,
            "jobs": 2,
            "threads_per_job": 1,
            "seeds": [42],
            "eval_seeds": [2026, 2027],
            "devices": ["cpu"],
            "timeout": None,
            "resume": False,
            "retry_failed": False,
            "dry_run": False,
            "output_dir": tmp_path / "study",
            "weights": dict(INITIAL_WEIGHTS),
            **overrides,
        }
    )


def test_exactly_one_term_and_matching_training_commands(tmp_path):
    args = arguments(tmp_path)
    variants, protocol = ablation_plan(args)
    assert [variant.name for variant in variants] == ["baseline", *COMPONENTS]
    assert protocol["architecture_config"]["attention_context"] == "all"
    for index, variant in enumerate(variants):
        active = [
            key
            for key, value in variant.settings.items()
            if key.endswith("_weight") and value > 0
        ]
        assert active == ([] if index == 0 else [f"{variant.name}_weight"])
        command = ablation_command(args, variant, 42, tmp_path / variant.name, "cpu")
        parsed = parse_args("ppo", argv=command[3:])
        assert parsed.reward_config.to_dict() == variant.settings
        assert parsed.architecture == "entity_mlp"
        assert parsed.pooling == "additive_attention"
        assert parsed.attention_context == "all"
        assert parsed.observation_encoding == "goal_aligned_conflict"
        assert parsed.speed_action == "absolute"
        assert parsed.eval_seeds == args.eval_seeds


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_ablation_requires_positive_finite_weights(value):
    with pytest.raises(ValueError):
        ablation_variants({**INITIAL_WEIGHTS, "progress": value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeds": []},
        {"seeds": [1, 1]},
        {"eval_seeds": [-1]},
        {"timesteps": 0},
        {"devices": []},
        {"retry_failed": True},
    ],
)
def test_invalid_protocol(tmp_path, overrides):
    with pytest.raises(ValueError):
        ablation_plan(arguments(tmp_path, **overrides))


def test_dry_run_does_not_create_artifacts(tmp_path, capsys):
    args = arguments(tmp_path, dry_run=True)
    assert run_reward_ablation(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["commands"]) == 7
    assert not args.output_dir.exists()


def test_cli_default_protocol_and_overrides(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        "bluesky_pabloramesc.reward_ablation.run_reward_ablation",
        lambda args: captured.append(args) or 0,
    )
    run_reward_ablation_cli(
        ["--output-dir", str(tmp_path / "study"), "--progress-weight", "0.2"]
    )
    assert captured[0].timesteps == 100_000
    assert captured[0].seeds == [42]
    assert captured[0].weights["progress"] == 0.2


def row(variant, seed, scenario, *, intrusion=2, success=1):
    return {
        "variant": variant,
        "training_seed": seed,
        "scenario_seed": scenario,
        "agent_index": 0,
        "return": 1000 if variant != "baseline" else 0,
        "base_return": 10,
        "total_reward": 10,
        "waypoint_reached": success,
        "timed_out": 1 - success,
        "intrusion_events": 1,
        "intrusion_time": intrusion,
        "restricted_area_events": 0,
        "time_in_restricted_area": 0,
        "sector_exit_events": 0,
        "time_outside_sector": 0,
        "flight_time": 100,
        **{f"reward_{name}": 0 for name in COMPONENTS},
    }


def test_summary_pairs_behavior_and_weights_training_seeds_equally():
    variants = ablation_variants(INITIAL_WEIGHTS)[:2]
    rows = [
        row(
            variant.name,
            seed,
            scenario,
            intrusion=seed + (3 if variant.name == "progress" else 0),
            success=int(variant.name == "baseline"),
        )
        for variant in variants
        for seed, scenarios in [(0, [1]), (2, [1, 2, 3])]
        for scenario in scenarios
    ]
    summary, paired = ablation_summary(rows, [], variants, [0, 2])
    assert summary[0]["mean_intrusion_time"] == 1
    assert summary[1]["mean_intrusion_time"] == 4
    assert summary[1]["std_intrusion_time"] == pytest.approx(2**0.5)
    assert "mean_successful_flight_time" not in summary[1]
    assert all("rank" not in item for item in summary)
    differences = [
        item["difference_from_baseline"]
        for item in paired
        if item["variant"] == "progress" and item["metric"] == "intrusion_time"
    ]
    assert differences == [3, 3]


def test_scheduler_validation_reports_and_resume(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    script = tmp_path / "worker.py"
    script.write_text("""import json, sys
from pathlib import Path
directory = Path(sys.argv[1]) / 'reports' / 'run'
directory.mkdir(parents=True)
(directory / 'run.json').write_text(sys.argv[2])
""")
    calls = []

    def command(args, variant, seed, directory, device):
        calls.append(variant.name)
        rows = [row(variant.name, seed, scenario) for scenario in args.eval_seeds]
        for item in rows:
            item.pop("base_return")
            item["reward_progress"] = 1 if variant.name == "progress" else 0
            item["total_reward"] = 10 + item["reward_progress"]
        _, protocol = ablation_plan(args)
        report = {
            "status": "completed",
            "seed": seed,
            "requested_timesteps": 4,
            "algorithm": "PPO",
            "control_config": protocol["control_config"],
            "architecture_config": protocol["architecture_config"],
            "reward_config": variant.settings,
            "training_seconds": 0.01,
            "actual_timesteps": 4,
            "steps_per_second": 400,
            "evaluation_scenarios": rows,
        }
        return [sys.executable, str(script), str(directory), json.dumps(report)]

    monkeypatch.setattr("bluesky_pabloramesc.reward_ablation.ablation_command", command)
    assert run_reward_ablation(args) == 0
    assert len(calls) == 7
    for name in (
        "comparison.json",
        "execution.json",
        "runs.json",
        "runs.csv",
        "episodes.csv",
        "summary.csv",
        "paired.csv",
        "comparison.png",
        "report.md",
    ):
        assert (args.output_dir / name).is_file()
    records = json.loads((args.output_dir / "runs.json").read_text())
    assert all(record["status"] == "completed" for record in records)
    args.resume = True
    assert run_reward_ablation(args) == 0
    assert len(calls) == 7
    variants, protocol = ablation_plan(args)
    record = next(item for item in records if item["variant"] == "progress")
    rows = ablation_evaluation(record, variants[1], protocol)
    assert rows[0]["base_return"] == 10
    with pytest.raises(ValueError, match="Reward configuration mismatch"):
        ablation_evaluation(
            record, replace(variants[1], settings=variants[0].settings), protocol
        )
    args.weights["progress"] = 0.2
    with pytest.raises(ValueError, match="Cannot change"):
        run_reward_ablation(args)
