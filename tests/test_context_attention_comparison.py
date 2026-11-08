"""New-only training reuses immutable attention evidence and matched scenarios."""

import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from bluesky_pabloramesc import architecture_comparison as scheduler
from bluesky_pabloramesc import context_attention_comparison as comparison
from bluesky_pabloramesc.actions import ControlConfig
from bluesky_pabloramesc.architecture_tree import attention_comparison_tree, expand_tree
from bluesky_pabloramesc.cli import run_context_attention_comparison_cli


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def snapshot(directory):
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def reference(tmp_path):
    directory = tmp_path / "reference"
    variants = expand_tree(attention_comparison_tree())
    protocol = {
        "algorithm": "ppo",
        "multi_agent": False,
        "timesteps": 100000,
        "seeds": [42],
        "eval_seeds": list(range(2026, 2036)),
        "control_config": ControlConfig(
            heading_action="relative",
            speed_action="absolute",
            observation_encoding="goal_aligned_conflict",
            observation_profile="control",
            speed_min=100,
            speed_max=500,
        ).to_dict(),
        "variants": [{"name": v.name, "settings": v.settings} for v in variants],
    }
    settings = {
        "learning_rate": 0.0003,
        "gamma": 0.995,
        "batch_size": 64,
        "n_steps": 2048,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "action_noise": None,
    }
    records = []
    for i, variant in enumerate(variants):
        run_dir = directory / "runs" / str(i)
        report = {
            "status": "completed",
            "algorithm": "PPO",
            "seed": 42,
            "requested_timesteps": 100000,
            "actual_timesteps": 100352,
            "training_seconds": 20.0,
            "steps_per_second": 5000.0,
            "control_config": protocol["control_config"],
            "architecture_config": variant.settings,
            "settings": settings,
            "evaluation_scenarios": [
                {
                    "scenario_seed": seed,
                    "agent_index": 0,
                    "return": float(-100 - i),
                    "waypoint_reached": 1,
                    "intrusion_events": 1,
                    "length": 50,
                }
                for seed in protocol["eval_seeds"]
            ],
        }
        report_path = run_dir / "reports" / "run" / "run.json"
        write_json(report_path, report)
        records.append(
            {
                "variant": variant.name,
                "training_seed": 42,
                "settings": variant.settings,
                "status": "completed",
                "returncode": 0,
                "attempt": 1,
                "directory": str(run_dir),
                "report_path": str(report_path),
                "device": "cpu",
            }
        )
    write_json(
        directory / "comparison.json",
        {"tree": attention_comparison_tree(), "protocol": protocol},
    )
    write_json(directory / "runs.json", records)
    write_json(
        directory / "execution.json",
        {"jobs": 3, "devices": ["cpu"], "threads_per_job": 1},
    )
    return Namespace(
        reference_dir=directory,
        output_dir=tmp_path / "comparison",
        resume=False,
        retry_failed=False,
        dry_run=False,
        timeout=None,
    )


@pytest.fixture
def synthetic_worker(reference, monkeypatch, tmp_path):
    calls = []
    options = {
        "fail": False,
        "mutation": {},
        "architecture": "context_entity_transformer",
    }
    template = json.loads(
        (
            reference.reference_dir / "runs" / "0" / "reports" / "run" / "run.json"
        ).read_text()
    )

    def command(args, variant, seed, directory, device):
        assert variant.settings["architecture"] == options["architecture"]
        assert args.timesteps == 100000 and seed == 42 and args.jobs == 1
        assert args.eval_seeds == list(range(2026, 2036))
        assert args.threads_per_job == 1 and device == "cpu"
        calls.append((variant.name, seed, directory))
        report = {
            **template,
            "architecture_config": variant.settings,
            "evaluation_scenarios": [
                {**r, "return": -90.0} for r in template["evaluation_scenarios"]
            ],
            **options["mutation"],
        }
        worker = """
import json, sys
from pathlib import Path
if sys.argv[3] == 'fail':
    sys.exit(7)
path = Path(sys.argv[1]) / 'reports' / 'run' / 'run.json'
path.parent.mkdir(parents=True)
path.write_text(sys.argv[2])
"""
        return [
            sys.executable,
            "-c",
            worker,
            str(directory),
            json.dumps(report),
            "fail" if options["fail"] else "ok",
        ]

    monkeypatch.setattr(scheduler, "architecture_command", command)
    return calls, options


def test_dry_run_reuses_three_results_and_emits_only_one_matched_command(
    reference, capsys
):
    before = snapshot(reference.reference_dir)
    reference.dry_run = True
    assert comparison.run_context_attention_comparison(reference) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["reused_runs"]) == 3 and len(plan["commands"]) == 1
    command = plan["commands"][0]
    assert command[command.index("--architecture") + 1] == "context_entity_transformer"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--timesteps") + 1] == "100000"
    for flag, value in {
        "--heading-action": "relative",
        "--speed-action": "absolute",
        "--observation-profile": "control",
        "--observation-encoding": "goal_aligned_conflict",
        "--speed-min": "100",
        "--speed-max": "500",
        "--decision-interval": "10",
        "--temporal": "none",
        "--pooling": "context_token",
        "--transformer-layers": "1",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert not reference.output_dir.exists()
    assert snapshot(reference.reference_dir) == before


def test_only_new_variant_runs_and_reports_all_four_then_resume_reuses_it(
    reference, synthetic_worker
):
    calls, _ = synthetic_worker
    before = snapshot(reference.reference_dir)
    assert comparison.run_context_attention_comparison(reference) == 0
    assert len(calls) == 1
    runs = json.loads((reference.output_dir / "runs.json").read_text())
    assert len(runs) == 4 and all(r["status"] == "completed" for r in runs)
    assert sum(r.get("reference", False) for r in runs) == 3
    summary = list(csv.DictReader((reference.output_dir / "summary.csv").open()))
    assert len(summary) == 4 and all(r["evaluation_episodes"] == "10" for r in summary)
    paired = list(csv.DictReader((reference.output_dir / "paired.csv").open()))
    new = next(r for r in paired if "context_transformer" in r["variant"])
    assert float(new["paired_difference"]) == 10.0
    assert (reference.output_dir / "comparison.png").stat().st_size > 0
    reference.resume = True
    assert comparison.run_context_attention_comparison(reference) == 0
    assert len(calls) == 1
    assert snapshot(reference.reference_dir) == before


def test_failed_new_run_retries_without_reference_training(reference, synthetic_worker):
    calls, options = synthetic_worker
    options["fail"] = True
    before = snapshot(reference.reference_dir)
    assert comparison.run_context_attention_comparison(reference) == 1
    original_log = (calls[0][2] / "process.log").read_bytes()
    reference.resume = True
    assert comparison.run_context_attention_comparison(reference) == 1
    assert len(calls) == 1
    reference.retry_failed = True
    options["fail"] = False
    assert comparison.run_context_attention_comparison(reference) == 0
    assert len(calls) == 2 and calls[-1][2].name == "attempt_2"
    assert (calls[0][2] / "process.log").read_bytes() == original_log
    assert snapshot(reference.reference_dir) == before


def test_new_metadata_fields_preserve_compatibility_with_older_references(
    reference, synthetic_worker
):
    _, options = synthetic_worker
    reference_report = (
        reference.reference_dir / "runs" / "0" / "reports" / "run" / "run.json"
    )
    settings = json.loads(reference_report.read_text())["settings"]
    options["mutation"] = {
        "settings": {**settings, "vf_coef": 0.5, "max_grad_norm": 0.5}
    }
    assert comparison.run_context_attention_comparison(reference) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        {"status": "failed"},
        {"seed": 1},
        {"actual_timesteps": 100353},
        {"evaluation_scenarios": []},
        {"settings": {"n_steps": 512}},
    ],
)
def test_bad_reference_stops_before_output_or_training(
    reference, synthetic_worker, mutation
):
    calls, _ = synthetic_worker
    path = reference.reference_dir / "runs" / "1" / "reports" / "run" / "run.json"
    write_json(path, {**json.loads(path.read_text()), **mutation})
    with pytest.raises(ValueError):
        comparison.run_context_attention_comparison(reference)
    assert calls == [] and not reference.output_dir.exists()


def test_missing_reference_never_triggers_training(reference, synthetic_worker):
    calls, _ = synthetic_worker
    (reference.reference_dir / "runs" / "1" / "reports" / "run" / "run.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        comparison.run_context_attention_comparison(reference)
    assert not calls and not reference.output_dir.exists()


def test_source_changes_are_rejected_on_resume(reference, synthetic_worker):
    calls, _ = synthetic_worker
    assert comparison.run_context_attention_comparison(reference) == 0
    path = reference.reference_dir / "runs" / "1" / "reports" / "run" / "run.json"
    report = json.loads(path.read_text())
    report["evaluation_scenarios"][0]["return"] -= 1
    write_json(path, report)
    reference.resume = reference.retry_failed = True
    with pytest.raises(ValueError, match="reference evidence"):
        comparison.run_context_attention_comparison(reference)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"settings": {"n_steps": 512}},
        {"actual_timesteps": 100353},
        {"evaluation_scenarios": []},
        {"architecture_config": {"architecture": "entity_mlp"}},
    ],
)
def test_mismatched_new_results_are_excluded(reference, synthetic_worker, mutation):
    _, options = synthetic_worker
    options["mutation"] = mutation
    assert comparison.run_context_attention_comparison(reference) == 1
    runs = json.loads((reference.output_dir / "runs.json").read_text())
    assert runs[-1]["status"] == "failed"
    summary = list(csv.DictReader((reference.output_dir / "summary.csv").open()))
    assert summary[-1]["completed_seeds"] == "0"


@pytest.mark.parametrize("location", ["same", "inside", "ancestor"])
def test_output_cannot_modify_reference_tree(reference, location):
    reference.output_dir = {
        "same": reference.reference_dir,
        "inside": reference.reference_dir / "new",
        "ancestor": reference.reference_dir.parent,
    }[location]
    with pytest.raises(ValueError, match="separate"):
        comparison.run_context_attention_comparison(reference)


def test_cli_defaults_dispatch_only_new_work(monkeypatch):
    captured = []
    monkeypatch.setattr(
        comparison,
        "run_context_attention_comparison",
        lambda args: captured.append(args) or 0,
    )
    run_context_attention_comparison_cli(["--dry-run"])
    assert captured[0].reference_dir == Path("experiments/results/attention_comparison")
    assert captured[0].output_dir == Path(
        "experiments/results/context_attention_comparison"
    )
    assert captured[0].dry_run
    with pytest.raises(SystemExit):
        run_context_attention_comparison_cli(["--timesteps", "200000"])
