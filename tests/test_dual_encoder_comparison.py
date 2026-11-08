"""The dual encoder trains once, reuses references, and exports matched tables."""

import csv
import json
from pathlib import Path

import pytest
from test_architecture_comparison import comparison_args
from test_context_attention_comparison import reference as reference  # noqa: PLC0414
from test_context_attention_comparison import snapshot, write_json
from test_context_attention_comparison import synthetic_worker as synthetic_worker  # noqa: PLC0414

from bluesky_pabloramesc import architecture_comparison as scheduler
from bluesky_pabloramesc import dual_encoder_comparison as comparison
from bluesky_pabloramesc.cli import run_dual_encoder_comparison_cli


@pytest.fixture
def dual_reference(reference, synthetic_worker):
    # Include a fourth, already-completed context-token variant as in real results.
    manifest_path = reference.reference_dir / "comparison.json"
    manifest = json.loads(manifest_path.read_text())
    records_path = reference.reference_dir / "runs.json"
    records = json.loads(records_path.read_text())
    node = {
        "name": "context",
        "settings": {
            "architecture": "context_entity_transformer",
            "pooling": "context_token",
            "transformer_layers": 1,
            "temporal": "none",
        },
    }
    manifest["tree"]["children"].append(node)
    variant = {"name": "attention_comparison/context", "settings": {**node["settings"]}}
    manifest["protocol"]["variants"].append(variant)
    report_path = (
        reference.reference_dir / "runs" / "context" / "reports" / "run" / "run.json"
    )
    report = json.loads(Path(records[0]["report_path"]).read_text())
    report["architecture_config"] = variant["settings"]
    write_json(report_path, report)
    records.append(
        {
            **records[0],
            "variant": variant["name"],
            "settings": variant["settings"],
            "directory": str(report_path.parents[2]),
            "report_path": str(report_path),
        }
    )
    write_json(manifest_path, manifest)
    write_json(records_path, records)
    synthetic_worker[1]["architecture"] = "dual_entity_transformer"
    return reference


def test_dry_run_reuses_four_and_matches_reference_protocol(dual_reference, capsys):
    args = dual_reference
    before = snapshot(args.reference_dir)
    args.dry_run = True
    assert comparison.run_dual_encoder_comparison(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["reused_runs"]) == 4 and len(plan["commands"]) == 1
    command = plan["commands"][0]
    for flag, value in {
        "--architecture": "dual_entity_transformer",
        "--transformer-layers": "2",
        "--pooling": "context_token",
        "--seed": "42",
        "--timesteps": "100000",
        "--observation-encoding": "goal_aligned_conflict",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert not args.output_dir.exists()
    assert snapshot(args.reference_dir) == before


def test_combined_report_resume_and_retry_preserve_references(
    dual_reference, synthetic_worker
):
    args = dual_reference
    calls, options = synthetic_worker
    before = snapshot(args.reference_dir)
    options["fail"] = True
    assert comparison.run_dual_encoder_comparison(args) == 1
    args.resume = args.retry_failed = True
    options["fail"] = False
    assert comparison.run_dual_encoder_comparison(args) == 0
    assert len(calls) == 2 and calls[-1][2].name == "attempt_2"
    assert all("dual_transformer" in call[0] for call in calls)
    assert comparison.run_dual_encoder_comparison(args) == 0
    assert len(calls) == 2
    summary = list(csv.DictReader((args.output_dir / "summary.csv").open()))
    assert len(summary) == 5 and all(
        row["evaluation_episodes"] == "10" for row in summary
    )
    paired = list(csv.DictReader((args.output_dir / "paired.csv").open()))
    new = next(row for row in paired if "dual_transformer" in row["variant"])
    assert float(new["paired_difference"]) == 10.0
    assert (args.output_dir / "report.md").is_file()
    assert (args.output_dir / "comparison.png").stat().st_size > 0
    assert snapshot(args.reference_dir) == before


@pytest.mark.parametrize(
    "mutation",
    [{"status": "failed"}, {"actual_timesteps": 10}, {"settings": {"n_steps": 4}}],
)
def test_bad_reference_stops_before_training(
    dual_reference, synthetic_worker, mutation
):
    args = dual_reference
    calls, _ = synthetic_worker
    path = args.reference_dir / "runs" / "context" / "reports" / "run" / "run.json"
    write_json(path, {**json.loads(path.read_text()), **mutation})
    with pytest.raises(ValueError):
        comparison.run_dual_encoder_comparison(args)
    assert not calls and not args.output_dir.exists()


def test_real_training_subprocess_exports_dual_architecture(tmp_path, monkeypatch):
    args = comparison_args(
        tmp_path, tree=Path("experiments/configs/dual_encoder.json"), seeds=[0]
    )
    original = scheduler.architecture_command

    def command(*values):
        result = original(*values)
        return [
            result[0],
            str(Path(__file__).with_name("architecture_training_worker.py")),
            *result[2:],
        ]

    monkeypatch.setattr(scheduler, "architecture_command", command)
    assert scheduler.run_architecture_comparison(args) == 0
    records = json.loads((args.output_dir / "runs.json").read_text())
    root = Path(records[0]["directory"])
    config = json.loads((root / "policy_architecture.json").read_text())
    assert config["architecture"] == "dual_entity_transformer"
    assert config["transformer_layers"] == 2
    assert (root / "policy_vecnormalize.pkl").is_file()
    report = json.loads(Path(records[0]["report_path"]).read_text())
    assert report["status"] == "completed" and report["actual_timesteps"] == 2
    assert (args.output_dir / "summary.csv").is_file()
    args.resume, args.tree = True, None
    assert scheduler.run_architecture_comparison(args) == 0
    assert json.loads((args.output_dir / "runs.json").read_text())[0]["reused"]


def test_cli_defaults_and_rejects_protocol_overrides(monkeypatch):
    captured = []
    monkeypatch.setattr(
        comparison,
        "run_dual_encoder_comparison",
        lambda args: captured.append(args) or 0,
    )
    run_dual_encoder_comparison_cli(["--dry-run"])
    assert captured[0].reference_dir == Path(
        "experiments/results/context_attention_comparison"
    )
    assert captured[0].output_dir == Path("experiments/results/dual_encoder_comparison")
    with pytest.raises(SystemExit):
        run_dual_encoder_comparison_cli(["--timesteps", "200000"])
