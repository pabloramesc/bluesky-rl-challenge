"""Nearest-edge attention runs reuse the known baseline without retraining it."""

import csv
import json
import sys
from pathlib import Path

import pytest
from test_context_attention_comparison import reference as reference  # noqa: PLC0414
from test_context_attention_comparison import snapshot, write_json

from bluesky_pabloramesc import attention_geometry_comparison as comparison
from bluesky_pabloramesc.actions import ControlConfig
from bluesky_pabloramesc.architecture_tree import ArchitectureVariant
from bluesky_pabloramesc.cli import run_attention_geometry_comparison_cli


def baseline_report(args):
    return args.reference_dir / "runs" / "0" / "reports" / "run" / "run.json"


@pytest.fixture
def nearest_worker(reference, monkeypatch):
    calls, options = [], {"fail": False, "mutation": {}}
    template = json.loads(baseline_report(reference).read_text())
    original = comparison.attention_geometry_command

    def command(args, variant, seed, directory, device):
        production = original(args, variant, seed, directory, device)
        calls.append(production)
        report = {
            **template,
            "control_config": ControlConfig(
                **{
                    **template["control_config"],
                    "geometry_representation": variant.name,
                }
            ).to_dict(),
            "evaluation_scenarios": [
                {**row, "return": row["return"] + 10}
                for row in template["evaluation_scenarios"]
            ],
            **options["mutation"],
        }
        worker = """
import sys
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

    monkeypatch.setattr(comparison, "attention_geometry_command", command)
    return calls, options


def test_plan_reuses_only_best_baseline_with_identical_controls(reference, capsys):
    before = snapshot(reference.reference_dir)
    reference.dry_run = True
    assert comparison.run_attention_geometry_comparison(reference) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["reused_runs"]) == 1
    assert plan["reused_runs"][0]["variant"] == "current"
    assert len(plan["commands"]) == 1
    command = plan["commands"][0]
    for flag, value in {
        "--geometry-representation": "nearest_edge",
        "--architecture": "entity_mlp",
        "--pooling": "additive_attention",
        "--attention-context": "all",
        "--temporal": "none",
        "--seed": "42",
        "--timesteps": "100000",
        "--observation-encoding": "goal_aligned_conflict",
        "--observation-profile": "control",
        "--heading-action": "relative",
        "--speed-action": "absolute",
        "--speed-min": "100",
        "--speed-max": "500",
        "--decision-interval": "10",
    }.items():
        assert command[command.index(flag) + 1] == value
    start = command.index("--eval-seeds") + 1
    assert command[start : start + 10] == list(map(str, range(2026, 2036)))
    assert plan["protocol"]["actual_timesteps"] == 100352
    controls = plan["protocol"]["control_configs"]
    default = ControlConfig(**controls["current"])
    nearest = controls["nearest_edge"]
    assert ControlConfig(**{**nearest, "geometry_representation": "current"}) == default
    assert snapshot(reference.reference_dir) == before
    assert not reference.output_dir.exists()


def test_run_exports_two_variants_and_resume_never_retrains_baseline(
    reference, nearest_worker
):
    calls, _ = nearest_worker
    before = snapshot(reference.reference_dir)
    assert comparison.run_attention_geometry_comparison(reference) == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--geometry-representation") + 1] == "nearest_edge"
    runs = json.loads((reference.output_dir / "runs.json").read_text())
    assert [r["variant"] for r in runs] == ["current", "nearest_edge"]
    assert runs[0]["reference"] and runs[0]["reused"]
    with (reference.output_dir / "summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 2
    assert all(row["evaluation_episodes"] == "10" for row in summary)
    with (reference.output_dir / "paired.csv").open() as stream:
        nearest = next(
            row for row in csv.DictReader(stream) if row["variant"] == "nearest_edge"
        )
    assert nearest["baseline"] == "current"
    assert float(nearest["paired_difference"]) == 10
    assert (reference.output_dir / "comparison.png").is_file()
    assert (reference.output_dir / "report.md").is_file()
    reference.resume = True
    assert comparison.run_attention_geometry_comparison(reference) == 0
    assert len(calls) == 1
    assert snapshot(reference.reference_dir) == before


def test_failed_nearest_run_can_retry_without_touching_baseline(
    reference, nearest_worker
):
    calls, options = nearest_worker
    before = snapshot(reference.reference_dir)
    options["fail"] = True
    assert comparison.run_attention_geometry_comparison(reference) == 1
    reference.resume = reference.retry_failed = True
    options["fail"] = False
    assert comparison.run_attention_geometry_comparison(reference) == 0
    assert len(calls) == 2
    runs = json.loads((reference.output_dir / "runs.json").read_text())
    assert runs[1]["attempt"] == 2
    assert snapshot(reference.reference_dir) == before


@pytest.mark.parametrize(
    "mutation",
    [
        {"status": "failed"},
        {"actual_timesteps": 10},
        {"evaluation_scenarios": []},
        {"settings": {}},
        {"reward_config": {"progress_weight": 1}},
    ],
)
def test_invalid_baseline_stops_before_training(reference, nearest_worker, mutation):
    calls, _ = nearest_worker
    path = baseline_report(reference)
    write_json(path, {**json.loads(path.read_text()), **mutation})
    with pytest.raises(ValueError):
        comparison.run_attention_geometry_comparison(reference)
    assert not calls and not reference.output_dir.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        {"control_config": ControlConfig().to_dict()},
        {"architecture_config": {"architecture": "mlp"}},
        {"settings": {"n_steps": 512}},
        {"actual_timesteps": 100353},
        {"evaluation_scenarios": []},
    ],
)
def test_incorrect_new_run_configuration_is_rejected(
    reference, nearest_worker, mutation
):
    calls, options = nearest_worker
    options["mutation"] = mutation
    assert comparison.run_attention_geometry_comparison(reference) == 1
    assert len(calls) == 1
    runs = json.loads((reference.output_dir / "runs.json").read_text())
    assert runs[1]["status"] == "failed" and runs[1]["error"]


def test_reference_mutation_rejected_on_resume(reference, nearest_worker):
    calls, _ = nearest_worker
    assert comparison.run_attention_geometry_comparison(reference) == 0
    path = baseline_report(reference)
    report = json.loads(path.read_text())
    report["evaluation_scenarios"][0]["return"] -= 1
    write_json(path, report)
    reference.resume = True
    with pytest.raises(ValueError, match="reference evidence"):
        comparison.run_attention_geometry_comparison(reference)
    assert len(calls) == 1


def test_baseline_command_is_forbidden():
    with pytest.raises(ValueError, match="never be retrained"):
        comparison.attention_geometry_command(
            None, ArchitectureVariant(("current",), {}), 42, Path("unused"), "cpu"
        )


def test_cli_defaults_to_known_attention_reference(monkeypatch):
    calls = []
    monkeypatch.setattr(
        comparison,
        "run_attention_geometry_comparison",
        lambda args: calls.append(args) or 0,
    )
    run_attention_geometry_comparison_cli(["--dry-run"])
    assert calls[0].reference_dir == Path("experiments/results/attention_comparison")
    assert calls[0].output_dir == Path(
        "experiments/results/attention_geometry_comparison"
    )
    with pytest.raises(SystemExit):
        run_attention_geometry_comparison_cli(["--timesteps", "200000"])


def test_ray_followup_plan_appends_rays_and_reuses_default(reference, capsys):
    before = snapshot(reference.reference_dir)
    reference.dry_run = True
    assert comparison.run_attention_ray_comparison(reference) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["commands"]) == len(plan["reused_runs"]) == 1
    assert plan["reused_runs"][0]["variant"] == "current"
    command = plan["commands"][0]
    for flag, value in {
        "--geometry-representation": "current_rays",
        "--geometry-ray-count": "24",
        "--geometry-ray-range-m": "100000",
        "--architecture": "entity_mlp",
        "--pooling": "additive_attention",
        "--attention-context": "all",
        "--observation-encoding": "goal_aligned_conflict",
        "--observation-profile": "control",
        "--speed-action": "absolute",
        "--seed": "42",
        "--timesteps": "100000",
    }.items():
        assert command[command.index(flag) + 1] == value
    controls = plan["protocol"]["control_configs"]
    rays = controls["current_rays"]
    assert ControlConfig(
        **{**rays, "geometry_representation": "current"}
    ) == ControlConfig(**controls["current"])
    assert plan["protocol"]["eval_seeds"] == list(range(2026, 2036))
    assert snapshot(reference.reference_dir) == before
    assert not reference.output_dir.exists()


def test_ray_followup_reports_and_resume(reference, nearest_worker):
    calls, _ = nearest_worker
    before = snapshot(reference.reference_dir)
    assert comparison.run_attention_ray_comparison(reference) == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--geometry-representation") + 1] == "current_rays"
    runs = json.loads((reference.output_dir / "runs.json").read_text())
    assert [r["variant"] for r in runs] == ["current", "current_rays"]
    with (reference.output_dir / "paired.csv").open() as stream:
        rays = next(
            row for row in csv.DictReader(stream) if row["variant"] == "current_rays"
        )
    assert rays["baseline"] == "current" and float(rays["paired_difference"]) == 10
    reference.resume = True
    assert comparison.run_attention_ray_comparison(reference) == 0
    assert len(calls) == 1
    assert snapshot(reference.reference_dir) == before


def test_ray_followup_rejects_replacement_geometry(reference, nearest_worker):
    calls, options = nearest_worker
    control = json.loads(baseline_report(reference).read_text())["control_config"]
    options["mutation"] = {
        "control_config": ControlConfig(
            **{
                **control,
                "geometry_representation": "ray_cast",
            }
        ).to_dict()
    }
    assert comparison.run_attention_ray_comparison(reference) == 1
    assert len(calls) == 1


def test_ray_cli_has_separate_output_directory(monkeypatch):
    from bluesky_pabloramesc.cli import run_attention_ray_comparison_cli

    calls = []
    monkeypatch.setattr(
        comparison, "run_attention_ray_comparison", lambda args: calls.append(args) or 0
    )
    run_attention_ray_comparison_cli(["--dry-run"])
    assert calls[0].reference_dir == Path("experiments/results/attention_comparison")
    assert calls[0].output_dir == Path("experiments/results/attention_ray_comparison")
