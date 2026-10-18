"""Paired experimental design and aggregation."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from bluesky_pabloramesc.comparison import comparison_command, run_comparison, summarize


def test_comparison_runs_all_variants_with_shared_scenarios(tmp_path, monkeypatch):
    commands = []

    def run(command, check):
        commands.append(command)
        root = Path(command[command.index("--results-dir") + 1]) / "run"
        root.mkdir(parents=True)
        heading = command[command.index("--heading-action") + 1]
        (root / "run.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "evaluation_scenarios": [
                        {
                            "scenario_seed": seed,
                            "agent_index": 0,
                            "return": 2.0 if heading == "absolute" else 1.0,
                        }
                        for seed in [100, 101]
                    ],
                }
            )
        )

    monkeypatch.setattr("bluesky_pabloramesc.comparison.subprocess.run", run)
    args = Namespace(
        algorithm="ppo",
        multi_agent=False,
        seeds=[0, 1],
        eval_seeds=[100, 101],
        output_dir=tmp_path / "compare",
        timesteps=8,
        speed_min=100,
        speed_max=500,
        device="cpu",
    )
    run_comparison(args)
    assert len(commands) == 8
    assert all(command[-3:] == ["--eval-seeds", "100", "101"] for command in commands)
    for name in (
        "episodes.csv",
        "summary.csv",
        "paired.csv",
        "comparison.png",
        "comparison.json",
    ):
        assert (args.output_dir / name).is_file()
    with pytest.raises(FileExistsError):
        run_comparison(args)


def test_uncertainty_uses_training_seeds_not_agent_samples():
    rows = [
        {"variant": variant, "training_seed": seed, "return": seed + offset}
        for variant, offset in [("relative_relative", 0), ("absolute_relative", 3)]
        for seed in [0, 2]
        for _ in range(10)
    ]
    summary, paired = summarize(rows)
    assert summary[0]["std_across_seed_means"] == pytest.approx(2**0.5)
    assert summary[1]["mean_return"] == 4
    assert [r["difference_from_relative_relative"] for r in paired] == [0, 0, 3, 3]


def test_multi_agent_command_uses_supported_entrypoint(tmp_path):
    command = comparison_command(
        "sac", True, "relative", "absolute", 42, [7], tmp_path, 8, 100, 500, "cpu"
    )
    assert command[2] == "experiments.train_zoo_sac"
    assert command[command.index("--observation-profile") + 1] == "control"


def test_resume_reuses_baseline_and_trains_only_missing_variants(tmp_path, monkeypatch):
    commands = []
    args = Namespace(
        algorithm="ppo",
        multi_agent=False,
        seeds=[0],
        eval_seeds=[100],
        output_dir=tmp_path / "compare",
        timesteps=8,
        speed_min=100,
        speed_max=500,
        device="cpu",
        resume=True,
    )
    args.output_dir.mkdir()
    original = {**vars(args), "seeds": [0, 1, 2, 3, 4], "eval_seeds": [100, 101]}
    manifest = args.output_dir / "comparison.json"
    manifest.write_text(json.dumps(original, default=str))
    original_text = manifest.read_text()

    def write_report(directory):
        report = directory / "reports" / "run"
        report.mkdir(parents=True)
        (report / "run.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "evaluation_scenarios": [
                        {"scenario_seed": seed, "agent_index": 0, "return": 1.0}
                        for seed in [100, 101]
                    ],
                }
            )
        )

    write_report(args.output_dir / "relative_relative" / "seed_0")

    def run(command, check):
        commands.append(command)
        report_parent = Path(command[command.index("--results-dir") + 1]).parent
        write_report(report_parent)

    monkeypatch.setattr("bluesky_pabloramesc.comparison.subprocess.run", run)
    run_comparison(args)
    assert len(commands) == 3
    assert manifest.read_text() == original_text
    assert (args.output_dir / "selection.json").is_file()
    commands.clear()
    run_comparison(args)
    assert commands == []
    import csv

    with (args.output_dir / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row["scenario_seed"] for row in rows} == {"100"}
    assert {row["training_seed"] for row in rows} == {"0"}
    args.timesteps = 16
    with pytest.raises(ValueError, match="Cannot change timesteps"):
        run_comparison(args)
    assert commands == []


def test_cached_results_reject_missing_scenarios_and_preserve_partial_runs(tmp_path):
    from bluesky_pabloramesc.comparison import completed_evaluation

    report = tmp_path / "reports" / "run"
    report.mkdir(parents=True)
    path = report / "run.json"
    path.write_text(json.dumps({"status": "interrupted", "actual_timesteps": 42}))
    with pytest.raises(ValueError, match="saved checkpoint is preserved"):
        completed_evaluation(tmp_path, [100])
    assert json.loads(path.read_text())["actual_timesteps"] == 42
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "evaluation_scenarios": [
                    {"scenario_seed": 100, "return": 1},
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="evaluation seeds are missing"):
        completed_evaluation(tmp_path, [101])


@pytest.mark.parametrize(
    "encodings",
    [
        None,
        ["upstream", "conflict"],
        ["ego_cartesian", "ego_cartesian_conflict"],
        ["goal_aligned", "goal_aligned_conflict"],
    ],
)
def test_observation_comparison_pairs_variants_and_reuses_results(
    tmp_path, monkeypatch, encodings
):
    import csv

    commands = []

    def run(command, check):
        commands.append(command)
        encoding = command[command.index("--observation-encoding") + 1]
        offset = {
            "upstream": 0,
            "ego_cartesian": 2,
            "goal_aligned": 3,
            "conflict": 4,
            "ego_cartesian_conflict": 6,
            "goal_aligned_conflict": 7,
        }[encoding]
        seed = int(command[command.index("--seed") + 1])
        root = Path(command[command.index("--results-dir") + 1]) / "run"
        root.mkdir(parents=True)
        (root / "run.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "evaluation_scenarios": [
                        {
                            "scenario_seed": scenario,
                            "agent_index": 0,
                            "return": seed + offset,
                        }
                        for scenario in [100, 101]
                    ],
                }
            )
        )

    monkeypatch.setattr("bluesky_pabloramesc.comparison.subprocess.run", run)
    args = Namespace(
        algorithm="ppo",
        multi_agent=False,
        seeds=[0, 1],
        eval_seeds=[100, 101],
        output_dir=tmp_path / "observations",
        timesteps=8,
        speed_min=100,
        speed_max=500,
        device="cpu",
        heading_action="relative",
        speed_action="relative",
        observation_profile="legacy",
        resume=False,
    )
    if encodings is not None:
        args.encodings = encodings
    run_comparison(args, observation=True)
    assert len(commands) == (8 if encodings is None else 4)
    for command in commands:
        assert command[command.index("--heading-action") + 1] == "relative"
        assert command[command.index("--speed-action") + 1] == "relative"
        assert command[command.index("--observation-profile") + 1] == "legacy"
        assert command[-3:] == ["--eval-seeds", "100", "101"]
    with (args.output_dir / "paired.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = [0, 0, 2, 2, 3, 3, 4, 4] if encodings is None else [0, 0, 4, 4]
    baseline = encodings[0] if encodings else "upstream"
    assert [float(row[f"difference_from_{baseline}"]) for row in rows] == expected
    args.resume = True
    commands.clear()
    run_comparison(args, observation=True)
    assert commands == []
    args.heading_action = "absolute"
    args.observation_profile = "control"
    with pytest.raises(ValueError, match="Cannot change"):
        run_comparison(args, observation=True)
    assert commands == []
