"""Four-way geometry comparisons preserve paired protocols and resume settings."""

import csv
import json
from pathlib import Path

import pytest

from bluesky_pabloramesc.cli import run_comparison_cli


def test_geometry_comparison_pairs_variants_and_reuses_reports(tmp_path, monkeypatch):
    commands = []

    def run(command, check):
        commands.append(command)
        directory = Path(command[command.index("--results-dir") + 1]) / "run"
        directory.mkdir(parents=True)
        representation = command[command.index("--geometry-representation") + 1]
        report = {
            "status": "completed",
            "evaluation_scenarios": [
                {
                    "scenario_seed": seed,
                    "agent_index": 0,
                    "return": 1.0 if representation == "current" else 3.0,
                    "restricted_area_events": 0,
                    "sector_exit_events": 0,
                }
                for seed in [100, 101]
            ],
        }
        (directory / "run.json").write_text(json.dumps(report))

    monkeypatch.setattr("bluesky_pabloramesc.comparison.subprocess.run", run)
    directory = tmp_path / "geometry"
    argv = [
        "--output-dir",
        str(directory),
        "--seeds",
        "0",
        "1",
        "--eval-seeds",
        "100",
        "101",
        "--timesteps",
        "8",
        "--geometry-ray-count",
        "8",
        "--geometry-ray-range-m",
        "50000",
        "--observation-encoding",
        "goal_aligned_conflict",
    ]
    run_comparison_cli(argv, geometry=True)
    assert len(commands) == 8
    assert {c[c.index("--geometry-representation") + 1] for c in commands} == {
        "current",
        "nearest_edge",
        "ray_cast",
        "both",
    }
    for command in commands:
        assert command[command.index("--geometry-ray-count") + 1] == "8"
        assert command[command.index("--geometry-ray-range-m") + 1] == "50000.0"
        assert (
            command[command.index("--observation-encoding") + 1]
            == "goal_aligned_conflict"
        )
        assert command[-3:] == ["--eval-seeds", "100", "101"]
    with (directory / "paired.csv").open() as stream:
        paired = list(csv.DictReader(stream))
    assert [float(row["difference_from_current"]) for row in paired] == [
        0,
        0,
        2,
        2,
        2,
        2,
        2,
        2,
    ]
    assert (directory / "comparison.png").is_file()
    run_comparison_cli([*argv, "--resume"], geometry=True)
    assert len(commands) == 8
    with pytest.raises(SystemExit):
        run_comparison_cli(
            [*argv, "--resume", "--geometry-ray-count", "16"], geometry=True
        )
    assert len(commands) == 8


def test_geometry_comparison_rejects_invalid_range_before_creating_output(tmp_path):
    directory = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        run_comparison_cli(
            ["--output-dir", str(directory), "--geometry-ray-range-m", "nan"],
            geometry=True,
        )
    assert not directory.exists()
