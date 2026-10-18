"""Conditional architecture expansion, bounded execution and paired reporting."""

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from bluesky_pabloramesc import architecture_comparison as comparison
from bluesky_pabloramesc.architecture_reporting import (
    architecture_summary,
    write_architecture_report,
)
from bluesky_pabloramesc.architecture_tree import (
    ArchitectureVariant,
    default_tree,
    expand_tree,
)


def test_default_tree_covers_all_spatial_pooling_and_attention_choices():
    variants = expand_tree(default_tree())
    assert len(variants) == 24
    assert variants[0].settings == {"architecture": "mlp", "temporal": "none"}
    assert {v.settings["architecture"] for v in variants} == {
        "mlp",
        "entity_mlp",
        "entity_transformer",
        "entity_gnn",
    }
    for architecture in ("entity_mlp", "entity_transformer", "entity_gnn"):
        selected = [
            v.settings for v in variants if v.settings["architecture"] == architecture
        ]
        assert {v["pooling"] for v in selected} == {
            "mean",
            "max",
            "mean_max",
            "additive_attention",
            *(["pma"] if architecture != "entity_mlp" else []),
        }
        assert {
            v["attention_context"]
            for v in selected
            if v["pooling"] == "additive_attention"
        } == {"ownship_goal", "all"}
    assert {
        v.settings["gnn_type"]
        for v in variants
        if v.settings["architecture"] == "entity_gnn"
    } == {"gcn", "gat"}


def test_temporal_options_cross_every_spatial_leaf_and_history():
    variants = expand_tree(
        default_tree(("none", "frame_stack", "gru", "lstm", "tcn"), (1, 8))
    )
    assert len(variants) == 24 * 9
    assert all(
        "history_length" not in v.settings
        for v in variants
        if v.settings["temporal"] == "none"
    )
    assert {
        v.settings["history_length"]
        for v in variants
        if v.settings["temporal"] == "gru"
    } == {1, 8}


def custom_tree():
    return {
        "name": "gnn",
        "settings": {"architecture": "entity_gnn", "pooling": "mean"},
        "sweep": {"gnn_type": ["gcn", "gat"], "gnn_layers": [1, 2]},
        "children": [
            {"name": "full", "settings": {"graph": "fully_connected"}},
            {
                "name": "distance",
                "settings": {"graph": "distance"},
                "sweep": {"graph_radius_nm": [5, 25]},
            },
            {"name": "knn", "settings": {"graph": "knn"}, "sweep": {"graph_k": [1, 3]}},
        ],
    }


def test_conditional_suboptions_are_inherited_and_cartesian():
    variants = expand_tree(custom_tree())
    assert len(variants) == 20
    assert len({v.name for v in variants}) == 20
    for variant in variants:
        settings = variant.settings
        assert ("graph_radius_nm" in settings) == (settings["graph"] == "distance")
        assert ("graph_k" in settings) == (settings["graph"] == "knn")
        assert settings["pooling"] == "mean"
    assert variants[0].name == "gnn/gnn_type_gcn/gnn_layers_1/full"


@pytest.mark.parametrize(
    "node",
    [
        None,
        {},
        {"name": "../escape"},
        {"name": "x", "unknown": 1},
        {"name": "x", "settings": {"device": "cpu"}},
        {"name": "x", "settings": {"history_length": True}},
        {"name": "x", "settings": {"graph_radius_nm": float("nan")}},
        {"name": "x", "sweep": {"pooling": []}},
        {"name": "x", "sweep": {"architecture": ["mlp", "mlp"]}},
        {"name": "x", "sweep": {"architecture": ["../mlp"]}},
        {"name": "x", "children": {}},
        {"name": "x", "settings": {"pooling": "pma"}},
        {"name": "x", "settings": {"architecture": "entity_mlp", "pooling": "pma"}},
        {"name": "x", "settings": {"graph_k": 3}},
        {"name": "x", "settings": {"history_length": 8}},
        {
            "name": "x",
            "settings": {"architecture": "entity_gnn", "graph": "knn", "graph_k": 0},
        },
    ],
)
def test_invalid_tree_fails_before_execution(node):
    with pytest.raises(ValueError):
        expand_tree(node)


def test_duplicate_effective_variants_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        expand_tree(
            {
                "name": "x",
                "children": [
                    {"name": "a"},
                    {"name": "b", "settings": {"architecture": "mlp"}},
                ],
            }
        )


def test_tree_depth_is_bounded():
    node = {"name": "leaf"}
    for _ in range(33):
        node = {"name": "branch", "children": [node]}
    with pytest.raises(ValueError, match="32 levels"):
        expand_tree(node)


def report_data():
    variants = [
        ArchitectureVariant((name,), {"architecture": "mlp"})
        for name in ("baseline", "better", "partial", "failed")
    ]
    rows, runs = [], []
    for variant in variants:
        for seed in (0, 1):
            complete = variant.name != "failed" and not (
                variant.name == "partial" and seed == 1
            )
            runs.append(
                {
                    "variant": variant.name,
                    "training_seed": seed,
                    "status": "completed" if complete else "failed",
                    "training_seconds": 2.0,
                }
            )
            if complete:
                for scenario in (100, 101):
                    for agent in range(10):
                        rows.append(
                            {
                                "variant": variant.name,
                                "training_seed": seed,
                                "scenario_seed": scenario,
                                "agent_index": agent,
                                "return": seed * 2
                                + (3 if variant.name == "better" else 0),
                                "waypoint_reached": 1.0,
                            }
                        )
    return rows, runs, variants


def test_reporting_weights_training_seeds_and_excludes_partial_ranks(tmp_path):
    rows, runs, variants = report_data()
    summary, paired = architecture_summary(rows, runs, variants, [0, 1])
    assert summary[0]["mean_return"] == 1
    assert summary[0]["std_return"] == pytest.approx(np.sqrt(2))
    assert summary[1]["mean_return"] == 4
    assert summary[1]["rank"] == 1 and summary[0]["rank"] == 2
    assert summary[1]["mean_waypoint_reached"] == 1
    assert summary[2]["status"] == "partial" and "rank" not in summary[2]
    assert summary[3]["status"] == "failed" and "mean_return" not in summary[3]
    assert [p["paired_difference"] for p in paired if p["variant"] == "better"] == [
        3,
        3,
    ]
    write_architecture_report(tmp_path, rows, runs, variants, [0, 1])
    for name in (
        "report.md",
        "comparison.png",
        "episodes.csv",
        "runs.csv",
        "summary.csv",
        "paired.csv",
    ):
        assert (tmp_path / name).stat().st_size > 0
    assert "partial" in (tmp_path / "report.md").read_text()


def test_reporting_handles_total_failure(tmp_path):
    _, runs, variants = report_data()
    for run in runs:
        run["status"] = "failed"
    summary = write_architecture_report(tmp_path, [], runs, variants, [0, 1])
    assert all(row["status"] == "failed" for row in summary)
    assert (
        "No completed" not in (tmp_path / "report.md").read_text()
    )  # Each failed variant remains listed.


def test_paired_report_requires_identical_scenario_and_agent_sets():
    rows, runs, variants = report_data()
    rows = [
        row
        for row in rows
        if not (row["variant"] == "better" and row["agent_index"] == 0)
    ]
    _, paired = architecture_summary(rows, runs, variants, [0, 1])
    assert not any(row["variant"] == "better" for row in paired)


def test_tree_json_roundtrip():
    original = custom_tree()
    assert expand_tree(json.loads(json.dumps(original))) == expand_tree(original)


def comparison_args(tmp_path, **changes):
    return Namespace(
        **{
            "tree": None,
            "temporal": None,
            "history_lengths": None,
            "algorithm": "ppo",
            "multi_agent": False,
            "seeds": [0, 1],
            "eval_seeds": [100, 101],
            "timesteps": 2,
            "jobs": 2,
            "threads_per_job": 1,
            "devices": ["cpu"],
            "timeout": None,
            "output_dir": tmp_path / "comparison",
            "resume": False,
            "retry_failed": False,
            "dry_run": False,
            "heading_action": "relative",
            "speed_action": "absolute",
            "observation_encoding": "goal_aligned_conflict",
            "observation_profile": "control",
            "decision_interval": 10,
            "speed_min": 100,
            "speed_max": 500,
            **changes,
        }
    )


@pytest.fixture
def subprocess_experiment(tmp_path, monkeypatch):
    """Real child processes emit production-shaped reports without simulator startup."""
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(
        json.dumps(
            {
                "name": "test",
                "children": [
                    {"name": "flat", "settings": {"architecture": "mlp"}},
                    {
                        "name": "entity",
                        "settings": {"architecture": "entity_mlp", "pooling": "mean"},
                    },
                ],
            }
        )
    )
    args = comparison_args(tmp_path, tree=tree_path)
    _, _, protocol = comparison.comparison_plan(args)
    trace = tmp_path / "trace.jsonl"
    worker = tmp_path / "worker.py"
    worker.write_text("""
import json
import os
import sys
import time
from pathlib import Path

payload = json.loads(sys.argv[1])
def event(kind):
    record = {"event": kind, "pid": os.getpid(), "parent": os.getppid(),
              "group": os.getpgrp(), "time": time.monotonic(),
              "threads": os.environ.get("OMP_NUM_THREADS"),
              "device": payload["device"], "seed": payload["seed"],
              "variant": payload["variant"]}
    with open(payload["trace"], "a") as stream:
        stream.write(json.dumps(record) + "\\n")
event("start")
try:
    time.sleep(payload["duration"])
    if payload["fail"]:
        print("synthetic training failure", flush=True)
        sys.exit(7)
    directory = Path(payload["directory"]) / "reports" / "run"
    directory.mkdir(parents=True)
    report = payload["report"]
    report.update(payload["mutation"])
    (directory / "run.json").write_text(json.dumps(report))
    print("synthetic training completed", flush=True)
finally:
    event("end")
""")
    options = {"fail": None, "duration": 0.15, "durations": {}, "mutation": {}}

    def command(current, variant, seed, directory, device):
        payload = {
            "directory": str(directory),
            "trace": str(trace),
            "seed": seed,
            "device": device,
            "variant": variant.name,
            "duration": options["durations"].get(
                (variant.name, seed), options["duration"]
            ),
            "fail": options["fail"] == (variant.name, seed),
            "mutation": options["mutation"],
            "report": {
                "status": "completed",
                "seed": seed,
                "requested_timesteps": current.timesteps,
                "actual_timesteps": current.timesteps,
                "algorithm": current.algorithm.upper(),
                "control_config": protocol["control_config"],
                "architecture_config": variant.settings,
                "training_seconds": 0.1,
                "steps_per_second": 20.0,
                "evaluation_scenarios": [
                    {
                        "scenario_seed": scenario,
                        "agent_index": agent,
                        "return": float(seed + 1),
                        "length": 3,
                    }
                    for scenario in current.eval_seeds
                    for agent in range(10 if current.multi_agent else 1)
                ],
            },
        }
        return [sys.executable, str(worker), json.dumps(payload)]

    monkeypatch.setattr(comparison, "architecture_command", command)
    return args, options, trace


def trace_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_real_subprocesses_respect_global_bound_and_device_slots(subprocess_experiment):
    args, _, trace = subprocess_experiment
    args.devices = ["cpu", "cuda:7"]
    args.threads_per_job = 2
    assert comparison.run_architecture_comparison(args) == 0
    events = trace_events(trace)
    active, peak = 0, 0
    for event in sorted(events, key=lambda row: row["time"]):
        active += 1 if event["event"] == "start" else -1
        peak = max(peak, active)
        assert 0 <= active <= 2
    assert active == 0 and peak == 2
    started = [row for row in events if row["event"] == "start"]
    assert len({row["pid"] for row in started}) == 4
    assert all(
        row["parent"] == os.getpid()
        and row["group"] == row["pid"]
        and row["threads"] == "2"
        for row in started
    )
    assert {row["device"] for row in started} == {"cpu", "cuda:7"}
    records = json.loads((args.output_dir / "runs.json").read_text())
    assert all(
        row["status"] == "completed" and row["actual_timesteps"] == 2 for row in records
    )
    assert all(
        Path(row["log_path"]).read_text().strip() == "synthetic training completed"
        for row in records
    )
    assert (args.output_dir / "report.md").is_file()


def test_resume_reuses_all_completed_jobs_without_launching(subprocess_experiment):
    args, _, trace = subprocess_experiment
    assert comparison.run_architecture_comparison(args) == 0
    before = trace.read_text()
    manifest = (args.output_dir / "comparison.json").read_bytes()
    args.resume, args.tree, args.jobs = True, None, 1
    assert comparison.run_architecture_comparison(args) == 0
    assert trace.read_text() == before
    assert (args.output_dir / "comparison.json").read_bytes() == manifest
    assert all(
        row["reused"] for row in json.loads((args.output_dir / "runs.json").read_text())
    )


def test_free_slot_refills_while_another_branch_is_still_running(subprocess_experiment):
    args, options, trace = subprocess_experiment
    options["duration"] = 0.05
    options["durations"] = {("test/flat", 0): 0.5}
    assert comparison.run_architecture_comparison(args) == 0
    events = trace_events(trace)
    slow_end = next(
        row["time"]
        for row in events
        if row["variant"] == "test/flat" and row["seed"] == 0 and row["event"] == "end"
    )
    other_start = next(
        row["time"]
        for row in events
        if row["variant"] == "test/entity" and row["event"] == "start"
    )
    assert other_start < slow_end


def test_failed_jobs_continue_and_explicit_retry_preserves_attempts(
    subprocess_experiment,
):
    args, options, trace = subprocess_experiment
    options["fail"] = ("test/flat", 1)
    assert comparison.run_architecture_comparison(args) == 1
    records = json.loads((args.output_dir / "runs.json").read_text())
    failed = next(row for row in records if row["status"] == "failed")
    original = Path(failed["directory"])
    assert failed["returncode"] == 7
    assert sum(row["status"] == "completed" for row in records) == 3
    args.resume = True
    before = trace.read_text()
    assert comparison.run_architecture_comparison(args) == 1
    assert trace.read_text() == before
    options["fail"], args.retry_failed = None, True
    assert comparison.run_architecture_comparison(args) == 0
    current = json.loads((args.output_dir / "runs.json").read_text())
    retried = next(
        row
        for row in current
        if row["variant"] == "test/flat" and row["training_seed"] == 1
    )
    assert retried["attempt"] == 2 and Path(retried["directory"]) != original
    assert "synthetic training failure" in (original / "process.log").read_text()
    assert len([row for row in trace_events(trace) if row["event"] == "start"]) == 5


def test_invalid_report_can_be_retried_in_a_fresh_attempt(subprocess_experiment):
    args, options, _ = subprocess_experiment
    options["mutation"] = {"evaluation_scenarios": []}
    assert comparison.run_architecture_comparison(args) == 1
    options["mutation"], args.resume, args.retry_failed = {}, True, True
    assert comparison.run_architecture_comparison(args) == 0
    records = json.loads((args.output_dir / "runs.json").read_text())
    assert all(row["attempt"] == 2 for row in records)


def test_corrupt_completed_cache_is_rejected_before_any_launch(subprocess_experiment):
    args, _, trace = subprocess_experiment
    assert comparison.run_architecture_comparison(args) == 0
    records = json.loads((args.output_dir / "runs.json").read_text())
    Path(records[-1]["report_path"]).write_text("{malformed")
    before = trace.read_text()
    args.resume = True
    with pytest.raises(ValueError):
        comparison.run_architecture_comparison(args)
    assert trace.read_text() == before


def test_resume_refuses_to_duplicate_a_live_training_process(subprocess_experiment):
    args, _, trace = subprocess_experiment
    assert comparison.run_architecture_comparison(args) == 0
    state = args.output_dir / "runs.json"
    records = json.loads(state.read_text())
    records[0].update(status="running", pid=os.getpid())
    state.write_text(json.dumps(records))
    args.resume = True
    before = trace.read_text()
    with pytest.raises(ValueError, match="still running"):
        comparison.run_architecture_comparison(args)
    assert trace.read_text() == before


def test_timeouts_stop_and_reap_children(subprocess_experiment):
    args, options, _ = subprocess_experiment
    options["duration"], args.timeout = 1, 0.1
    assert comparison.run_architecture_comparison(args) == 1
    records = json.loads((args.output_dir / "runs.json").read_text())
    assert all(
        "timeout" in row["error"] and row["status"] == "failed" for row in records
    )
    assert all(not comparison._process_alive(row["pid"]) for row in records)


def test_interrupt_cleans_active_children_and_saves_partial_report(
    subprocess_experiment, monkeypatch
):
    args, options, _ = subprocess_experiment
    options["duration"] = 5

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(comparison, "sleep", interrupt)
    assert comparison.run_architecture_comparison(args) == 130
    records = json.loads((args.output_dir / "runs.json").read_text())
    assert sum(row["status"] == "interrupted" for row in records) == 2
    assert sum(row["status"] == "queued" for row in records) == 2
    assert all(not comparison._process_alive(row.get("pid")) for row in records)
    assert (args.output_dir / "report.md").is_file()


def test_missing_executable_is_recorded_and_reported(
    subprocess_experiment, monkeypatch
):
    args, _, _ = subprocess_experiment

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("missing worker")

    monkeypatch.setattr(comparison.subprocess, "Popen", missing)
    assert comparison.run_architecture_comparison(args) == 1
    records = json.loads((args.output_dir / "runs.json").read_text())
    assert all(row["error"] == "missing worker" for row in records)


@pytest.mark.parametrize(
    "mutation",
    [
        {"status": "interrupted"},
        {"seed": 999},
        {"requested_timesteps": 999},
        {"algorithm": "SAC"},
        {"control_config": {}},
        {"architecture_config": {}},
        {"architecture_config": []},
        {"actual_timesteps": 1},
        {"training_seconds": -1},
        {"evaluation_scenarios": []},
        {
            "evaluation_scenarios": [
                {"scenario_seed": 100, "agent_index": 0, "return": float("nan")}
            ]
        },
        {"evaluation_scenarios": [{"scenario_seed": 100}]},
        {"training_seconds": float("inf")},
    ],
)
def test_invalid_reports_are_failed_and_do_not_enter_summary(
    subprocess_experiment, mutation
):
    args, options, _ = subprocess_experiment
    options["mutation"] = mutation
    assert comparison.run_architecture_comparison(args) == 1
    records = json.loads((args.output_dir / "runs.json").read_text())
    assert all(row["status"] == "failed" for row in records)


@pytest.mark.parametrize(
    "changes",
    [
        {"jobs": 0},
        {"threads_per_job": True},
        {"timesteps": -1},
        {"seeds": []},
        {"seeds": [0, 0]},
        {"seeds": [-1]},
        {"eval_seeds": [1, 1]},
        {"devices": []},
        {"devices": [""]},
        {"jobs": 1, "devices": ["cpu", "cuda:0"]},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"retry_failed": True},
        {"history_lengths": [1, 8]},
        {"multi_agent": True, "algorithm": "ddpg"},
        {"multi_agent": True, "decision_interval": 5},
    ],
)
def test_invalid_protocol_creates_no_outputs(tmp_path, changes):
    args = comparison_args(tmp_path, **changes)
    with pytest.raises(ValueError):
        comparison.run_architecture_comparison(args)
    assert not args.output_dir.exists()


def test_resume_rejects_protocol_changes_before_launching(subprocess_experiment):
    args, _, trace = subprocess_experiment
    assert comparison.run_architecture_comparison(args) == 0
    before = trace.read_text()
    args.resume, args.timesteps = True, 100
    with pytest.raises(ValueError, match="Cannot change"):
        comparison.run_architecture_comparison(args)
    assert trace.read_text() == before


def test_dry_run_has_no_files_or_processes(tmp_path, monkeypatch, capsys):
    args = comparison_args(tmp_path, dry_run=True)
    monkeypatch.setattr(
        comparison.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("dry run started a process"),
    )
    assert comparison.run_architecture_comparison(args) == 0
    assert not args.output_dir.exists()
    assert "24 variants" in capsys.readouterr().out


def test_comparison_lock_prevents_concurrent_coordinators(tmp_path):
    with (
        comparison._comparison_lock(tmp_path),
        pytest.raises(ValueError, match="Another comparison"),
        comparison._comparison_lock(tmp_path),
    ):
        pytest.fail("acquired the lock twice")


@pytest.mark.parametrize(
    "algorithm,multi_agent,module",
    [
        ("ppo", False, "train_single_ppo"),
        ("sac", False, "train_single_sac"),
        ("td3", False, "train_single_td3"),
        ("ddpg", False, "train_single_ddpg"),
        ("ppo", True, "train_zoo_ppo"),
        ("sac", True, "train_zoo_sac"),
    ],
)
def test_production_commands_use_existing_builders_and_complete_settings(
    tmp_path, algorithm, multi_agent, module
):
    from bluesky_pabloramesc.cli import parse_args

    args = comparison_args(tmp_path, algorithm=algorithm, multi_agent=multi_agent)
    for variant in expand_tree(
        default_tree(("none", "gru"), (3,)), algorithm, multi_agent
    ):
        command = comparison.architecture_command(args, variant, 7, tmp_path, "cpu")
        assert command[:3] == [sys.executable, "-m", f"experiments.{module}"]
        parsed = parse_args(algorithm, multi_agent, command[3:])
        assert parsed.seed == 7 and parsed.eval_seeds == [100, 101]
        assert parsed.model_path == tmp_path / "policy"
        assert parsed.observation_encoding == "goal_aligned_conflict"
        assert all(
            getattr(parsed, key) == value for key, value in variant.settings.items()
        )


def test_architecture_cli_defaults_and_optional_temporal_tree(tmp_path, monkeypatch):
    from bluesky_pabloramesc.cli import run_architecture_comparison_cli

    captured = []
    monkeypatch.setattr(
        comparison,
        "run_architecture_comparison",
        lambda args: captured.append(args) or 0,
    )
    run_architecture_comparison_cli(["--output-dir", str(tmp_path / "out")])
    assert captured[0].jobs == 1 and captured[0].devices == ["cpu"]
    assert len(comparison.comparison_plan(captured[0])[1]) == 24
    run_architecture_comparison_cli(
        ["--temporal", "none", "gru", "--history-lengths", "1", "8", "--jobs", "2"]
    )
    assert len(comparison.comparison_plan(captured[1])[1]) == 72


def test_actual_training_subprocesses_aggregate_and_resume(tmp_path, monkeypatch):
    import csv

    tree_path = tmp_path / "tree.json"
    tree_path.write_text(
        json.dumps(
            {
                "name": "integration",
                "children": [
                    {"name": "flat", "settings": {"architecture": "mlp"}},
                    {
                        "name": "attention_gru",
                        "settings": {
                            "architecture": "entity_mlp",
                            "pooling": "additive_attention",
                            "attention_context": "all",
                            "temporal": "gru",
                            "history_length": 3,
                        },
                    },
                    {
                        "name": "transformer",
                        "settings": {
                            "architecture": "entity_transformer",
                            "pooling": "pma",
                        },
                    },
                    {
                        "name": "gat_knn_tcn",
                        "settings": {
                            "architecture": "entity_gnn",
                            "gnn_type": "gat",
                            "graph": "knn",
                            "graph_k": 1,
                            "pooling": "mean",
                            "temporal": "tcn",
                            "history_length": 3,
                        },
                    },
                ],
            }
        )
    )
    args = comparison_args(tmp_path, tree=tree_path, seeds=[0])
    original_command = comparison.architecture_command

    def synthetic_command(*command_args):
        command = original_command(*command_args)
        return [
            command[0],
            str(Path(__file__).with_name("architecture_training_worker.py")),
            *command[2:],
        ]

    monkeypatch.setattr(comparison, "architecture_command", synthetic_command)
    assert comparison.run_architecture_comparison(args) == 0
    records = json.loads((args.output_dir / "runs.json").read_text())
    assert len(records) == 4
    for record in records:
        root = Path(record["directory"])
        assert (root / "policy.zip").is_file()
        assert (root / "policy_vecnormalize.pkl").is_file()
        assert (root / "policy_architecture.json").is_file()
        report = json.loads(Path(record["report_path"]).read_text())
        assert report["status"] == "completed" and report["actual_timesteps"] == 2
    with (args.output_dir / "episodes.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 8
    args.resume, args.tree = True, None
    assert comparison.run_architecture_comparison(args) == 0
    assert all(
        row["reused"] for row in json.loads((args.output_dir / "runs.json").read_text())
    )
