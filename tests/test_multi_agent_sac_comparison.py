"""Matched SAC scheduling, strict report reuse, and safety/completion summaries."""

import json
from copy import deepcopy

import pytest
from test_entities import TrafficEnv

from bluesky_pabloramesc.cli import parse_sac_comparison_args
from bluesky_pabloramesc.multi_agent import N_AGENTS
from bluesky_pabloramesc.multi_agent_sac import CentralizedAttentionQ, LocalAttentionQ
from bluesky_pabloramesc.multi_agent_sac_comparison import (
    ALGORITHM_CLASSES,
    run_sac_comparison,
    run_sac_comparison_worker,
    sac_comparison_command,
    sac_comparison_evaluation,
    sac_comparison_plan,
    write_sac_comparison_report,
)


def arguments(tmp_path, *options):
    return parse_sac_comparison_args(
        [
            "--output-dir",
            str(tmp_path / "study"),
            "--joint-steps",
            "3",
            "--train-freq",
            "2",
            "--gradient-steps",
            "1",
            "--learning-starts",
            "0",
            "--buffer-size",
            "8",
            "--batch-size",
            "2",
            "--hidden-sizes",
            "8",
            "--eval-seeds",
            "2026",
            *options,
        ]
    )


def completed_report(tmp_path, variant, protocol):
    report_path = tmp_path / variant.name / "reports/run/run.json"
    report_path.parent.mkdir(parents=True)
    policy = deepcopy(protocol["settings"]["policy_kwargs"])
    policy["critic_class"] = str(
        CentralizedAttentionQ if variant.name == "masac" else LocalAttentionQ
    )
    for role in ("actor_kwargs", "critic_kwargs"):
        policy[role].update(
            layout=TrafficEnv().observation_space.entity_layout, attention_context="all"
        )
    rows = [
        {
            "scenario_seed": 2026,
            "agent_index": agent,
            "return": float(agent),
            "length": 2,
            "waypoint_reached": 1,
            "intrusion_time": 0,
            "time_in_restricted_area": 0,
            "time_outside_sector": 0,
            "flight_time": 20,
            "total_reward": float(agent),
            "intrusion_events": 0,
            "restricted_area_events": 0,
            "sector_exit_events": 0,
            "timed_out": 0,
        }
        for agent in range(N_AGENTS)
    ]
    report = {
        "status": "completed",
        "seed": 42,
        "requested_timesteps": protocol["timesteps"],
        "algorithm": ALGORITHM_CLASSES[variant.name],
        "control_config": protocol["control_config"],
        "reward_config": protocol["reward_config"],
        "architecture_config": variant.settings,
        "evaluation_scenarios": rows,
        "training_seconds": 1,
        "actual_timesteps": 4 * N_AGENTS,
        "steps_per_second": 40,
        "joint_steps": 4,
        "active_timesteps": 30,
        "parameter_count": 100,
        "gradient_updates": 2,
        "checkpoint_interval": protocol["checkpoint_interval"],
        "settings": {
            k: v for k, v in protocol["settings"].items() if k != "policy_kwargs"
        },
        "policy_kwargs": policy,
    }
    report_path.write_text(json.dumps(report))
    record = {
        "directory": str(report_path.parents[2]),
        "training_seed": 42,
        "variant": variant.name,
        "status": "completed",
    }
    return record, report_path, report


def test_commands_match_protocol_and_dry_run_is_read_only(tmp_path, capsys):
    args = arguments(tmp_path, "--seeds", "42", "43", "--dry-run")
    variants, protocol = sac_comparison_plan(args)
    assert [v.name for v in variants] == ["shared_sac", "isac", "masac"]
    assert protocol["settings"]["learning_starts"] == 0
    for variant in variants:
        command = sac_comparison_command(
            args, variant, 43, tmp_path / variant.name, "cpu"
        )
        worker = parse_sac_comparison_args(command[3:])
        assert worker.worker == variant.name
        assert worker.seeds == [43]
        assert worker.eval_seeds == args.eval_seeds
        assert sac_comparison_plan(worker)[1]["settings"] == protocol["settings"]
    assert run_sac_comparison(args) == 0
    assert len(json.loads(capsys.readouterr().out)["commands"]) == 6
    assert not args.output_dir.exists()


@pytest.mark.parametrize("method", ["shared_sac", "isac", "masac"])
def test_completed_reports_require_matched_algorithms_budgets_and_settings(
    tmp_path, method
):
    variants, protocol = sac_comparison_plan(arguments(tmp_path))
    variant = next(v for v in variants if v.name == method)
    record, path, report = completed_report(tmp_path, variant, protocol)
    assert len(sac_comparison_evaluation(record, variant, protocol)) == N_AGENTS
    for name, value in [
        ("algorithm", "SAC"),
        ("joint_steps", 5),
        ("actual_timesteps", 50),
        ("gradient_updates", None),
        ("gradient_updates", 99),
        ("parameter_count", None),
        ("reward_config", {}),
        ("settings", {**report["settings"], "gradient_steps": 99}),
        ("evaluation_scenarios", report["evaluation_scenarios"][:-1]),
        ("policy_kwargs", {**report["policy_kwargs"], "critic_class": "wrong"}),
    ]:
        invalid = deepcopy(report)
        invalid[name] = value
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            sac_comparison_evaluation(record, variant, protocol)


def test_reports_preserve_pairing_and_table_metrics(tmp_path):
    args = arguments(tmp_path)
    variants, protocol = sac_comparison_plan(args)
    records, episodes = [], []
    for variant in variants:
        record, _, _ = completed_report(tmp_path, variant, protocol)
        episodes.extend(sac_comparison_evaluation(record, variant, protocol))
        records.append(record)
    write_sac_comparison_report(tmp_path, episodes, records, variants, [42])
    body = (tmp_path / "report.md").read_text()
    assert "Multi-agent SAC comparison" in body
    assert "Valid transitions" in body
    assert "focal action" in body
    assert "shared_sac" in (tmp_path / "paired.csv").read_text()
    assert "gradient_updates" in (tmp_path / "runs.csv").read_text()


@pytest.mark.parametrize(
    "options",
    [
        ["--seeds", "42", "42"],
        ["--learning-starts", "-1"],
        ["--initial-alpha", "0"],
        ["--tau", "2"],
        ["--learning-rate", "nan"],
        ["--retry-failed"],
    ],
)
def test_invalid_protocol_fails_before_launch(tmp_path, options):
    with pytest.raises(ValueError):
        sac_comparison_plan(arguments(tmp_path, *options))


def test_resume_rejects_changed_settings(tmp_path):
    args = arguments(tmp_path, "--resume")
    _, protocol = sac_comparison_plan(args)
    args.output_dir.mkdir()
    (args.output_dir / "comparison.json").write_text(json.dumps({"protocol": protocol}))
    args.gradient_steps = 99
    with pytest.raises(ValueError, match="Cannot change"):
        run_sac_comparison(args)


def test_actual_worker_uses_training_workflow_and_validates_its_report(
    tmp_path, monkeypatch
):
    """Train with real normalization/checkpoint/report code on synthetic traffic."""
    from test_multi_agent_attention import EntityJointTraffic, make_attention_env

    from bluesky_pabloramesc import multi_agent_sac_comparison as comparison

    original_step = EntityJointTraffic.step

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = original_step(
            self, actions
        )
        for info in infos.values():
            info.update(
                flight_time=20.0,
                waypoint_reached=1.0,
                total_reward=1.0,
                intrusion_events=0.0,
                intrusion_time=0.0,
                restricted_area_events=0.0,
                time_in_restricted_area=0.0,
                sector_exit_events=0.0,
                time_outside_sector=0.0,
            )
        return observations, rewards, terminated, truncated, infos

    monkeypatch.setattr(EntityJointTraffic, "step", step)
    monkeypatch.setattr(
        comparison, "make_mappo_env", lambda **kwargs: make_attention_env()
    )
    monkeypatch.setattr(comparison, "N_AGENTS", 2)
    monkeypatch.setattr("bluesky_pabloramesc.multi_agent.N_AGENTS", 2)
    for method in ALGORITHM_CLASSES:
        args = arguments(
            tmp_path, "--worker", method, "--output-dir", str(tmp_path / method)
        )
        variants, protocol = sac_comparison_plan(args)
        run_sac_comparison_worker(args)
        variant = next(v for v in variants if v.name == method)
        record = {"directory": str(args.output_dir), "training_seed": 42}
        assert len(sac_comparison_evaluation(record, variant, protocol)) == 2
