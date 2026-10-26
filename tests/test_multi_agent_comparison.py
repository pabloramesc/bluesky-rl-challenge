"""Matched experiment commands, strict result reuse, and paired reporting."""

import json
from copy import deepcopy

import pytest

from bluesky_pabloramesc.cli import parse_multi_agent_comparison_args
from bluesky_pabloramesc.multi_agent import N_AGENTS
from bluesky_pabloramesc.multi_agent_comparison import (
    ALGORITHM_CLASSES,
    multi_agent_command,
    multi_agent_evaluation,
    multi_agent_plan,
    run_multi_agent_comparison,
    write_multi_agent_report,
)


def arguments(tmp_path, *options):
    return parse_multi_agent_comparison_args(
        [
            "--output-dir",
            str(tmp_path / "study"),
            "--joint-steps",
            "3",
            "--rollout-steps",
            "2",
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--hidden-sizes",
            "8",
            "--eval-seeds",
            "2026",
            *options,
        ]
    )


def completed_report(tmp_path, variant, protocol):
    directory = tmp_path / variant.name
    report_path = directory / "reports" / "run" / "run.json"
    report_path.parent.mkdir(parents=True)
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
        "architecture_config": variant.settings.copy(),
        "evaluation_scenarios": rows,
        "training_seconds": 1,
        "actual_timesteps": 4 * N_AGENTS,
        "steps_per_second": 4 * N_AGENTS,
        "joint_steps": 4,
        "active_timesteps": 3 * N_AGENTS,
        "parameter_count": 100,
        "settings": {
            key: value
            for key, value in protocol["settings"].items()
            if key != "policy_kwargs"
        },
        "policy_kwargs": deepcopy(protocol["settings"]["policy_kwargs"]),
    }
    if variant.settings["architecture"] == "entity_mlp":
        from test_entities import TrafficEnv

        from bluesky_pabloramesc.multi_agent_attention import (
            AdditiveAttentionActor,
            CentralizedAdditiveAttentionCritic,
            LocalAdditiveAttentionCritic,
        )

        report["policy_kwargs"].update(
            actor_class=str(AdditiveAttentionActor),
            critic_class=str(
                CentralizedAdditiveAttentionCritic
                if variant.name == "mappo"
                else LocalAdditiveAttentionCritic
            ),
        )
        for role in ("actor_kwargs", "critic_kwargs"):
            report["policy_kwargs"][role].update(
                layout=TrafficEnv().observation_space.entity_layout,
                attention_context=variant.settings["attention_context"],
            )
    report_path.write_text(json.dumps(report))
    record = {
        "directory": str(directory),
        "training_seed": 42,
        "variant": variant.name,
        "status": "completed",
    }
    return record, report_path, report


@pytest.mark.parametrize("architecture", ["mlp", "entity_mlp"])
def test_all_commands_match_protocol_and_use_separate_workers(tmp_path, architecture):
    args = arguments(tmp_path, "--seeds", "42", "43", "--architecture", architecture)
    variants, protocol = multi_agent_plan(args)
    assert [variant.name for variant in variants] == ["shared_ppo", "ippo", "mappo"]
    assert protocol["timesteps"] == 3 * N_AGENTS
    for variant in variants:
        command = multi_agent_command(args, variant, 43, tmp_path / variant.name, "cpu")
        worker = parse_multi_agent_comparison_args(command[3:])
        assert command[2] == "experiments.compare_multi_agent"
        assert worker.worker == variant.name
        assert worker.seeds == [43]
        assert worker.eval_seeds == args.eval_seeds
        assert worker.joint_steps == args.joint_steps
        assert worker.rollout_steps == args.rollout_steps
        assert worker.hidden_sizes == [8]
        assert worker.architecture == architecture
        assert worker.attention_context == args.attention_context
        assert variant.settings["architecture"] == architecture
        assert worker.output_dir == tmp_path / variant.name


@pytest.mark.parametrize(
    "option",
    [
        ["--seeds", "1", "1"],
        ["--eval-seeds", "-1"],
        ["--timeout", "nan"],
        ["--retry-failed"],
        ["--jobs", "1", "--devices", "cpu", "cuda:0"],
    ],
)
def test_invalid_protocol_rejected(tmp_path, option):
    with pytest.raises(ValueError):
        multi_agent_plan(arguments(tmp_path, *option))


def test_dry_run_is_read_only(tmp_path, capsys):
    args = arguments(tmp_path, "--dry-run")
    assert run_multi_agent_comparison(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["commands"]) == 3
    assert not args.output_dir.exists()


@pytest.mark.parametrize("method", ["shared_ppo", "ippo", "mappo"])
@pytest.mark.parametrize("architecture", ["mlp", "entity_mlp"])
def test_completed_evaluations_require_matched_settings_and_budgets(
    tmp_path, method, architecture
):
    variants, protocol = multi_agent_plan(
        arguments(tmp_path, "--architecture", architecture)
    )
    variant = next(v for v in variants if v.name == method)
    record, path, report = completed_report(tmp_path, variant, protocol)
    rows = multi_agent_evaluation(record, variant, protocol)
    assert len(rows) == N_AGENTS
    assert record["joint_steps"] == 4
    for name, value in [
        ("algorithm", "PPO"),
        ("joint_steps", 5),
        ("actual_timesteps", 5 * N_AGENTS),
        ("parameter_count", None),
        ("reward_config", {}),
        ("evaluation_scenarios", report["evaluation_scenarios"][:-1]),
        ("settings", {**report["settings"], "gae_lambda": 0.5}),
        (
            "policy_kwargs",
            {**report["policy_kwargs"], "actor_kwargs": {"hidden_sizes": [16]}},
        ),
    ]:
        invalid = deepcopy(report)
        invalid[name] = value
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            multi_agent_evaluation(record, variant, protocol)


@pytest.mark.parametrize("architecture", ["mlp", "entity_mlp"])
def test_reports_pair_against_shared_policy_and_preserve_costs(tmp_path, architecture):
    args = arguments(tmp_path, "--architecture", architecture)
    variants, protocol = multi_agent_plan(args)
    records, episodes = [], []
    for variant in variants:
        record, _, _ = completed_report(tmp_path, variant, protocol)
        episodes.extend(multi_agent_evaluation(record, variant, protocol))
        records.append(record)
    write_multi_agent_report(tmp_path, episodes, records, variants, [42])
    report = (tmp_path / "report.md").read_text()
    assert "Multi-agent PPO comparison" in report
    assert "Valid transitions" in report
    assert "IPPO has separate networks" in report
    if architecture == "entity_mlp":
        assert "masked additive attention" in report
    assert "shared_ppo" in (tmp_path / "paired.csv").read_text()
    assert "parameter_count" in (tmp_path / "runs.csv").read_text()
    assert (tmp_path / "comparison.png").is_file()


def test_resume_rejects_changed_protocol_before_launch(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--resume")
    _, protocol = multi_agent_plan(args)
    args.output_dir.mkdir()
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"protocol": {**protocol, "joint_steps": 99}})
    )

    def unexpected(*args, **kwargs):
        pytest.fail("must reject before launching")

    monkeypatch.setattr(
        "bluesky_pabloramesc.multi_agent_comparison._schedule", unexpected
    )
    with pytest.raises(ValueError, match="Cannot change"):
        run_multi_agent_comparison(args)


@pytest.mark.parametrize(
    "field", ["actor_class", "critic_class", "actor_kwargs", "critic_kwargs"]
)
def test_attention_report_rejects_wrong_networks_or_context(tmp_path, field):
    variants, protocol = multi_agent_plan(
        arguments(tmp_path, "--architecture", "entity_mlp")
    )
    variant = variants[0]
    record, path, report = completed_report(tmp_path, variant, protocol)
    if field.endswith("class"):
        report["policy_kwargs"][field] = "MLP"
    else:
        report["policy_kwargs"][field]["attention_context"] = "ownship_goal"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        multi_agent_evaluation(record, variant, protocol)


def test_attention_resume_rejects_mlp_protocol(tmp_path):
    args = arguments(tmp_path)
    _, protocol = multi_agent_plan(args)
    args.output_dir.mkdir()
    (args.output_dir / "comparison.json").write_text(json.dumps({"protocol": protocol}))
    args = arguments(tmp_path, "--resume", "--architecture", "entity_mlp")
    with pytest.raises(ValueError, match="Cannot change"):
        run_multi_agent_comparison(args)
