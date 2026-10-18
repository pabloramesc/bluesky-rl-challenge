"""Temporal CLI restoration and full training/checkpoint workflows."""

import json
from functools import partial
from unittest.mock import Mock

import numpy as np
import pytest
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from test_temporal_matrix import METHODS, TemporalTraffic, spatial_cases

from bluesky_pabloramesc import cli, models
from bluesky_pabloramesc.temporal import HistoryWindow
from bluesky_pabloramesc.training import (
    architecture_config_path,
    load_architecture_config,
    replay,
    train,
)


def saved_configuration(case, method):
    config = {
        "architecture": case["architecture"],
        "pooling": case["pooling"],
        "temporal": method,
        "history_length": 2,
    }
    if case["pooling"] == "additive_attention":
        config["attention_context"] = case["attention_context"]
    if case["architecture"] == "entity_transformer":
        config["transformer_layers"] = case["transformer_layers"]
    if case["architecture"] == "entity_gnn":
        config.update(
            {
                key: case[key]
                for key in ("gnn_type", "gnn_layers", "graph", "graph_radius_nm")
            }
        )
        if case["graph"] == "knn":
            config["graph_k"] = case["graph_k"]
    return config


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=name) for name, case in spatial_cases()]
)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode", ["resume", "evaluate", "replay"])
def test_all_temporal_cli_configurations_restore(
    case, method, mode, tmp_path, monkeypatch
):
    path = tmp_path / "policy"
    config = saved_configuration(case, method)
    architecture_config_path(path).write_text(json.dumps(config))
    workflow = Mock(return_value=(0.0, 0.0))
    monkeypatch.setattr(cli, "train" if mode == "resume" else "replay", workflow)
    monkeypatch.setattr("sys.argv", ["trainer", mode, "--model-path", str(path)])
    cli.run_cli("ppo")
    factory = workflow.call_args.args[1]
    assert factory.keywords["history_length"] == 2
    if case["architecture"] != "mlp":
        assert factory.keywords["entity_observations"]
    if case.get("graph") == "knn":
        assert factory.keywords["graph_k"] == case["graph_k"]
    if case.get("graph") == "distance":
        assert factory.keywords["graph_radius_nm"] == case["graph_radius_nm"]
    if mode == "resume":
        builder = workflow.call_args.args[0]
        for key, value in config.items():
            assert (
                builder.keywords.get(
                    key, {"architecture": "mlp", "pooling": "mean_max"}.get(key)
                )
                == value
            )
    else:
        assert workflow.call_args.kwargs["render_mode"] == (
            "human" if mode == "replay" else None
        )


WORKFLOW_CASES = [
    {
        "architecture": "entity_mlp",
        "pooling": "additive_attention",
        "attention_context": "all",
    },
    {"architecture": "entity_transformer", "pooling": "pma", "transformer_layers": 2},
    {
        "architecture": "entity_gnn",
        "gnn_type": "gcn",
        "gnn_layers": 3,
        "graph": "distance",
        "graph_radius_nm": 2.0,
        "graph_k": 1,
        "pooling": "additive_attention",
        "attention_context": "ownship_goal",
    },
    {
        "architecture": "entity_gnn",
        "gnn_type": "gat",
        "gnn_layers": 2,
        "graph": "knn",
        "graph_radius_nm": 2.0,
        "graph_k": 1,
        "pooling": "pma",
    },
]


def workflow_environment(case, render_mode=None):
    factory = partial(
        TemporalTraffic, graph=case.get("graph", "fully_connected"), duration=2
    )
    return HistoryWindow(DummyVecEnv([factory]), 2)


@pytest.mark.parametrize("case", WORKFLOW_CASES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("algorithm", [PPO, SAC])
def test_temporal_spatial_metadata_evaluate_resume_and_replay(
    case, method, algorithm, small_training_budgets, tmp_path
):
    builder = partial(
        models.build_ppo if algorithm == PPO else models.build_sac,
        **case,
        temporal=method,
        history_length=2,
    )
    factory = partial(workflow_environment, case)
    path = tmp_path / "policy"
    mean, std = train(
        builder,
        factory,
        path,
        4,
        episodes=1,
        device="cpu",
        results_dir=tmp_path / "reports",
    )
    assert np.isfinite([mean, std]).all()
    expected = saved_configuration(case, method)
    assert load_architecture_config(path) == expected
    assert np.isfinite(
        replay(algorithm, factory, path, episodes=1, device="cpu", render_mode=None)
    ).all()
    resumed = tmp_path / "resumed"
    train(
        None,
        factory,
        resumed,
        4,
        episodes=1,
        device="cpu",
        resume_from=path,
        model_type=algorithm,
        results_dir=tmp_path / "reports",
    )
    assert load_architecture_config(resumed) == expected
    reports = [
        json.loads(path.read_text())
        for path in (tmp_path / "reports").glob("*/run.json")
    ]
    for report in reports:
        assert report["architecture_config"] == expected
    if algorithm == SAC:
        continued = next(report for report in reports if "resumed_from" in report)
        assert continued["replay_buffer_restored"]
