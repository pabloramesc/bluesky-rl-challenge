"""Expand conditional architecture trees into reproducible training variants."""

import json
import math
import re
from dataclasses import dataclass
from itertools import product

ARCHITECTURE_KEYS = (
    "architecture",
    "pooling",
    "attention_context",
    "transformer_layers",
    "gnn_type",
    "gnn_layers",
    "graph",
    "graph_radius_nm",
    "graph_k",
    "temporal",
    "history_length",
)


@dataclass(frozen=True)
class ArchitectureVariant:
    """A named leaf and its inherited, validated architecture settings."""

    path: tuple[str, ...]
    settings: dict

    @property
    def name(self):
        return "/".join(self.path)


def setting_arguments(settings):
    """Convert known architecture settings into shell-free CLI arguments."""
    return [
        argument
        for key, value in settings.items()
        for argument in (f"--{key.replace('_', '-')}", str(value))
    ]


def effective_settings(parsed):
    """Keep only settings that affect the selected model, including defaults."""
    settings = {"architecture": parsed.architecture, "temporal": parsed.temporal}
    if parsed.temporal != "none":
        settings["history_length"] = parsed.history_length
    if parsed.architecture != "mlp":
        settings["pooling"] = parsed.pooling
        if parsed.pooling == "additive_attention":
            settings["attention_context"] = parsed.attention_context
    if parsed.architecture == "entity_transformer":
        settings["transformer_layers"] = parsed.transformer_layers
    if parsed.architecture == "entity_gnn":
        for key in ("gnn_type", "gnn_layers", "graph"):
            settings[key] = getattr(parsed, key)
        if parsed.graph == "distance":
            settings["graph_radius_nm"] = parsed.graph_radius_nm
        if parsed.graph == "knn":
            settings["graph_k"] = parsed.graph_k
    return settings


def _settings(values):
    if not isinstance(values, dict) or set(values) - set(ARCHITECTURE_KEYS):
        raise ValueError(f"Settings must use architecture keys: {ARCHITECTURE_KEYS}")
    for value in values.values():
        if type(value) not in (str, int, float) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            raise ValueError("Architecture settings must be finite scalar values")
    return values


def expand_tree(tree, algorithm="ppo", multi_agent=False):
    """Inherit node settings and expand Cartesian sweeps, then visit child choices.

    Nodes have a name and optional settings, sweep, and children. Every terminal
    node becomes a variant; seed replication belongs to the execution layer.
    Validate every leaf through the production training CLI before launching jobs.
    """
    from .cli import parse_args

    variants, names, configurations = [], set(), set()

    def visit(node, inherited, path):
        if not isinstance(node, dict) or set(node) - {
            "name",
            "settings",
            "sweep",
            "children",
        }:
            raise ValueError("Tree nodes accept name, settings, sweep, and children")
        name = node.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name
        ):
            raise ValueError(
                "Node names must be safe path components (1–80 characters)"
            )
        if len(path) >= 32:
            raise ValueError("Architecture tree exceeds 32 levels")
        settings = {**inherited, **_settings(node.get("settings", {}))}
        sweep = node.get("sweep", {})
        if not isinstance(sweep, dict) or set(sweep) - set(ARCHITECTURE_KEYS):
            raise ValueError("Sweep must map architecture keys to nonempty lists")
        for key, values in sweep.items():
            if not isinstance(values, list) or not values:
                raise ValueError("Sweep values must be nonempty lists")
            for value in values:
                _settings({key: value})
            if len({json.dumps(value) for value in values}) != len(values):
                raise ValueError("Sweep values must be unique")
        children = node.get("children", [])
        if type(children) is not list:
            raise ValueError("Node children must be a list")
        for values in product(*sweep.values()):
            selected = dict(zip(sweep, values))
            labels = tuple(f"{key}_{value}" for key, value in selected.items())
            if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", label) for label in labels):
                raise ValueError("Sweep values must form safe path components")
            branch = (*path, name, *labels)
            merged = {**settings, **selected}
            if children:
                for child in children:
                    visit(child, merged, branch)
                continue
            try:
                parsed = parse_args(algorithm, multi_agent, setting_arguments(merged))
            except SystemExit as error:
                raise ValueError(
                    f"Invalid architecture settings at {'/'.join(branch)}"
                ) from error
            canonical = effective_settings(parsed)
            identity = json.dumps(canonical, sort_keys=True)
            if branch in names or identity in configurations:
                raise ValueError(f"Duplicate architecture variant: {'/'.join(branch)}")
            names.add(branch)
            configurations.add(identity)
            variants.append(ArchitectureVariant(branch, canonical))

    visit(tree, {}, ())
    return variants


def default_tree(temporal=("none",), history_lengths=(8,)):
    """Compare pooling/context choices with default-depth, fully connected encoders.

    Distance/kNN and additional layer counts can be added through a JSON tree.
    Temporal methods are appended to each spatial leaf to form matched variants.
    """

    def pooling_nodes(pma=False):
        nodes = [
            {"name": p, "settings": {"pooling": p}} for p in ("mean", "max", "mean_max")
        ]
        nodes.append(
            {
                "name": "additive_attention",
                "settings": {"pooling": "additive_attention"},
                "sweep": {"attention_context": ["ownship_goal", "all"]},
            }
        )
        if pma:
            nodes.append({"name": "pma", "settings": {"pooling": "pma"}})
        return nodes

    tree = {
        "name": "architectures",
        "children": [
            {"name": "mlp", "settings": {"architecture": "mlp"}},
            {
                "name": "entity_mlp",
                "settings": {"architecture": "entity_mlp"},
                "children": pooling_nodes(),
            },
            {
                "name": "entity_transformer",
                "settings": {"architecture": "entity_transformer"},
                "children": pooling_nodes(True),
            },
            {
                "name": "entity_gnn",
                "settings": {"architecture": "entity_gnn"},
                "sweep": {"gnn_type": ["gcn", "gat"]},
                "children": pooling_nodes(True),
            },
        ],
    }

    def add_temporal(node):
        if node.get("children"):
            for child in node["children"]:
                add_temporal(child)
        else:
            node["children"] = [
                {
                    "name": method,
                    "settings": {"temporal": method},
                    **(
                        {"sweep": {"history_length": list(history_lengths)}}
                        if method != "none"
                        else {}
                    ),
                }
                for method in temporal
            ]

    add_temporal(tree)
    return tree
