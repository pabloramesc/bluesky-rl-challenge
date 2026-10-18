# Custom implementations

Put reusable code in `bluesky_pabloramesc/` and experiment entry points in
`experiments/`.

| Module | Purpose |
| --- | --- |
| `models.py` | PPO/SAC/TD3/DDPG MLP architectures and hyperparameters |
| `single_agent.py` | Single-aircraft Gymnasium environment |
| `multi_agent.py` | PettingZoo wrappers and vector environment |
| `training.py` | Training, checkpoints, evaluation, and replay |
| `reporting.py` | Run metadata, raw episode histories, and performance plots |
| `cli.py` | Shared command-line options |
| `actions.py` | Validated action configuration and heading/CAS target mappings |
| `observations.py` | Geometry encodings, optional conflict features, ownship state, and shared environment hooks |
| `comparison.py` | Isolated action/observation ablations and paired reports |
| `architecture_tree.py` | Conditional architecture options, inherited settings, and Cartesian sweeps |
| `architecture_reporting.py` | Architecture CSVs, seed-paired differences, ranked tables, and plots |
| `architecture_comparison.py` | Bounded subprocess scheduling, resume/retry, logs, and report collection |

Add your own modules alongside these as needed. For example, after creating
`agents.py`, import it with `from bluesky_pabloramesc.agents import MyAgent`.

Create `.venv` using the [setup guide](../README.md#setup), then install the
package and development tools from the repository root:

```bash
source .venv/bin/activate
uv pip install -e ./vendor/bluesky-gym -e ".[dev]"
python -m pytest
```

Editable installation makes source changes available without reinstalling.

`entities.py` implements shared aircraft MLP encoding, masked mean/max pooling,
additive attention with ownship/goal or full context, and slot-shared
normalization. Select it with `--architecture entity_mlp`;
`--architecture mlp` remains the default.

`--architecture entity_transformer` adds configurable aircraft self-attention
(`--transformer-layers`, default 1). It reuses all MLP pooling methods and adds
`--pooling pma` for a Set Transformer variant with one learned pooling query.

`--architecture entity_gnn` adds GCN/GAT message passing with configurable depth
and graph connectivity. Fully connected graphs are the default; distance and
k-nearest (`--graph knn --graph-k 3`) graphs compute binary adjacency before
running normalization. All existing pooling
methods are supported.

`temporal.py` provides per-agent raw history windows, shared temporal normalization,
and frame-stack/GRU/LSTM/causal-TCN encoders. Use `--temporal` independently of
`--architecture`; `--history-length` defaults to 8 for temporal modes.

Architecture comparison trees use JSON objects with `name`, optional `settings`,
optional `sweep` (setting names mapped to lists), and optional `children`.
Children inherit ancestor settings. Sweeps form a Cartesian product at their
node; child branches select conditional options. Every leaf is validated using
the training CLI before execution, and duplicate effective configurations are
rejected. For example, a conditional graph branch can be written as:

```json
{
  "name": "graphs",
  "settings": {"architecture": "entity_gnn", "pooling": "mean"},
  "sweep": {"gnn_type": ["gcn", "gat"]},
  "children": [
    {"name": "full", "settings": {"graph": "fully_connected"}},
    {
      "name": "knn",
      "settings": {"graph": "knn"},
      "sweep": {"graph_k": [1, 3, 5]}
    }
  ]
}
```

`expand_tree(tree)` returns eight named variants for this example. The reporting
module averages evaluation scenarios and agents within each training seed before
measuring variation across training seeds. Its first variant is the paired
baseline. Incomplete variants remain visible without receiving a rank.
Run these through `python -m experiments.compare_architectures`; see the
[comparison guide](../experiments/README.md#architecture-comparisons) for resource
limits, conditional tree examples, temporal sweeps and result reuse.
