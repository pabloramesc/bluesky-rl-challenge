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
