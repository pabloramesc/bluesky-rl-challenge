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
| `observations.py` | Geometry encodings, ownship features, and shared environment hooks |
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
