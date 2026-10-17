# Custom implementations

Put reusable code in `bluesky_pabloramesc/` and experiment entry points in
`experiments/`.

| Module | Purpose |
| --- | --- |
| `models.py` | PPO/SAC MLP architectures and hyperparameters |
| `single_agent.py` | Single-aircraft Gymnasium environment |
| `multi_agent.py` | PettingZoo wrappers and vector environment |
| `training.py` | Training, checkpoints, evaluation, and replay |
| `cli.py` | Shared command-line options |

Add your own modules alongside these as needed. For example, after creating
`agents.py`, import it with `from bluesky_pabloramesc.agents import MyAgent`.

Install the package and development tools from the repository root:

```bash
python -m pip install -e ./vendor/bluesky-gym -e ".[dev]"
python -m pytest
```

Editable installation makes source changes available without reinstalling.
