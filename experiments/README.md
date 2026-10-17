# Personal agents and experiments

All code in this directory is personal work built on the upstream TU Delft
competition environments. Run commands from the repository root.

## Setup and training

```bash
python -m pip install -e ./vendor/bluesky-gym -e ".[dev]"
```

### Single-agent MLP baselines

PPO and SAC control one aircraft in the Gymnasium `CompetitionEnv`. Both use
two hidden layers of 256 units. PPO uses tanh actor/value networks; SAC uses
ReLU actor/twin-Q networks. Edit the builders in
[`models.py`](../src/bluesky_pabloramesc/models.py) to customize them.
These baselines use Stable-Baselines3 for the learning algorithms.

```bash
python -m experiments.train_single_ppo --timesteps 2000000
python -m experiments.train_single_sac --timesteps 2000000
python -m experiments.train_single_ppo evaluate --episodes 100 --eval-seed 10000
python -m experiments.train_single_ppo replay --episodes 3
python -m experiments.train_single_sac replay --episodes 3
```

Install the project before running these commands. Models
default to `experiments/models/ppo_single_mlp.zip` and `sac_single_mlp.zip`,
each with a matching `_vecnormalize.pkl` file. Keep both files for replay.
Use `--model-path`, `--seed`, and `--device` to configure training. Evaluation
uses a separate `--eval-seed` (default `10000`) in all modes. `evaluate` loads a
checkpoint without rendering; `replay` renders it. These modes also accept
`--stats-path` for an explicit normalization file. Use the same evaluation seed,
episode count, and environment settings when comparing policies; this does not
yet replace an explicit fixed scenario suite or multi-seed analysis. Training ends
with evaluation on raw rewards, with normalization statistics frozen. Run one
environment per process because BlueSky shares simulator state.

### Multi-agent baselines

```bash
python -m experiments.train_zoo_ppo 2000000
python -m experiments.train_zoo_sac 2000000
```

PPO and SAC share wrappers in
[`multi_agent.py`](../src/bluesky_pabloramesc/multi_agent.py). Their default models
and normalization sidecars are written to `experiments/models/`.

```bash
python -m experiments.train_zoo_ppo replay
python -m experiments.train_zoo_sac replay
```

The earlier upstream-example checkpoint is preserved as
`experiments/models/ppo_competition.zip`; it is distinct from the tuned trainers'
models and has no normalization sidecar.

Put custom agents, algorithms, and reusable utilities in
[`src/bluesky_pabloramesc/`](../src/README.md). Keep training scripts, configs,
and experiment notes here. Models, results, and virtual environments are ignored
by Git; store large outputs separately when sharing.

## Checks

```bash
python -m pytest
ruff check src experiments tests
ruff format --check src experiments tests
```

Tests use short synthetic environments to check policy updates, checkpoint
round trips, frozen evaluation statistics, cleanup, and multi-agent padding.
They do not start BlueSky or measure training quality.

Implementation status and remaining comparisons live in [ROADMAP.md](../ROADMAP.md).
