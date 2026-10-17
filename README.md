# bluesky-pabloramesc

Personal models, agents, and training experiments built on TU Delft BlueSky-Gym.

```text
src/bluesky_pabloramesc/  Models, environments, and shared training logic
experiments/             Training entry points and experiment configuration
tests/                   Fast pytest checks
vendor/bluesky-gym/      TU Delft repository (Git submodule pinned to a commit)
pyproject.toml           Development package and dependencies
```

Upstream attribution, license, examples, and documentation live in
[vendor/bluesky-gym](vendor/bluesky-gym/README.md).
See [UPSTREAM.md](UPSTREAM.md) for the pinned revision and update workflow.

See [ROADMAP.md](ROADMAP.md) for implementation status and planned comparisons.

Use [src/](src/README.md) for your own implementations and import them from
training scripts in `experiments/`.

## Setup

Use uv to create the environment and install both packages. Run these commands
from a shell where `uv` is available:

```bash
git clone --recurse-submodules https://github.com/pabloramesc/bluesky-rl-challenge.git bluesky-pabloramesc
cd bluesky-pabloramesc
uv venv .venv
source .venv/bin/activate
uv pip install -e ./vendor/bluesky-gym -e ".[dev]"
```

For an existing checkout with `.venv` already created, run from the repository
root:

```bash
git submodule update --init --recursive
source .venv/bin/activate
uv pip install -e ./vendor/bluesky-gym -e ".[dev]"
```

The single `.venv` serves both packages:

- `-e ./vendor/bluesky-gym` installs the upstream environments.
- `-e ".[dev]"` installs your own `bluesky-pabloramesc` package from
  `src/bluesky_pabloramesc/`, the `experiments` entry points, and pytest and Ruff.

Both packages are editable, so source changes take effect without reinstalling.
uv-created environments do not need pip installed; use `uv pip install` to
manage their packages. Activate `.venv` in each new shell before running training
or checks.

Your package is imported as `bluesky_pabloramesc`; see
[src/README.md](src/README.md) for its modules.

## Run

Run commands from this repository's root:

```bash
python -m experiments.train_single_ppo --timesteps 100000
python -m experiments.train_single_sac --timesteps 100000
python -m experiments.train_single_td3 --timesteps 100000
python -m experiments.train_single_ddpg --timesteps 100000
python -m experiments.train_zoo_ppo 100000
python -m experiments.train_zoo_sac 100000
```

To add more training to an existing checkpoint:

```bash
python -m experiments.train_single_ppo resume --timesteps 100000
```

`--timesteps` adds steps to the saved total. See
[resuming training](experiments/README.md#resuming-training) for other algorithms,
replay buffers, and saving to a new path.

The `train_single_*` scripts train single-agent MLP policies; `train_zoo_*`
scripts train shared policies in the multi-agent environment.

Training defaults to 100,000 steps and displays SB3's tqdm/Rich progress bar.
Periodic checkpoints are saved every 10,000 steps at rollout boundaries. Ctrl+C
saves the current policy, normalization, replay buffer, and partial run report.
Resume it with the same algorithm's `resume` command.

See [experiments/README.md](experiments/README.md) for training details.
Generated checkpoints and results are kept under `experiments/models/` and
`experiments/results/`, ignored by Git. Each training run saves a unique report
with dates, training time, evaluation scores, episode CSVs, performance plots,
and its own checkpoint. See [run reports](experiments/README.md#reviewing-a-training-run).

The setup above includes development tools. Run `python -m pytest`. See the [checks](experiments/README.md#checks) for linting
and formatting commands.
