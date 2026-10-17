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

Clone your repository with its dependency:

```bash
git clone --recurse-submodules https://github.com/pabloramesc/bluesky-rl-challenge.git bluesky-pabloramesc
cd bluesky-pabloramesc
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./vendor/bluesky-gym -e .
```

For an existing clone, run `git submodule update --init --recursive` first.
Alternatively, `uv sync` installs the personal package and local upstream dependency.

## Run

Run commands from this repository's root:

```bash
python -m experiments.train_single_ppo --timesteps 2000000
python -m experiments.train_single_sac --timesteps 2000000
python -m experiments.train_zoo_ppo 2000000
python -m experiments.train_zoo_sac 2000000
```

The `train_single_*` scripts train single-agent MLP policies; `train_zoo_*`
scripts train shared policies in the multi-agent environment.

See [experiments/README.md](experiments/README.md) for training details.
Generated checkpoints and results are kept under `experiments/models/` and
`experiments/results/`, ignored by Git.

Install development tools with `python -m pip install -e ".[dev]"`, then run
`python -m pytest`. See the [checks](experiments/README.md#checks) for linting
and formatting commands.