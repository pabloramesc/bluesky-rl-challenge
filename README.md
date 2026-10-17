# BlueSky agents development repository

Personal models, agents, and training experiments built on TU Delft BlueSky-Gym.

```text
experiments/          Your trainers, wrappers, adapters, and configuration
vendor/bluesky-gym/    TU Delft repository (Git submodule pinned to a commit)
pyproject.toml        Your development package and local dependency
```

Upstream attribution, license, examples, and documentation live in
[vendor/bluesky-gym](vendor/bluesky-gym/README.md).
See [UPSTREAM.md](UPSTREAM.md) for the pinned revision and update workflow.

## Setup

Clone your repository with its dependency:

```bash
git clone --recurse-submodules https://github.com/pabloramesc/bluesky-rl-challenge.git
cd bluesky-rl-challenge
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./vendor/bluesky-gym -e .
```

For an existing clone, run `git submodule update --init --recursive` first.
Alternatively, `uv sync` installs the personal package and local upstream dependency.

## Run

Run commands from this repository's root:

```bash
python -m experiments.train_zoo_ppo 2000000
python -m experiments.train_zoo_sac 2000000
python -m experiments.train_zoo_mappo_marllib 2000000 --num-gpus 1
```

MAPPO requires the separate legacy `.venv-marllib` runtime; see
[experiments/README.md](experiments/README.md). Generated checkpoints and results
are kept under `experiments/models/` and `experiments/results/`, ignored by Git.

## Git remote

The development repository uses
https://github.com/pabloramesc/bluesky-rl-challenge.git as `origin`.
Commit and push personal changes from this root:

```bash
git add .
git commit -m "Describe your changes"
git push origin main
```

The upstream submodule has its own `origin` pointing to TU Delft.
