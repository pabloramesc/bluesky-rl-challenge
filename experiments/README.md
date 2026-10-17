# Personal agents and experiments

All code in this directory is personal work built on the upstream TU Delft
competition environments. Run commands from the repository root.

## Setup and training

Install the upstream package into your active Python environment:

```bash
python -m pip install -e ./vendor/bluesky-gym -e .
python -m experiments.train_zoo_ppo 2000000
python -m experiments.train_zoo_sac 2000000
```

PPO and SAC share wrappers in `train_zoo_marl_common.py`. Their default models
and normalization sidecars are written to `experiments/models/`.

```bash
python -m experiments.train_zoo_ppo replay
python -m experiments.train_zoo_sac replay
```

The earlier upstream-example checkpoint is preserved as
`experiments/models/ppo_competition.zip`; it is distinct from the tuned trainers'
models and has no normalization sidecar.

## MAPPO / MARLlib

MARLlib uses the separate legacy `.venv-marllib` runtime described in the trainer
module. Keep it at the repository root. Install the upstream source into that runtime
as a source path (upstream package metadata requires Python >=3.10):

```bash
.venv-marllib/bin/python - <<'PYTHON'
from pathlib import Path
import sysconfig
source = Path("vendor/bluesky-gym").resolve()
(Path(sysconfig.get_path("purelib")) / "bluesky_gym_source.pth").write_text(str(source) + "\n")
PYTHON
```

Preserve the existing MARLlib/Ray dependency versions and patches. If MARLlib is unavailable in the current
interpreter, the trainer automatically uses `.venv-marllib/bin/python` when present.

```bash
python -m experiments.train_zoo_mappo_marllib 2000000 --num-gpus 1
```

`train_zoo_marllib_env.py` adapts the environment for MARLlib;
`marllib_bluesky_competition.yaml` contains its configuration. The BlueSky runtime
working directory stays under `.venv-marllib/`. Training results default to
`experiments/results/marllib_mappo/`; use `--results-dir` to override this.
Existing result logs are preserved verbatim and may contain old absolute paths.

Models, results, and virtual environments are ignored by Git. Add new agent code,
configs, and experiment notes here; store large outputs separately when sharing.
