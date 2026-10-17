"""Train genuine CTDE MAPPO on CompetitionZooEnv using MARLlib 1.0.3.

The shared actor receives only its aircraft's 125 features (124 flattened
environment features plus an active mask). During training, the centralized
critic additionally receives the aligned 1,250-feature joint state. Execution
therefore remains decentralized.

MARLlib is a legacy stack and must be installed in a separate Python 3.8/3.9
environment following its official instructions, including ``add_patch.py``.
It is not compatible with this repository's Python 3.12 ``.venv``.

Run:
    python -m experiments.train_zoo_mappo_marllib [total_timesteps]
    python -m experiments.train_zoo_mappo_marllib 2000000 --num-gpus 1

Only one rollout worker is used by default. BlueSky is process-global, so do
not configure multiple environments per Ray worker.
"""

import argparse
from importlib.util import find_spec
import os
from pathlib import Path
import sys


def use_isolated_marllib_runtime():
    """Re-exec with the local legacy interpreter when run from the main venv."""
    if find_spec("marllib") is not None:
        return
    legacy_python = (
        Path(__file__).resolve().parents[1] / ".venv-marllib" / "bin" / "python"
    )
    if legacy_python.exists():
        os.execv(
            str(legacy_python),
            [str(legacy_python), "-m", "experiments.train_zoo_mappo_marllib", *sys.argv[1:]],
        )


use_isolated_marllib_runtime()

from .train_zoo_marllib_env import make_marllib_env


DEFAULT_TIMESTEPS = 2_000_000
ENV_CONFIG_PATH = Path(__file__).with_name("marllib_bluesky_competition.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "total_timesteps",
        nargs="?",
        type=int,
        default=DEFAULT_TIMESTEPS,
    )
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parent / "results" / "marllib_mappo"),
    )
    return parser.parse_args()


def train(args):
    marl, env = make_marllib_env(ENV_CONFIG_PATH.resolve())

    mappo = marl.algos.mappo(
        hyperparam_source="common",
        use_gae=True,
        **{
            "lambda": 0.97,
            "num_sgd_iter": 10,
            "vf_loss_coeff": 0.5,
            "lr": 3e-4,
            "entropy_coeff": 0.01,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "batch_mode": "truncate_episodes",
        },
    )
    model = marl.build_model(
        env,
        mappo,
        {
            "core_arch": "mlp",
            "encode_layer": "256-256-128",
        },
    )

    return mappo.fit(
        env,
        model,
        stop={"timesteps_total": args.total_timesteps},
        local_mode=False,
        num_workers=1,
        num_gpus=args.num_gpus,
        num_gpus_per_worker=0,
        share_policy="all",
        seed=args.seed,
        evaluation_interval=None,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_end=True,
        local_dir=args.results_dir,
    )


if __name__ == "__main__":
    train(parse_args())
