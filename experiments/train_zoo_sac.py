"""Train a parameter-shared SAC policy on the multi-agent competition env.

The ten PettingZoo agents are exposed as ten vector slots backed by one
BlueSky simulation.  SAC then learns one shared stochastic policy from all
agents' transitions.  The settings below account for the wider observation,
long horizon, correlated joint steps, and the memory cost of storing ten
observations per replay-buffer row.

Run:
    .venv/bin/python -m experiments.train_zoo_sac [total_timesteps]
    .venv/bin/python -m experiments.train_zoo_sac replay [model_path] [normalizer_path]
"""

import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from .train_zoo_marl_common import EVAL_EPISODES, SEED, evaluate, make_vec_env


MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "sac_marl_competition")
Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
DEFAULT_TIMESTEPS = 2_000_000
GAMMA = 0.995


def normalizer_path(model_path):
    """Return the VecNormalize sidecar path belonging to a model path."""
    stem = model_path[:-4] if model_path.endswith(".zip") else model_path
    return f"{stem}_vecnormalize.pkl"


def make_training_env():
    return VecNormalize(
        make_vec_env(),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )


def evaluate_without_updating_stats(venv, n_episodes, model=None):
    """Evaluate on raw episode returns without changing running statistics."""
    old_training, old_norm_reward = venv.training, venv.norm_reward
    venv.training = False
    venv.norm_reward = False
    try:
        return evaluate(venv, n_episodes, model=model)
    finally:
        venv.training = old_training
        venv.norm_reward = old_norm_reward


def build_model(venv):
    """Construct SAC for ten-agent parameter-sharing training."""
    return SAC(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=20_000,
        batch_size=512,
        tau=0.005,
        gamma=GAMMA,
        # Each joint step inserts ten agent transitions.  Batched collection
        # reduces temporal correlation; 32 updates per 80 new transitions is
        # a practical update-to-data ratio for this expensive simulator.
        train_freq=(8, "step"),
        gradient_steps=32,
        ent_coef="auto_0.1",
        target_entropy="auto",
        use_sde=True,
        sde_sample_freq=4,
        use_sde_at_warmup=False,
        policy_kwargs={
            # The critics also consume the joint (observation, action) input,
            # so they get more capacity than the actor.
            "net_arch": {
                "pi": [256, 256, 128],
                "qf": [512, 512, 256],
            },
            "activation_fn": torch.nn.ReLU,
            "n_critics": 2,
            "share_features_extractor": False,
            "optimizer_kwargs": {"eps": 1e-5},
        },
        seed=SEED,
        verbose=1,
        device="auto",
    )


def train(total_timesteps):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    venv = make_training_env()
    try:
        base_mean, base_std = evaluate_without_updating_stats(
            venv, EVAL_EPISODES
        )
        print(
            f"\nBEFORE (random policy): mean episode return = "
            f"{base_mean:8.2f} +/- {base_std:.2f}"
        )

        model = build_model(venv)
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        model.save(MODEL_PATH)
        venv.save(normalizer_path(MODEL_PATH))

        trained_mean, trained_std = evaluate_without_updating_stats(
            venv, EVAL_EPISODES, model=model
        )
        print(
            f"\nBEFORE (random policy): mean episode return = "
            f"{base_mean:8.2f} +/- {base_std:.2f}"
        )
        print(
            f"AFTER ({total_timesteps} steps, greedy): mean episode return = "
            f"{trained_mean:8.2f} +/- {trained_std:.2f}"
        )
        print("learned something" if trained_mean > base_mean else "no improvement")
        print(
            f"saved policy to {MODEL_PATH}.zip and normalization statistics to "
            f"{normalizer_path(MODEL_PATH)}"
        )
    finally:
        venv.close()


def replay(model_path=MODEL_PATH, stats_path=None, n_episodes=3, seed=SEED):
    """Render complete joint episodes using the saved observation statistics."""
    stats_path = stats_path or normalizer_path(model_path)
    venv = VecNormalize.load(stats_path, make_vec_env(render_mode="human"))
    venv.training = False
    venv.norm_reward = False
    model = SAC.load(model_path, env=venv)
    venv.seed(seed)

    completed = 0
    obs = venv.reset()
    try:
        while completed < n_episodes:
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = venv.step(actions)
            if bool(np.all(dones)):
                completed += 1
    finally:
        venv.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "replay":
        replay(
            model_path=args[1] if len(args) > 1 else MODEL_PATH,
            stats_path=args[2] if len(args) > 2 else None,
        )
    else:
        train(int(args[0]) if args else DEFAULT_TIMESTEPS)
