"""MLP baselines: edit these builders to customize networks and learning settings."""

import torch
from stable_baselines3 import PPO, SAC

GAMMA = 0.995


def build_ppo(env, seed=0, device="auto", **overrides):
    """Independent 256x256 tanh MLPs for the actor and value function."""
    settings = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": GAMMA,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_kwargs": {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": torch.nn.Tanh,
        },
        "seed": seed,
        "device": device,
        "verbose": 1,
    }
    settings.update(overrides)
    return PPO("MlpPolicy", env, **settings)


def build_sac(env, seed=0, device="auto", **overrides):
    """256x256 ReLU MLP actor and twin Q critics for continuous actions."""
    settings = {
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": GAMMA,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "policy_kwargs": {
            "net_arch": {"pi": [256, 256], "qf": [256, 256]},
            "activation_fn": torch.nn.ReLU,
            "n_critics": 2,
        },
        "seed": seed,
        "device": device,
        "verbose": 1,
    }
    settings.update(overrides)
    return SAC("MlpPolicy", env, **settings)


def build_multi_ppo(venv, seed=0, device="auto"):
    """Construct PPO for ten-agent parameter-sharing training."""
    return PPO(
        "MlpPolicy",
        venv,
        # 1,024 joint steps produce 10,240 samples.  A 512 batch therefore
        # divides the buffer exactly while mixing experience across agents.
        n_steps=1_024,
        batch_size=512,
        n_epochs=10,
        gamma=GAMMA,
        gae_lambda=0.97,
        learning_rate=3e-4,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            # Separate heads prevent the value objective from forcing the
            # shared actor into the same representation bottleneck.
            "net_arch": {
                "pi": [256, 256, 128],
                "vf": [256, 256, 128],
            },
            "activation_fn": torch.nn.Tanh,
            "ortho_init": True,
            "log_std_init": -0.5,
            "optimizer_kwargs": {"eps": 1e-5},
        },
        seed=seed,
        verbose=1,
        device=device,
    )


def build_multi_sac(venv, seed=0, device="auto"):
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
            # The critics also consume the (observation, action) input,
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
        seed=seed,
        verbose=1,
        device=device,
    )
