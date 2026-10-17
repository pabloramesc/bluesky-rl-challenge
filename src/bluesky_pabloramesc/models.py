"""MLP baselines: edit these builders to customize networks and learning settings."""

import numpy as np
import torch
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

GAMMA = 0.995


def build_ppo(env, seed=0, device="auto", **overrides):
    """Build PPO with separate 256x256 tanh actor and value networks.

    Args:
        env: Continuous-action environment used for training.
        seed: Random seed for initialization and training.
        device: PyTorch device name, or "auto".
        **overrides: SB3 constructor settings that replace the defaults.

    Returns:
        Configured SB3 PPO model.
    """
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
    """Build SAC with a 256x256 ReLU actor and twin Q critics.

    Args:
        env: Continuous-action environment used for training.
        seed: Random seed for initialization and training.
        device: PyTorch device name, or "auto".
        **overrides: SB3 constructor settings that replace the defaults.

    Returns:
        Configured SB3 SAC model.
    """
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


def _build_deterministic(model_type, env, seed, device, overrides):
    """Build a deterministic actor-critic with shared baseline settings.

    Args:
        model_type: SB3 TD3 or DDPG class.
        env: Continuous-action environment used for training.
        seed: Random seed for initialization and training.
        device: PyTorch device name, or "auto".
        overrides: SB3 constructor settings that replace the defaults.

    Returns:
        Configured model with Gaussian action noise.
    """
    action_shape = env.action_space.shape
    settings = {
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": GAMMA,
        "train_freq": 1,
        "gradient_steps": 1,
        # SB3 adds this noise in normalized [-1, 1] action coordinates.
        "action_noise": NormalActionNoise(
            mean=np.zeros(action_shape), sigma=0.1 * np.ones(action_shape)
        ),
        "policy_kwargs": {
            "net_arch": {"pi": [256, 256], "qf": [256, 256]},
            "activation_fn": torch.nn.ReLU,
        },
        "seed": seed,
        "device": device,
        "verbose": 1,
    }
    settings.update(overrides)
    return model_type("MlpPolicy", env, **settings)


def build_td3(env, seed=0, device="auto", **overrides):
    """Build TD3 with 256x256 ReLU networks and Gaussian exploration.

    Args:
        env: Continuous-action environment used for training.
        seed: Random seed for initialization and training.
        device: PyTorch device name, or "auto".
        **overrides: SB3 constructor settings that replace the defaults.

    Returns:
        Configured SB3 TD3 model with twin critics and delayed actor updates.
    """
    return _build_deterministic(TD3, env, seed, device, overrides)


def build_ddpg(env, seed=0, device="auto", **overrides):
    """Build DDPG with 256x256 ReLU networks and Gaussian exploration.

    Args:
        env: Continuous-action environment used for training.
        seed: Random seed for initialization and training.
        device: PyTorch device name, or "auto".
        **overrides: SB3 constructor settings that replace the defaults.

    Returns:
        Configured SB3 DDPG model with one critic.
    """
    return _build_deterministic(DDPG, env, seed, device, overrides)


def build_multi_ppo(venv, seed=0, device="auto"):
    """Build PPO for ten-agent parameter sharing.

    Args:
        venv: Vector environment with one slot per aircraft.
        seed: Random seed for initialization and training.
        device: PyTorch device name, or "auto".

    Returns:
        Configured SB3 PPO model shared across agent slots.
    """
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
    """Build SAC for ten-agent parameter sharing.

    Args:
        venv: Vector environment with one slot per aircraft.
        seed: Random seed for initialization and training.
        device: PyTorch device name, or "auto".

    Returns:
        Configured SB3 SAC model shared across agent slots.
    """
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
