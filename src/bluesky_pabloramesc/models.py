"""Policy builders for flat MLPs and entity MLP, Transformer, and GNN encoders."""

import numpy as np
import torch
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from .entities import (
    ATTENTION_CONTEXTS,
    ATTENTION_POOLING,
    GNN_TYPES,
    GRAPH_TYPES,
    POOLING_METHODS,
    TRANSFORMER_ARCHITECTURES,
)

GAMMA = 0.995


def build_mappo(env, seed=0, device="auto", **overrides):
    """Build custom MAPPO with a shared local actor and centralized critic.

    Override policy_kwargs.actor_class/critic_class and their kwargs to change
    representations without changing rollout collection or the PPO update.
    Batch sizes count joint timesteps, each containing all fixed agent slots.
    """
    from .mappo import MAPPO

    return _build_joint_ppo(MAPPO, env, seed, device, **overrides)


def build_ippo(env, seed=0, device="auto", **overrides):
    """Build separate actors, local critics, and optimizers for every aircraft."""
    from .ippo import IPPO

    return _build_joint_ppo(IPPO, env, seed, device, **overrides)


def build_shared_ppo(env, seed=0, device="auto", **overrides):
    """Build the shared local-critic control with matched joint PPO settings."""
    from .ippo import SharedPPO

    return _build_joint_ppo(SharedPPO, env, seed, device, **overrides)


def _build_joint_ppo(
    algorithm,
    env,
    seed,
    device,
    architecture="mlp",
    pooling=None,
    attention_context="all",
    **overrides,
):
    """Use matched PPO settings with flat MLPs or additive aircraft attention."""
    if architecture not in ("mlp", "entity_mlp"):
        raise ValueError("Joint PPO supports mlp or entity_mlp architectures")
    if architecture == "mlp" and pooling not in (None, "mean_max"):
        raise ValueError("Flat MLPs do not support attention pooling")
    if architecture == "entity_mlp" and pooling not in (None, "additive_attention"):
        raise ValueError("Joint entity MLPs require additive_attention pooling")
    if attention_context not in ATTENTION_CONTEXTS:
        raise ValueError(f"Unknown attention context: {attention_context}")

    settings = {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": GAMMA,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "value_clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "actor_kwargs": {"hidden_sizes": (256, 256)},
            "critic_kwargs": {"hidden_sizes": (256, 256)},
            "log_std_init": -0.5,
        },
        "seed": seed,
        "device": device,
        "verbose": 1,
    }
    settings.update(overrides)
    if architecture == "entity_mlp":
        from .mappo import MAPPO
        from .multi_agent_attention import (
            AdditiveAttentionActor,
            CentralizedAdditiveAttentionCritic,
            LocalAdditiveAttentionCritic,
        )

        layout = getattr(env.observation_space, "entity_layout", None)
        if layout is None:
            raise ValueError("Additive attention requires entity_observations=True")
        policy = dict(settings["policy_kwargs"])
        if "actor_class" in policy or "critic_class" in policy:
            raise ValueError("Custom network classes require architecture='mlp'")
        policy.update(
            actor_class=AdditiveAttentionActor,
            critic_class=(
                CentralizedAdditiveAttentionCritic
                if algorithm is MAPPO
                else LocalAdditiveAttentionCritic
            ),
        )
        for role in ("actor_kwargs", "critic_kwargs"):
            policy[role] = {
                **policy.get(role, {}),
                "layout": layout,
                "attention_context": attention_context,
            }
        settings["policy_kwargs"] = policy
    return algorithm(env=env, **settings)


def architecture_settings(
    env,
    architecture,
    pooling,
    policy_kwargs,
    attention_context="ownship_goal",
    transformer_layers=1,
    gnn_type="gat",
    gnn_layers=1,
    graph="fully_connected",
    graph_radius_nm=25.0,
    graph_k=3,
    temporal="none",
    history_length=8,
):
    """Select the flat baseline or a separately learned actor/critic set encoder."""
    if architecture not in (
        "mlp",
        "entity_mlp",
        "entity_gnn",
        *TRANSFORMER_ARCHITECTURES,
    ):
        raise ValueError(f"Unknown architecture: {architecture}")
    if attention_context not in ATTENTION_CONTEXTS:
        raise ValueError(f"Unknown attention context: {attention_context}")
    if pooling is None:
        pooling = (
            "context_token"
            if architecture == "context_entity_transformer"
            else "mean_max"
        )
    if (pooling == "context_token") != (architecture == "context_entity_transformer"):
        raise ValueError(
            "context_entity_transformer requires context_token pooling exclusively"
        )
    if pooling not in POOLING_METHODS:
        raise ValueError(f"Unknown pooling: {pooling}")
    if pooling == "pma" and architecture not in ("entity_transformer", "entity_gnn"):
        raise ValueError("PMA pooling requires entity_transformer or entity_gnn")
    if not isinstance(transformer_layers, int) or transformer_layers <= 0:
        raise ValueError("transformer_layers must be a positive integer")
    if gnn_type not in GNN_TYPES or graph not in GRAPH_TYPES:
        raise ValueError("Unknown GNN type or graph")
    if not isinstance(gnn_layers, int) or gnn_layers <= 0:
        raise ValueError("gnn_layers must be a positive integer")
    if not np.isfinite(graph_radius_nm) or graph_radius_nm <= 0:
        raise ValueError("graph_radius_nm must be finite and positive")
    if type(graph_k) is not int or graph_k <= 0:
        raise ValueError("graph_k must be a positive integer")
    from stable_baselines3.common.torch_layers import FlattenExtractor

    from .temporal import TEMPORAL_METHODS, TemporalFeatures

    if temporal not in TEMPORAL_METHODS:
        raise ValueError(f"Unknown temporal encoder: {temporal}")
    if type(history_length) is not int or history_length <= 0:
        raise ValueError("history_length must be a positive integer")
    if temporal == "none" and hasattr(env.observation_space, "temporal_frame_space"):
        raise ValueError("HistoryWindow requires a temporal encoder")
    if architecture == "mlp" and temporal == "none":
        return policy_kwargs
    from .entities import (
        ContextEntityTransformer,
        EntityGNN,
        EntityMLP,
        EntityTransformer,
    )

    settings = {
        **policy_kwargs,
        "features_extractor_class": {
            "mlp": FlattenExtractor,
            "entity_mlp": EntityMLP,
            "entity_transformer": EntityTransformer,
            "context_entity_transformer": ContextEntityTransformer,
            "entity_gnn": EntityGNN,
        }[architecture],
        "features_extractor_kwargs": {
            **(
                {"layout": env.observation_space.entity_layout}
                if architecture != "mlp"
                else {}
            ),
            "pooling": pooling,
            **(
                {
                    "gnn_type": gnn_type,
                    "gnn_layers": gnn_layers,
                    "graph": graph,
                    "graph_radius_nm": graph_radius_nm,
                    "graph_k": graph_k,
                }
                if architecture == "entity_gnn"
                else {}
            ),
            **(
                {"transformer_layers": transformer_layers}
                if architecture in TRANSFORMER_ARCHITECTURES
                else {}
            ),
            **(
                {"attention_context": attention_context}
                if pooling in ATTENTION_POOLING
                else {}
            ),
        },
        "share_features_extractor": False,
    }
    if architecture == "mlp":
        settings["features_extractor_kwargs"] = {}
    if temporal != "none":
        frame_space = getattr(env.observation_space, "temporal_frame_space", None)
        if frame_space is None:
            raise ValueError("Temporal policies require a HistoryWindow environment")
        settings["features_extractor_kwargs"] = {
            "frame_space": frame_space,
            "frame_encoder_class": settings["features_extractor_class"],
            "frame_encoder_kwargs": settings["features_extractor_kwargs"],
            "temporal": temporal,
            "history_length": history_length,
        }
        settings["features_extractor_class"] = TemporalFeatures
    return settings


def build_ppo(
    env,
    seed=0,
    device="auto",
    architecture="mlp",
    pooling=None,
    attention_context="ownship_goal",
    transformer_layers=1,
    gnn_type="gat",
    gnn_layers=1,
    graph="fully_connected",
    graph_radius_nm=25.0,
    graph_k=3,
    temporal="none",
    history_length=8,
    **overrides,
):
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
    settings["policy_kwargs"] = architecture_settings(
        env,
        architecture,
        pooling,
        settings["policy_kwargs"],
        attention_context,
        transformer_layers,
        gnn_type,
        gnn_layers,
        graph,
        graph_radius_nm,
        graph_k,
        temporal,
        history_length,
    )
    return PPO("MlpPolicy", env, **settings)


def build_sac(
    env,
    seed=0,
    device="auto",
    architecture="mlp",
    pooling=None,
    attention_context="ownship_goal",
    transformer_layers=1,
    gnn_type="gat",
    gnn_layers=1,
    graph="fully_connected",
    graph_radius_nm=25.0,
    graph_k=3,
    temporal="none",
    history_length=8,
    **overrides,
):
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
    settings["policy_kwargs"] = architecture_settings(
        env,
        architecture,
        pooling,
        settings["policy_kwargs"],
        attention_context,
        transformer_layers,
        gnn_type,
        gnn_layers,
        graph,
        graph_radius_nm,
        graph_k,
        temporal,
        history_length,
    )
    return SAC("MlpPolicy", env, **settings)


def _build_deterministic(
    model_type,
    env,
    seed,
    device,
    overrides,
    architecture,
    pooling,
    attention_context,
    transformer_layers,
    gnn_type,
    gnn_layers,
    graph,
    graph_radius_nm,
    graph_k,
    temporal,
    history_length,
):
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
    settings["policy_kwargs"] = architecture_settings(
        env,
        architecture,
        pooling,
        settings["policy_kwargs"],
        attention_context,
        transformer_layers,
        gnn_type,
        gnn_layers,
        graph,
        graph_radius_nm,
        graph_k,
        temporal,
        history_length,
    )
    return model_type("MlpPolicy", env, **settings)


def build_td3(
    env,
    seed=0,
    device="auto",
    architecture="mlp",
    pooling=None,
    attention_context="ownship_goal",
    transformer_layers=1,
    gnn_type="gat",
    gnn_layers=1,
    graph="fully_connected",
    graph_radius_nm=25.0,
    graph_k=3,
    temporal="none",
    history_length=8,
    **overrides,
):
    """Build TD3 with 256x256 ReLU networks and Gaussian exploration.

    Args:
        env: Continuous-action environment used for training.
        seed: Random seed for initialization and training.
        device: PyTorch device name, or "auto".
        **overrides: SB3 constructor settings that replace the defaults.

    Returns:
        Configured SB3 TD3 model with twin critics and delayed actor updates.
    """
    return _build_deterministic(
        TD3,
        env,
        seed,
        device,
        overrides,
        architecture,
        pooling,
        attention_context,
        transformer_layers,
        gnn_type,
        gnn_layers,
        graph,
        graph_radius_nm,
        graph_k,
        temporal,
        history_length,
    )


def build_ddpg(
    env,
    seed=0,
    device="auto",
    architecture="mlp",
    pooling=None,
    attention_context="ownship_goal",
    transformer_layers=1,
    gnn_type="gat",
    gnn_layers=1,
    graph="fully_connected",
    graph_radius_nm=25.0,
    graph_k=3,
    temporal="none",
    history_length=8,
    **overrides,
):
    """Build DDPG with 256x256 ReLU networks and Gaussian exploration.

    Args:
        env: Continuous-action environment used for training.
        seed: Random seed for initialization and training.
        device: PyTorch device name, or "auto".
        **overrides: SB3 constructor settings that replace the defaults.

    Returns:
        Configured SB3 DDPG model with one critic.
    """
    return _build_deterministic(
        DDPG,
        env,
        seed,
        device,
        overrides,
        architecture,
        pooling,
        attention_context,
        transformer_layers,
        gnn_type,
        gnn_layers,
        graph,
        graph_radius_nm,
        graph_k,
        temporal,
        history_length,
    )


def build_multi_ppo(
    venv,
    seed=0,
    device="auto",
    architecture="mlp",
    pooling=None,
    attention_context="ownship_goal",
    transformer_layers=1,
    gnn_type="gat",
    gnn_layers=1,
    graph="fully_connected",
    graph_radius_nm=25.0,
    graph_k=3,
    temporal="none",
    history_length=8,
):
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
        policy_kwargs=architecture_settings(
            venv,
            architecture,
            pooling,
            {
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
            attention_context,
            transformer_layers,
            gnn_type,
            gnn_layers,
            graph,
            graph_radius_nm,
            graph_k,
            temporal,
            history_length,
        ),
        seed=seed,
        verbose=1,
        device=device,
    )


def build_multi_sac(
    venv,
    seed=0,
    device="auto",
    architecture="mlp",
    pooling=None,
    attention_context="ownship_goal",
    transformer_layers=1,
    gnn_type="gat",
    gnn_layers=1,
    graph="fully_connected",
    graph_radius_nm=25.0,
    graph_k=3,
    temporal="none",
    history_length=8,
):
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
        policy_kwargs=architecture_settings(
            venv,
            architecture,
            pooling,
            {
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
            attention_context,
            transformer_layers,
            gnn_type,
            gnn_layers,
            graph,
            graph_radius_nm,
            graph_k,
            temporal,
            history_length,
        ),
        seed=seed,
        verbose=1,
        device=device,
    )
