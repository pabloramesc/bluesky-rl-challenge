"""Local aircraft attention and an active-agent attention critic for joint PPO."""

import numpy as np
from gymnasium import spaces
from torch import cat, nn

from .entities import AdditiveAttention, EntityMLP
from .mappo import mlp


def aircraft_encoder(observation_dim, layout, attention_context):
    """Reuse the single-agent entity MLP and its masked additive readout."""
    return EntityMLP(
        spaces.Box(-np.inf, np.inf, (observation_dim,), dtype=np.float32),
        layout,
        pooling="additive_attention",
        attention_context=attention_context,
    )


def local_features(encoder, observations):
    """Preserve arbitrary leading batch/agent axes around the SB3 extractor."""
    features = encoder(observations.reshape(-1, observations.shape[-1]))
    return features.reshape(*observations.shape[:-1], encoder.features_dim)


class AdditiveAttentionActor(nn.Module):
    """Condition local Gaussian means on context and attended visible aircraft."""

    def __init__(
        self,
        observation_dim,
        action_dim,
        layout,
        hidden_sizes=(256, 256),
        attention_context="all",
    ):
        super().__init__()
        self.features_extractor = aircraft_encoder(
            observation_dim, layout, attention_context
        )
        self.network = mlp(
            self.features_extractor.features_dim, action_dim, hidden_sizes
        )
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observations):
        return self.network(local_features(self.features_extractor, observations))


class LocalAdditiveAttentionCritic(nn.Module):
    """Estimate each value using only that aircraft's local set observation."""

    def __init__(
        self,
        observation_dim,
        n_agents,
        layout,
        hidden_sizes=(256, 256),
        attention_context="all",
    ):
        super().__init__()
        self.features_extractor = aircraft_encoder(
            observation_dim, layout, attention_context
        )
        self.network = mlp(self.features_extractor.features_dim, 1, hidden_sizes)

    def forward(self, observations, active):
        observations = observations.masked_fill(~active.unsqueeze(-1), 0)
        features = local_features(self.features_extractor, observations)
        return self.network(features).squeeze(-1).masked_fill(~active, 0)


class CentralizedAdditiveAttentionCritic(nn.Module):
    """Attend over active agents, conditioned on each focal agent's local features.

    First encode each local aircraft set with additive attention. A separate
    64-feature agent encoder and additive pool summarize all active agents
    (including the focal agent). Concatenate that summary with the focal local
    features before predicting its value. No agent positional embeddings are used.
    """

    def __init__(
        self,
        observation_dim,
        n_agents,
        layout,
        hidden_sizes=(256, 256),
        attention_context="all",
    ):
        super().__init__()
        self.features_extractor = aircraft_encoder(
            observation_dim, layout, attention_context
        )
        size = self.features_extractor.features_dim
        self.agent_encoder = nn.Sequential(nn.Linear(size, 64), nn.ReLU())
        self.attention = AdditiveAttention(size)
        self.network = mlp(size + 64, 1, hidden_sizes)

    def forward(self, observations, active):
        observations = observations.masked_fill(~active.unsqueeze(-1), 0)
        features = local_features(self.features_extractor, observations)
        batch, agents, size = features.shape
        encoded = self.agent_encoder(features).masked_fill(~active.unsqueeze(-1), 0)
        # One context-conditioned pool per focal agent, with the same active set.
        pooled = self.attention(
            encoded[:, None].expand(-1, agents, -1, -1).reshape(-1, agents, 64),
            features.reshape(-1, size),
            active[:, None].expand(-1, agents, -1).reshape(-1, agents),
        ).reshape(batch, agents, 64)
        return (
            self.network(cat((features, pooled), dim=-1))
            .squeeze(-1)
            .masked_fill(~active, 0)
        )
