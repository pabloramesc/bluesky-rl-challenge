"""Aircraft and obstacle set encoders with slot-shared normalization."""

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn

ATTENTION_POOLING = ("additive_attention", "ego_attention")
POOLING_METHODS = (
    "mean",
    "max",
    "mean_max",
    *ATTENTION_POOLING,
    "pma",
    "context_token",
)
CONTEXT_TOKEN_ARCHITECTURES = (
    "context_entity_transformer",
    "dual_entity_transformer",
)
TRANSFORMER_ARCHITECTURES = ("entity_transformer", *CONTEXT_TOKEN_ARCHITECTURES)
ATTENTION_CONTEXTS = ("ownship_goal", "all")
GNN_TYPES = ("gcn", "gat")
GRAPH_TYPES = ("fully_connected", "distance", "knn")

AIRCRAFT_FIELDS = (
    "intruder_distance",
    "x_r",
    "y_r",
    "vx_r",
    "vy_r",
    "cos_track",
    "sin_track",
    "tcpa",
    "dcpa",
    "closing_rate",
)


def entity_layout(space):
    """Locate aircraft, optional obstacles, masks, and context in a flat Dict."""
    offsets = {}
    offset = 0
    for key, field in space.spaces.items():
        size = int(np.prod(field.shape))
        offsets[key] = list(range(offset, offset + size))
        offset += size
    fields = [offsets[key] for key in AIRCRAFT_FIELDS if key in offsets]
    mask = offsets["intruder_present"]
    if not fields or any(len(field) != len(mask) for field in fields):
        raise ValueError("Aircraft fields must have the same number of slots")
    aircraft = np.array(fields).T.tolist()
    excluded = {index for field in fields for index in field} | set(mask)
    excluded.update(offsets.get("graph_adjacency", []))
    obstacle_layout = {}
    if "obstacle_present" in offsets:
        obstacle_mask = offsets["obstacle_present"]
        obstacle_fields = [
            offsets[key]
            for key in (
                "obstacle_distance",
                "obstacle_cos_bearing",
                "obstacle_sin_bearing",
                "obstacle_x",
                "obstacle_y",
                "obstacle_radius",
            )
            if key in offsets
        ]
        if not obstacle_fields or any(
            len(field) != len(obstacle_mask) for field in obstacle_fields
        ):
            raise ValueError("Obstacle fields must have the same number of slots")
        excluded.update(obstacle_mask)
        obstacle_indices = {index for field in obstacle_fields for index in field}
        obstacle_layout = {
            "obstacles": np.array(obstacle_fields).T.tolist(),
            "obstacle_mask": obstacle_mask,
            "state_context": [
                index
                for index in range(offset)
                if index not in excluded and index not in obstacle_indices
            ],
        }
    return {
        "aircraft": aircraft,
        "mask": mask,
        "context": [index for index in range(offset) if index not in excluded],
        "graph_edges": offsets.get("graph_adjacency", []),
        "ownship_goal": [
            index
            for key, indices in offsets.items()
            if key.startswith(("own_", "goal_", "waypoint_"))
            or key in ("airspeed", "cos_drift", "sin_drift")
            for index in indices
        ],
        **obstacle_layout,
    }


class AdditiveAttention(nn.Module):
    """Score aircraft against context and return a masked weighted average."""

    def __init__(self, context_size):
        super().__init__()
        self.entity_projection = nn.Linear(64, 64)
        self.context_projection = nn.Linear(context_size, 64, bias=False)
        self.score = nn.Linear(64, 1, bias=False)

    def weights(self, encoded, context, present):
        """Return normalized weights, or all zeros for empty traffic."""
        hidden = self.entity_projection(encoded) + self.context_projection(
            context
        ).unsqueeze(1)
        scores = self.score(torch.tanh(hidden)).squeeze(-1)
        scores = scores.masked_fill(~present, -torch.inf)
        # Avoid softmax over all -inf, which would produce NaNs and bad gradients.
        scores = torch.where(present.any(1, keepdim=True), scores, 0)
        return torch.softmax(scores, dim=1).masked_fill(~present, 0)

    def forward(self, encoded, context, present):
        weights = self.weights(encoded, context, present)
        return (encoded * weights.unsqueeze(-1)).sum(1)


class EntityMLP(BaseFeaturesExtractor):
    """Encode each aircraft with shared weights, then pool occupied slots."""

    supported_pooling = tuple(
        method for method in POOLING_METHODS if method not in ("pma", "context_token")
    )

    def __init__(
        self,
        observation_space,
        layout,
        pooling="mean_max",
        attention_context="ownship_goal",
    ):
        if attention_context not in ATTENTION_CONTEXTS:
            raise ValueError(f"Unknown attention context: {attention_context}")
        if pooling not in self.supported_pooling:
            raise ValueError(f"Unknown pooling: {pooling}")
        size = 128 if pooling == "mean_max" else 64
        super().__init__(observation_space, size + len(layout["context"]))
        self.pooling = pooling
        self.register_buffer("aircraft_indices", torch.tensor(layout["aircraft"]))
        self.register_buffer("mask_indices", torch.tensor(layout["mask"]))
        self.register_buffer(
            "context_indices", torch.tensor(layout["context"], dtype=torch.long)
        )
        self.encoder = nn.Sequential(
            nn.Linear(len(layout["aircraft"][0]), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        if pooling in ATTENTION_POOLING:
            self.attention_context = attention_context
            query_indices = (
                layout["context"]
                if attention_context == "all"
                else layout["ownship_goal"]
            )
            if not query_indices:
                raise ValueError(
                    "Attention requires observation features for its context"
                )
            self.register_buffer(
                "query_indices", torch.tensor(query_indices, dtype=torch.long)
            )
            attention_type = (
                AdditiveAttention
                if pooling == "additive_attention"
                else EgoCrossAttention
            )
            self.attention = attention_type(len(query_indices))

    def encode_entities(self, entities, present):
        """Encode aircraft independently; subclasses may add interactions."""
        return self.encoder(entities)

    def forward(self, observations):
        """Return context and a finite, permutation-invariant traffic summary."""
        present = observations[:, self.mask_indices] > 0.5
        entities = observations[:, self.aircraft_indices]
        entities = entities.masked_fill(~present.unsqueeze(-1), 0)
        encoded = self.encode_entities(entities, present)
        return self.pool_entities(observations, encoded, present)

    def pool_entities(self, observations, encoded, present):
        """Combine node embeddings with context using the selected readout."""
        mask = present.unsqueeze(-1)
        pooled = []
        if self.pooling == "pma":
            pooled.append(self.pma(encoded, present))
        if self.pooling in ATTENTION_POOLING:
            pooled.append(
                self.attention(encoded, observations[:, self.query_indices], present)
            )
        if self.pooling in ("mean", "mean_max"):
            pooled.append((encoded * mask).sum(1) / mask.sum(1).clamp(min=1))
        if self.pooling in ("max", "mean_max"):
            maximum = encoded.masked_fill(~mask, -torch.inf).amax(1)
            pooled.append(torch.where(present.any(1, keepdim=True), maximum, 0))
        return torch.cat([observations[:, self.context_indices], *pooled], dim=1)


def attention_padding_mask(present):
    """Expose one dummy slot for empty rows so attention stays finite."""
    padding = ~present.clone()
    padding[:, 0] &= present.any(dim=1)
    return padding


class EgoCrossAttention(nn.Module):
    """Read the aircraft set with one context-conditioned, four-head query."""

    def __init__(self, context_size):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(context_size, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        self.attention = nn.MultiheadAttention(64, 4, dropout=0, batch_first=True)
        self.attention_norm = nn.LayerNorm(64)
        self.feed_forward = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.output_norm = nn.LayerNorm(64)

    def forward(self, encoded, context, present):
        """Return a 64-feature summary, or zero for observations without traffic."""
        query = self.query_encoder(context).unsqueeze(1)
        values = encoded.masked_fill(~present.unsqueeze(-1), 0)
        attended, _ = self.attention(
            query,
            values,
            values,
            key_padding_mask=attention_padding_mask(present),
            need_weights=False,
        )
        summary = self.attention_norm(query + attended)
        summary = self.output_norm(summary + self.feed_forward(summary)).squeeze(1)
        return summary.masked_fill(~present.any(dim=1, keepdim=True), 0)


class PoolingMultiheadAttention(nn.Module):
    """Pool a set with one learned seed and a residual attention block."""

    def __init__(self):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 1, 64))
        nn.init.xavier_uniform_(self.seed)
        self.input_projection = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.attention = nn.MultiheadAttention(64, 4, dropout=0, batch_first=True)
        self.attention_norm = nn.LayerNorm(64)
        self.feed_forward = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.output_norm = nn.LayerNorm(64)

    def forward(self, encoded, present):
        """Produce one 64-feature summary; empty traffic produces zeros."""
        values = self.input_projection(encoded).masked_fill(~present.unsqueeze(-1), 0)
        query = self.seed.expand(encoded.shape[0], -1, -1)
        attended, _ = self.attention(
            query,
            values,
            values,
            key_padding_mask=attention_padding_mask(present),
            need_weights=False,
        )
        summary = self.attention_norm(query + attended)
        summary = self.output_norm(summary + self.feed_forward(summary)).squeeze(1)
        return summary.masked_fill(~present.any(dim=1, keepdim=True), 0)


def transformer_blocks(layers, *, norm_first=False):
    """Build independently initialized spatial attention blocks."""
    if type(layers) is not int or layers <= 0:
        raise ValueError("transformer_layers must be a positive integer")
    return nn.ModuleList(
        [
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=0,
                activation="relu",
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(layers)
        ]
    )


class EntityTransformer(EntityMLP):
    """Refine aircraft embeddings with self-attention before set pooling."""

    supported_pooling = tuple(p for p in POOLING_METHODS if p != "context_token")

    def __init__(
        self,
        observation_space,
        layout,
        pooling="mean_max",
        attention_context="ownship_goal",
        transformer_layers=1,
    ):
        if not isinstance(transformer_layers, int) or transformer_layers <= 0:
            raise ValueError("transformer_layers must be a positive integer")
        super().__init__(observation_space, layout, pooling, attention_context)
        self.transformer_layers = transformer_layers
        self.transformer = transformer_blocks(transformer_layers)
        if pooling == "pma":
            self.pma = PoolingMultiheadAttention()

    def encode_entities(self, entities, present):
        """Use bidirectional attention without position embeddings or causal masks."""
        encoded = self.encoder(entities).masked_fill(~present.unsqueeze(-1), 0)
        padding = attention_padding_mask(present)
        for layer in self.transformer:
            encoded = layer(encoded, src_key_padding_mask=padding)
            encoded = encoded.masked_fill(~present.unsqueeze(-1), 0)
        return encoded


class ContextEntityTransformer(EntityMLP):
    """Jointly attend over full context and aircraft; read the context token."""

    supported_pooling = ("context_token",)

    def __init__(
        self,
        observation_space,
        layout,
        pooling="context_token",
        transformer_layers=1,
    ):
        if not layout["context"]:
            raise ValueError("Context Transformer requires observation context")
        super().__init__(observation_space, layout, pooling)
        self.transformer_layers = transformer_layers
        self.context_encoder = nn.Sequential(
            nn.Linear(len(layout["context"]), 64), nn.ReLU(), nn.Linear(64, 64)
        )
        self.transformer = transformer_blocks(transformer_layers, norm_first=True)
        self.output_norm = nn.LayerNorm(64)

    def encode_tokens(self, observations):
        """Keep context visible even when every aircraft slot is absent."""
        context = observations[:, self.context_indices]
        present = observations[:, self.mask_indices] > 0.5
        aircraft = observations[:, self.aircraft_indices]
        aircraft = aircraft.masked_fill(~present.unsqueeze(-1), 0)
        tokens = torch.cat(
            [self.context_encoder(context).unsqueeze(1), self.encoder(aircraft)], dim=1
        )
        padding = torch.cat([torch.zeros_like(present[:, :1]), ~present], dim=1)
        for layer in self.transformer:
            tokens = layer(tokens, src_key_padding_mask=padding)
            tokens = tokens.masked_fill(padding.unsqueeze(-1), 0)
        return self.output_norm(tokens).masked_fill(padding.unsqueeze(-1), 0)

    def forward(self, observations):
        """Return raw context plus its traffic-conditioned token representation."""
        return torch.cat(
            [
                observations[:, self.context_indices],
                self.encode_tokens(observations)[:, 0],
            ],
            dim=1,
        )


class ContextSetEncoder(nn.Module):
    """Encode an unordered entity set and read its ownship/goal context token."""

    def __init__(self, entity_size, context_size, layers):
        super().__init__()
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        self.transformer = transformer_blocks(layers, norm_first=True)
        self.output_norm = nn.LayerNorm(64)

    def forward(self, entities, present, context):
        entities = entities.masked_fill(~present.unsqueeze(-1), 0)
        tokens = torch.cat(
            [self.context_encoder(context).unsqueeze(1), self.entity_encoder(entities)],
            dim=1,
        )
        padding = torch.cat([present.new_zeros((present.shape[0], 1)), ~present], dim=1)
        for layer in self.transformer:
            tokens = layer(tokens, src_key_padding_mask=padding)
            tokens = tokens.masked_fill(padding.unsqueeze(-1), 0)
        return self.output_norm(tokens[:, 0])


class DualEntityTransformer(BaseFeaturesExtractor):
    """Fuse independent traffic/obstacle latents with raw ownship and context."""

    def __init__(
        self,
        observation_space,
        layout,
        pooling="context_token",
        transformer_layers=1,
    ):
        if pooling != "context_token":
            raise ValueError("Dual Transformer requires context_token pooling")
        if not layout.get("obstacles") or not layout.get("obstacle_mask"):
            raise ValueError("Dual Transformer requires obstacle observations and mask")
        if not layout.get("ownship_goal"):
            raise ValueError("Dual Transformer requires ownship/goal context")
        super().__init__(observation_space, len(layout["state_context"]) + 128)
        self.pooling = pooling
        self.transformer_layers = transformer_layers
        for name, indices in (
            ("aircraft_indices", layout["aircraft"]),
            ("mask_indices", layout["mask"]),
            ("obstacle_indices", layout["obstacles"]),
            ("obstacle_mask_indices", layout["obstacle_mask"]),
            ("context_indices", layout["state_context"]),
            ("query_indices", layout["ownship_goal"]),
        ):
            self.register_buffer(name, torch.tensor(indices, dtype=torch.long))
        self.traffic_encoder = ContextSetEncoder(
            len(layout["aircraft"][0]), len(layout["ownship_goal"]), transformer_layers
        )
        self.obstacle_encoder = ContextSetEncoder(
            len(layout["obstacles"][0]), len(layout["ownship_goal"]), transformer_layers
        )

    def forward(self, observations):
        context = observations[:, self.query_indices]
        traffic = self.traffic_encoder(
            observations[:, self.aircraft_indices],
            observations[:, self.mask_indices] > 0.5,
            context,
        )
        obstacles = self.obstacle_encoder(
            observations[:, self.obstacle_indices],
            observations[:, self.obstacle_mask_indices] > 0.5,
            context,
        )
        return torch.cat(
            [observations[:, self.context_indices], traffic, obstacles], dim=1
        )


class GraphConvolution(nn.Module):
    """GCN update with symmetric degree normalization and self-loop adjacency."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(64, 64, bias=False)
        self.bias = nn.Parameter(torch.zeros(64))

    def forward(self, encoded, adjacency):
        degree_scale = adjacency.sum(-1).clamp(min=1).rsqrt()
        normalized = (
            adjacency.to(encoded.dtype)
            * degree_scale.unsqueeze(-1)
            * degree_scale.unsqueeze(-2)
        )
        return torch.relu(normalized @ self.projection(encoded) + self.bias)


class GraphAttention(nn.Module):
    """Original multi-head GAT with concatenated heads and masked neighbors."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(64, 64, bias=False)
        self.source_score = nn.Parameter(torch.empty(4, 16))
        self.target_score = nn.Parameter(torch.empty(4, 16))
        nn.init.xavier_uniform_(self.source_score)
        nn.init.xavier_uniform_(self.target_score)
        self.bias = nn.Parameter(torch.zeros(64))

    def forward(self, encoded, adjacency):
        batch, nodes, _ = encoded.shape
        values = self.projection(encoded).reshape(batch, nodes, 4, 16).transpose(1, 2)
        source = (values * self.source_score[None, :, None, :]).sum(-1)
        target = (values * self.target_score[None, :, None, :]).sum(-1)
        scores = nn.functional.leaky_relu(
            source.unsqueeze(-1) + target.unsqueeze(-2), 0.2
        )
        edges = adjacency.unsqueeze(1)
        scores = scores.masked_fill(~edges, -torch.inf)
        scores = torch.where(edges.any(-1, keepdim=True), scores, 0)
        weights = torch.softmax(scores, dim=-1).masked_fill(~edges, 0)
        combined = (weights @ values).transpose(1, 2).reshape(batch, nodes, 64)
        return nn.functional.elu(combined + self.bias)


class EntityGNN(EntityMLP):
    """Pass messages between observed aircraft, then reuse invariant set pooling."""

    supported_pooling = tuple(p for p in POOLING_METHODS if p != "context_token")

    def __init__(
        self,
        observation_space,
        layout,
        pooling="mean_max",
        attention_context="ownship_goal",
        gnn_type="gat",
        gnn_layers=1,
        graph="fully_connected",
        graph_radius_nm=25.0,
        graph_k=3,
    ):
        if gnn_type not in GNN_TYPES:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        if graph not in GRAPH_TYPES:
            raise ValueError(f"Unknown graph: {graph}")
        if not isinstance(gnn_layers, int) or gnn_layers <= 0:
            raise ValueError("gnn_layers must be a positive integer")
        if not np.isfinite(graph_radius_nm) or graph_radius_nm <= 0:
            raise ValueError("graph_radius_nm must be finite and positive")
        if type(graph_k) is not int or graph_k <= 0:
            raise ValueError("graph_k must be a positive integer")
        super().__init__(observation_space, layout, pooling, attention_context)
        self.graph_k = graph_k
        self.gnn_type, self.gnn_layers = gnn_type, gnn_layers
        self.graph, self.graph_radius_nm = graph, graph_radius_nm
        if graph != "fully_connected":
            nodes = len(layout["aircraft"])
            if len(layout.get("graph_edges", [])) != nodes * nodes:
                raise ValueError(
                    "Sparse graphs require raw-geometry adjacency observations"
                )
            self.register_buffer(
                "edge_indices", torch.tensor(layout["graph_edges"], dtype=torch.long)
            )
        layer_type = GraphConvolution if gnn_type == "gcn" else GraphAttention
        self.graph_layers = nn.ModuleList([layer_type() for _ in range(gnn_layers)])
        if pooling == "pma":
            self.pma = PoolingMultiheadAttention()

    def encode_entities(self, entities, present):
        """Use a fully connected graph unless raw geometry edges are supplied."""
        adjacency = present.unsqueeze(-1) & present.unsqueeze(-2)
        return self.message_passing(entities, present, adjacency)

    def message_passing(self, entities, present, adjacency):
        encoded = self.encoder(entities).masked_fill(~present.unsqueeze(-1), 0)
        for layer in self.graph_layers:
            encoded = layer(encoded, adjacency).masked_fill(~present.unsqueeze(-1), 0)
        return encoded

    def forward(self, observations):
        if self.graph == "fully_connected":
            return super().forward(observations)
        # Sparse graph edges bypass normalization and are excluded from policy context.
        present = observations[:, self.mask_indices] > 0.5
        adjacency = (
            observations[:, self.edge_indices].reshape(
                -1, present.shape[1], present.shape[1]
            )
            > 0.5
        )
        adjacency &= present.unsqueeze(-1) & present.unsqueeze(-2)
        diagonal = torch.eye(present.shape[1], device=present.device, dtype=torch.bool)
        adjacency |= diagonal.unsqueeze(0) & present.unsqueeze(-1)
        entities = observations[:, self.aircraft_indices].masked_fill(
            ~present.unsqueeze(-1), 0
        )
        encoded = self.message_passing(entities, present, adjacency)
        return self.pool_entities(observations, encoded, present)


class EntityRunningMeanStd(RunningMeanStd):
    """Share each entity feature's statistics across occupied slots only."""

    def __init__(self, shape, layout):
        super().__init__(shape=shape)
        self.layout = layout
        self.context_rms = RunningMeanStd(
            shape=(len(layout.get("state_context", layout["context"])),)
        )
        self.aircraft_rms = RunningMeanStd(shape=(len(layout["aircraft"][0]),))
        if "obstacles" in layout:
            self.obstacle_rms = RunningMeanStd(shape=(len(layout["obstacles"][0]),))

    def update(self, observations):
        """Update context normally and aircraft using pooled occupied samples."""
        context = self.layout.get("state_context", self.layout["context"])
        aircraft = np.asarray(self.layout["aircraft"])
        present = observations[:, self.layout["mask"]] > 0.5
        self.context_rms.update(observations[:, context])
        samples = observations[:, aircraft][present]
        if len(samples):
            self.aircraft_rms.update(samples)
        self.mean[context] = self.context_rms.mean
        self.var[context] = self.context_rms.var
        self.mean[aircraft] = self.aircraft_rms.mean
        self.var[aircraft] = self.aircraft_rms.var
        if "obstacles" in self.layout:
            obstacles = np.asarray(self.layout["obstacles"])
            present = observations[:, self.layout["obstacle_mask"]] > 0.5
            samples = observations[:, obstacles][present]
            if len(samples):
                self.obstacle_rms.update(samples)
            self.mean[obstacles] = self.obstacle_rms.mean
            self.var[obstacles] = self.obstacle_rms.var
        self.count = self.context_rms.count


class EntityVecNormalize(VecNormalize):
    """Persist shared slot statistics while preserving binary presence masks."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.obs_rms = EntityRunningMeanStd(
            env.observation_space.shape, env.observation_space.entity_layout
        )

    def _normalize_obs(self, obs, obs_rms):
        result = super()._normalize_obs(obs, obs_rms)
        binary = (
            obs_rms.layout["mask"]
            + obs_rms.layout.get("obstacle_mask", [])
            + obs_rms.layout.get("graph_edges", [])
        )
        result[..., binary] = obs[..., binary]
        return result

    def _unnormalize_obs(self, obs, obs_rms):
        result = super()._unnormalize_obs(obs, obs_rms)
        binary = (
            obs_rms.layout["mask"]
            + obs_rms.layout.get("obstacle_mask", [])
            + obs_rms.layout.get("graph_edges", [])
        )
        result[..., binary] = obs[..., binary]
        return result
