"""Fixed-window temporal encoders and raw-history normalization."""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .entities import EntityRunningMeanStd

TEMPORAL_METHODS = ("none", "frame_stack", "gru", "lstm", "tcn")


class HistoryWindow(VecEnvWrapper):
    """Keep oldest-to-newest raw observations plus a binary frame-validity column."""

    def __init__(self, env, history_length=8):
        if type(history_length) is not int or history_length <= 0:
            raise ValueError("history_length must be a positive integer")
        frame_space = env.observation_space
        if not isinstance(frame_space, spaces.Box) or len(frame_space.shape) != 1:
            raise ValueError("HistoryWindow requires flat Box observations")
        low = np.tile(np.append(np.minimum(frame_space.low, 0), 0), (history_length, 1))
        high = np.tile(
            np.append(np.maximum(frame_space.high, 0), 1), (history_length, 1)
        )
        space = spaces.Box(
            low.astype(frame_space.dtype),
            high.astype(frame_space.dtype),
            dtype=frame_space.dtype,
        )
        space.temporal_frame_space = frame_space
        if hasattr(frame_space, "entity_layout"):
            space.entity_layout = frame_space.entity_layout
        super().__init__(env, observation_space=space)
        self.history = np.zeros((env.num_envs, *space.shape), dtype=space.dtype)

    def reset(self):
        observations = self.venv.reset()
        self.history.fill(0)
        self.history[:, -1, :-1] = observations
        self.history[:, -1, -1] = 1
        return self.history.copy()

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        self.history = np.roll(self.history, -1, axis=1)
        for index, done in enumerate(dones):
            if done:
                if "terminal_observation" in infos[index]:
                    terminal = self.history[index].copy()
                    terminal[-1, :-1] = infos[index]["terminal_observation"]
                    terminal[-1, -1] = 1
                    infos[index]["terminal_observation"] = terminal
                self.history[index].fill(0)
            elif infos[index].get("_temporal_inactive", False):
                self.history[index].fill(0)
                continue
            self.history[index, -1, :-1] = observations[index]
            self.history[index, -1, -1] = 1
        return self.history.copy(), rewards, dones, infos


class HistoryRunningMeanStd(RunningMeanStd):
    """Share observation statistics across time; update using only the latest frame."""

    def __init__(self, space):
        super().__init__(shape=space.shape)
        frame_space = space.temporal_frame_space
        self.frame_rms = (
            EntityRunningMeanStd(frame_space.shape, frame_space.entity_layout)
            if hasattr(frame_space, "entity_layout")
            else RunningMeanStd(shape=frame_space.shape)
        )
        layout = getattr(frame_space, "entity_layout", {})
        self.binary = (
            layout.get("mask", [])
            + layout.get("graph_edges", [])
            + [space.shape[-1] - 1]
        )

    def update(self, observations):
        latest = observations[:, -1, :]
        samples = latest[latest[:, -1] > 0.5, :-1]
        if len(samples):
            self.frame_rms.update(samples)
        self.mean[:, :-1] = self.frame_rms.mean
        self.var[:, :-1] = self.frame_rms.var
        self.count = self.frame_rms.count


class HistoryVecNormalize(VecNormalize):
    """Normalize history consistently while preserving validity, presence, and edges."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.obs_rms = HistoryRunningMeanStd(env.observation_space)

    def _normalize_obs(self, obs, obs_rms):
        result = super()._normalize_obs(obs, obs_rms)
        result[..., obs_rms.binary] = obs[..., obs_rms.binary]
        return np.where(obs[..., -1:] > 0.5, result, 0)

    def _unnormalize_obs(self, obs, obs_rms):
        result = super()._unnormalize_obs(obs, obs_rms)
        result[..., obs_rms.binary] = obs[..., obs_rms.binary]
        return np.where(obs[..., -1:] > 0.5, result, 0)


class TemporalFeatures(BaseFeaturesExtractor):
    """Encode frames spatially, then summarize a bounded history with fresh state."""

    def __init__(
        self,
        observation_space,
        frame_space,
        frame_encoder_class,
        frame_encoder_kwargs,
        temporal="gru",
        history_length=8,
    ):
        if temporal not in TEMPORAL_METHODS or temporal == "none":
            raise ValueError(f"Unknown temporal encoder: {temporal}")
        if observation_space.shape != (history_length, frame_space.shape[0] + 1):
            raise ValueError("Temporal encoder and history observation shapes disagree")
        frame_encoder = frame_encoder_class(frame_space, **frame_encoder_kwargs)
        width = frame_encoder.features_dim
        features_dim = history_length * (width + 1) if temporal == "frame_stack" else 64
        super().__init__(observation_space, features_dim)
        self.frame_encoder = frame_encoder
        self.temporal, self.history_length = temporal, history_length
        if temporal in ("gru", "lstm"):
            self.recurrent = {"gru": nn.GRU, "lstm": nn.LSTM}[temporal](
                width, 64, batch_first=True
            )
        if temporal == "tcn":
            self.input_projection = nn.Linear(width, 64)
            layers, receptive_field, dilation = [], 1, 1
            while receptive_field < max(2, history_length):
                layers.append(nn.Conv1d(64, 64, kernel_size=3, dilation=dilation))
                receptive_field += 2 * dilation
                dilation *= 2
            self.convolutions = nn.ModuleList(layers)

    def forward(self, observations):
        batch, length, _ = observations.shape
        valid = observations[..., -1] > 0.5
        frames = observations[..., :-1].masked_fill(~valid.unsqueeze(-1), 0)
        encoded = self.frame_encoder(frames.reshape(batch * length, -1)).reshape(
            batch, length, -1
        )
        encoded = encoded.masked_fill(~valid.unsqueeze(-1), 0)
        if self.temporal == "frame_stack":
            return torch.cat([encoded.flatten(1), valid.to(encoded.dtype)], dim=1)
        if self.temporal in ("gru", "lstm"):
            # Move valid frames to the start, preserving chronological order.
            order = (~valid).to(torch.int64).argsort(dim=1, stable=True)
            compact = encoded.gather(1, order.unsqueeze(-1).expand_as(encoded))
            lengths = valid.sum(1).clamp(min=1).cpu()
            packed = pack_padded_sequence(
                compact, lengths, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.recurrent(packed)
            if self.temporal == "lstm":
                hidden = hidden[0]
            summary = hidden[-1]
        else:
            sequence = (
                self.input_projection(encoded)
                .masked_fill(~valid.unsqueeze(-1), 0)
                .transpose(1, 2)
            )
            for convolution in self.convolutions:
                sequence = torch.relu(
                    convolution(
                        nn.functional.pad(sequence, (2 * convolution.dilation[0], 0))
                    )
                )
                sequence = sequence.masked_fill(~valid.unsqueeze(1), 0)
            summary = sequence[:, :, -1]
        return summary.masked_fill(~valid.any(1, keepdim=True), 0)
