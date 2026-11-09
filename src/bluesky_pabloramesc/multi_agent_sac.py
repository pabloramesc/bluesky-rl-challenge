"""Masked off-policy SAC learners for one fixed-slot, continuous-action world.

SharedSAC shares local networks; ISAC owns independent local learners; MASAC
shares actors and uses centralized, focal-agent-conditioned twin Q critics.
Replay stores raw joint transitions and normalizes only when sampling.
"""

from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.utils import polyak_update
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .mappo import mlp
from .multi_agent_attention import aircraft_encoder, local_features


class AttentionSACActor(nn.Module):
    """Local state-dependent Gaussian with reparameterized, tanh-bounded actions."""

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
            self.features_extractor.features_dim, 2 * action_dim, hidden_sizes
        )

    def sample(self, observations, deterministic=False):
        mean, log_std = self.network(
            local_features(self.features_extractor, observations)
        ).chunk(2, -1)
        distribution = Normal(mean, log_std.clamp(-20, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        actions = raw.tanh()
        # Stable log(1 - tanh(raw)^2), including saturated actions.
        correction = 2 * (np.log(2) - raw - F.softplus(-2 * raw))
        log_prob = (distribution.log_prob(raw) - correction).sum(-1)
        return actions, log_prob


class LocalAttentionQ(nn.Module):
    """A shared local Q(o_i, a_i), with no cross-agent inputs."""

    def __init__(
        self,
        observation_dim,
        action_dim,
        n_agents,
        layout,
        hidden_sizes=(256, 256),
        attention_context="all",
    ):
        super().__init__()
        self.features_extractor = aircraft_encoder(
            observation_dim, layout, attention_context
        )
        self.network = mlp(
            self.features_extractor.features_dim + action_dim, 1, hidden_sizes
        )

    def forward(self, observations, actions, active, detach_peers=False):
        observations = observations.masked_fill(~active.unsqueeze(-1), 0)
        actions = actions.masked_fill(~active.unsqueeze(-1), 0)
        features = local_features(self.features_extractor, observations)
        return (
            self.network(torch.cat((features, actions), -1))
            .squeeze(-1)
            .masked_fill(~active, 0)
        )


class CentralizedAttentionQ(nn.Module):
    """Per-agent Q values attending over active observation/action embeddings.

    For actor losses, detach peers inside each focal readout so that Q_i sends
    action gradients only through a_i, even when all agents share one actor.
    """

    def __init__(
        self,
        observation_dim,
        action_dim,
        n_agents,
        layout,
        hidden_sizes=(256, 256),
        attention_context="all",
    ):
        super().__init__()
        from .entities import AdditiveAttention

        self.features_extractor = aircraft_encoder(
            observation_dim, layout, attention_context
        )
        size = self.features_extractor.features_dim + action_dim
        self.agent_encoder = nn.Sequential(nn.Linear(size, 64), nn.ReLU())
        self.attention = AdditiveAttention(size)
        self.network = mlp(size + 64, 1, hidden_sizes)

    def forward(self, observations, actions, active, detach_peers=False):
        observations = observations.masked_fill(~active.unsqueeze(-1), 0)
        actions = actions.masked_fill(~active.unsqueeze(-1), 0)
        features = torch.cat(
            (local_features(self.features_extractor, observations), actions), -1
        )
        batch, agents, size = features.shape
        encoded = self.agent_encoder(features).masked_fill(~active.unsqueeze(-1), 0)
        peers = encoded[:, None].expand(-1, agents, -1, -1)
        if detach_peers:
            focal = torch.eye(agents, dtype=torch.bool, device=actions.device)[
                None, :, :, None
            ]
            peers = torch.where(focal, peers, peers.detach())
        pooled = self.attention(
            peers.reshape(-1, agents, 64),
            features.reshape(-1, size),
            active[:, None].expand(-1, agents, -1).reshape(-1, agents),
        ).reshape(batch, agents, 64)
        return (
            self.network(torch.cat((features, pooled), -1))
            .squeeze(-1)
            .masked_fill(~active, 0)
        )


class SACPolicy(BasePolicy):
    """Separate actor, twin online/target critics, and temperature optimizers."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_agents,
        actor_kwargs=None,
        critic_kwargs=None,
        critic_class=LocalAttentionQ,
        initial_alpha=0.1,
    ):
        super().__init__(observation_space, action_space, squash_output=True)
        observation_dim, action_dim = observation_space.shape[0], action_space.shape[0]
        self.actor = AttentionSACActor(
            observation_dim, action_dim, **(actor_kwargs or {})
        )
        self.critic = nn.ModuleList(
            critic_class(observation_dim, action_dim, n_agents, **(critic_kwargs or {}))
            for _ in range(2)
        )
        self.critic_target = deepcopy(self.critic).requires_grad_(False)
        self.log_ent_coef = nn.Parameter(torch.tensor(float(np.log(initial_alpha))))
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr_schedule(1), eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_schedule(1), eps=1e-5
        )
        self.ent_coef_optimizer = torch.optim.Adam(
            [self.log_ent_coef], lr=lr_schedule(1), eps=1e-5
        )

    def _predict(self, observation, deterministic=False):
        return self.actor.sample(observation, deterministic)[0]


class IndependentSACPolicy(BasePolicy):
    """Keep every network, target, temperature, and optimizer slot-specific."""

    def __init__(
        self, observation_space, action_space, lr_schedule, n_agents, **kwargs
    ):
        super().__init__(observation_space, action_space, squash_output=True)
        self.n_agents = n_agents
        self.learners = nn.ModuleList(
            SACPolicy(
                observation_space, action_space, lr_schedule, n_agents=1, **kwargs
            )
            for _ in range(n_agents)
        )

    def _predict(self, observation, deterministic=False):
        if observation.shape[-2] != self.n_agents:
            raise ValueError("ISAC prediction requires all agent slots in fixed order")
        return torch.stack(
            [
                learner._predict(observation[..., index, :], deterministic)
                for index, learner in enumerate(self.learners)
            ],
            -2,
        )


class JointReplayBuffer:
    """CPU circular buffer whose capacity and minibatches count joint decisions."""

    def __init__(self, capacity, n_agents, observation_dim, action_dim):
        self.capacity = capacity
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
        self.data = {}

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add(self, **transition):
        if not self.data:
            self.data = {
                name: np.zeros(
                    (self.capacity, *np.shape(value)), dtype=np.asarray(value).dtype
                )
                for name, value in transition.items()
            }
        for name, value in transition.items():
            self.data[name][self.pos] = value
        self.pos += 1
        if self.pos == self.capacity:
            self.full, self.pos = True, 0

    def sample(self, batch_size, device, normalizer=None):
        indices = np.random.randint(len(self), size=batch_size)
        batch = {name: value[indices].copy() for name, value in self.data.items()}
        if normalizer is not None:
            for name in ("observations", "next_observations"):
                shape = batch[name].shape
                batch[name] = normalizer.normalize_obs(
                    batch[name].reshape(-1, shape[-1])
                ).reshape(shape)
            batch["rewards"] = normalizer.normalize_reward(batch["rewards"])
        return {
            name: torch.as_tensor(value, device=device) for name, value in batch.items()
        }


def soft_targets(rewards, terminated, next_active, next_q, next_log_prob, alpha, gamma):
    """Bootstrap time limits from the final state, but never true terminations."""
    bootstrap = next_active & ~terminated
    soft_value = (next_q - alpha * next_log_prob).masked_fill(~bootstrap, 0)
    return rewards + gamma * soft_value


class SharedSAC(BaseAlgorithm):
    """Shared local SAC with individual boundaries and raw joint replay.

    num_timesteps and learning_starts count agent slots. train_freq, buffer_size,
    and batch_size count joint decisions. Each gradient step updates every ISAC
    learner once, or the shared learner once. Rewards remain individual.
    """

    def __init__(
        self,
        policy=SACPolicy,
        env=None,
        learning_rate=3e-4,
        buffer_size=30_000,
        learning_starts=20_000,
        batch_size=64,
        train_freq=8,
        gradient_steps=32,
        gamma=0.995,
        tau=0.005,
        target_entropy=None,
        policy_kwargs=None,
        seed=0,
        device="auto",
        verbose=0,
        _init_setup_model=True,
    ):
        for name, value in (
            ("buffer_size", buffer_size),
            ("batch_size", batch_size),
            ("train_freq", train_freq),
            ("gradient_steps", gradient_steps),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(learning_starts) is not int or learning_starts < 0:
            raise ValueError("learning_starts must be a nonnegative integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be between zero and one")
        if not np.isfinite(tau) or not 0 < tau <= 1:
            raise ValueError("tau must be between zero (exclusive) and one")
        if target_entropy is not None and not np.isfinite(target_entropy):
            raise ValueError("target_entropy must be finite")
        settings = policy_kwargs or {}
        if (
            not np.isfinite(settings.get("initial_alpha", 0.1))
            or settings.get("initial_alpha", 0.1) <= 0
        ):
            raise ValueError("initial_alpha must be finite and positive")
        self.buffer_size, self.batch_size = buffer_size, batch_size
        self.learning_starts = learning_starts
        self.train_freq, self.gradient_steps = train_freq, gradient_steps
        self.gamma, self.tau = gamma, tau
        self.target_entropy = target_entropy
        self.n_agents = None
        self.active_timesteps = 0
        super().__init__(
            policy,
            env,
            learning_rate,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=verbose,
            support_multi_env=True,
            supported_action_spaces=(spaces.Box,),
        )
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        if len(self.observation_space.shape) != 1 or len(self.action_space.shape) != 1:
            raise ValueError("Joint SAC requires flat Box observations and actions")
        if (
            not np.isfinite(self.action_space.low).all()
            or not np.isfinite(self.action_space.high).all()
            or np.any(self.action_space.high <= self.action_space.low)
        ):
            raise ValueError("Joint SAC requires finite, nonempty action bounds")
        if self.n_agents is None:
            self.n_agents = self.n_envs
        if self.n_envs != self.n_agents:
            raise ValueError(
                "Checkpoint agent count does not match the joint environment"
            )
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            n_agents=self.n_agents,
            **self.policy_kwargs,
        ).to(self.device)
        if self.target_entropy is None:
            self.target_entropy = -float(self.action_space.shape[0])
        self.replay_buffer = JointReplayBuffer(
            self.buffer_size,
            self.n_agents,
            self.observation_space.shape[0],
            self.action_space.shape[0],
        )
        self._active = np.ones(self.n_agents, dtype=bool)

    def _get_torch_save_params(self):
        prefixes = (
            [f"policy.learners.{i}" for i in range(self.n_agents)]
            if isinstance(self.policy, IndependentSACPolicy)
            else ["policy"]
        )
        return ["policy"] + [
            f"{prefix}.{role}_optimizer"
            for prefix in prefixes
            for role in ("actor", "critic", "ent_coef")
        ], []

    def save_replay_buffer(self, path):
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(self, path):
        buffer = load_from_pkl(path, self.verbose)
        expected = (
            self.buffer_size,
            self.n_agents,
            self.observation_space.shape[0],
            self.action_space.shape[0],
        )
        if (
            not isinstance(buffer, JointReplayBuffer)
            or (
                buffer.capacity,
                buffer.n_agents,
                buffer.observation_dim,
                buffer.action_dim,
            )
            != expected
        ):
            raise ValueError("Replay buffer does not match this joint environment")
        self.replay_buffer = buffer

    def _store_transition(self, new_obs, rewards, dones, infos, actions):
        if not np.all(dones == dones[0]):
            raise ValueError("Joint SAC requires synchronized joint resets")
        required = {"_mappo_active", "_mappo_terminated", "_mappo_truncated"}
        if any(not required.issubset(info) for info in infos):
            raise ValueError(
                "Joint SAC needs agent boundary metadata; use make_mappo_env"
            )
        next_active = np.array([info["_mappo_active"] for info in infos], dtype=bool)
        terminated = np.array([info["_mappo_terminated"] for info in infos], dtype=bool)
        truncated = np.array([info["_mappo_truncated"] for info in infos], dtype=bool)
        normalizer = self._vec_normalize_env
        observations = self._last_original_obs if normalizer else self._last_obs
        next_observations = normalizer.get_original_obs() if normalizer else new_obs
        raw_rewards = normalizer.get_original_reward() if normalizer else rewards
        if dones[0]:
            next_observations = np.stack(
                [
                    info.get(
                        "_joint_terminal_observation", info["terminal_observation"]
                    )
                    for info in infos
                ]
            )
            if normalizer and not all(
                "_joint_terminal_observation" in info for info in infos
            ):
                next_observations = normalizer.unnormalize_obs(next_observations)
        self.replay_buffer.add(
            observations=np.asarray(observations, dtype=np.float32),
            next_observations=np.asarray(next_observations, dtype=np.float32),
            actions=actions.astype(np.float32),
            rewards=np.asarray(raw_rewards, dtype=np.float32),
            active=self._active,
            next_active=next_active | (truncated & ~terminated),
            terminated=terminated,
        )
        self._last_original_obs = (
            normalizer.get_original_obs() if normalizer else new_obs.copy()
        )
        return next_active

    def _update_learner(self, learner, batch):
        active = batch["active"]
        if not active.any():
            return None
        for optimizer in (
            learner.actor_optimizer,
            learner.critic_optimizer,
            learner.ent_coef_optimizer,
        ):
            self._update_learning_rate(optimizer)
        alpha = learner.log_ent_coef.detach().exp()
        with torch.no_grad():
            next_actions, next_log_prob = learner.actor.sample(
                batch["next_observations"].masked_fill(
                    ~batch["next_active"].unsqueeze(-1), 0
                )
            )
            next_q = torch.minimum(
                *[
                    q(batch["next_observations"], next_actions, batch["next_active"])
                    for q in learner.critic_target
                ]
            )
            target = soft_targets(
                batch["rewards"],
                batch["terminated"],
                batch["next_active"],
                next_q,
                next_log_prob,
                alpha,
                self.gamma,
            )
        values = [
            q(batch["observations"], batch["actions"], active) for q in learner.critic
        ]
        critic_loss = 0.5 * sum(F.mse_loss(q[active], target[active]) for q in values)
        learner.critic_optimizer.zero_grad()
        critic_loss.backward()
        learner.critic_optimizer.step()
        learner.critic.requires_grad_(False)
        try:
            actions, log_prob = learner.actor.sample(
                batch["observations"].masked_fill(~active.unsqueeze(-1), 0)
            )
            q_value = torch.minimum(
                *[
                    q(batch["observations"], actions, active, detach_peers=True)
                    for q in learner.critic
                ]
            )
            actor_loss = (alpha * log_prob - q_value)[active].mean()
            learner.actor_optimizer.zero_grad()
            actor_loss.backward()
            learner.actor_optimizer.step()
        finally:
            learner.critic.requires_grad_(True)
        alpha_loss = -(
            learner.log_ent_coef * (log_prob.detach() + self.target_entropy)[active]
        ).mean()
        learner.ent_coef_optimizer.zero_grad()
        alpha_loss.backward()
        learner.ent_coef_optimizer.step()
        polyak_update(
            learner.critic.parameters(), learner.critic_target.parameters(), self.tau
        )
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "ent_coef_loss": alpha_loss.item(),
            "ent_coef": alpha.item(),
        }

    def train(self, gradient_steps, batch_size):
        self.policy.set_training_mode(True)
        metrics = []
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(
                batch_size, self.device, self._vec_normalize_env
            )
            if isinstance(self.policy, IndependentSACPolicy):
                for index, learner in enumerate(self.policy.learners):
                    selected = batch["active"][:, index]
                    if selected.any():
                        own = {
                            name: value[selected, index : index + 1]
                            for name, value in batch.items()
                        }
                        metrics.append(self._update_learner(learner, own))
            else:
                metrics.append(self._update_learner(self.policy, batch))
            self._n_updates += 1
        metrics = [row for row in metrics if row is not None]
        for name in metrics[0] if metrics else ():
            self.logger.record(f"train/{name}", np.mean([row[name] for row in metrics]))
        self.logger.record("train/n_updates", self._n_updates)

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=4,
        tb_log_name="run",
        reset_num_timesteps=True,
        progress_bar=False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        if log_interval is not None and log_interval <= 0:
            raise ValueError("log_interval must be positive or None")
        if self._last_episode_starts.all():
            self._active = np.ones(self.n_agents, dtype=bool)
        if reset_num_timesteps:
            self.active_timesteps = 0
        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:
            self.policy.set_training_mode(False)
            callback.on_rollout_start()
            for _ in range(self.train_freq):
                if self.num_timesteps < self.learning_starts:
                    actions = np.random.uniform(
                        -1, 1, (self.n_agents, *self.action_space.shape)
                    ).astype(np.float32)
                else:
                    executed, _ = self.predict(self._last_obs)
                    actions = self.policy.scale_action(executed)
                actions[~self._active] = 0
                new_obs, rewards, dones, infos = self.env.step(
                    self.policy.unscale_action(actions)
                )
                dones = np.asarray(dones, dtype=bool)
                self.num_timesteps += self.n_agents
                self.active_timesteps += int(self._active.sum())
                next_active = self._store_transition(
                    new_obs, rewards, dones, infos, actions
                )
                self._last_obs, self._last_episode_starts = new_obs, dones
                self._active = (
                    np.ones(self.n_agents, dtype=bool) if dones[0] else next_active
                )
                self._update_info_buffer(infos, dones)
                self._update_current_progress_remaining(
                    self.num_timesteps, total_timesteps
                )
                callback.update_locals(locals())
                if not callback.on_step():
                    callback.on_training_end()
                    return self
            callback.on_rollout_end()
            if self.num_timesteps >= self.learning_starts:
                self.train(self.gradient_steps, self.batch_size)
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.record("replay/joint_steps", len(self.replay_buffer))
            collection = self.num_timesteps // (self.n_agents * self.train_freq)
            if log_interval is not None and collection % log_interval == 0:
                self.logger.dump(step=self.num_timesteps)
        callback.on_training_end()
        return self


class ISAC(SharedSAC):
    """Independent local actors, twin critics, targets, and temperatures per slot."""

    def __init__(self, policy=IndependentSACPolicy, env=None, **kwargs):
        super().__init__(policy=policy, env=env, **kwargs)


class MASAC(SharedSAC):
    """Shared local actor with centralized additive-attention twin Q critics."""

    def __init__(self, policy=SACPolicy, env=None, policy_kwargs=None, **kwargs):
        settings = dict(policy_kwargs or {})
        settings.setdefault("critic_class", CentralizedAttentionQ)
        super().__init__(policy=policy, env=env, policy_kwargs=settings, **kwargs)
