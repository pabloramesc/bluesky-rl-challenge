"""Custom MAPPO updates with decentralized actors and a centralized value function.

SB3's on-policy base supplies callbacks, logging, prediction, and serialization.
Rollout collection, agent boundaries, GAE, networks, and PPO losses live here.
One vector slot is one agent in a single joint environment, not an independent
environment. Batches retain the joint agent axis for interchangeable critics.
"""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from torch import nn
from torch.distributions import Normal


def mlp(input_dim, output_dim, hidden_sizes):
    """Build a tanh MLP with a linear output layer."""
    layers = []
    for width in hidden_sizes:
        if type(width) is not int or width <= 0:
            raise ValueError("hidden_sizes must contain positive integers")
        layers.extend((nn.Linear(input_dim, width), nn.Tanh()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    """Map each local observation to Gaussian action means, sharing parameters."""

    def __init__(self, observation_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.network = mlp(observation_dim, action_dim, hidden_sizes)
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observations):
        """Accept [..., observation_dim] and return [..., action_dim]."""
        return self.network(observations)


class CentralizedMLPCritic(nn.Module):
    """Predict one value per fixed agent slot from masked joint observations."""

    def __init__(self, observation_dim, n_agents, hidden_sizes=(256, 256)):
        super().__init__()
        self.network = mlp(n_agents * (observation_dim + 1), n_agents, hidden_sizes)

    def forward(self, observations, active):
        """Map [batch, agents, features] and [batch, agents] to per-agent values.

        Fixed output slots distinguish agents. The explicit mask distinguishes
        departed agents from active aircraft with zero-valued observations.
        """
        observations = observations.masked_fill(~active.unsqueeze(-1), 0)
        state = torch.cat((observations.flatten(1), active.float()), dim=-1)
        return self.network(state)


class MAPPOPolicy(BasePolicy):
    """Separate actor and critic modules with small, architecture-neutral contracts.

    Replace actor_class and/or critic_class (and their kwargs) to add attention,
    Transformer, or graph encoders. Actors see only local observations; critics
    receive the complete joint tensor and active-agent mask. Prediction requires
    only the actor and works on single observations or arbitrary local batches.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_agents,
        actor_class=MLPActor,
        critic_class=CentralizedMLPCritic,
        actor_kwargs=None,
        critic_kwargs=None,
        log_std_init=-0.5,
    ):
        super().__init__(observation_space, action_space)
        observation_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self.actor = actor_class(observation_dim, action_dim, **(actor_kwargs or {}))
        self.critic = critic_class(observation_dim, n_agents, **(critic_kwargs or {}))
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr_schedule(1), eps=1e-5
        )

    def distribution(self, observations):
        """Use an unsquashed Gaussian; collection clips only the executed actions."""
        return Normal(self.actor(observations), self.log_std.exp())

    def _predict(self, observation, deterministic=False):
        distribution = self.distribution(observation)
        return distribution.mean if deterministic else distribution.sample()

    def evaluate_actions(self, observations, active, actions):
        """Evaluate joint minibatches without removing their agent dimension."""
        distribution = self.distribution(observations)
        return (
            self.critic(observations, active),
            distribution.log_prob(actions).sum(-1),
            distribution.entropy().sum(-1),
        )


class JointRolloutBuffer:
    """Keep joint timesteps together and exclude padding from GAE and losses."""

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.transitions = []
        self.data = {}

    def add(self, **transition):
        self.transitions.append(
            {name: np.array(value, copy=True) for name, value in transition.items()}
        )

    def compute_returns(self, gamma, gae_lambda):
        """Bootstrap truncations but stop traces at every individual boundary."""
        data = {
            name: np.stack([step[name] for step in self.transitions])
            for name in self.transitions[0]
        }
        advantages = np.zeros_like(data["rewards"], dtype=np.float32)
        following_advantage = np.zeros_like(advantages[0])
        for step in reversed(range(len(self.transitions))):
            delta = (
                data["rewards"][step]
                + gamma * data["next_values"][step]
                - data["old_values"][step]
            )
            following_advantage = (
                delta
                + gamma * gae_lambda * data["continuations"][step] * following_advantage
            ) * data["active"][step]
            advantages[step] = following_advantage
        data["advantages"] = advantages
        data["returns"] = advantages + data["old_values"]
        self.data = {
            name: torch.as_tensor(value, device=self.device)
            for name, value in data.items()
        }
        self.transitions.clear()

    def batches(self, joint_batch_size):
        """Shuffle joint timesteps; never split a critic's joint observation."""
        indices = torch.randperm(len(self.data["observations"]), device=self.device)
        for start in range(0, len(indices), joint_batch_size):
            selected = indices[start : start + joint_batch_size]
            yield {name: value[selected] for name, value in self.data.items()}


def clipped_losses(batch, values, log_probs, entropy, clip_range, value_clip_range):
    """Calculate PPO losses and diagnostics using only active agent transitions."""
    active = batch["active"]
    log_ratio = (log_probs - batch["old_log_probs"])[active]
    ratio = log_ratio.exp()
    advantages = batch["advantages"][active]
    policy_loss = -torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip_range, 1 + clip_range) * advantages
    ).mean()
    values = values[active]
    errors = (values - batch["returns"][active]).square()
    if value_clip_range is not None:
        old_values = batch["old_values"][active]
        clipped_values = old_values + (values - old_values).clamp(
            -value_clip_range, value_clip_range
        )
        errors = torch.maximum(
            errors, (clipped_values - batch["returns"][active]).square()
        )
    return {
        "policy_loss": policy_loss,
        "value_loss": 0.5 * errors.mean(),
        "entropy_loss": -entropy[active].mean(),
        "approx_kl": ((ratio - 1) - log_ratio).mean().detach(),
        "clip_fraction": ((ratio - 1).abs() > clip_range).float().mean().detach(),
    }


class MAPPO(OnPolicyAlgorithm):
    """Parameter-sharing MAPPO for one fixed-slot, continuous-action joint world.

    n_steps and batch_size count joint timesteps. num_timesteps counts all agent
    slots, including padding, consistently with the existing SB3 experiment CLI.
    Individual rewards train per-agent values; rewards are not averaged by team.
    """

    def __init__(
        self,
        policy=MAPPOPolicy,
        env=None,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        value_clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        policy_kwargs=None,
        seed=0,
        device="auto",
        verbose=0,
        _init_setup_model=True,
    ):
        for name, value in (
            ("n_steps", n_steps),
            ("batch_size", batch_size),
            ("n_epochs", n_epochs),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        for name, value in (
            ("learning_rate", learning_rate),
            ("clip_range", clip_range),
            ("value_clip_range", value_clip_range),
            ("max_grad_norm", max_grad_norm),
            ("target_kl", target_kl),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (("ent_coef", ent_coef), ("vf_coef", vf_coef)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.target_kl = target_kl
        self.n_agents = None
        self.active_timesteps = 0
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=verbose,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=(spaces.Box,),
        )

    def _setup_model(self):
        if (
            not isinstance(self.observation_space, spaces.Box)
            or len(self.observation_space.shape) != 1
        ):
            raise ValueError("MAPPO requires flat Box observations")
        if len(self.action_space.shape) != 1:
            raise ValueError("MAPPO requires one-dimensional Box actions")
        if self.n_agents is None:
            self.n_agents = self.n_envs
        if self.n_envs != self.n_agents:
            raise ValueError(
                "Checkpoint agent count does not match the joint environment"
            )
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.rollout_buffer = JointRolloutBuffer(self.device)
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            n_agents=self.n_agents,
            **self.policy_kwargs,
        ).to(self.device)
        self._active = np.ones(self.n_agents, dtype=bool)

    def _setup_learn(self, *args, **kwargs):
        result = super()._setup_learn(*args, **kwargs)
        if self._last_episode_starts.all():
            self._active = np.ones(self.n_agents, dtype=bool)
        return result

    def _values(self, observations, active):
        observations = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        active = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            return (
                self.policy.critic(observations.unsqueeze(0), active.unsqueeze(0))[0]
                .cpu()
                .numpy()
            )

    def _next_values(self, observations, dones, infos):
        if not np.all(dones == dones[0]):
            raise ValueError(
                "MAPPO needs synchronized joint resets, not independent vector environments"
            )
        required = {"_mappo_active", "_mappo_terminated", "_mappo_truncated"}
        if any(not required.issubset(info) for info in infos):
            raise ValueError("MAPPO needs agent boundary metadata; use make_mappo_env")
        active = np.array([info["_mappo_active"] for info in infos], dtype=bool)
        terminated = np.array([info["_mappo_terminated"] for info in infos], dtype=bool)
        truncated = np.array([info["_mappo_truncated"] for info in infos], dtype=bool)
        if dones[0]:
            # VecEnv auto-resets; its terminal observations have already been
            # normalized by VecNormalize using the current running statistics.
            observations = np.stack([info["terminal_observation"] for info in infos])
        values = self._values(observations, active | (truncated & ~terminated))
        values[terminated] = 0
        return values, active

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        for _ in range(n_rollout_steps):
            observations = torch.as_tensor(
                self._last_obs, dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                distribution = self.policy.distribution(observations)
                sampled_actions = distribution.sample()
                log_probs = distribution.log_prob(sampled_actions).sum(-1).cpu().numpy()
                actions = sampled_actions.cpu().numpy()
            values = self._values(self._last_obs, self._active)
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )
            clipped_actions[~self._active] = np.clip(
                0, self.action_space.low, self.action_space.high
            )
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            dones = np.asarray(dones, dtype=bool)
            self.num_timesteps += env.num_envs
            self.active_timesteps += int(self._active.sum())
            next_values, next_active = self._next_values(new_obs, dones, infos)
            rollout_buffer.add(
                observations=self._last_obs,
                actions=actions,
                rewards=rewards,
                active=self._active,
                old_values=values,
                old_log_probs=log_probs,
                next_values=next_values,
                continuations=next_active & ~dones,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._active = (
                np.ones(self.n_agents, dtype=bool) if dones[0] else next_active
            )
            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())
            if not callback.on_step():
                return False
        rollout_buffer.compute_returns(self.gamma, self.gae_lambda)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self):
        """Run masked, clipped PPO updates over joint minibatches."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        data = self.rollout_buffer.data
        advantages = data["advantages"][data["active"]]
        if advantages.numel() > 1:
            data["advantages"][data["active"]] = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        metrics = []
        stop = False
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.batches(self.batch_size):
                if not batch["active"].any():
                    continue
                values, log_probs, entropy = self.policy.evaluate_actions(
                    batch["observations"], batch["active"], batch["actions"]
                )
                losses = clipped_losses(
                    batch,
                    values,
                    log_probs,
                    entropy,
                    self.clip_range,
                    self.value_clip_range,
                )
                metrics.append({name: value.item() for name, value in losses.items()})
                if (
                    self.target_kl is not None
                    and losses["approx_kl"] > 1.5 * self.target_kl
                ):
                    stop = True
                    break
                loss = (
                    losses["policy_loss"]
                    + self.vf_coef * losses["value_loss"]
                    + self.ent_coef * losses["entropy_loss"]
                )
                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if stop:
                break
        for name in metrics[0] if metrics else ():
            self.logger.record(f"train/{name}", np.mean([row[name] for row in metrics]))
        self.logger.record("train/n_updates", self._n_updates)
