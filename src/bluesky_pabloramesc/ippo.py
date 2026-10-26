"""Separate PPO learners and a shared local-critic control for one joint world."""

import numpy as np
import torch
from stable_baselines3.common.policies import BasePolicy
from torch import nn
from torch.distributions import Normal

from .mappo import MAPPO, MAPPOPolicy, MLPActor, clipped_losses, mlp


class LocalMLPCritic(nn.Module):
    """Apply the same local value function independently along the agent axis."""

    def __init__(self, observation_dim, n_agents, hidden_sizes=(256, 256)):
        super().__init__()
        self.network = mlp(observation_dim, 1, hidden_sizes)

    def forward(self, observations, active):
        observations = observations.masked_fill(~active.unsqueeze(-1), 0)
        return self.network(observations).squeeze(-1).masked_fill(~active, 0)


class IPPOPolicy(BasePolicy):
    """Own one actor, local critic, action variance, and optimizer per fixed slot.

    Batched prediction requires every slot in the environment's stable order.
    For an individual aircraft, use learners[index].predict(local_observation).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_agents,
        actor_class=MLPActor,
        critic_class=LocalMLPCritic,
        actor_kwargs=None,
        critic_kwargs=None,
        log_std_init=-0.5,
    ):
        super().__init__(observation_space, action_space)
        self.n_agents = n_agents
        self.learners = nn.ModuleList(
            MAPPOPolicy(
                observation_space,
                action_space,
                lr_schedule,
                n_agents=1,
                actor_class=actor_class,
                critic_class=critic_class,
                actor_kwargs=actor_kwargs,
                critic_kwargs=critic_kwargs,
                log_std_init=log_std_init,
            )
            for _ in range(n_agents)
        )

    def distribution(self, observations):
        if observations.ndim < 2 or observations.shape[-2] != self.n_agents:
            raise ValueError("IPPO prediction requires all agent slots in fixed order")
        distributions = [
            learner.distribution(observations[..., index, :])
            for index, learner in enumerate(self.learners)
        ]
        return Normal(
            torch.stack([item.loc for item in distributions], dim=-2),
            torch.stack([item.scale for item in distributions], dim=-2),
        )

    def critic(self, observations, active):
        return torch.stack(
            [
                learner.critic(observations[..., index, :], active[..., index])
                for index, learner in enumerate(self.learners)
            ],
            dim=-1,
        )

    def _predict(self, observation, deterministic=False):
        distribution = self.distribution(observation)
        return distribution.mean if deterministic else distribution.sample()


class IPPO(MAPPO):
    """Collect one joint rollout, then update each PPO using only its own samples.

    Reuse MAPPO's individual boundary handling and GAE. No actor, critic,
    optimizer state, advantage normalization, or KL stopping is shared.
    """

    def __init__(self, policy=IPPOPolicy, env=None, **kwargs):
        super().__init__(policy=policy, env=env, **kwargs)

    def _get_torch_save_params(self):
        return ["policy"] + [
            f"policy.learners.{index}.optimizer" for index in range(self.n_agents)
        ], []

    def _train_agent(self, index):
        learner = self.policy.learners[index]
        self._update_learning_rate(learner.optimizer)
        joint = self.rollout_buffer.data
        selected = joint["active"][:, index]
        data = {name: value[:, index][selected] for name, value in joint.items()}
        count = int(selected.sum())
        if not count:
            return [], 0
        advantages = data["advantages"]
        if count > 1:
            data["advantages"] = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        metrics = []
        for epoch in range(self.n_epochs):
            indices = torch.randperm(count, device=self.device)
            for start in range(0, count, self.batch_size):
                selected = indices[start : start + self.batch_size]
                batch = {name: value[selected] for name, value in data.items()}
                values, log_probs, entropy = learner.evaluate_actions(
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
                    return metrics, epoch + 1
                loss = (
                    losses["policy_loss"]
                    + self.vf_coef * losses["value_loss"]
                    + self.ent_coef * losses["entropy_loss"]
                )
                learner.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(learner.parameters(), self.max_grad_norm)
                learner.optimizer.step()
        return metrics, self.n_epochs

    def train(self):
        self.policy.set_training_mode(True)
        metrics, epochs = [], []
        for index in range(self.n_agents):
            current, completed = self._train_agent(index)
            metrics.extend(current)
            epochs.append(completed)
        self._n_updates += max(epochs, default=0)
        for name in metrics[0] if metrics else ():
            self.logger.record(f"train/{name}", np.mean([row[name] for row in metrics]))
        self.logger.record("train/n_updates", self._n_updates)


class SharedPPO(MAPPO):
    """Shared actor and local critic with the same masked collector as IPPO/MAPPO."""

    def __init__(self, policy=MAPPOPolicy, env=None, policy_kwargs=None, **kwargs):
        settings = dict(policy_kwargs or {})
        settings.setdefault("critic_class", LocalMLPCritic)
        super().__init__(policy=policy, env=env, policy_kwargs=settings, **kwargs)
