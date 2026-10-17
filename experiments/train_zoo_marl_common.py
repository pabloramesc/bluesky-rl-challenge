"""Shared environment plumbing for the tuned PPO and SAC MARL trainers.

This module deliberately lives separately from ``train_zoo.py`` so the
original minimal example remains unchanged.
"""

import numpy as np
import supersuit as ss
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper
from stable_baselines3.common.vec_env import VecMonitor
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from bluesky_zoo.competition_v0 import CompetitionZooEnv


SEED = 0
N_AGENTS = 10
EVAL_EPISODES = 12


class FlattenObs(BaseParallelWrapper):
    """Flatten each agent's Dict observation into a one-dimensional Box."""

    def __init__(self, env):
        super().__init__(env)
        self._flat = {
            agent: spaces.flatten_space(env.observation_space(agent))
            for agent in env.possible_agents
        }

    def observation_space(self, agent):
        return self._flat[agent]

    def _flatten(self, observations):
        return {
            agent: spaces.flatten(self.env.observation_space(agent), observation)
            for agent, observation in observations.items()
        }

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._flatten(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return (
            self._flatten(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )


class JointEpisodeBlackDeath(BaseParallelWrapper):
    """Keep fixed agent slots and finish them together at joint episode end.

    SuperSuit 3.11's black-death wrapper checks ``termination & truncation``
    for joint completion. Those flags are mutually exclusive in this env, so
    the underlying env's empty ``agents`` list is used instead.
    """

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return observations, infos

    def step(self, actions):
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        observations, rewards, _, _, infos = self.env.step(active_actions)
        joint_done = not self.env.agents

        for agent in self.possible_agents:
            if agent not in observations:
                space = self.observation_space(agent)
                observations[agent] = np.zeros(space.shape, dtype=space.dtype)
                rewards[agent] = 0.0
                infos[agent] = {}

        terminations = {agent: joint_done for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        if joint_done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos


class SeedableSB3VecEnvWrapper(SB3VecEnvWrapper):
    """Bridge SB3's deferred seeding API to SuperSuit reset(seed=...)."""

    def __init__(self, venv):
        super().__init__(venv)
        self._pending_seed = None

    def seed(self, seed=None):
        self._pending_seed = seed
        return [seed] * self.num_envs

    def reset(self):
        observations, self.reset_infos = self.venv.reset(seed=self._pending_seed)
        self._pending_seed = None
        return observations


def make_vec_env(render_mode=None):
    """Create ten agent slots backed by one process-global BlueSky env."""
    env = CompetitionZooEnv(render_mode=render_mode, n_agents=N_AGENTS)
    env = FlattenObs(env)
    env = JointEpisodeBlackDeath(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = SeedableSB3VecEnvWrapper(env)
    return VecMonitor(env)


def evaluate(venv, n_episodes, model=None):
    """Return mean and std of per-agent episode returns."""
    returns = []
    obs = venv.reset()
    while len(returns) < n_episodes:
        if model is None:
            actions = np.stack(
                [venv.action_space.sample() for _ in range(venv.num_envs)]
            )
        else:
            actions, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = venv.step(actions)
        for info in infos:
            if "episode" in info:
                returns.append(info["episode"]["r"])
    return float(np.mean(returns)), float(np.std(returns))
