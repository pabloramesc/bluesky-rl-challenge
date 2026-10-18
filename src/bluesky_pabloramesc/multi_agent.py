"""Fixed-slot PettingZoo wrappers for shared-policy training."""

import numpy as np
import supersuit as ss
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper
from stable_baselines3.common.vec_env import VecMonitor
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

N_AGENTS = 10


class FlattenObs(BaseParallelWrapper):
    """Flatten each agent's Dict observation into a one-dimensional Box."""

    def __init__(self, env):
        """Cache the flattened observation space for each possible agent.

        Args:
            env: Parallel PettingZoo environment with flattenable observations.
        """
        super().__init__(env)
        self._flat = {
            agent: spaces.flatten_space(env.observation_space(agent))
            for agent in env.possible_agents
        }

    def observation_space(self, agent):
        """Return the cached flat observation space for an agent.

        Args:
            agent: Agent identifier.

        Returns:
            Gymnasium Box for the flattened observation.
        """
        return self._flat[agent]

    def _flatten(self, observations):
        """Flatten each supplied observation using its original agent space.

        Args:
            observations: Mapping of agent identifiers to original observations.

        Returns:
            Mapping of agent identifiers to flat observation arrays.
        """
        return {
            agent: spaces.flatten(self.env.observation_space(agent), observation)
            for agent, observation in observations.items()
        }

    def reset(self, seed=None, options=None):
        """Reset the environment and flatten initial observations.

        Args:
            seed: Environment seed, or None to leave RNG initialization unchanged.
            options: Reset options forwarded to the underlying environment.

        Returns:
            Pair of flattened observations and per-agent reset info mappings.
        """
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._flatten(observations), infos

    def step(self, actions):
        """Step the environment and flatten returned observations.

        Args:
            actions: Mapping of active agent identifiers to actions.

        Returns:
            Observations, rewards, terminations, truncations, and infos by agent.
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return (
            self._flatten(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )


class JointEpisodeBlackDeath(BaseParallelWrapper):
    """Keep fixed agent slots until the underlying joint episode finishes.

    Missing observations and rewards are padded with zeros. Joint completion
    uses the underlying agents list because termination and truncation flags
    are mutually exclusive in this environment.
    """

    def __init__(self, env, track_history=False):
        super().__init__(env)
        self.track_history = track_history

    def reset(self, seed=None, options=None):
        """Reset the environment and restore all possible agent slots.

        Args:
            seed: Environment seed, or None to leave RNG initialization unchanged.
            options: Reset options forwarded to the underlying environment.

        Returns:
            Initial observations and per-agent reset info mappings.
        """
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return observations, infos

    def step(self, actions):
        """Step active agents, pad missing slots, and synchronize completion.

        Args:
            actions: Mapping of fixed agent slots to actions; inactive actions are ignored.

        Returns:
            Padded observations, rewards, joint terminations, false truncations, and infos.
        """
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        observations, rewards, _, _, infos = self.env.step(active_actions)
        joint_done = not self.env.agents

        for agent in self.possible_agents:
            if agent not in observations:
                space = self.observation_space(agent)
                observations[agent] = np.zeros(space.shape, dtype=space.dtype)
                rewards[agent] = 0.0
                infos[agent] = {}

        if self.track_history:
            for agent in self.possible_agents:
                infos[agent]["_temporal_inactive"] = agent not in self.env.agents

        terminations = {agent: joint_done for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        if joint_done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos


class SeedableSB3VecEnvWrapper(SB3VecEnvWrapper):
    """Apply SB3 deferred seeds to the next SuperSuit environment reset."""

    def __init__(self, venv):
        """Wrap a SuperSuit vector environment with deferred seeding.

        Args:
            venv: SuperSuit vector environment compatible with the SB3 adapter.
        """
        super().__init__(venv)
        self._pending_seed = None

    def seed(self, seed=None):
        """Store a seed to apply on the next reset.

        Args:
            seed: Seed passed to SuperSuit on reset, including None.

        Returns:
            List repeating the requested seed for each vector slot.
        """
        self._pending_seed = seed
        return [seed] * self.num_envs

    def reset(self):
        """Reset using the pending seed, then clear it.

        Returns:
            Vector observations; reset infos are stored in reset_infos.
        """
        observations, self.reset_infos = self.venv.reset(seed=self._pending_seed)
        self._pending_seed = None
        return observations


def make_vec_env(
    render_mode=None,
    control_config=None,
    entity_observations=False,
    graph_radius_nm=None,
    graph_k=None,
    history_length=None,
):
    """Create ten monitored agent slots backed by one BlueSky simulator.

    Create only one BlueSky environment per process; simulator state is global.

    Args:
        render_mode: Rendering mode, or None for headless operation.
        control_config: ControlConfig, or None for the unchanged legacy interface.

    Returns:
        VecMonitor wrapping a seeded SB3-compatible shared-policy environment.
    """
    from bluesky_zoo.competition_v0 import CompetitionZooEnv

    from .actions import ControlConfig
    from .observations import ControlHooks, extend_observation_space

    class ControlledCompetitionZooEnv(ControlHooks, CompetitionZooEnv):
        def __init__(self):
            if graph_radius_nm is not None and graph_k is not None:
                raise ValueError(
                    "Select either a distance radius or k-nearest neighbors"
                )
            if (
                graph_radius_nm is not None or graph_k is not None
            ) and not entity_observations:
                raise ValueError("Graph observations require entity_observations=True")
            self.graph_k = graph_k
            self.graph_radius_nm = graph_radius_nm
            self.entity_observations = entity_observations
            self.control_config = control_config or ControlConfig()
            super().__init__(render_mode=render_mode, n_agents=N_AGENTS)
            self.observation_spaces = {
                agent: extend_observation_space(
                    space,
                    self.control_config,
                    entity_observations,
                    **({"graph_k": graph_k} if graph_k is not None else {}),
                    **(
                        {"graph_radius_nm": graph_radius_nm}
                        if graph_radius_nm is not None
                        else {}
                    ),
                )
                for agent, space in self.observation_spaces.items()
            }

    env = ControlledCompetitionZooEnv()
    if entity_observations:
        from .entities import entity_layout

        layout = entity_layout(env.observation_space(env.possible_agents[0]))
    env = FlattenObs(env)
    env = JointEpisodeBlackDeath(env, track_history=history_length is not None)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = SeedableSB3VecEnvWrapper(env)
    env = VecMonitor(env)
    if entity_observations:
        env.observation_space.entity_layout = layout
    if history_length is not None:
        from .temporal import HistoryWindow

        env = HistoryWindow(env, history_length)
    return env
