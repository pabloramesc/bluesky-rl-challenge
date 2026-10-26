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
    are mutually exclusive in this environment. With track_agents, infos retain
    individual completion flags and next-step activity for MAPPO. Final rewards
    and infos are preserved even if a departing agent omits its observation.
    """

    def __init__(self, env, track_history=False, track_agents=False):
        super().__init__(env)
        self.track_history = track_history
        self.track_agents = track_agents

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
        active_agents = set(self.env.agents)
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        observations, rewards, terminated, truncated, infos = self.env.step(
            active_actions
        )
        joint_done = not self.env.agents

        for agent in self.possible_agents:
            if (
                self.track_agents
                and truncated.get(agent, False)
                and agent not in observations
            ):
                raise ValueError("Truncated agents must provide a final observation")
            if agent not in observations:
                space = self.observation_space(agent)
                observations[agent] = np.zeros(space.shape, dtype=space.dtype)
            rewards.setdefault(agent, 0.0)
            infos.setdefault(agent, {})
            if self.track_agents:
                is_truncated = bool(truncated.get(agent, False))
                is_terminated = bool(terminated.get(agent, False)) or (
                    agent in active_agents
                    and agent not in self.env.agents
                    and not is_truncated
                )
                infos[agent].update(
                    _mappo_active=agent in self.env.agents,
                    _mappo_terminated=is_terminated,
                    _mappo_truncated=is_truncated,
                )

        if self.track_history:
            for agent in self.possible_agents:
                infos[agent]["_temporal_inactive"] = agent not in self.env.agents

        terminations = {agent: joint_done for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        if joint_done:
            self.agents = []
            # SuperSuit merges reset infos over terminal infos during auto-reset.
            # Preserve the departing episode's metrics until the SB3 adapter.
            for info in infos.values():
                info["_joint_final_info"] = dict(info)
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

    def step_wait(self):
        """Restore terminal metrics overwritten by SuperSuit's reset-info merge."""
        observations, rewards, dones, infos = super().step_wait()
        for info in infos:
            info.update(info.pop("_joint_final_info", {}))
        return observations, rewards, np.asarray(dones, dtype=bool), infos


def make_vec_env(
    render_mode=None,
    control_config=None,
    entity_observations=False,
    graph_radius_nm=None,
    graph_k=None,
    history_length=None,
    track_agents=False,
    reward_config=None,
):
    """Create ten monitored agent slots backed by one BlueSky simulator.

    Create only one BlueSky environment per process; simulator state is global.

    Args:
        render_mode: Rendering mode, or None for headless operation.
        control_config: ControlConfig, or None for the unchanged legacy interface.
        track_agents: Include individual boundaries and activity in agent infos.
        reward_config: RewardConfig, or None to keep all new reward weights zero.

    Returns:
        VecMonitor wrapping a seeded SB3-compatible shared-policy environment.
    """
    from bluesky_zoo.competition_v0 import CompetitionZooEnv

    from .actions import ControlConfig
    from .observations import ControlHooks, extend_observation_space
    from .rewards import RewardConfig, RewardHooks

    class ControlledCompetitionZooEnv(RewardHooks, ControlHooks, CompetitionZooEnv):
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
            self.reward_config = reward_config or RewardConfig()
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
    env = JointEpisodeBlackDeath(
        env, track_history=history_length is not None, track_agents=track_agents
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = SeedableSB3VecEnvWrapper(env)
    env = VecMonitor(env)
    if entity_observations:
        env.observation_space.entity_layout = layout
    if history_length is not None:
        from .temporal import HistoryWindow

        env = HistoryWindow(env, history_length)
    return env


def make_mappo_env(
    render_mode=None,
    control_config=None,
    reward_config=None,
    entity_observations=False,
):
    """Create one joint simulator with fixed slots and individual MAPPO boundaries."""
    return make_vec_env(
        render_mode=render_mode,
        control_config=control_config,
        track_agents=True,
        reward_config=reward_config,
        entity_observations=entity_observations,
    )
