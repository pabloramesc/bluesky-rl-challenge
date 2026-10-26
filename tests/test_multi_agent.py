import gymnasium as gym
import numpy as np
import pytest
from pettingzoo import ParallelEnv

from bluesky_pabloramesc.multi_agent import FlattenObs, JointEpisodeBlackDeath


class DepartingAgents(ParallelEnv):
    """One aircraft leaves before the joint episode ends."""

    def __init__(self):
        self.possible_agents = ["first", "second"]
        self.metadata = {}

    def observation_space(self, agent):
        return gym.spaces.Dict({"position": gym.spaces.Box(-1, 1, (2,), np.float32)})

    def action_space(self, agent):
        return gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        return self.observations(), {agent: {} for agent in self.agents}

    def observations(self):
        return {agent: {"position": np.ones(2, np.float32)} for agent in self.agents}

    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.agents = self.agents[1:]
        return (
            self.observations(),
            {agent: 1.0 for agent in self.agents},
            {agent: False for agent in self.agents},
            {agent: False for agent in self.agents},
            {agent: {} for agent in self.agents},
        )


def test_departed_agent_is_padded_until_joint_episode_end():
    env = JointEpisodeBlackDeath(FlattenObs(DepartingAgents()))
    observations, _ = env.reset(seed=1)
    assert env.observation_space("first").contains(observations["first"])
    actions = {agent: np.zeros(1, np.float32) for agent in env.possible_agents}

    observations, rewards, terminated, truncated, _ = env.step(actions)
    np.testing.assert_array_equal(observations["first"], np.zeros(2))
    assert rewards == {"first": 0.0, "second": 1.0}
    assert not any(terminated.values())
    assert not any(truncated.values())

    _, _, terminated, _, _ = env.step(actions)
    assert all(terminated.values())
    assert env.agents == []
    env.reset()
    assert env.agents == env.possible_agents


def test_mappo_metadata_and_terminal_rewards_survive_missing_observations():
    class FinalReward(DepartingAgents):
        def step(self, actions):
            observations, rewards, terminated, truncated, infos = super().step(actions)
            rewards["first"] = 5.0
            infos["first"] = {"waypoint_reached": True}
            return observations, rewards, terminated, truncated, infos

    env = JointEpisodeBlackDeath(FlattenObs(FinalReward()), track_agents=True)
    env.reset()
    _, rewards, _, _, infos = env.step(
        {agent: np.zeros(1, np.float32) for agent in env.agents}
    )
    assert rewards["first"] == 5
    assert infos["first"]["waypoint_reached"]
    assert infos["first"]["_mappo_terminated"]
    assert not infos["first"]["_mappo_active"]
    assert not infos["first"]["_mappo_truncated"]
    assert infos["second"]["_mappo_active"]


def test_mappo_truncation_requires_final_observation():
    class MissingFinalObservation(DepartingAgents):
        def step(self, actions):
            observations, rewards, terminated, truncated, infos = super().step(actions)
            truncated["first"] = True
            return observations, rewards, terminated, truncated, infos

    env = JointEpisodeBlackDeath(
        FlattenObs(MissingFinalObservation()), track_agents=True
    )
    env.reset()
    with pytest.raises(ValueError, match="final observation"):
        env.step({agent: np.zeros(1, np.float32) for agent in env.agents})
