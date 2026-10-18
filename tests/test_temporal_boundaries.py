"""Independent references for temporal boundaries, replay, and vectorized lifecycles."""

import numpy as np
import pytest
import supersuit as ss
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from test_temporal import CounterEnv
from test_temporal_matrix import TemporalTraffic, make_environment

from bluesky_pabloramesc.multi_agent import (
    FlattenObs,
    JointEpisodeBlackDeath,
    SeedableSB3VecEnvWrapper,
)
from bluesky_pabloramesc.temporal import (
    HistoryRunningMeanStd,
    HistoryVecNormalize,
    HistoryWindow,
    TemporalFeatures,
)


class DepartingTraffic(ParallelEnv):
    """Ten slots depart in three waves, followed by a joint automatic reset."""

    render_mode = None

    def __init__(self):
        self.metadata = {"render_modes": []}
        self.possible_agents = [f"aircraft_{index}" for index in range(10)]
        self.episodes = 0

    def observation_space(self, agent):
        return spaces.Dict({"state": spaces.Box(-10000, 10000, (2,), dtype=np.float32)})

    def action_space(self, agent):
        return spaces.Box(-1, 1, (2,), dtype=np.float32)

    def observations(self):
        return {
            agent: {
                "state": np.array(
                    [
                        10 * self.episodes + self.steps,
                        self.possible_agents.index(agent),
                    ],
                    np.float32,
                )
            }
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.episodes += 1
        self.agents = self.possible_agents.copy()
        return self.observations(), {agent: {} for agent in self.agents}

    def step(self, actions):
        assert set(actions) == set(self.agents)
        previous = self.agents.copy()
        self.steps += 1
        self.agents = (
            self.possible_agents[4:]
            if self.steps == 1
            else self.possible_agents[7:]
            if self.steps == 2
            else []
        )
        return (
            self.observations(),
            {agent: 1.0 for agent in previous},
            {agent: agent not in self.agents for agent in previous},
            {agent: False for agent in previous},
            {agent: {} for agent in previous},
        )


def make_departing_history(length):
    env = JointEpisodeBlackDeath(FlattenObs(DepartingTraffic()), track_history=True)
    env = SeedableSB3VecEnvWrapper(ss.pettingzoo_env_to_vec_env_v1(env))
    return HistoryWindow(VecMonitor(env), length)


@pytest.mark.parametrize("length", [1, 2, 8])
def test_supersuit_departures_and_joint_autoreset(length):
    env = make_departing_history(length)
    try:
        initial = env.reset()
        np.testing.assert_array_equal(initial[:, -1, -1], np.ones(10))
        actions = np.zeros((10, 2), np.float32)
        first, _, _, _ = env.step(actions)
        assert (first[:4] == 0).all()
        second, _, _, _ = env.step(actions)
        assert (second[:7] == 0).all()
        reset, _, done, infos = env.step(actions)
        assert done.all()
        assert (reset[:, :-1] == 0).all()
        np.testing.assert_array_equal(reset[:, -1, -1], np.ones(10))
        np.testing.assert_array_equal(reset[:, -1, 0], np.full(10, 20))
        # Already-departed aircraft must not gain a fictitious valid terminal frame.
        for info in infos[:7]:
            assert (info["terminal_observation"] == 0).all()
        np.testing.assert_array_equal(initial[:, -1, 0], np.full(10, 10))
        np.testing.assert_array_equal(second[7:, -1, 0], np.full(3, 12))
    finally:
        env.close()


@pytest.mark.parametrize("length", [1, 2, 8, 17])
def test_history_terminal_timeout_and_returned_arrays_do_not_alias(length):
    env = HistoryWindow(
        DummyVecEnv(
            [lambda: TemporalTraffic(entities=False, duration=2, truncated=True)]
        ),
        length,
    )
    try:
        first = env.reset()
        first_copy = first.copy()
        env.step(np.zeros((1, 2)))
        following, _, dones, infos = env.step(np.zeros((1, 2)))
        assert dones[0] and infos[0]["TimeLimit.truncated"]
        terminal = infos[0]["terminal_observation"]
        assert terminal.shape == env.observation_space.shape
        assert terminal[-1, -1] == 1
        assert not np.array_equal(terminal, following[0])
        np.testing.assert_array_equal(first, first_copy)
        terminal_copy = terminal.copy()
        env.step(np.zeros((1, 2)))
        np.testing.assert_array_equal(terminal, terminal_copy)
        assert not np.shares_memory(following, env.history)
    finally:
        env.close()


@pytest.mark.parametrize("method", ["gru", "lstm"])
@pytest.mark.parametrize("history", [1, 2, 8, 17])
def test_recurrent_batch_matches_individual_unpadded_sequences(method, history):
    torch.manual_seed(23)
    env = HistoryWindow(DummyVecEnv([CounterEnv]), history)
    try:
        encoder = TemporalFeatures(
            env.observation_space,
            env.observation_space.temporal_frame_space,
            FlattenExtractor,
            {},
            method,
            history,
        )
        obs = torch.zeros(4, history, 2)
        lengths = [1, history, 0, min(3, history)]
        for index, length in enumerate(lengths):
            if length:
                obs[index, -length:, 0] = torch.arange(1, length + 1)
                obs[index, -length:, 1] = 1
        actual = encoder(obs)
        for index, length in enumerate(lengths):
            if length:
                _, hidden = encoder.recurrent(obs[index : index + 1, -length:, :1])
                expected = hidden[0][-1, 0] if method == "lstm" else hidden[-1, 0]
                torch.testing.assert_close(actual[index], expected)
            else:
                assert (actual[index] == 0).all()
        order = [2, 0, 3, 1]
        torch.testing.assert_close(encoder(obs[order]), actual[order])
    finally:
        env.close()


@pytest.mark.parametrize("history", [1, 2, 3, 4, 8, 17])
def test_tcn_receptive_field_reaches_oldest_observation(history):
    env = HistoryWindow(DummyVecEnv([CounterEnv]), history)
    try:
        encoder = TemporalFeatures(
            env.observation_space,
            env.observation_space.temporal_frame_space,
            FlattenExtractor,
            {},
            "tcn",
            history,
        )
        with torch.no_grad():
            encoder.input_projection.weight.fill_(1)
            encoder.input_projection.bias.zero_()
            for layer in encoder.convolutions:
                layer.weight.fill_(0.01)
                layer.bias.zero_()
        obs = torch.ones(1, history, 2, requires_grad=True)
        encoder(obs).sum().backward()
        assert obs.grad[0, 0, 0] > 0
    finally:
        env.close()


@pytest.mark.parametrize("graph", ["fully_connected", "distance", "knn"])
def test_history_normalization_counts_each_latest_occupied_sample_once(graph):
    env = make_environment({"architecture": "entity_gnn", "graph": graph}, 8, slots=3)
    try:
        env.reset()
        for _ in range(4):
            normalized, _, _, _ = env.step(np.zeros((3, 2)))
        rms = env.obs_rms.frame_rms
        assert rms.context_rms.count == pytest.approx(15.0001)
        assert rms.aircraft_rms.count == pytest.approx(30.0001)
        layout = env.observation_space.entity_layout
        raw = env.get_original_obs()
        binary = layout["mask"] + layout["graph_edges"] + [raw.shape[-1] - 1]
        np.testing.assert_array_equal(normalized[..., binary], raw[..., binary])
        env.training = False
        env.step(np.zeros((3, 2)))
        assert rms.context_rms.count == pytest.approx(15.0001)
        assert rms.aircraft_rms.count == pytest.approx(30.0001)
    finally:
        env.close()


def test_normalization_ignores_inactive_latest_frames_and_nan_padding():
    env = HistoryWindow(DummyVecEnv([CounterEnv]), 8)
    try:
        stats = HistoryRunningMeanStd(env.observation_space)
        obs = np.zeros((3, 8, 2))
        obs[:, :-1, 0] = np.nan
        obs[:, -1, 0] = [2.0, 4.0, np.nan]
        obs[:, -1, 1] = [1, 1, 0]
        stats.update(obs)
        assert stats.count == pytest.approx(2.0001)
        np.testing.assert_allclose(stats.mean[:, 0], 3.0, atol=0.001)
        assert np.isfinite(stats.mean).all() and np.isfinite(stats.var).all()
    finally:
        env.close()


@pytest.mark.parametrize("history", [0, -1, True, 2.5])
def test_temporal_features_rejects_invalid_history_lengths(history):
    env = HistoryWindow(DummyVecEnv([CounterEnv]), 1)
    try:
        with pytest.raises(ValueError, match="history_length"):
            TemporalFeatures(
                env.observation_space,
                env.observation_space.temporal_frame_space,
                FlattenExtractor,
                {},
                "gru",
                history,
            )
    finally:
        env.close()


@pytest.mark.parametrize("method", ["frame_stack", "gru", "lstm", "tcn"])
@pytest.mark.parametrize("truncated", [False, True])
def test_off_policy_terminal_windows_and_timeout_bootstrapping(
    method, truncated, small_training_budgets, tmp_path
):
    from bluesky_pabloramesc.models import build_sac

    class EndCounter(CounterEnv):
        def step(self, action):
            obs, reward, done, _, info = super().step(action)
            return obs, reward, done and not truncated, done and truncated, info

    env = HistoryVecNormalize(HistoryWindow(DummyVecEnv([lambda: EndCounter(2)]), 3))
    try:
        model = build_sac(env, temporal=method, history_length=3, device="cpu")
        model.learn(2)
        buffer = model.replay_buffer
        terminal = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(buffer.next_observations[1, 0], terminal, atol=1e-6)
        # Reset observations do not replace the final next-observation in replay.
        assert not np.array_equal(
            buffer.next_observations[1, 0], env.get_original_obs()[0]
        )
        sample = buffer._get_samples(np.array([1]), env=env)
        np.testing.assert_allclose(
            sample.next_observations.numpy()[0], env.normalize_obs(terminal), atol=1e-6
        )
        assert sample.dones.item() == (0.0 if truncated else 1.0)
        before = buffer.observations.copy()
        model.save_replay_buffer(tmp_path / "buffer.pkl")
        model.replay_buffer.observations.fill(np.nan)
        model.load_replay_buffer(tmp_path / "buffer.pkl")
        np.testing.assert_array_equal(model.replay_buffer.observations, before)
    finally:
        env.close()


@pytest.mark.parametrize("method", ["frame_stack", "gru", "lstm", "tcn"])
@pytest.mark.parametrize("truncated", [False, True])
def test_ppo_bootstraps_timeouts_with_terminal_history(
    method, truncated, small_training_budgets, monkeypatch
):
    from bluesky_pabloramesc.models import GAMMA, build_ppo

    class EndCounter(CounterEnv):
        def step(self, action):
            obs, reward, done, _, info = super().step(action)
            return obs, reward, done and not truncated, done and truncated, info

    env = HistoryVecNormalize(
        HistoryWindow(DummyVecEnv([lambda: EndCounter(2)]), 3),
        norm_obs=False,
        norm_reward=False,
    )
    try:
        model = build_ppo(env, temporal=method, history_length=3, device="cpu")
        captured = []

        def fixed_value(obs):
            captured.append(obs.detach().cpu().numpy().copy())
            return torch.full((obs.shape[0], 1), 5.0, device=obs.device)

        monkeypatch.setattr(model.policy, "predict_values", fixed_value)
        model.learn(2)
        assert model.rollout_buffer.rewards[1, 0] == pytest.approx(
            1 + GAMMA * 5 if truncated else 1
        )
        if truncated:
            np.testing.assert_array_equal(captured[0][0], [[0, 1], [1, 1], [2, 1]])
    finally:
        env.close()
