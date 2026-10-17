import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class ShortEpisode(gym.Env):
    """Continuous control with predictable rewards and short episodes."""

    observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def close(self):
        self.closed = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.full(3, 0.5, dtype=np.float32), 2.0, False, self.steps == 3, {}


@pytest.fixture(scope="session", autouse=True)
def limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def env_factory():
    def make_env(render_mode=None, n_envs=1):
        return DummyVecEnv([lambda: Monitor(ShortEpisode()) for _ in range(n_envs)])

    return make_env
