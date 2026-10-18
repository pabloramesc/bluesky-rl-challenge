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


@pytest.fixture
def small_training_budgets(monkeypatch):
    """Keep actual builders and policy settings; shrink only collection/update budgets."""
    from stable_baselines3 import DDPG, PPO, SAC, TD3

    from bluesky_pabloramesc import models

    for name, algorithm in (("PPO", PPO), ("SAC", SAC), ("TD3", TD3), ("DDPG", DDPG)):

        def constructor(policy, env, _algorithm=algorithm, **kwargs):
            kwargs.update(batch_size=max(2, env.num_envs), verbose=0)
            if _algorithm == PPO:
                kwargs.update(n_steps=2, n_epochs=1)
            else:
                kwargs.update(
                    buffer_size=64, learning_starts=0, train_freq=1, gradient_steps=1
                )
            return _algorithm(policy, env, **kwargs)

        monkeypatch.setattr(models, name, constructor)
