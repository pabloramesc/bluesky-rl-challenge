"""Run the production training CLI with synthetic traffic in subprocess tests."""

import runpy
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from test_temporal_matrix import TemporalTraffic

from bluesky_pabloramesc import models, single_agent
from bluesky_pabloramesc.actions import ControlConfig
from bluesky_pabloramesc.temporal import HistoryWindow


def make_env(
    render_mode=None,
    control_config=None,
    entity_observations=False,
    graph_radius_nm=None,
    graph_k=None,
    history_length=None,
):
    config = control_config or ControlConfig()
    graph = (
        "distance"
        if graph_radius_nm is not None
        else "knn"
        if graph_k is not None
        else "fully_connected"
    )
    env = DummyVecEnv(
        [
            lambda: Monitor(
                TemporalTraffic(
                    config.observation_encoding,
                    config.observation_profile,
                    graph,
                    entity_observations,
                )
            )
        ]
    )
    return HistoryWindow(env, history_length) if history_length is not None else env


def small_ppo(policy, env, **kwargs):
    kwargs.update(n_steps=2, n_epochs=1, batch_size=2, verbose=0)
    return PPO(policy, env, **kwargs)


if __name__ == "__main__":
    single_agent.make_env = make_env
    models.PPO = small_ppo
    module = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_module(module, run_name="__main__")
