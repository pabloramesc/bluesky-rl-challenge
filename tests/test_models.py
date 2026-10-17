import numpy as np
import pytest

from bluesky_pabloramesc.models import (
    build_ddpg,
    build_multi_ppo,
    build_multi_sac,
    build_td3,
)


@pytest.mark.parametrize("builder", [build_multi_ppo, build_multi_sac])
def test_shared_policy_predicts_for_all_aircraft(builder, env_factory):
    env = env_factory(n_envs=10)
    try:
        model = builder(env, device="cpu")
        actions, _ = model.predict(env.reset(), deterministic=True)
        assert model.n_envs == 10
        assert actions.shape == (10, 2)
        assert np.isfinite(actions).all()
    finally:
        env.close()


@pytest.mark.parametrize(
    ("builder", "critics", "delay"), [(build_td3, 2, 2), (build_ddpg, 1, 1)]
)
def test_deterministic_baseline_settings(builder, critics, delay, env_factory):
    env = env_factory()
    try:
        model = builder(env, device="cpu", verbose=0)
        assert model.critic.n_critics == critics
        assert model.policy_delay == delay
        noise = model.action_noise()
        assert noise.shape == env.action_space.shape
        assert np.isfinite(noise).all()
        assert np.all(model.action_noise._sigma > 0)
    finally:
        env.close()
