import numpy as np
import pytest

from bluesky_pabloramesc.models import build_multi_ppo, build_multi_sac


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
