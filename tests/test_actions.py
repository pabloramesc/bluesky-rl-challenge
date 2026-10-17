"""Control mappings, observation contracts, and simulator hook integration."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from gymnasium import spaces

from bluesky_pabloramesc.actions import KNOTS_PER_MPS, ControlConfig, action_targets
from bluesky_pabloramesc.observations import (
    ControlHooks,
    extend_observation_space,
    ownship_features,
)


@pytest.mark.parametrize(
    "heading,action,expected", [(350, 1, 35), (10, -1, -35), (90, 0, 90)]
)
def test_relative_heading_wraps_from_measured_state(heading, action, expected):
    assert action_targets([action, 0], heading, 250, ControlConfig()) == (expected, 250)


@pytest.mark.parametrize("action,heading", [(-1, 0), (0, -180), (1, 0), (-0.5, 90)])
def test_absolute_heading(action, heading):
    config = ControlConfig(heading_action="absolute", observation_profile="control")
    assert action_targets([action, 0], 123, 250, config)[0] == heading


def test_speed_modes_share_bounds_and_units():
    relative = ControlConfig(speed_min=200, speed_max=300)
    absolute = ControlConfig(speed_action="absolute", speed_min=200, speed_max=300)
    assert action_targets([0, 1], 0, 298, relative)[1] == 300
    assert action_targets([0, -1], 0, 201, relative)[1] == 200
    for action, expected in [(-1, 200), (0, 250), (1, 300)]:
        assert action_targets([0, action], 0, 275, absolute)[1] == expected
    assert action_targets([0, 1], 0, 250, ControlConfig())[1] == pytest.approx(
        250 + 20 / 3
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"heading_action": "absolute"},
        {"speed_action": "absolute"},
        {"speed_min": 100},
        {"speed_min": 200, "speed_max": 100},
        {"speed_min": 100, "speed_max": float("inf")},
        {"heading_delta": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ControlConfig(**kwargs)


@pytest.mark.parametrize("action", [[np.nan, 0], [0], [0, np.inf]])
def test_invalid_actions(action):
    with pytest.raises(ValueError):
        action_targets(action, 0, 250, ControlConfig())


def test_observation_space_matches_features():
    base = spaces.Dict({"original": spaces.Box(-1, 1, shape=(1,), dtype=np.float64)})
    assert extend_observation_space(base, ControlConfig()) is base
    config = ControlConfig(observation_profile="control")
    extended = extend_observation_space(base, config)
    observation = {"original": np.zeros(1), **ownship_features(90, 300)}
    assert extended.contains(observation)
    assert spaces.flatten(extended, observation).shape == (4,)
    assert observation["own_heading_sin"][0] == pytest.approx(1)
    assert observation["own_cas"][0] == 0
    assert list(base.spaces) == ["original"]


def test_hooks_read_fresh_aircraft_index_and_issue_cas_commands(monkeypatch):
    simulator = SimpleNamespace(
        traf=SimpleNamespace(
            id2idx=Mock(return_value=1), hdg=[0, 350], cas=[0, 250 / KNOTS_PER_MPS]
        ),
        stack=SimpleNamespace(stack=Mock()),
    )
    monkeypatch.setitem(sys.modules, "bluesky", simulator)

    class Base:
        def _get_obs(self, ac_id):
            return {"original": np.zeros(1)}

    class Env(ControlHooks, Base):
        control_config = ControlConfig(observation_profile="control")

    env = Env()
    env._get_action("AC2", [1, 0])
    calls = simulator.stack.stack.call_args_list
    assert calls[0].args[0] == "HDG AC2 35.0"
    assert float(calls[1].args[0].split()[-1]) == pytest.approx(250)
    observation = env._get_obs("AC2")
    assert observation["own_cas"][0] == pytest.approx(-0.5)
    assert simulator.traf.id2idx.call_count == 2
