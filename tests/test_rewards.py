"""Numerical reward references and simulator-independent lifecycle checks."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from bluesky_pabloramesc.actions import KNOTS_PER_MPS, ControlConfig
from bluesky_pabloramesc.geometry import (
    closest_approach,
    local_meters,
    signed_clearance,
)
from bluesky_pabloramesc.observations import ControlHooks
from bluesky_pabloramesc.rewards import (
    COMPONENTS,
    RewardConfig,
    RewardHooks,
    clearance_cost,
    predictive_risk,
    smoothness_cost,
)


@pytest.mark.parametrize(
    "settings",
    [
        {"progress_weight": -1},
        {"predictive_weight": float("nan")},
        {"time_weight": float("inf")},
        {"smoothness_weight": True},
        {"progress_scale_m": 0},
        {"prediction_horizon_s": -1},
        {"separation_distance_m": 0},
        {"obstacle_buffer_m": 0},
        {"sector_buffer_m": 0},
        {"heading_scale_deg": 0},
        {"speed_scale_knots": 0},
        {"schema_version": 2},
        {"schema_version": True},
    ],
)
def test_invalid_reward_config(settings):
    with pytest.raises(ValueError):
        RewardConfig(**settings)


def test_config_roundtrip_and_opt_in_defaults():
    assert not RewardConfig().enabled
    config = RewardConfig(progress_weight=0.25, prediction_horizon_s=60)
    assert config.enabled
    assert RewardConfig(**config.to_dict()) == config


def test_predictive_risk_matches_tcpa_dcpa_formula():
    # Head-on collision, offset pass, receding, stationary, and beyond horizon.
    positions = np.array([[10, 0], [10, 5], [10, 0], [10, 0], [30, 0]])
    velocities = np.array([[-1, 0], [-1, 0], [1, 0], [0, 0], [-1, 0]])
    tcpa, dcpa = closest_approach(positions, velocities)
    np.testing.assert_allclose(tcpa, [10, 10, -10, 0, 30])
    np.testing.assert_allclose(dcpa, [0, 5, 10, 10, 0])
    assert predictive_risk(positions, velocities, 20, 10) == pytest.approx(0.5 + 0.125)
    # Summation counts simultaneous threats separately.
    assert predictive_risk(positions[:2], velocities[:2], 20, 10) == pytest.approx(
        0.625
    )


@pytest.mark.parametrize("time", [0, 20, 21, -1])
def test_predictive_risk_excludes_nonfuture_or_out_of_horizon(time):
    assert predictive_risk([[time, 0]], [[-1, 0]], 20, 10) == 0


def test_empty_and_near_parallel_traffic_are_finite():
    assert predictive_risk(np.empty((0, 2)), np.empty((0, 2)), 120, 1000) == 0
    assert predictive_risk([[1, 0]], [[1e-5, 0]], 120, 1000) == 0


def test_clearance_uses_edges_and_signed_safe_side():
    # A narrow, long obstacle: centre distance would miss the near side edge.
    polygon = Polygon([(0, -100), (2, -100), (2, 100), (0, 100)])
    outside = signed_clearance(Point(-1, 0), polygon, inside_positive=False)
    inside = signed_clearance(Point(1, 0), polygon, inside_positive=False)
    assert outside == 1
    assert inside == -1
    assert clearance_cost(outside, 2) == 0.25
    assert clearance_cost(0, 2) == 1
    assert clearance_cost(inside, 2) == 2.25
    assert clearance_cost(3, 2) == 0
    assert signed_clearance(Point(1, 0), polygon, inside_positive=True) == 1
    assert signed_clearance(Point(-1, 0), polygon, inside_positive=True) == -1


def test_concave_sector_clearance():
    polygon = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
    assert signed_clearance(Point(2, 2), polygon, inside_positive=True) == -1
    assert signed_clearance(Point(0.5, 2), polygon, inside_positive=True) == 0.5


def test_local_projection_uses_east_north_metres_and_wraps_longitude():
    np.testing.assert_allclose(
        local_meters([[1, 0], [0, 1]], (0, 0)), [[0, 111120], [111120, 0]]
    )
    np.testing.assert_allclose(local_meters([60, 1], (60, 0)), [55560, 0])
    np.testing.assert_allclose(local_meters([0, -179], (0, 179)), [222240, 0])


def test_smoothness_wraps_heading_uses_physical_units_and_command_interval():
    config = RewardConfig(heading_scale_deg=2, speed_scale_knots=10)
    assert smoothness_cost(None, (179, 100), 0, config) == 0
    assert smoothness_cost((179, 100), (-179, 110), 10, config) == pytest.approx(0.2)
    assert smoothness_cost((179, 100), (-179, 110), 5, config) == pytest.approx(0.4)
    assert smoothness_cost((179, 100), (179, 100), 10, config) == 0
    with pytest.raises(ValueError, match="positive elapsed"):
        smoothness_cost((0, 100), (1, 100), 0, config)


class MetricEnvironment:
    """Minimal upstream protocol: metrics and rewards are separate."""

    sim_dt = 1

    def reset(self):
        self.agents = ["first", "second"]
        self.metrics = {}
        self.distances = dict.fromkeys(self.agents, 1000.0)
        for agent in self.agents:
            self._init_metrics(agent)
            self._update_metrics(agent, dt=0)

    def _init_metrics(self, ac_id):
        self.metrics[ac_id] = {"flight_time": 0.0, "total_reward": 0.0}

    def _update_metrics(self, ac_id, dt=None):
        self.metrics[ac_id]["flight_time"] += self.sim_dt if dt is None else dt

    def _get_reward(self, ac_id):
        return 1.0

    def _get_info(self, ac_id):
        return dict(self.metrics[ac_id])


class SyntheticRewards(RewardHooks, MetricEnvironment):
    def __init__(self, config):
        self.reward_config = config
        self.reset()

    def _goal_distance(self, ac_id):
        return self.distances[ac_id]

    def _sample_costs(self, ac_id):
        return {"predictive": 0.5, "obstacle": 0.25, "sector": 0.125}


def test_all_components_sum_and_reward_reads_are_idempotent():
    config = RewardConfig(
        progress_weight=2,
        progress_scale_m=100,
        predictive_weight=3,
        obstacle_weight=4,
        sector_weight=5,
        time_weight=0.1,
        smoothness_weight=6,
        heading_scale_deg=2,
        speed_scale_knots=10,
    )
    env = SyntheticRewards(config)
    env._on_action_target("first", 179, 100)
    for _ in range(3):
        env.distances["first"] -= 10
        env._update_metrics("first")
    expected = {
        "progress": 0.6,
        "predictive": -4.5,
        "obstacle": -3,
        "sector": -1.875,
        "time": -0.3,
        "smoothness": 0,
    }
    assert env._get_info("first")["reward_components"] == pytest.approx(expected)
    assert env._get_reward("first") == pytest.approx(1 + sum(expected.values()))
    assert env._get_reward("first") == pytest.approx(1 + sum(expected.values()))
    env._on_action_target("first", -179, 110)
    env.distances["first"] += 10  # Moving away is penalized.
    env._update_metrics("first")
    second = env._get_info("first")["reward_components"]
    assert second["progress"] == pytest.approx(-0.2)
    assert second["smoothness"] == pytest.approx(-6 * 2 / 3)
    assert env._get_info("first")["reward_component_totals"] == pytest.approx(
        {name: expected[name] + second[name] for name in COMPONENTS}
    )
    assert env.metrics["first"] == {"flight_time": 4, "total_reward": 0}


@pytest.mark.parametrize("interval", [5, 10, 20])
def test_integrated_costs_and_progress_are_decision_interval_independent(interval):
    env = SyntheticRewards(
        RewardConfig(
            progress_weight=1,
            progress_scale_m=100,
            predictive_weight=1,
            obstacle_weight=1,
            sector_weight=1,
            time_weight=1,
        )
    )
    accumulated = dict.fromkeys(COMPONENTS, 0.0)
    for _ in range(20 // interval):
        env._on_action_target("first", 0, 100)
        for _ in range(interval):
            env.distances["first"] -= 1
            env._update_metrics("first")
        for name, reward in env._get_info("first")["reward_components"].items():
            accumulated[name] += reward
    assert accumulated == pytest.approx(
        {
            "progress": 0.2,
            "predictive": -10,
            "obstacle": -5,
            "sector": -2.5,
            "time": -20,
            "smoothness": 0,
        }
    )


def test_early_agent_completion_and_reset_preserve_independent_state():
    env = SyntheticRewards(RewardConfig(time_weight=1, smoothness_weight=1))
    for agent in env.agents:
        env._on_action_target(agent, 0, 100)
    for _ in range(3):
        for agent in env.agents:
            env._update_metrics(agent)
    final_reward = env._get_reward("first")
    final_info = env._get_info("first")
    env.agents.remove("first")
    for _ in range(7):
        env._update_metrics("second")
    assert final_reward == -2
    assert final_info["reward_component_totals"]["time"] == -3
    assert env._get_info("second")["reward_components"]["time"] == -10
    env.reset()
    env._on_action_target("first", 90, 200)
    assert env._get_info("first")["reward_component_totals"] == dict.fromkeys(
        COMPONENTS, 0
    )
    assert env._reward_geometry is None


def test_smoothness_is_equivalent_for_relative_and_absolute_actions(monkeypatch):
    traffic = SimpleNamespace(
        hdg=np.array([0.0]),
        cas=np.array([200 / KNOTS_PER_MPS]),
        id2idx=lambda ac_id: 0,
    )
    commands = []
    monkeypatch.setitem(
        sys.modules,
        "bluesky",
        SimpleNamespace(traf=traffic, stack=SimpleNamespace(stack=commands.append)),
    )

    class CommandRewards(SyntheticRewards, ControlHooks):
        pass

    relative = CommandRewards(RewardConfig(smoothness_weight=1))
    relative.control_config = ControlConfig(speed_min=100, speed_max=500)
    absolute = CommandRewards(RewardConfig(smoothness_weight=1))
    absolute.control_config = ControlConfig(
        heading_action="absolute",
        speed_action="absolute",
        observation_profile="control",
        speed_min=100,
        speed_max=500,
    )
    relative._get_action("first", [1, 1])
    absolute._get_action("first", [-0.75, (200 + 20 / 3 - 100) / 200 - 1])
    np.testing.assert_allclose(
        relative._reward_state["first"]["command"],
        absolute._reward_state["first"]["command"],
    )
    for _ in range(10):
        relative._update_metrics("first")
        absolute._update_metrics("first")
    relative._get_action("first", [0.5, 0])
    absolute._get_action("first", [-0.875, -0.5])
    for env in (relative, absolute):
        assert env._get_info("first")["reward_components"][
            "smoothness"
        ] == pytest.approx(-0.125)
    assert len(commands) == 8


def test_zero_weights_leave_reward_and_info_unchanged():
    env = SyntheticRewards(RewardConfig())
    env._on_action_target("first", 0, 100)
    env._update_metrics("first")
    assert env._get_reward("first") == 1
    assert env._get_info("first") == {"flight_time": 1, "total_reward": 0}


@pytest.mark.parametrize("multi_agent", [False, True])
def test_physical_sampling_uses_live_traffic_and_polygon_edges(
    monkeypatch, multi_agent
):
    # Degrees around the equator encode simple metric geometry exactly here.
    traffic = SimpleNamespace(
        lat=np.array([0, 5 / 111120]),
        lon=np.array([0, 10 / 111120]),
        gseast=np.array([0, -1]),
        gsnorth=np.zeros(2),
        id2idx=lambda ac_id: {"first": 0, "second": 1}.get(ac_id, -1),
    )
    monkeypatch.setitem(sys.modules, "bluesky", SimpleNamespace(traf=traffic))
    env = SyntheticRewards(
        RewardConfig(
            predictive_weight=1,
            obstacle_weight=1,
            sector_weight=1,
            prediction_horizon_s=20,
            separation_distance_m=10,
            obstacle_buffer_m=10,
            sector_buffer_m=10,
        )
    )
    env.scenario = SimpleNamespace(
        center=(0, 0),
        sector=np.array([[-20, -20], [-20, 20], [20, 20], [20, -20]]) / 111120,
        obstacles=[
            SimpleNamespace(
                vertices=np.array([[-100, 4], [-100, 6], [100, 6], [100, 4]]) / 111120
            )
        ],
    )
    if multi_agent:
        env.possible_agents = ["first", "second"]
        env.agents = ["first", "second", "missing"]
    else:
        env.intruder_ids = ["second", "missing"]
    # Exercise the actual adapter, bypassing SyntheticRewards' fixed samples.
    costs = RewardHooks._sample_costs(env, "first")
    assert costs == pytest.approx({"predictive": 0.125, "obstacle": 0.36, "sector": 0})
    traffic.lat = traffic.lat[:1]
    traffic.lon = traffic.lon[:1]
    traffic.gseast = traffic.gseast[:1]
    traffic.gsnorth = traffic.gsnorth[:1]
    traffic.id2idx = lambda ac_id: 0 if ac_id == "first" else -1
    env._reward_geometry = None
    env.scenario.obstacles = []
    costs = RewardHooks._sample_costs(env, "first")
    assert costs == dict.fromkeys(("predictive", "obstacle", "sector"), 0)
