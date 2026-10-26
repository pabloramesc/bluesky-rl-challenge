"""Optional reward engineering, independent of observations and scored metrics."""

import math
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import closest_approach, local_meters, signed_clearance


@dataclass(frozen=True)
class RewardConfig:
    """Nonnegative weights; physical scales are metres, seconds, degrees and knots."""

    progress_weight: float = 0.0
    predictive_weight: float = 0.0
    obstacle_weight: float = 0.0
    sector_weight: float = 0.0
    time_weight: float = 0.0
    smoothness_weight: float = 0.0
    progress_scale_m: float = 1852.0
    prediction_horizon_s: float = 120.0
    separation_distance_m: float = 6 * 1852.0
    obstacle_buffer_m: float = 1852.0
    sector_buffer_m: float = 1852.0
    heading_scale_deg: float = 45.0
    speed_scale_knots: float = 20 / 3
    schema_version: int = 1

    def __post_init__(self):
        for name, value in self.to_dict().items():
            if name == "schema_version":
                if type(value) is not int or value != 1:
                    raise ValueError("unsupported reward schema version")
                continue
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (not name.endswith("_weight") and value == 0):
                raise ValueError(
                    f"{name} must be {'nonnegative' if name.endswith('_weight') else 'positive'}"
                )

    @property
    def enabled(self):
        """Whether any shaping component is enabled."""
        return any(getattr(self, f"{name}_weight") > 0 for name in COMPONENTS)

    def to_dict(self):
        """Return checkpoint-safe configuration."""
        return asdict(self)


COMPONENTS = ("progress", "predictive", "obstacle", "sector", "time", "smoothness")


def predictive_risk(position, velocity, horizon_s, separation_m):
    """Sum imminent future conflict risks over all supplied aircraft."""
    tcpa, dcpa = closest_approach(position, velocity)
    risk = (1 - tcpa / horizon_s) * np.maximum(0, 1 - dcpa / separation_m) ** 2
    return float(np.sum(np.where((tcpa > 0) & (tcpa < horizon_s), risk, 0)))


def clearance_cost(clearance_m, buffer_m):
    """Squared warning-buffer penetration, increasing beyond the boundary."""
    return max(0.0, 1 - clearance_m / buffer_m) ** 2


def smoothness_cost(previous, current, elapsed_s, config):
    """Squared normalized command rate integrated over the command interval."""
    if previous is None:
        return 0.0
    if elapsed_s <= 0:
        raise ValueError("successive commands require positive elapsed time")
    heading_delta = (current[0] - previous[0] + 180) % 360 - 180
    speed_delta = current[1] - previous[1]
    return float(
        (
            (heading_delta / config.heading_scale_deg) ** 2
            + (speed_delta / config.speed_scale_knots) ** 2
        )
        / elapsed_s
    )


class RewardHooks:
    """Sample physical costs each simulator second in either competition API.

    Subclasses set reward_config before upstream initialization. State is keyed by
    aircraft ID because BlueSky traffic indices shift when another aircraft leaves.
    """

    def reset(self, *args, **kwargs):
        self._reward_state = {}
        self._reward_geometry = None
        return super().reset(*args, **kwargs)

    def _goal_distance(self, ac_id):
        import bluesky as bs
        from core.tools import kwikqdrdist

        index = bs.traf.id2idx(ac_id)
        goal = (
            self._goal[ac_id]
            if hasattr(self, "_goal")
            else (self.goal_lat, self.goal_lon)
        )
        return (
            float(kwikqdrdist(bs.traf.lat[index], bs.traf.lon[index], *goal)[1]) * 1852
        )

    def _init_metrics(self, ac_id):
        super()._init_metrics(ac_id)
        if self.reward_config.enabled:
            self._reward_state[ac_id] = {
                "totals": dict.fromkeys(COMPONENTS, 0.0),
                "baseline": dict.fromkeys(COMPONENTS, 0.0),
                "distance": self._goal_distance(ac_id)
                if self.reward_config.progress_weight
                else 0.0,
                "command": None,
                "command_time": 0.0,
            }

    def _on_action_target(self, ac_id, heading, speed):
        if not self.reward_config.enabled:
            return
        state = self._reward_state[ac_id]
        state["baseline"] = dict(state["totals"])
        now = self.metrics[ac_id]["flight_time"]
        config = self.reward_config
        if config.smoothness_weight:
            state["totals"]["smoothness"] -= config.smoothness_weight * smoothness_cost(
                state["command"], (heading, speed), now - state["command_time"], config
            )
        state["command"] = (heading, speed)
        state["command_time"] = now

    def _geometry(self):
        from shapely.geometry import Polygon

        if self._reward_geometry is None:
            center = self.scenario.center
            self._reward_geometry = (
                Polygon(local_meters(self.scenario.sector, center)),
                [
                    Polygon(local_meters(obstacle.vertices, center))
                    for obstacle in self.scenario.obstacles
                ],
            )
        return self._reward_geometry

    def _sample_costs(self, ac_id):
        import bluesky as bs
        from shapely.geometry import Point

        config = self.reward_config
        costs = dict.fromkeys(("predictive", "obstacle", "sector"), 0.0)
        index = bs.traf.id2idx(ac_id)
        position = local_meters(
            (bs.traf.lat[index], bs.traf.lon[index]), self.scenario.center
        )
        if config.predictive_weight:
            others = (
                self.agents if hasattr(self, "possible_agents") else self.intruder_ids
            )
            indices = [bs.traf.id2idx(other) for other in others if other != ac_id]
            indices = np.asarray([other for other in indices if other >= 0], dtype=int)
            positions = local_meters(
                np.column_stack([bs.traf.lat[indices], bs.traf.lon[indices]]),
                self.scenario.center,
            )
            velocities = np.column_stack(
                [bs.traf.gseast[indices], bs.traf.gsnorth[indices]]
            )
            velocity = np.array([bs.traf.gseast[index], bs.traf.gsnorth[index]])
            costs["predictive"] = predictive_risk(
                positions - position,
                velocities - velocity,
                config.prediction_horizon_s,
                config.separation_distance_m,
            )
        if config.obstacle_weight or config.sector_weight:
            sector, obstacles = self._geometry()
            point = Point(position)
            if config.obstacle_weight and obstacles:
                clearance = min(
                    signed_clearance(point, polygon, inside_positive=False)
                    for polygon in obstacles
                )
                costs["obstacle"] = clearance_cost(clearance, config.obstacle_buffer_m)
            if config.sector_weight:
                clearance = signed_clearance(point, sector, inside_positive=True)
                costs["sector"] = clearance_cost(clearance, config.sector_buffer_m)
        return costs

    def _update_metrics(self, ac_id, dt=None):
        super()._update_metrics(ac_id, dt=dt)
        elapsed = self.sim_dt if dt is None else dt
        if not self.reward_config.enabled or elapsed == 0:
            return
        config = self.reward_config
        state = self._reward_state[ac_id]
        totals = state["totals"]
        if config.progress_weight:
            distance = self._goal_distance(ac_id)
            totals["progress"] += (
                config.progress_weight
                * (state["distance"] - distance)
                / config.progress_scale_m
            )
            state["distance"] = distance
        if config.predictive_weight or config.obstacle_weight or config.sector_weight:
            for name, cost in self._sample_costs(ac_id).items():
                totals[name] -= getattr(config, f"{name}_weight") * cost * elapsed
        totals["time"] -= config.time_weight * elapsed

    def _reward_components(self, ac_id):
        state = self._reward_state[ac_id]
        return {
            name: state["totals"][name] - state["baseline"][name] for name in COMPONENTS
        }

    def _get_reward(self, ac_id):
        reward = super()._get_reward(ac_id)
        if self.reward_config.enabled:
            reward += sum(self._reward_components(ac_id).values())
        return reward

    def _get_info(self, ac_id):
        info = super()._get_info(ac_id)
        if self.reward_config.enabled:
            info["reward_components"] = self._reward_components(ac_id)
            info["reward_component_totals"] = dict(self._reward_state[ac_id]["totals"])
        return info
