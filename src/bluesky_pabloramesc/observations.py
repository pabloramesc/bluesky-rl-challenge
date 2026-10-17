"""Shared observation and command hooks for the competition environments."""

import numpy as np
from gymnasium import spaces

from .actions import KNOTS_PER_MPS, action_targets

OBSERVATION_ENCODINGS = ("upstream", "ego_cartesian", "goal_aligned")


def encode_geometry(observation, encoding):
    """Re-express existing geometry, retaining upstream scales and slot order.

    Upstream polar angles are heading minus bearing. Cartesian axes point
    forward/right for ego encoding and toward/right of the goal for goal encoding.
    No simulator state or predictive features are added.
    """
    if encoding == "upstream":
        return observation
    if encoding not in OBSERVATION_ENCODINGS:
        raise ValueError(f"unknown observation encoding: {encoding}")
    result = dict(observation)
    goal_cos = observation["cos_drift"]
    goal_sin = -observation["sin_drift"]

    def rotate(x, y):
        if encoding == "ego_cartesian":
            return x, y
        return x * goal_cos + y * goal_sin, -x * goal_sin + y * goal_cos

    groups = [
        (
            "obstacle",
            "obstacle_distance",
            "obstacle_cos_bearing",
            "obstacle_sin_bearing",
        ),
        (
            "sector_point",
            "sector_point_distance",
            "sector_point_cos_bearing",
            "sector_point_sin_bearing",
        ),
    ]
    if encoding == "ego_cartesian":
        groups.append(("goal", "waypoint_distance", "cos_drift", "sin_drift"))
    for prefix, distance_key, cos_key, sin_key in groups:
        distance = result.pop(distance_key)
        x = distance * result.pop(cos_key)
        y = -distance * result.pop(sin_key)
        result[f"{prefix}_x"], result[f"{prefix}_y"] = rotate(x, y)

    if encoding == "goal_aligned":
        for x_key, y_key in (
            ("x_r", "y_r"),
            ("vx_r", "vy_r"),
            ("cos_track", "sin_track"),
        ):
            result[x_key], result[y_key] = rotate(
                observation[x_key], observation[y_key]
            )
    return result


def extend_observation_space(space, config):
    """Append ownship state before flattening, preserving legacy spaces exactly."""
    if (
        config.observation_encoding == "upstream"
        and config.observation_profile == "legacy"
    ):
        return space
    fields = dict(space.spaces)
    if config.observation_encoding != "upstream":
        sample = {key: np.zeros(value.shape) for key, value in fields.items()}
        sample["cos_drift"][:] = 1
        encoded = encode_geometry(sample, config.observation_encoding)
        fields = {
            key: fields[key]
            if key in fields
            else spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float64)
            for key, value in encoded.items()
        }
    if config.observation_profile == "control":
        for key in ("own_heading_sin", "own_heading_cos"):
            fields[key] = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
        fields["own_cas"] = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
    return spaces.Dict(fields)


def ownship_features(heading, cas_knots):
    """Encode circular heading and CAS, centered at 300 kt with 100 kt scale."""
    radians = np.deg2rad(heading)
    return {
        "own_heading_sin": np.array([np.sin(radians)]),
        "own_heading_cos": np.array([np.cos(radians)]),
        "own_cas": np.array([(cas_knots - 300) / 100]),
    }


class ControlHooks:
    """Share upstream observation/action extension points without owning a simulator."""

    def _get_obs(self, ac_id):
        observation = encode_geometry(
            super()._get_obs(ac_id), self.control_config.observation_encoding
        )
        if self.control_config.observation_profile == "control":
            import bluesky as bs

            index = bs.traf.id2idx(ac_id)
            observation.update(
                ownship_features(bs.traf.hdg[index], bs.traf.cas[index] * KNOTS_PER_MPS)
            )
        return observation

    def _get_action(self, ac_id, action):
        import bluesky as bs

        index = bs.traf.id2idx(ac_id)
        heading, speed = action_targets(
            action,
            bs.traf.hdg[index],
            bs.traf.cas[index] * KNOTS_PER_MPS,
            self.control_config,
        )
        bs.stack.stack(f"HDG {ac_id} {heading}")
        bs.stack.stack(f"SPD {ac_id} {speed}")
