"""Shared observation and command hooks for the competition environments."""

import numpy as np
from gymnasium import spaces

from .actions import KNOTS_PER_MPS, action_targets

DEFAULT_OBSERVATION_ENCODINGS = (
    "upstream",
    "ego_cartesian",
    "goal_aligned",
    "conflict",
)
OBSERVATION_ENCODINGS = DEFAULT_OBSERVATION_ENCODINGS + (
    "ego_cartesian_conflict",
    "goal_aligned_conflict",
)


def conflict_features(observation, pos_norm=1_000_000, spd_norm=150):
    """Constant-velocity horizontal CPA in existing slots, with finite padding.

    TCPA is signed and scaled/clipped to +/-600 s. DCPA is the minimum
    future distance (current range for receding/parallel traffic), scaled by
    10 km. Closing rate is positive when approaching, scaled by 150 m/s.
    Track components distinguish occupied slots even at coincident positions.
    """
    x = np.asarray(observation["x_r"], dtype=float) * pos_norm
    y = np.asarray(observation["y_r"], dtype=float) * pos_norm
    vx = np.asarray(observation["vx_r"], dtype=float) * spd_norm
    vy = np.asarray(observation["vy_r"], dtype=float) * spd_norm
    present = (observation["cos_track"] ** 2 + observation["sin_track"] ** 2) > 0
    distance = np.hypot(x, y)
    speed_squared = vx * vx + vy * vy
    dot = x * vx + y * vy
    tcpa = np.divide(
        -dot, speed_squared, out=np.zeros_like(x), where=speed_squared > 1e-6
    )
    future_time = np.maximum(tcpa, 0)
    dcpa = np.hypot(x + future_time * vx, y + future_time * vy)
    closing = np.divide(-dot, distance, out=np.zeros_like(x), where=distance > 0)
    return {
        "tcpa": np.where(present, np.clip(tcpa / 600, -1, 1), 0),
        "dcpa": np.where(present, dcpa / 10_000, 0),
        "closing_rate": np.where(present, closing / 150, 0),
        "intruder_present": present.astype(np.float64),
    }


def encode_geometry(observation, encoding, pos_norm=1_000_000, spd_norm=150):
    """Re-express existing geometry, retaining upstream scales and slot order.

    Upstream polar angles are heading minus bearing. Cartesian axes point
    forward/right for ego encoding and toward/right of the goal for goal encoding.
    The conflict encoding appends constant-velocity predictive features.
    """
    if encoding == "upstream":
        return observation
    if encoding not in OBSERVATION_ENCODINGS:
        raise ValueError(f"unknown observation encoding: {encoding}")
    if encoding == "conflict" or encoding.endswith("_conflict"):
        base_encoding = (
            "upstream" if encoding == "conflict" else encoding.removesuffix("_conflict")
        )
        return {
            **encode_geometry(observation, base_encoding),
            **conflict_features(observation, pos_norm, spd_norm),
        }
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


def extend_observation_space(
    space, config, entity_observations=False, graph_radius_nm=None
):
    """Append ownship state before flattening, preserving legacy spaces exactly."""
    if (
        config.observation_encoding == "upstream"
        and config.observation_profile == "legacy"
        and not entity_observations
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
    if entity_observations:
        fields["intruder_present"] = spaces.Box(
            0, 1, shape=fields["x_r"].shape, dtype=np.float64
        )
    if graph_radius_nm is not None:
        nodes = fields["x_r"].shape[0]
        fields["graph_adjacency"] = spaces.Box(
            0, 1, shape=(nodes, nodes), dtype=np.float64
        )
    return spaces.Dict(fields)


def distance_adjacency(observation, pos_norm, radius_nm):
    """Connect occupied aircraft pairs using pre-normalization Cartesian geometry."""
    positions = np.stack([observation["x_r"], observation["y_r"]], axis=-1) * pos_norm
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    present = observation["intruder_present"] > 0.5
    return (
        (distances <= radius_nm * 1852) & present[:, None] & present[None, :]
    ).astype(np.float64)


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
        observation = super()._get_obs(ac_id)
        encoding = self.control_config.observation_encoding
        if encoding == "conflict" or encoding.endswith("_conflict"):
            observation = encode_geometry(
                observation,
                encoding,
                self.intruder_obs.pos_norm,
                self.intruder_obs.spd_norm,
            )
        else:
            observation = encode_geometry(observation, encoding)
        if self.control_config.observation_profile == "control":
            import bluesky as bs

            index = bs.traf.id2idx(ac_id)
            observation.update(
                ownship_features(bs.traf.hdg[index], bs.traf.cas[index] * KNOTS_PER_MPS)
            )
        if getattr(self, "entity_observations", False):
            observation["intruder_present"] = (
                observation["cos_track"] ** 2 + observation["sin_track"] ** 2 > 0
            ).astype(np.float64)
        radius = getattr(self, "graph_radius_nm", None)
        if radius is not None:
            observation["graph_adjacency"] = distance_adjacency(
                observation, self.intruder_obs.pos_norm, radius
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
