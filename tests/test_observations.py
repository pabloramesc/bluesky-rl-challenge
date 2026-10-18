"""Geometry encodings preserve information, frame conventions, and padding."""

import numpy as np
import pytest
from gymnasium import spaces

from bluesky_pabloramesc.actions import ControlConfig
from bluesky_pabloramesc.observations import (
    OBSERVATION_ENCODINGS,
    encode_geometry,
    extend_observation_space,
)


def geometry():
    return {
        "cos_drift": np.array([0.0]),
        "sin_drift": np.array([-1.0]),  # Goal to the right of ownship.
        "waypoint_distance": np.array([2.0]),
        "obstacle_distance": np.array([3.0, 0.0]),
        "obstacle_cos_bearing": np.array([1.0, 0.0]),
        "obstacle_sin_bearing": np.array([0.0, 0.0]),
        "sector_point_distance": np.array([4.0]),
        "sector_point_cos_bearing": np.array([0.0]),
        "sector_point_sin_bearing": np.array([-1.0]),
        "x_r": np.array([5.0, 0.0]),
        "y_r": np.array([0.0, 0.0]),
        "vx_r": np.array([0.0, 0.0]),
        "vy_r": np.array([1.0, 0.0]),
        "cos_track": np.array([0.0, 0.0]),
        "sin_track": np.array([1.0, 0.0]),
        "airspeed": np.array([0.5]),
    }


def test_ego_geometry_signs_scales_and_padding():
    original = geometry()
    encoded = encode_geometry(original, "ego_cartesian")
    np.testing.assert_allclose(encoded["goal_x"], [0])
    np.testing.assert_allclose(encoded["goal_y"], [2])
    np.testing.assert_allclose(encoded["obstacle_x"], [3, 0])
    np.testing.assert_allclose(encoded["obstacle_y"], [0, 0])
    np.testing.assert_allclose(encoded["sector_point_y"], [4])
    np.testing.assert_array_equal(encoded["x_r"], original["x_r"])
    assert "waypoint_distance" in original
    assert "waypoint_distance" not in encoded


def test_goal_frame_rotates_positions_velocities_and_tracks_together():
    encoded = encode_geometry(geometry(), "goal_aligned")
    np.testing.assert_allclose(encoded["obstacle_x"], [0, 0])
    np.testing.assert_allclose(encoded["obstacle_y"], [-3, 0])
    np.testing.assert_allclose(encoded["sector_point_x"], [4])
    np.testing.assert_allclose(encoded["y_r"], [-5, 0])
    np.testing.assert_allclose(encoded["vx_r"], [1, 0])
    np.testing.assert_allclose(encoded["cos_track"], [1, 0])
    np.testing.assert_allclose(encoded["sin_track"], [0, 0])
    np.testing.assert_allclose(encoded["sin_drift"], [-1])
    np.testing.assert_allclose(encoded["waypoint_distance"], [2])


@pytest.mark.parametrize("encoding", OBSERVATION_ENCODINGS)
@pytest.mark.parametrize("profile", ["legacy", "control"])
def test_encoding_space_matches_flattened_observation(encoding, profile):
    from bluesky_pabloramesc.observations import ownship_features

    original = geometry()
    space = spaces.Dict(
        {
            key: spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float64)
            for key, value in original.items()
        }
    )
    config = ControlConfig(observation_profile=profile, observation_encoding=encoding)
    encoded_space = extend_observation_space(space, config)
    encoded = encode_geometry(original, encoding)
    if profile == "control":
        encoded = {**encoded, **ownship_features(359, 250)}
    assert encoded_space.contains(encoded)
    assert np.isfinite(spaces.flatten(encoded_space, encoded)).all()


def test_polar_wraparound_is_continuous_and_zero_distance_is_finite():
    observations = []
    for angle in [-180 + 1e-6, 180 - 1e-6]:
        original = geometry()
        original["cos_drift"] = np.cos(np.deg2rad([angle]))
        original["sin_drift"] = np.sin(np.deg2rad([angle]))
        original["waypoint_distance"][:] = 0
        observations.append(encode_geometry(original, "goal_aligned"))
    for key in observations[0]:
        np.testing.assert_allclose(
            observations[0][key], observations[1][key], atol=1e-6
        )
        assert np.isfinite(observations[0][key]).all()


def test_goal_frame_is_invariant_to_a_change_of_body_axes():
    original = geometry()
    rotated = {key: value.copy() for key, value in original.items()}
    angle = np.deg2rad(37)
    c, s = np.cos(angle), np.sin(angle)
    # Re-express the same scene after rotating the ownship axes by 37 degrees.
    for cos_key, sin_key in (
        ("cos_drift", "sin_drift"),
        ("obstacle_cos_bearing", "obstacle_sin_bearing"),
        ("sector_point_cos_bearing", "sector_point_sin_bearing"),
    ):
        x, y = original[cos_key], -original[sin_key]
        rotated[cos_key] = c * x + s * y
        rotated[sin_key] = s * x - c * y
    for x_key, y_key in (("x_r", "y_r"), ("vx_r", "vy_r"), ("cos_track", "sin_track")):
        x, y = original[x_key], original[y_key]
        rotated[x_key] = c * x + s * y
        rotated[y_key] = -s * x + c * y
    expected = encode_geometry(original, "goal_aligned")
    actual = encode_geometry(rotated, "goal_aligned")
    for key in expected.keys() - {"cos_drift", "sin_drift"}:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)


@pytest.mark.parametrize(
    "position, velocity, expected",
    [
        ((10000, 0), (-100, 0), (100, 0, 100)),  # Head on.
        ((10000, 0), (100, 0), (-100, 10000, -100)),  # Receding.
        ((10000, 10000), (-100, 0), (100, 10000, 100 / np.sqrt(2))),
        ((10000, 0), (0, 0), (0, 10000, 0)),  # Parallel.
        ((0, 0), (0, 0), (0, 0, 0)),  # Coincident.
        ((10000, 0), (-1e-8, 0), (0, 10000, 1e-8)),
    ],
)
def test_conflict_analytic_encounters(position, velocity, expected):
    from bluesky_pabloramesc.observations import conflict_features

    observation = {
        "x_r": np.array([position[0] / 1000, 0]),
        "y_r": np.array([position[1] / 1000, 0]),
        "vx_r": np.array([velocity[0] / 50, 0]),
        "vy_r": np.array([velocity[1] / 50, 0]),
        "cos_track": np.array([1.0, 0]),
        "sin_track": np.zeros(2),
    }
    features = conflict_features(observation, pos_norm=1000, spd_norm=50)
    for key, value, scale in zip(
        ("tcpa", "dcpa", "closing_rate"), expected, (600, 10000, 150)
    ):
        np.testing.assert_allclose(features[key], [value / scale, 0], atol=1e-12)
    np.testing.assert_array_equal(features["intruder_present"], [1, 0])


def test_conflict_time_is_clipped_without_changing_cpa_distance():
    original = geometry()
    original["x_r"] = np.array([1.0, 0])
    original["vx_r"] = np.array([-1 / 150, 0])
    original["vy_r"] = np.zeros(2)
    encoded = encode_geometry(original, "conflict")
    np.testing.assert_allclose(encoded["tcpa"], [1, 0])
    np.testing.assert_allclose(encoded["dcpa"], [0, 0])
    for key in original:
        np.testing.assert_array_equal(encoded[key], original[key])


def test_conflict_hook_uses_upstream_scales_and_preserves_slot_order():
    from types import SimpleNamespace

    from bluesky_pabloramesc.observations import ControlHooks

    original = geometry()
    original["x_r"] = np.array([10.0, 20.0])
    original["vx_r"] = np.array([-2.0, 2.0])
    original["vy_r"] = np.zeros(2)
    original["cos_track"] = np.ones(2)

    class Base:
        def _get_obs(self, ac_id):
            return original

    class Env(ControlHooks, Base):
        control_config = ControlConfig(observation_encoding="conflict")
        intruder_obs = SimpleNamespace(pos_norm=1000, spd_norm=50)

    encoded = Env()._get_obs("AC1")
    np.testing.assert_allclose(encoded["tcpa"], [100 / 600, -200 / 600])
    np.testing.assert_allclose(encoded["dcpa"], [0, 2])
    np.testing.assert_allclose(encoded["closing_rate"], [100 / 150, -100 / 150])


def test_conflict_features_are_invariant_to_rotation():
    original = geometry()
    rotated = {key: value.copy() for key, value in original.items()}
    angle = np.deg2rad(37)
    c, s = np.cos(angle), np.sin(angle)
    for x_key, y_key in (("x_r", "y_r"), ("vx_r", "vy_r")):
        rotated[x_key] = c * original[x_key] + s * original[y_key]
        rotated[y_key] = -s * original[x_key] + c * original[y_key]
    expected = encode_geometry(original, "conflict")
    actual = encode_geometry(rotated, "conflict")
    for key in ("tcpa", "dcpa", "closing_rate", "intruder_present"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)


@pytest.mark.parametrize("base", ["ego_cartesian", "goal_aligned"])
def test_combined_encoding_only_appends_conflict_features(base):
    original = geometry()
    expected_base = encode_geometry(original, base)
    expected_conflict = encode_geometry(original, "conflict")
    combined = encode_geometry(original, f"{base}_conflict")
    assert (
        combined.keys()
        == expected_conflict.keys() - original.keys() | expected_base.keys()
    )
    for key, value in expected_base.items():
        np.testing.assert_array_equal(combined[key], value)
    for key in ("tcpa", "dcpa", "closing_rate", "intruder_present"):
        np.testing.assert_array_equal(combined[key], expected_conflict[key])
