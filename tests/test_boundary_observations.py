"""Exact boundary geometry, observation wiring, and checkpoint settings."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces
from shapely.geometry import LineString, Point, Polygon
from test_observations import geometry

from bluesky_pabloramesc.actions import ControlConfig
from bluesky_pabloramesc.boundary_observations import (
    GEOMETRY_REPRESENTATIONS,
    boundary_features,
)
from bluesky_pabloramesc.geometry import (
    PolygonGeometry,
    nearest_boundary,
    ray_distances,
)
from bluesky_pabloramesc.observations import (
    OBSERVATION_ENCODINGS,
    ControlHooks,
    encode_geometry,
    extend_observation_space,
)


def scenario(obstacles=()):
    def latlon(vertices):
        return np.asarray(vertices, dtype=float)[:, ::-1] / 111120

    return SimpleNamespace(
        center=(0, 0),
        sector=latlon([(-20, -20), (20, -20), (20, 20), (-20, 20)]),
        obstacles=[
            SimpleNamespace(vertices=latlon(vertices)) for vertices in obstacles
        ],
    )


def original():
    return {
        **geometry(),
        "obstacle_radius": np.array([1.0, 0.0]),
        "inside_sector": np.ones(1),
    }


def test_nearest_point_uses_segment_interior_and_safe_side():
    polygon = Polygon([(0, -100), (2, -100), (2, 100), (0, 100)])
    offset, clearance = nearest_boundary(Point(-1, 0), polygon, inside_positive=False)
    np.testing.assert_allclose(offset, [1, 0])
    assert clearance == 1
    assert nearest_boundary(Point(1, 0), polygon, inside_positive=False)[1] == -1
    assert nearest_boundary(Point(0, 0), polygon, inside_positive=False)[1] == 0
    concave = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
    assert nearest_boundary(Point(2, 2), concave, inside_positive=True)[1] == -1


def test_rays_cardinal_corner_collinear_boundary_and_no_hit():
    polygon = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    directions = np.array([(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)])
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    distance, hits = ray_distances(Point(0, 0), polygon.boundary, directions, 10)
    np.testing.assert_allclose(distance, [2, 2, 2, 2, np.sqrt(8)])
    np.testing.assert_array_equal(hits, 1)
    distance, hits = ray_distances(Point(0, 0), polygon.boundary, directions, 1)
    np.testing.assert_array_equal(distance, 1)
    np.testing.assert_array_equal(hits, 0)
    assert ray_distances(Point(2, 0), polygon.boundary, [(1, 0)], 10)[0][0] == 0
    distance, hits = ray_distances(
        Point(0, 0), LineString([(2, 0), (4, 0)]), [(1, 0)], 2
    )
    assert distance[0] == 2 and hits[0] == 1  # Inclusive range and overlap.
    distance, hits = ray_distances(Point(-4, 0), polygon.boundary, [(1, 0)], 10)
    assert distance[0] == 2 and hits[0] == 1  # Outside sector: first entry.


def test_nearest_order_padding_and_ray_directions():
    world = PolygonGeometry(
        scenario(
            [
                [(10, -2), (12, -2), (12, 2), (10, 2)],
                [(-2, 5), (2, 5), (2, 7), (-2, 7)],
            ]
        )
    )
    config = ControlConfig(
        geometry_representation="both", geometry_ray_count=4, geometry_ray_range_m=100
    )
    features = boundary_features(world, (0, 0), 0, config, 3)
    np.testing.assert_allclose(
        features["obstacle_nearest_x"] * 100000, [5, 0, 0], atol=1e-8
    )
    np.testing.assert_allclose(
        features["obstacle_nearest_y"] * 100000, [0, 10, 0], atol=1e-8
    )
    np.testing.assert_array_equal(features["obstacle_present"], [1, 1, 0])
    np.testing.assert_allclose(
        features["obstacle_ray_distance"], [0.05, 0.1, 1, 1], atol=1e-8
    )
    np.testing.assert_array_equal(features["obstacle_ray_hit"], [1, 1, 0, 0])
    np.testing.assert_allclose(features["sector_ray_distance"], 0.2, atol=1e-8)
    rotated = boundary_features(world, (0, 0), 90, config, 3)
    np.testing.assert_allclose(
        rotated["obstacle_ray_distance"], [0.1, 1, 1, 0.05], atol=1e-8
    )
    unsafe = boundary_features(world, (0, 6), 0, config, 3)
    assert unsafe["inside_obstacle"][0] == 1
    assert unsafe["obstacle_clearance"][0] < 0


def test_empty_obstacles_remain_finite_and_explicit():
    features = boundary_features(
        PolygonGeometry(scenario()),
        (0, 0),
        0,
        ControlConfig(geometry_representation="both"),
        2,
    )
    np.testing.assert_array_equal(features["obstacle_present"], 0)
    np.testing.assert_array_equal(features["obstacle_ray_distance"], 1)
    np.testing.assert_array_equal(features["obstacle_ray_hit"], 0)
    assert all(np.isfinite(value).all() for value in features.values())


@pytest.mark.parametrize("encoding", OBSERVATION_ENCODINGS)
@pytest.mark.parametrize("representation", GEOMETRY_REPRESENTATIONS)
@pytest.mark.parametrize("obstacle_tokens", [False, True])
def test_hook_space_matches_observation_and_reset_invalidates_cache(
    monkeypatch, encoding, representation, obstacle_tokens
):
    monkeypatch.setitem(
        sys.modules,
        "bluesky",
        SimpleNamespace(
            traf=SimpleNamespace(
                id2idx=lambda _: 0,
                lat=[0],
                lon=[0],
                hdg=[0],
                cas=[150],
            )
        ),
    )

    class Base:
        def _get_obs(self, ac_id):
            return original()

        def reset(self, world):
            self.scenario = world
            return self._get_obs("a")

    class Env(ControlHooks, Base):
        intruder_obs = SimpleNamespace(pos_norm=1_000_000, spd_norm=150)
        obstacle_observations = obstacle_tokens
        entity_observations = True
        control_config = ControlConfig(
            observation_encoding=encoding,
            geometry_representation=representation,
            observation_profile="control",
        )

    env = Env()
    raw_space = spaces.Dict(
        {
            key: spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float64)
            for key, value in original().items()
        }
    )
    if obstacle_tokens and representation == "ray_cast":
        with pytest.raises(ValueError, match="Obstacle token"):
            extend_observation_space(
                raw_space, env.control_config, obstacle_observations=True
            )
        return
    space = extend_observation_space(
        raw_space,
        env.control_config,
        entity_observations=True,
        obstacle_observations=obstacle_tokens,
    )
    obs = env.reset(scenario())
    assert space.contains(obs)
    if representation != "current":
        assert "obstacle_radius" not in obs
        cache = env._boundary_geometry
        obs = env.reset(scenario([[(-1, -1), (1, -1), (1, 1), (-1, 1)]]))
        assert env._boundary_geometry is not cache
        assert obs["inside_obstacle"][0] == 1
        if representation in ("nearest_edge", "both"):
            from bluesky_pabloramesc.entities import EntityRunningMeanStd, entity_layout

            flat = spaces.flatten(space, obs)
            layout = entity_layout(space)
            indices = np.asarray(layout["obstacles"])
            assert indices.shape == (2, 3)
            assert not set(indices.ravel()) & set(layout["state_context"])
            rms = EntityRunningMeanStd(flat.shape, layout)
            rms.update(flat[None])
            assert rms.obstacle_rms.count == pytest.approx(1.0001)
            np.testing.assert_array_equal(flat[layout["obstacle_mask"]], [1, 0])
    else:
        encoded = encode_geometry(original(), encoding)
        for key, value in encoded.items():
            np.testing.assert_array_equal(obs[key], value)


@pytest.mark.parametrize(
    "settings",
    [
        {"geometry_representation": "invalid"},
        {"geometry_ray_count": 0},
        {"geometry_ray_count": True},
        {"geometry_ray_range_m": float("nan")},
        {"geometry_ray_range_m": 0},
    ],
)
def test_invalid_geometry_settings(settings):
    with pytest.raises(ValueError):
        ControlConfig(**settings)


def test_checkpoint_geometry_settings_reload_and_cannot_be_overridden(tmp_path):
    import json

    from bluesky_pabloramesc.cli import parse_args
    from bluesky_pabloramesc.training import control_config_path

    path = tmp_path / "policy"
    config = ControlConfig(geometry_representation="both", geometry_ray_count=8)
    control_config_path(path).write_text(json.dumps(config.to_dict()))
    args = parse_args("ppo", argv=["evaluate", "--model-path", str(path)])
    assert args.control_config == config
    with pytest.raises(SystemExit):
        parse_args(
            "ppo",
            argv=[
                "evaluate",
                "--model-path",
                str(path),
                "--geometry-representation",
                "current",
            ],
        )


def test_current_defaults_preserve_legacy_metadata():
    settings = ControlConfig().to_dict()
    assert not any(key.startswith("geometry_") for key in settings)
    assert ControlConfig(**settings) == ControlConfig()
    alternative = ControlConfig(geometry_representation="nearest_edge")
    assert alternative.to_dict()["geometry_representation"] == "nearest_edge"
    assert ControlConfig(**alternative.to_dict()) == alternative
