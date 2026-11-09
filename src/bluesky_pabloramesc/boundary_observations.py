"""Exact polygon boundary observations, independent of traffic encoding."""

import numpy as np
from gymnasium import spaces

from .geometry import nearest_boundary, ray_distances

GEOMETRY_REPRESENTATIONS = ("current", "nearest_edge", "ray_cast", "both")
GEOMETRY_CHOICES = (*GEOMETRY_REPRESENTATIONS, "current_rays")
DISTANCE_SCALE_M = 100_000.0


def boundary_space(fields, config, obstacle_tokens=False):
    """Replace or extend circle/perimeter fields with fixed-size observations."""
    if config.geometry_representation == "current":
        return fields
    nearest = config.geometry_representation in ("nearest_edge", "both")
    append_rays = config.geometry_representation == "current_rays"
    if obstacle_tokens and not nearest and not append_rays:
        raise ValueError(
            "Obstacle token encoders require nearest_edge, both, or current_rays"
        )
    count = fields["obstacle_radius"].shape[0]
    result = (
        dict(fields)
        if append_rays
        else {
            key: field
            for key, field in fields.items()
            if not key.startswith(("obstacle_", "sector_point_"))
        }
    )

    def add(key, size, low=-np.inf, high=np.inf):
        result[key] = spaces.Box(low, high, shape=(size,), dtype=np.float64)

    if not append_rays:
        add("inside_sector", 1, 0, 1)
    add("inside_obstacle", 1, 0, 1)
    if nearest:
        for key in ("obstacle_nearest_x", "obstacle_nearest_y", "obstacle_clearance"):
            add(key, count)
        add("obstacle_present", count, 0, 1)
        for key in ("sector_nearest_x", "sector_nearest_y", "sector_clearance"):
            add(key, 1)
    if config.geometry_representation in ("ray_cast", "both", "current_rays"):
        for prefix in ("obstacle", "sector"):
            add(f"{prefix}_ray_distance", config.geometry_ray_count, 0, 1)
            add(f"{prefix}_ray_hit", config.geometry_ray_count, 0, 1)
    return result


def boundary_features(geometry, position, heading, config, obstacle_count):
    """Encode nearest points and clockwise rays in ownship forward/right axes.

    Nearest displacements and signed clearances use a fixed 100 km scale. Rays
    use their configured maximum range. Polygon slots are ordered by unsigned
    boundary distance, with scenario order breaking ties; presence is explicit.
    """
    from shapely.geometry import Point

    point = Point(position)
    angle = np.deg2rad(heading)
    forward = np.array([np.sin(angle), np.cos(angle)])
    right = np.array([np.cos(angle), -np.sin(angle)])
    result = {
        "inside_obstacle": np.array(
            [float(any(p.covers(point) for p in geometry.obstacles))]
        ),
    }
    if config.geometry_representation != "current_rays":
        result["inside_sector"] = np.array([float(geometry.sector.covers(point))])
    if config.geometry_representation in ("nearest_edge", "both"):
        for prefix, polygons, count, inside_positive in (
            ("obstacle", geometry.obstacles, obstacle_count, False),
            ("sector", [geometry.sector], 1, True),
        ):
            samples = [
                nearest_boundary(point, polygon, inside_positive=inside_positive)
                for polygon in polygons
            ]
            if prefix == "obstacle":
                samples.sort(key=lambda sample: abs(sample[1]))
                result["obstacle_present"] = np.zeros(count)
                result["obstacle_present"][: min(count, len(samples))] = 1
            x, y, clearance = np.zeros((3, count))
            for index, (offset, distance) in enumerate(samples[:count]):
                x[index] = offset @ forward / DISTANCE_SCALE_M
                y[index] = offset @ right / DISTANCE_SCALE_M
                clearance[index] = distance / DISTANCE_SCALE_M
            result[f"{prefix}_nearest_x"] = x
            result[f"{prefix}_nearest_y"] = y
            result[f"{prefix}_clearance"] = clearance
    if config.geometry_representation in ("ray_cast", "both", "current_rays"):
        angles = (
            np.arange(config.geometry_ray_count) * 2 * np.pi / config.geometry_ray_count
        )
        directions = np.cos(angles)[:, None] * forward + np.sin(angles)[:, None] * right
        for prefix, boundary in (
            ("obstacle", geometry.obstacle_boundary),
            ("sector", geometry.sector.boundary),
        ):
            distances, hits = ray_distances(
                point, boundary, directions, config.geometry_ray_range_m
            )
            result[f"{prefix}_ray_distance"] = distances / config.geometry_ray_range_m
            result[f"{prefix}_ray_hit"] = hits
    return result
