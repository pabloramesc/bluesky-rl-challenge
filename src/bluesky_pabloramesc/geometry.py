"""Physical geometry shared by observations and reward shaping."""

import numpy as np


def closest_approach(position, velocity):
    """Return signed TCPA and future DCPA for relative vectors in metres and m/s."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    speed_squared = np.sum(velocity**2, axis=-1)
    tcpa = np.divide(
        -np.sum(position * velocity, axis=-1),
        speed_squared,
        out=np.zeros_like(speed_squared),
        where=speed_squared > 1e-6,
    )
    dcpa = np.linalg.norm(position + np.maximum(tcpa, 0)[..., None] * velocity, axis=-1)
    return tcpa, dcpa


def local_meters(lat_lon, center):
    """Project lat/lon to east/north metres in a fixed local tangent approximation."""
    coordinates = np.asarray(lat_lon, dtype=float)
    delta = coordinates - center
    longitude = (delta[..., 1] + 180) % 360 - 180
    return np.stack(
        [longitude * np.cos(np.deg2rad(center[0])), delta[..., 0]], axis=-1
    ) * (60 * 1852)


def signed_clearance(point, polygon, *, inside_positive):
    """Distance to polygon edges, positive on the requested safe side."""
    distance = point.distance(polygon.boundary)
    return distance if polygon.covers(point) == inside_positive else -distance


class PolygonGeometry:
    """Static scenario polygons in a shared east/north metre projection."""

    def __init__(self, scenario):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        self.center = np.asarray(scenario.center, dtype=float)
        self.sector = Polygon(local_meters(scenario.sector, self.center))
        self.obstacles = [
            Polygon(local_meters(obstacle.vertices, self.center))
            for obstacle in scenario.obstacles
        ]
        for polygon in [self.sector, *self.obstacles]:
            if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                raise ValueError(
                    "Geometry observations require valid nonempty polygons"
                )
        self.obstacle_boundary = unary_union([p.boundary for p in self.obstacles])


def nearest_boundary(point, polygon, *, inside_positive):
    """Return displacement to the actual boundary and signed safe-side clearance."""
    from shapely.ops import nearest_points

    nearest = nearest_points(point, polygon.boundary)[1]
    displacement = np.array([nearest.x - point.x, nearest.y - point.y])
    return displacement, signed_clearance(
        point, polygon, inside_positive=inside_positive
    )


def ray_distances(point, boundary, directions, max_range):
    """Intersect finite unit-direction rays, including tangencies and overlaps.

    No hit returns max_range and a false mask. A boundary origin returns zero;
    from an unsafe position a hit can be an exit, so retain occupancy separately.
    """
    from shapely.geometry import LineString

    directions = np.asarray(directions, dtype=float)
    origin = np.array([point.x, point.y])
    distances = np.full(len(directions), float(max_range))
    hits = np.zeros(len(directions), dtype=np.float64)
    if boundary.is_empty:
        return distances, hits
    for index, direction in enumerate(directions):
        ray = LineString([origin, origin + max_range * direction])
        intersection = ray.intersection(boundary)
        if not intersection.is_empty:
            distances[index] = min(max_range, point.distance(intersection))
            hits[index] = 1
    return distances, hits
