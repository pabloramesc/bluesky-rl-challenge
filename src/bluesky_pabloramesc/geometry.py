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
