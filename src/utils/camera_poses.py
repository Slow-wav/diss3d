import math
from typing import Iterable


def _normalize(vector: Iterable[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return [value / norm for value in values]


def _cross(a: Iterable[float], b: Iterable[float]) -> list[float]:
    ax, ay, az = a
    bx, by, bz = b
    return [
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ]


def _subtract(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def look_at_camera_matrix(
    position: Iterable[float],
    target: Iterable[float] = (0.0, 0.0, 0.0),
    up: Iterable[float] = (0.0, 0.0, 1.0),
) -> list[list[float]]:
    position_vec = [float(value) for value in position]
    forward = _normalize(_subtract(target, position_vec))
    right = _cross(forward, up)

    right_norm = math.sqrt(sum(value * value for value in right))
    if right_norm == 0:
        right = _cross(forward, (0.0, 1.0, 0.0))

    right = _normalize(right)
    true_up = _normalize(_cross(right, forward))

    return [
        [right[0], true_up[0], -forward[0], position_vec[0]],
        [right[1], true_up[1], -forward[1], position_vec[1]],
        [right[2], true_up[2], -forward[2], position_vec[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def generate_orbit_camera_poses(
    num_views: int,
    radius: float,
    elevation_deg: float,
    azimuth_start_deg: float = 0.0,
    orbit_degrees: float = 360.0,
) -> list[dict[str, object]]:
    if num_views <= 0:
        raise ValueError("num_views must be greater than zero.")

    elevation_rad = math.radians(elevation_deg)
    poses: list[dict[str, object]] = []

    for index in range(num_views):
        if math.isclose(orbit_degrees, 360.0) and num_views > 1:
            step_fraction = index / num_views
        elif num_views == 1:
            step_fraction = 0.0
        else:
            step_fraction = index / (num_views - 1)

        azimuth_deg = azimuth_start_deg + (orbit_degrees * step_fraction)
        azimuth_rad = math.radians(azimuth_deg)

        x = radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = radius * math.sin(elevation_rad)
        position = [round(x, 6), round(y, 6), round(z, 6)]

        poses.append(
            {
                "view_index": index,
                "radius": radius,
                "elevation_deg": elevation_deg,
                "azimuth_deg": round(azimuth_deg, 6),
                "position": position,
                "transform_matrix": look_at_camera_matrix(position),
            }
        )

    return poses
