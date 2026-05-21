import math
import os
from pathlib import Path
from typing import Any

from .camera_poses import look_at_camera_matrix


ORBIT_CENTER = (0.0, 0.0, 0.0)


def camera_angle_x_from_mm(focal_length_mm: float, sensor_width_mm: float) -> float:
    return 2.0 * math.atan(sensor_width_mm / (2.0 * focal_length_mm))


def rotate_point_around_z(point: tuple[float, float, float], angle_rad: float) -> tuple[float, float, float]:
    x, y, z = point
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    return (
        (x * cos_theta) - (y * sin_theta),
        (x * sin_theta) + (y * cos_theta),
        z,
    )


def generate_standard_orbit_positions(
    num_views: int,
    radius: float,
    elevation_ratio: float,
) -> list[tuple[float, float, float]]:
    if num_views <= 0:
        raise ValueError("num_views must be greater than zero.")
    if radius <= 0:
        raise ValueError("radius must be greater than zero.")

    camera_start = (0.0, -radius, elevation_ratio * radius)
    angle_step = (2.0 * math.pi) / num_views

    positions: list[tuple[float, float, float]] = []
    for view_index in range(num_views):
        camera_position = rotate_point_around_z(camera_start, view_index * angle_step)
        positions.append(
            (
                round(camera_position[0], 6),
                round(camera_position[1], 6),
                round(camera_position[2], 6),
            )
        )

    return positions


def list_ordered_png_images(images_dir: Path, num_views: int) -> list[Path]:
    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    if len(image_paths) < num_views:
        raise ValueError(
            f"Expected at least {num_views} PNG images in {images_dir}, but found {len(image_paths)}."
        )
    return image_paths[:num_views]


def build_orbit_frames(
    images_dir: Path,
    output_path: Path,
    num_views: int,
    radius: float,
    elevation_ratio: float,
) -> list[dict[str, Any]]:
    image_paths = list_ordered_png_images(images_dir, num_views)
    camera_positions = generate_standard_orbit_positions(num_views, radius, elevation_ratio)

    frames: list[dict[str, Any]] = []
    for image_path, camera_position in zip(image_paths, camera_positions):
        relative_path = Path(os.path.relpath(image_path, output_path.parent)).as_posix()
        frames.append(
            {
                "file_path": relative_path,
                "transform_matrix": look_at_camera_matrix(camera_position, target=ORBIT_CENTER),
            }
        )

    return frames


def build_orbit_transforms_payload(
    images_dir: Path,
    output_path: Path,
    num_views: int,
    width: int,
    height: int,
    focal_length_mm: float,
    sensor_width_mm: float,
    radius: float,
    elevation_ratio: float,
) -> dict[str, Any]:
    return {
        "camera_angle_x": round(camera_angle_x_from_mm(focal_length_mm, sensor_width_mm), 6),
        "w": width,
        "h": height,
        "frames": build_orbit_frames(images_dir, output_path, num_views, radius, elevation_ratio),
    }
