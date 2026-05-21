import math
import os
from pathlib import Path
from typing import Any

from .io import write_json


def _camera_intrinsics(
    image_width: int,
    image_height: int,
    focal_length_px: float | None = None,
    fov_deg: float | None = None,
) -> dict[str, float]:
    if focal_length_px is None and fov_deg is None:
        raise ValueError("Either focal_length_px or fov_deg must be provided.")

    if focal_length_px is None:
        focal_length_px = 0.5 * image_width / math.tan(math.radians(fov_deg) / 2.0)

    camera_angle_x = 2.0 * math.atan((0.5 * image_width) / focal_length_px)
    camera_angle_y = 2.0 * math.atan((0.5 * image_height) / focal_length_px)

    return {
        "w": int(image_width),
        "h": int(image_height),
        "fl_x": float(round(focal_length_px, 6)),
        "fl_y": float(round(focal_length_px, 6)),
        "cx": float(image_width / 2.0),
        "cy": float(image_height / 2.0),
        "camera_angle_x": float(round(camera_angle_x, 6)),
        "camera_angle_y": float(round(camera_angle_y, 6)),
        "fov_deg": float(round(math.degrees(camera_angle_x), 6)),
    }


def write_nerf_transforms_json(
    output_path: Path,
    image_paths: list[Path],
    poses: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    focal_length_px: float | None = None,
    fov_deg: float | None = None,
) -> Path:
    if len(image_paths) != len(poses):
        raise ValueError("image_paths and poses must have the same length.")

    intrinsics = _camera_intrinsics(
        image_width=image_width,
        image_height=image_height,
        focal_length_px=focal_length_px,
        fov_deg=fov_deg,
    )

    frames = []
    for image_path, pose in zip(image_paths, poses):
        relative_path = os.path.relpath(image_path, output_path.parent)
        frames.append(
            {
                "file_path": Path(relative_path).as_posix(),
                "transform_matrix": pose["transform_matrix"],
                "view_index": pose["view_index"],
                "azimuth_deg": pose["azimuth_deg"],
                "elevation_deg": pose["elevation_deg"],
            }
        )

    payload = {
        **intrinsics,
        "frames": frames,
    }
    return write_json(output_path, payload)


def write_camera_pose_sidecar_json(
    output_path: Path,
    poses: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    focal_length_px: float | None = None,
    fov_deg: float | None = None,
) -> Path:
    intrinsics = _camera_intrinsics(
        image_width=image_width,
        image_height=image_height,
        focal_length_px=focal_length_px,
        fov_deg=fov_deg,
    )

    payload = {
        "camera_model": "mock_orbital_pinhole",
        "intrinsics": intrinsics,
        "poses": poses,
    }
    return write_json(output_path, payload)
