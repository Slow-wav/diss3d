#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "This script must be run from Blender, for example:\n"
        "blender your_scene.blend --background --python scripts/export_blender_real_view_transforms.py -- "
        "--images-dir /abs/path/to/real_views --output /abs/path/to/transforms.json --frame-start 1 --frame-end 96"
    ) from exc


def blender_cli_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export exact Blender camera poses to a NeRF-style transforms.json."
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing the rendered PNG images in frame order.")
    parser.add_argument("--output", required=True, help="Output transforms.json path.")
    parser.add_argument("--frame-start", type=int, required=True, help="First frame to export.")
    parser.add_argument("--frame-end", type=int, required=True, help="Last frame to export, inclusive.")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step between exported poses.")
    parser.add_argument(
        "--camera-name",
        default=None,
        help="Optional camera object name. Defaults to the active scene camera.",
    )
    parser.add_argument(
        "--file-pattern",
        default=None,
        help="Optional explicit file pattern such as '{frame:04d}.png'. If omitted, sorted PNG files from --images-dir are used.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Allow a different number of PNG files than exported frames when using --file-pattern.",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional sidecar manifest JSON path describing the export settings.",
    )
    return parser.parse_args(blender_cli_args())


def render_dimensions(scene: bpy.types.Scene) -> tuple[int, int]:
    scale = scene.render.resolution_percentage / 100.0
    width = int(round(scene.render.resolution_x * scale))
    height = int(round(scene.render.resolution_y * scale))
    return width, height


def camera_intrinsics(scene: bpy.types.Scene, camera_obj: bpy.types.Object) -> dict[str, float]:
    camera = camera_obj.data
    width, height = render_dimensions(scene)

    if camera.type != "PERSP":
        raise ValueError(f"Only perspective cameras are supported, got: {camera.type}")

    lens_mm = float(camera.lens)
    sensor_width_mm = float(camera.sensor_width)
    sensor_height_mm = float(camera.sensor_height)

    if camera.sensor_fit == "VERTICAL":
        camera_angle_y = 2.0 * math.atan(sensor_height_mm / (2.0 * lens_mm))
        fl_y = 0.5 * height / math.tan(camera_angle_y / 2.0)
        fl_x = fl_y
        camera_angle_x = 2.0 * math.atan((0.5 * width) / fl_x)
    else:
        camera_angle_x = 2.0 * math.atan(sensor_width_mm / (2.0 * lens_mm))
        fl_x = 0.5 * width / math.tan(camera_angle_x / 2.0)
        fl_y = fl_x
        camera_angle_y = 2.0 * math.atan((0.5 * height) / fl_y)

    return {
        "w": width,
        "h": height,
        "fl_x": round(float(fl_x), 6),
        "fl_y": round(float(fl_y), 6),
        "cx": round(width / 2.0, 6),
        "cy": round(height / 2.0, 6),
        "camera_angle_x": round(float(camera_angle_x), 6),
        "camera_angle_y": round(float(camera_angle_y), 6),
        "fov_deg": round(float(math.degrees(camera_angle_x)), 6),
        "lens_mm": round(lens_mm, 6),
        "sensor_width_mm": round(sensor_width_mm, 6),
        "sensor_height_mm": round(sensor_height_mm, 6),
    }


def resolve_camera(scene: bpy.types.Scene, camera_name: str | None) -> bpy.types.Object:
    if camera_name:
        camera_obj = bpy.data.objects.get(camera_name)
        if camera_obj is None:
            raise ValueError(f"Camera object not found: {camera_name}")
    else:
        camera_obj = scene.camera
        if camera_obj is None:
            raise ValueError("The active Blender scene has no camera.")

    if camera_obj.type != "CAMERA":
        raise ValueError(f"Selected object is not a camera: {camera_obj.name}")

    return camera_obj


def resolve_image_paths(images_dir: Path, frame_numbers: list[int], file_pattern: str | None) -> list[Path]:
    if file_pattern is not None:
        image_paths = [images_dir / file_pattern.format(frame=frame, index=index + 1) for index, frame in enumerate(frame_numbers)]
    else:
        image_paths = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")

    return image_paths


def matrix_rows(matrix: object) -> list[list[float]]:
    rows: list[list[float]] = []
    for row_index in range(4):
        rows.append([round(float(matrix[row_index][column_index]), 12) for column_index in range(4)])
    return rows


def relative_file_path(image_path: Path, output_path: Path) -> str:
    return Path(os.path.relpath(image_path, output_path.parent)).as_posix()


def build_frames(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    output_path: Path,
    image_paths: list[Path],
    frame_numbers: list[int],
) -> list[dict[str, object]]:
    if len(image_paths) != len(frame_numbers):
        raise ValueError(
            f"Image count ({len(image_paths)}) does not match exported frame count ({len(frame_numbers)})."
        )

    original_frame = scene.frame_current
    frames: list[dict[str, object]] = []
    try:
        for index, (frame_number, image_path) in enumerate(zip(frame_numbers, image_paths)):
            if not image_path.exists():
                raise FileNotFoundError(f"Missing rendered image for frame export: {image_path}")
            scene.frame_set(frame_number)
            frames.append(
                {
                    "file_path": relative_file_path(image_path, output_path),
                    "transform_matrix": matrix_rows(camera_obj.matrix_world),
                    "frame": int(frame_number),
                    "view_index": index,
                }
            )
    finally:
        scene.frame_set(original_frame)

    return frames


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_pose_summary(frames: list[dict[str, object]]) -> dict[str, object]:
    if not frames:
        return {
            "frame_count": 0,
            "duplicate_pose_pairs": 0,
            "unique_translation_count": 0,
            "z_bands": [],
        }

    translations: list[tuple[float, float, float]] = []
    duplicate_pose_pairs = 0
    previous_matrix: list[list[float]] | None = None

    for frame in frames:
        matrix = frame["transform_matrix"]
        translations.append(
            (
                round(float(matrix[0][3]), 6),
                round(float(matrix[1][3]), 6),
                round(float(matrix[2][3]), 6),
            )
        )
        if previous_matrix is not None and matrix == previous_matrix:
            duplicate_pose_pairs += 1
        previous_matrix = matrix

    x_values = [translation[0] for translation in translations]
    y_values = [translation[1] for translation in translations]
    z_values = [translation[2] for translation in translations]

    return {
        "frame_count": len(frames),
        "duplicate_pose_pairs": duplicate_pose_pairs,
        "unique_translation_count": len(set(translations)),
        "z_bands": sorted(set(z_values)),
        "translation_range": {
            "x": [min(x_values), max(x_values)],
            "y": [min(y_values), max(y_values)],
            "z": [min(z_values), max(z_values)],
        },
    }


def main() -> None:
    args = parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be greater than zero.")
    if args.frame_end < args.frame_start:
        raise ValueError("--frame-end must be greater than or equal to --frame-start.")

    scene = bpy.context.scene
    camera_obj = resolve_camera(scene, args.camera_name)
    output_path = Path(args.output).resolve()
    images_dir = Path(args.images_dir).resolve()
    frame_numbers = list(range(args.frame_start, args.frame_end + 1, args.frame_step))
    image_paths = resolve_image_paths(images_dir, frame_numbers, args.file_pattern)

    if not args.allow_count_mismatch and len(image_paths) != len(frame_numbers):
        raise ValueError(
            f"Resolved {len(image_paths)} image files but {len(frame_numbers)} frame poses would be exported. "
            "Either fix the image directory / frame range or use --file-pattern."
        )

    if len(image_paths) > len(frame_numbers):
        image_paths = image_paths[: len(frame_numbers)]

    intrinsics = camera_intrinsics(scene, camera_obj)
    frames = build_frames(scene, camera_obj, output_path, image_paths, frame_numbers)
    pose_summary = build_pose_summary(frames)
    payload = {
        "camera_angle_x": intrinsics["camera_angle_x"],
        "camera_angle_y": intrinsics["camera_angle_y"],
        "w": intrinsics["w"],
        "h": intrinsics["h"],
        "fl_x": intrinsics["fl_x"],
        "fl_y": intrinsics["fl_y"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "frames": frames,
    }
    write_json(output_path, payload)

    if args.manifest_output:
        manifest_path = Path(args.manifest_output).resolve()
        write_json(
            manifest_path,
            {
                "script": "export_blender_real_view_transforms.py",
                "camera_name": camera_obj.name,
                "images_dir": str(images_dir),
                "output": str(output_path),
                "frame_start": args.frame_start,
                "frame_end": args.frame_end,
                "frame_step": args.frame_step,
                "frame_count": len(frame_numbers),
                "exported_frame_count": len(frames),
                "file_pattern": args.file_pattern,
                "intrinsics": intrinsics,
                "pose_summary": pose_summary,
            },
        )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "frame_count": len(frames),
                "camera_name": camera_obj.name,
                "images_dir": str(images_dir),
                "pose_summary": pose_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
