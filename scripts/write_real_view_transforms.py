#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import (
    build_orbit_transforms_payload,
    load_object_metadata,
    resolve_object_metadata_path,
    resolve_object_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a NeRF-style transforms.json for Blender-rendered real views."
    )
    parser.add_argument(
        "--object_id",
        default=None,
        help="Object folder name under data/objects/, for example 'shark'.",
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Directory containing orbit-ordered images such as 0001.png to 0032.png.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the generated transforms JSON file.",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=32,
        help="Number of orbit views to include.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Image height in pixels.",
    )
    parser.add_argument(
        "--focal_length_mm",
        type=float,
        default=50.0,
        help="Camera focal length in millimetres.",
    )
    parser.add_argument(
        "--sensor_width_mm",
        type=float,
        default=36.0,
        help="Camera sensor width in millimetres.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Object-specific orbit distance d. Overrides metadata if provided.",
    )
    parser.add_argument(
        "--elevation_ratio",
        type=float,
        default=0.25,
        help="Z height as a ratio of radius. Camera start is (0, -radius, elevation_ratio * radius).",
    )
    parser.add_argument(
        "--number_of_orbits",
        type=int,
        default=1,
        help="Study metadata only. Records how many elevation rings are represented by this dataset.",
    )
    return parser.parse_args()


def resolve_object_paths(
    object_id: str | None,
    images_dir_arg: str | None,
    output_arg: str | None,
) -> tuple[Path, Path, Path | None]:
    if object_id is not None:
        object_root = resolve_object_root(PROJECT_ROOT, object_id)
        metadata_path = resolve_object_metadata_path(PROJECT_ROOT, object_id)
        images_dir = (object_root / "real_views_master_5ring").resolve() if images_dir_arg is None else (PROJECT_ROOT / images_dir_arg).resolve()
        output_path = (
            object_root / "real_views_master_5ring" / "transforms.json"
            if output_arg is None
            else (PROJECT_ROOT / output_arg).resolve()
        )
        return images_dir, output_path, metadata_path

    if images_dir_arg is None or output_arg is None:
        raise ValueError("Either provide --object_id, or provide both --images_dir and --output.")

    return (PROJECT_ROOT / images_dir_arg).resolve(), (PROJECT_ROOT / output_arg).resolve(), None


def resolve_camera_parameters(
    radius_arg: float | None,
    elevation_ratio_arg: float,
    metadata: dict[str, Any] | None,
) -> tuple[float, float]:
    metadata = metadata or {}

    radius = radius_arg if radius_arg is not None else metadata.get("radius")
    if radius is None:
        raise ValueError("No radius provided. Set --radius or add 'radius' to the object metadata file.")

    elevation_ratio = float(metadata.get("elevation_ratio", elevation_ratio_arg))
    if radius_arg is not None:
        radius = float(radius_arg)
    else:
        radius = float(radius)

    return radius, elevation_ratio


def main() -> None:
    args = parse_args()
    images_dir, output_path, metadata_path = resolve_object_paths(args.object_id, args.images_dir, args.output)
    metadata = load_object_metadata(metadata_path) if metadata_path is not None and metadata_path.exists() else None
    radius, elevation_ratio = resolve_camera_parameters(args.radius, args.elevation_ratio, metadata)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_orbit_transforms_payload(
        images_dir=images_dir,
        output_path=output_path,
        num_views=args.num_views,
        width=args.width,
        height=args.height,
        focal_length_mm=args.focal_length_mm,
        sensor_width_mm=args.sensor_width_mm,
        radius=radius,
        elevation_ratio=elevation_ratio,
    )
    payload["number_of_orbits"] = int(args.number_of_orbits)
    payload["views_per_orbit"] = int(args.num_views // args.number_of_orbits) if args.number_of_orbits > 0 else None

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
